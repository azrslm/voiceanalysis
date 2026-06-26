
import logging
import asyncio
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from dotenv import load_dotenv

from agents import (
    BedrockConfig,
    create_all_agents,
    AgentEvaluationResult,
    CriteriaEvaluation,
    BaseEvaluationAgent
)
from evaluation_criteria import get_total_max_score
from langchain_core.prompts import ChatPromptTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# environment variables
load_dotenv()
DEFAULT_AWS_PROFILE = os.getenv("AWS_PROFILE", "").strip()
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# Cost cutting: optionally send only the first N seconds of transcript to early-phase agents.
# Set OPENING_VERIFICATION_MAX_SECONDS=120 (or any number) to enable.
OPENING_VERIFICATION_MAX_SECONDS = float(os.getenv("OPENING_VERIFICATION_MAX_SECONDS", "0") or 0)
OPENING_VERIFICATION_FALLBACK_CHARS = int(os.getenv("OPENING_VERIFICATION_FALLBACK_CHARS", "4000") or 4000)


def _slice_transcript_to_first_seconds(transcript: str, max_seconds: float) -> str:
    """
    Slice a transcript to the first `max_seconds` based on leading timestamps "MM:SS".
    Falls back to first N chars if timestamps aren't present.
    """
    t = str(transcript or "")
    if not t.strip():
        return ""
    try:
        max_s = float(max_seconds or 0.0)
    except Exception:
        max_s = 0.0
    if max_s <= 0.0:
        return t

    lines = t.splitlines()
    out: List[str] = []
    saw_ts = False

    ts_re = re.compile(r"^\s*(\d{2}):(\d{2})\b")
    for line in lines:
        m = ts_re.match(line)
        if m:
            saw_ts = True
            try:
                mm = int(m.group(1))
                ss = int(m.group(2))
                sec = (mm * 60) + ss
            except Exception:
                sec = 0
            if sec > max_s:
                break
        out.append(line)

    sliced = "\n".join(out).strip()
    if saw_ts and sliced:
        return sliced

    # Fallback when transcript has no timestamps (still cut token usage)
    return t[: max(1, OPENING_VERIFICATION_FALLBACK_CHARS)].strip()


@dataclass
class ComprehensiveEvaluationReport:
    """Complete evaluation report combining all agent results"""
    transcript_id: str
    evaluation_timestamp: str
    detected_agent_name: str = ""
    # Agent roster enrichment (optional)
    detected_agent_canonical_name: str = ""
    detected_agent_skills: List[str] = field(default_factory=list)
    detected_agent_match_score: float = 0.0
    # Call insights (intent + issues + articulation coaching)
    call_classification: str = ""
    call_subject: str = ""
    call_issues: List[str] = field(default_factory=list)
    articulation_suggestions: Dict[str, str] = field(default_factory=dict)
    agent_results: Dict[str, AgentEvaluationResult] = field(default_factory=dict)
    # Raw totals (based on rubric score/max_score points)
    raw_total_score: float = 0.0
    raw_max_possible_score: float = 0.0
    raw_percentage_score: float = 0.0

    # Weighted overall (uses `weight` as importance; normalized to 0-100)
    weight_sum: float = 0.0
    total_score: float = 0.0
    max_possible_score: float = 0.0
    percentage_score: float = 0.0
    overall_rating: str = ""
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    # Language + translation metadata
    detected_languages: List[str] = field(default_factory=list)
    original_transcript: str = ""
    evaluation_transcript: str = ""
    was_translated: bool = False
    translation_notes: str = ""
    
    def calculate_totals(self):
        """
        Calculate totals.

        - Raw: sums rubric points (score/max_score) across criteria.
        - Weighted: overall % uses each criterion's `weight` as importance and normalizes to 0-100:
            percent = (Σ weight * (score/max_score) / Σ weight) * 100
        """
        # Raw totals (existing behavior)
        self.raw_total_score = sum(r.total_score for r in self.agent_results.values())
        self.raw_max_possible_score = sum(r.max_possible_score for r in self.agent_results.values())
        if self.raw_max_possible_score > 0:
            self.raw_percentage_score = (self.raw_total_score / self.raw_max_possible_score) * 100
        else:
            self.raw_percentage_score = 0.0

        # Weighted overall percentage
        weighted_sum = 0.0
        weight_sum = 0.0
        for r in (self.agent_results or {}).values():
            for e in getattr(r, "evaluations", []) or []:
                try:
                    w = float(getattr(e, "weight", 0.0) or 0.0)
                    score = float(getattr(e, "score", 0.0) or 0.0)
                    max_score = float(getattr(e, "max_score", 0.0) or 0.0)
                except Exception:
                    continue

                if w <= 0.0 or max_score == 0.0:
                    continue

                weighted_sum += w * (score / max_score)
                weight_sum += w

        self.weight_sum = weight_sum

        if weight_sum > 0.0:
            weighted_percent = (weighted_sum / weight_sum) * 100.0
            # Present weighted score as 0-100 points for UI consistency
            self.total_score = weighted_percent
            self.max_possible_score = 100.0
            self.percentage_score = weighted_percent
        else:
            # Fallback to raw if weights are missing
            self.total_score = self.raw_total_score
            self.max_possible_score = self.raw_max_possible_score
            self.percentage_score = self.raw_percentage_score
        
      
        if self.percentage_score >= 90:
            self.overall_rating = "Excellent"
        elif self.percentage_score >= 80:
            self.overall_rating = "Good"
        elif self.percentage_score >= 70:
            self.overall_rating = "Satisfactory"
        elif self.percentage_score >= 60:
            self.overall_rating = "Needs Improvement"
        else:
            self.overall_rating = "Unsatisfactory"
    
    def identify_strengths_and_improvements(self):
        """Identify strengths and areas for improvement"""
        self.strengths = []
        self.areas_for_improvement = []
        
        for agent_name, result in self.agent_results.items():
            for evaluation in result.evaluations:
                if evaluation.rating in ["Desirable", "Expected"]:
                    if evaluation.evidence:
                        self.strengths.append(
                            f"{evaluation.criteria_name}: {evaluation.reasoning}"
                        )
                elif evaluation.rating in ["Undesirable", "Basic"]:
                    self.areas_for_improvement.append(
                        f"{evaluation.criteria_name}: {evaluation.reasoning}"
                    )
    
    def generate_summary(self):
        """Generate a concise (~200 words) executive summary of the evaluation."""

        def _truncate_words(text: str, max_words: int = 200) -> str:
            words = (text or "").split()
            if len(words) <= max_words:
                return text.strip()
            return " ".join(words[:max_words]).strip() + "…"

        # Flatten all criteria evaluations across agents
        all_evals: List[CriteriaEvaluation] = []
        for r in (self.agent_results or {}).values():
            all_evals.extend(getattr(r, "evaluations", []) or [])

        # Strengths: full-score criteria (prefer higher max_score)
        strengths = [e for e in all_evals if float(getattr(e, "score", 0.0)) >= float(getattr(e, "max_score", 0.0)) - 1e-9]
        strengths.sort(key=lambda e: float(getattr(e, "max_score", 0.0)), reverse=True)
        top_strengths = [e.criteria_name for e in strengths[:3]]

        # Improvements: largest score gaps
        gaps = []
        for e in all_evals:
            try:
                gap = float(e.max_score) - float(e.score)
            except Exception:
                gap = 0.0
            if gap > 1e-6:
                gaps.append((gap, e))
        gaps.sort(key=lambda x: x[0], reverse=True)
        top_improvements = [e.criteria_name for _, e in gaps[:3]]

        agent_name = (self.detected_agent_name or "").strip()
        agent_part = f" Agent identified: {agent_name}." if agent_name else ""

        parts: List[str] = []
        parts.append(
            f"Call evaluation for {self.transcript_id} completed on {self.evaluation_timestamp}.{agent_part} "
            f"Overall score is {self.total_score:.2f}/{self.max_possible_score:.2f} ({self.percentage_score:.1f}%), "
            f"resulting in an overall rating of {self.overall_rating}."
        )

        if top_strengths:
            parts.append(
                "Key strengths observed were: "
                + ", ".join(top_strengths)
                + "."
            )
        else:
            parts.append("Key strengths could not be strongly evidenced across high-weight criteria in this call.")

        if top_improvements:
            parts.append(
                "Largest improvement opportunities were: "
                + ", ".join(top_improvements)
                + "."
            )

        parts.append(
            "Recommended focus for coaching: reinforce a consistent opening and verification flow, "
            "maintain professional tonality and active listening, explain holds and next steps clearly, "
            "and close by confirming resolution and inviting any further questions. "
            "Track progress by re-evaluating the same criteria on the next calls and aiming to move "
            "from Expected to Desirable where applicable."
        )

        self.summary = _truncate_words(" ".join(parts), max_words=200)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "transcript_id": self.transcript_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "detected_agent_name": self.detected_agent_name,
            "detected_agent_canonical_name": self.detected_agent_canonical_name,
            "detected_agent_skills": self.detected_agent_skills,
            "detected_agent_match_score": self.detected_agent_match_score,
            "call_classification": self.call_classification,
            "call_subject": self.call_subject,
            "call_issues": self.call_issues,
            "articulation_suggestions": self.articulation_suggestions,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
            "percentage_score": self.percentage_score,
            "overall_rating": self.overall_rating,
            "summary": self.summary,
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
            "detected_languages": self.detected_languages,
            "original_transcript": self.original_transcript,
            "evaluation_transcript": self.evaluation_transcript,
            "was_translated": self.was_translated,
            "translation_notes": self.translation_notes,
            "agent_results": {
                name: {
                    "agent_name": result.agent_name,
                    "total_score": result.total_score,
                    "max_possible_score": result.max_possible_score,
                    "evaluations": [
                        {
                            "criteria_id": e.criteria_id,
                            "criteria_name": e.criteria_name,
                            "rating": e.rating,
                            "score": e.score,
                            "max_score": e.max_score,
                            "weight": e.weight,
                            "evidence": e.evidence,
                            "reasoning": e.reasoning
                        }
                        for e in result.evaluations
                    ]
                }
                for name, result in self.agent_results.items()
            }
        }


class EvaluationOrchestrator:
    """
    ✨ AI-Powered Orchestrator for coordinating multiple evaluation agents
    
    This orchestrator manages the evaluation workflow:
    1. Receives transcript from the transcription service
    2. Dispatches transcript to specialized agents
    3. Collects and aggregates results
    4. Generates comprehensive evaluation report
    """
    
    def __init__(self, config: BedrockConfig):
        """Initialize the orchestrator with AWS Bedrock configuration"""
        self.config = config
        self.agents: Dict[str, BaseEvaluationAgent] = {}
        self._initialized = False
        logger.info("✨ EvaluationOrchestrator initialized")
    
    def initialize_agents(self):
        """Initialize all evaluation agents"""
        if not self._initialized:
            logger.info("✨ Initializing evaluation agents...")
            self.agents = create_all_agents(self.config)
            self._initialized = True
            logger.info(f"✨ {len(self.agents)} agents initialized successfully")
    
    def evaluate_transcript(
        self, 
        transcript: str, 
        transcript_id: Optional[str] = None,
        parallel: bool = True
    ) -> ComprehensiveEvaluationReport:
        """
        Evaluate a transcript using all specialized agents
        
        Args:
            transcript: The call transcript to evaluate
            transcript_id: Optional identifier for the transcript
            parallel: Whether to run agents in parallel (default True)
        
        Returns:
            ComprehensiveEvaluationReport with all evaluation results
        """
    
        self.initialize_agents()
        
     
        if transcript_id is None:
            transcript_id = f"EVAL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info(f"✨ Starting evaluation for transcript: {transcript_id}")
        
      
        report = ComprehensiveEvaluationReport(
            transcript_id=transcript_id,
            evaluation_timestamp=datetime.now().isoformat(),
        )

        # Detect agent name + roster skills BEFORE evaluation, so downstream agents can use SME context.
        try:
            report.detected_agent_name = self._extract_agent_name(transcript)
        except Exception as e:
            logger.error(f"Error extracting agent name: {e}")
            report.detected_agent_name = "Unknown"

        # Optional: enrich agent name with domain skills from roster CSV/XLSX/S3
        try:
            from agent_roster import load_agent_roster, match_agent_name, rank_agent_candidates

            roster = load_agent_roster()
            entry, score = match_agent_name(report.detected_agent_name, roster)
            if entry is not None:
                report.detected_agent_canonical_name = entry.name
                report.detected_agent_skills = list(entry.skills)
                report.detected_agent_match_score = float(score or 0.0)
            else:
                # Debug: show top candidates to understand why matching failed (e.g., wrong sheet/path or ambiguity).
                try:
                    candidates = rank_agent_candidates(report.detected_agent_name, roster, top_n=5)
                    if candidates:
                        cand_str = ", ".join(
                            f"{e.name}({s:.2f})" for s, e in candidates
                        )
                        logger.info(
                            "Agent roster: no confident match for '%s'. Top candidates: %s",
                            report.detected_agent_name,
                            cand_str,
                        )
                    else:
                        logger.info(
                            "Agent roster: no confident match for '%s' (no candidates).",
                            report.detected_agent_name,
                        )
                except Exception:
                    pass
        except Exception:
            # Never fail evaluation because roster mapping is missing/broken.
            pass

        try:
            roster_path = os.getenv("AGENT_ROSTER_PATH", "agent_roster.csv")
        except Exception:
            roster_path = "agent_roster.csv"

        logger.info(
            "Agent roster context: extracted='%s' canonical='%s' skills=%s match_score=%.3f roster='%s'",
            report.detected_agent_name,
            report.detected_agent_canonical_name or "",
            ",".join(report.detected_agent_skills or []),
            float(report.detected_agent_match_score or 0.0),
            roster_path,
        )

        ctx = {
            "agent_extracted_name": report.detected_agent_name,
            "agent_canonical_name": report.detected_agent_canonical_name or report.detected_agent_name,
            "agent_skills": report.detected_agent_skills or [],
        }
        
        if parallel:
            report.agent_results = self._evaluate_parallel(transcript, context=ctx)
        else:
            report.agent_results = self._evaluate_sequential(transcript, context=ctx)
        
        report.calculate_totals()
        report.identify_strengths_and_improvements()
        report.generate_summary()

        # ------------------------------------------------------------------
        # Call insights: intent classification + customer issues + articulation coaching
        # ------------------------------------------------------------------
        try:
            from agents import CallInsightsAgent

            # Prefer a full-context agent LLM (avoid early-phase overrides)
            preferred_agent = (
                self.agents.get("soft_skills")
                or self.agents.get("enquiry_resolution")
                or self.agents.get("wrap_up")
                or next(iter(self.agents.values()))
            )
            llm = preferred_agent.llm

            # Compact issues: only Basic/Undesirable criteria with minimal evidence
            evaluation_issues: List[Dict[str, Any]] = []
            for _agent_key, res in (report.agent_results or {}).items():
                cat_name = getattr(res, "agent_name", _agent_key) or _agent_key
                for ev in getattr(res, "evaluations", []) or []:
                    rating = str(getattr(ev, "rating", "") or "")
                    if rating not in ("Basic", "Undesirable"):
                        continue
                    evidence = getattr(ev, "evidence", None) or []
                    if isinstance(evidence, list):
                        evidence = evidence[:1]
                    else:
                        evidence = [str(evidence)]
                    evaluation_issues.append(
                        {
                            "category": cat_name,
                            "criterion": getattr(ev, "criteria_name", "") or "",
                            "rating": rating,
                            "evidence": evidence,
                        }
                    )

            agent_skills_str = ", ".join(report.detected_agent_skills or []) or "Unknown"
            insights_agent = CallInsightsAgent(llm)
            insights = insights_agent.analyze(
                transcript=transcript,
                agent_extracted_name=report.detected_agent_name or "Unknown",
                agent_canonical_name=(report.detected_agent_canonical_name or report.detected_agent_name or "Unknown"),
                agent_skills=agent_skills_str,
                evaluation_issues=evaluation_issues,
            )

            report.call_classification = str((insights or {}).get("classification") or "").strip()
            report.call_subject = str((insights or {}).get("subject") or "").strip()
            report.call_issues = (insights or {}).get("issues") or []
            if not isinstance(report.call_issues, list):
                report.call_issues = [str(report.call_issues)]

            report.articulation_suggestions = (insights or {}).get("articulation_suggestions") or {}
            if not isinstance(report.articulation_suggestions, dict):
                report.articulation_suggestions = {}

            logger.info(
                "Call insights: classification='%s' subject='%s' issues=%d",
                report.call_classification,
                (report.call_subject[:80] + "…") if report.call_subject and len(report.call_subject) > 80 else report.call_subject,
                len(report.call_issues or []),
            )
        except Exception as e:
            logger.warning(f"Call insights agent failed: {type(e).__name__}: {e}")
        
        logger.info(f"✨ Evaluation complete. Score: {report.percentage_score:.1f}%")
        
        return report
    
    def _evaluate_parallel(self, transcript: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, AgentEvaluationResult]:
        """Run all agents in parallel"""
        results = {}

        # Optional per-agent transcript shortening for cost cutting
        early_transcript = (
            _slice_transcript_to_first_seconds(transcript, OPENING_VERIFICATION_MAX_SECONDS)
            if OPENING_VERIFICATION_MAX_SECONDS > 0
            else transcript
        )
        agent_transcripts = {
            "opening_greeting": early_transcript,
            "verification": early_transcript,
        }
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
      
            future_to_agent = {}
            for agent_name, agent in self.agents.items():
                t = agent_transcripts.get(agent_name, transcript)
                future_to_agent[executor.submit(agent.evaluate, t, context)] = agent_name
            
       
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    logger.info(f"✨ {agent_name} agent completed")
                except Exception as e:
                    logger.error(f"✨ Error in {agent_name} agent: {str(e)}")
               
                    results[agent_name] = self.agents[agent_name]._create_default_result()
        
        return results
    
    def _evaluate_sequential(self, transcript: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, AgentEvaluationResult]:
        """Run all agents sequentially"""
        results = {}

        early_transcript = (
            _slice_transcript_to_first_seconds(transcript, OPENING_VERIFICATION_MAX_SECONDS)
            if OPENING_VERIFICATION_MAX_SECONDS > 0
            else transcript
        )
        
        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"✨ Running {agent_name} agent...")
                if agent_name in ("opening_greeting", "verification"):
                    result = agent.evaluate(early_transcript, context)
                else:
                    result = agent.evaluate(transcript, context)
                results[agent_name] = result
            except Exception as e:
                logger.error(f"✨ Error in {agent_name} agent: {str(e)}")
                results[agent_name] = agent._create_default_result()
        
        return results
    
    def evaluate_specific_category(
        self, 
        transcript: str, 
        category: str,
        transcript_id: Optional[str] = None
    ) -> Optional[AgentEvaluationResult]:
        """
        Evaluate a transcript using only a specific agent category
        
        Args:
            transcript: The call transcript to evaluate
            category: The agent category to use (e.g., 'soft_skills', 'verification')
            transcript_id: Optional identifier for the transcript
        
        Returns:
            AgentEvaluationResult from the specified agent
        """
        self.initialize_agents()
        
        if category not in self.agents:
            logger.error(f"Unknown agent category: {category}")
            return None
        
        logger.info(f"✨ Running {category} evaluation...")
        return self.agents[category].evaluate(transcript, None)

    def _extract_agent_name(self, transcript: str) -> str:
        """
        Use the underlying LLM to extract the call center agent's name
        from the transcript, if mentioned.
        
        Enhanced version with:
        - Pre-filtering to CSO/Agent lines only
        - Better prompt with more greeting pattern examples
        - Improved error handling
        """
        if not self.agents:
            return "Unknown"

        # Pre-filter to only include CSO/Agent lines for more focused extraction
        cso_lines = []
        for line in transcript.split('\n'):
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['**cso**:', 'cso:', '**agent**:', 'agent:', 'spk_0']):
                cso_lines.append(line)
        
        # Focus on first 10 lines where introduction usually happens
        intro_section = '\n'.join(cso_lines[:10]) if cso_lines else transcript[:2000]
        
        if not intro_section.strip():
            return "Unknown"

        # Prefer a "full-context" agent for name extraction so optional cheap early-phase models
        # (opening/verification) don't accidentally degrade extraction quality.
        preferred_agent = (
            self.agents.get("soft_skills")
            or self.agents.get("enquiry_resolution")
            or self.agents.get("wrap_up")
            or next(iter(self.agents.values()))
        )
        llm = preferred_agent.llm

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert at extracting the CALL CENTER AGENT's name from call transcripts.\n\n"
                        "IMPORTANT RULES:\n"
                        "- The agent is the CSO (Customer Service Officer), NOT the customer/caller.\n"
                        "- Agent names typically appear in the opening greeting.\n"
                        "- Look for these common introduction patterns:\n"
                        "  * 'Good morning/afternoon, this is [NAME]'\n"
                        "  * 'Thank you for calling [COMPANY], my name is [NAME]'\n"
                        "  * 'Thank you for calling, I'm [NAME]'\n"
                        "  * '[COMPANY] hotline, [NAME] speaking'\n"
                        "  * 'Hi, this is [NAME] from [COMPANY]'\n"
                        "  * 'Hello, [NAME] here, how may I help you?'\n"
                        "  * 'You're speaking with [NAME]'\n"
                        "  * 'My name is [NAME], how can I assist you?'\n\n"
                        "OUTPUT FORMAT:\n"
                        "- Return ONLY the agent's name (first name, or full name if available).\n"
                        "- Do NOT include titles like 'Mr.', 'Ms.', 'CSO', etc.\n"
                        "- Do NOT include the company name.\n"
                        "- If the name is unclear or not mentioned, respond with exactly: Unknown\n"
                        "- Examples of valid responses: 'Sarah', 'John Tan', 'Mary Lee', 'Unknown'\n"
                    ),
                ),
                ("human", "Extract the agent's name from this transcript excerpt:\n\n{transcript}"),
            ]
        )

        try:
            response = (prompt | llm).invoke({"transcript": intro_section})
            name_raw = getattr(response, "content", str(response)).strip()
            first_line = name_raw.splitlines()[0].strip().strip('"').strip("'").strip()

            # Validation checks
            if not first_line:
                return "Unknown"
            if "unknown" in first_line.lower():
                return "Unknown"
            if len(first_line) > 50:  # Name shouldn't be too long
                return "Unknown"
            if any(word in first_line.lower() for word in ['customer', 'caller', 'policyholder', 'ph', 'sorry', 'cannot', 'not mentioned', 'no name']):
                return "Unknown"
            
            # Clean up any remaining artifacts
            name_clean = first_line.replace('CSO:', '').replace('Agent:', '').strip()
            
            return name_clean if name_clean else "Unknown"
            
        except Exception as e:
            logger.warning(f"Failed to extract agent name: {e}")
            return "Unknown"


def create_orchestrator(
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    model_id: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> EvaluationOrchestrator:
    """
    Factory function to create an EvaluationOrchestrator
    
    Args:
        profile_name: Deprecated. Kept for backward compatibility.
        region_name: AWS Region (falls back to env AWS_REGION or ap-southeast-1)
        model_id: Bedrock model ID (falls back to env BEDROCK_MODEL_ID)
        aws_access_key_id: IAM access key ID
        aws_secret_access_key: IAM secret access key
    
    Returns:
        Configured EvaluationOrchestrator instance
    """
    profile = profile_name or DEFAULT_AWS_PROFILE
    region = region_name or DEFAULT_AWS_REGION or "ap-southeast-1"
    model = model_id or DEFAULT_BEDROCK_MODEL_ID
    access_key = (aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", "")).strip()
    secret_key = (aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")).strip()

    # Diagnostics: helps catch accidental imports of a different `agents` module.
    try:
        import agents as _agents_mod  # local file should win; if not, log where it's coming from
        logger.info(f"Using agents module from: {getattr(_agents_mod, '__file__', 'unknown')}")
    except Exception:
        pass

    # Be robust to older/mismatched BedrockConfig definitions that may not accept kwargs.
    try:
        config = BedrockConfig(
            profile_name=profile,
            region_name=region,
            model_id=model,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )
    except TypeError as e:
        logger.warning(f"BedrockConfig constructor rejected kwargs ({e}); using fallback config object.")
        try:
            # Try creating and setting attrs (may fail if class uses __slots__=())
            cfg = BedrockConfig()
            try:
                setattr(cfg, "profile_name", profile)
                setattr(cfg, "region_name", region)
                setattr(cfg, "model_id", model)
            except Exception:
                pass

            # Ensure required attrs exist; else fallback to SimpleNamespace
            if not all(hasattr(cfg, k) for k in ("profile_name", "region_name", "model_id")):
                raise AttributeError("BedrockConfig instance missing required attributes")
            config = cfg
        except Exception:
            from types import SimpleNamespace
            config = SimpleNamespace(
                profile_name=profile,
                region_name=region,
                model_id=model,
                aws_access_key_id=access_key or None,
                aws_secret_access_key=secret_key or None,
            )
    
    return EvaluationOrchestrator(config)

class DetailedReportGenerator:
    """Generates detailed HTML and markdown reports from evaluation results"""
    
    @staticmethod
    def generate_markdown_report(report: ComprehensiveEvaluationReport) -> str:
        """Generate a detailed markdown report"""
        md = f"""# ✨ AI-Generated Call Quality Evaluation Report

## Overview
- **Transcript ID:** {report.transcript_id}
- **Evaluation Date:** {report.evaluation_timestamp}
- **Overall Score:** {report.total_score:.2f} / {report.max_possible_score:.2f}
- **Percentage:** {report.percentage_score:.1f}%
- **Overall Rating:** {report.overall_rating}

---

## Detailed Evaluation by Category

"""
        for agent_name, result in report.agent_results.items():
            agent_pct = (result.total_score / result.max_possible_score * 100) if result.max_possible_score > 0 else 0
            md += f"### {result.agent_name}\n"
            md += f"**Score:** {result.total_score:.2f} / {result.max_possible_score:.2f} ({agent_pct:.1f}%)\n\n"
            
            md += "| Criteria | Rating | Score | Evidence |\n"
            md += "|----------|--------|-------|----------|\n"
            
            for eval in result.evaluations:
                evidence_str = "; ".join(eval.evidence[:2]) if eval.evidence else "N/A"
                if len(evidence_str) > 100:
                    evidence_str = evidence_str[:97] + "..."
                md += f"| {eval.criteria_name} | {eval.rating} | {eval.score:.2f}/{eval.max_score:.2f} | {evidence_str} |\n"
            
            md += "\n"
        
        
        if report.strengths:
            md += "## Strengths\n\n"
            for strength in report.strengths:
                md += f"- ✓ {strength}\n"
            md += "\n"
        
        
        if report.areas_for_improvement:
            md += "## Areas for Improvement\n\n"
            for area in report.areas_for_improvement:
                md += f"- ⚠ {area}\n"
            md += "\n"
        
        md += "---\n*✨ This report was generated by AI using AWS Bedrock Claude Sonnet 3.5*\n"
        
        return md

    @staticmethod
    def generate_excel_report(
        report: ComprehensiveEvaluationReport,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> io.BytesIO:
        """Generate an Excel report with call insights + criteria details."""
        import pandas as pd
        rows: List[Dict[str, Any]] = []

        def _short_criteria_id(criteria_id: str) -> str:
            return criteria_id.split(".", 1)[1] if isinstance(criteria_id, str) and "." in criteria_id else (criteria_id or "")

        def _format_timestamp(seconds: float) -> str:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        def _format_time_range(start: float, end: float) -> str:
            return f"{_format_timestamp(start)} → {_format_timestamp(end)}"

        def _normalize_text(t: str) -> str:
            if not t:
                return ""
            t = t.lower()
            return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t).split())

        def _find_timeframe_for_evidence(ev_text: str) -> Optional[str]:
            if not ev_text or not segments:
                return None
            ev_norm = _normalize_text(ev_text)
            if not ev_norm:
                return None
            for seg in segments:
                seg_text = seg.get("text", "")
                seg_norm = _normalize_text(seg_text)
                if ev_norm and seg_norm and ev_norm in seg_norm:
                    try:
                        start = float(seg.get("start", 0.0))
                        end = float(seg.get("end", start))
                        return _format_time_range(start, end)
                    except Exception:
                        return None
            return None

        for agent_key, result in report.agent_results.items():
            for eval in result.evaluations:
                evidence_str = " | ".join(eval.evidence) if getattr(eval, "evidence", None) else ""
                timeframes = []
                for ev_txt in (getattr(eval, "evidence", None) or []):
                    tf = _find_timeframe_for_evidence(ev_txt)
                    timeframes.append(tf or "")
                evidence_timeframe = " | ".join([t for t in timeframes if t]) if any(timeframes) else ""
                rows.append(
                    {
                        "Agent Category": result.agent_name,
                        "Criteria ID": _short_criteria_id(eval.criteria_id),
                        "Criteria Name": eval.criteria_name,
                        "Rating": eval.rating,
                        "Weight (%)": eval.weight,
                        "Score": eval.score,
                        "Max Score": eval.max_score,
                        "Evidence": evidence_str,
                        "Evidence Timeframe": evidence_timeframe,
                        "AI Remarks": eval.reasoning or "",
                    }
                )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Summary sheet (includes call intent + roster mapping)
            summary_rows = [
                {
                    "Transcript ID": report.transcript_id,
                    "Evaluation Timestamp": report.evaluation_timestamp,
                    "Overall Score": report.total_score,
                    "Overall %": report.percentage_score,
                    "Overall Rating": report.overall_rating,
                    "Detected Agent Name": report.detected_agent_name,
                    "Matched Agent Name": report.detected_agent_canonical_name,
                    "Agent Skills (LI/GI/HI)": ", ".join(report.detected_agent_skills or []),
                    "Name Match Score": report.detected_agent_match_score,
                    "Call Classification": report.call_classification,
                    "Call Subject": report.call_subject,
                    "Customer Issues": " | ".join(report.call_issues or []),
                }
            ]
            pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")

            # Evaluation sheet (per-criterion)
            df = pd.DataFrame(rows)
            df.to_excel(writer, index=False, sheet_name="Evaluation")
        output.seek(0)
        return output
    
    @staticmethod
    def generate_html_report(report: ComprehensiveEvaluationReport) -> str:
        """Generate a detailed HTML report"""
        
        rating_colors = {
            "Excellent": "#28a745",
            "Good": "#17a2b8",
            "Satisfactory": "#ffc107",
            "Needs Improvement": "#fd7e14",
            "Unsatisfactory": "#dc3545"
        }
        rating_color = rating_colors.get(report.overall_rating, "#6c757d")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ Call Quality Evaluation Report</title>
    <style>
        :root {{
            --primary-color: #1a1a2e;
            --secondary-color: #16213e;
            --accent-color: #0f3460;
            --highlight-color: #e94560;
            --text-color: #eee;
            --rating-color: {rating_color};
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: var(--text-color);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, var(--highlight-color), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 8px solid var(--rating-color);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: 1rem auto;
            background: rgba(255,255,255,0.05);
        }}
        
        .score-circle .percentage {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--rating-color);
        }}
        
        .score-circle .rating {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 1rem;
            opacity: 0.8;
        }}
        
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .section h2 {{
            color: var(--highlight-color);
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }}
        
        .agent-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        
        .agent-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .agent-score {{
            background: var(--accent-color);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            background: rgba(255,255,255,0.05);
            font-weight: 600;
        }}
        
        .rating-badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }}
        
        .rating-desirable {{ background: #28a745; }}
        .rating-expected {{ background: #17a2b8; }}
        .rating-basic {{ background: #ffc107; color: #333; }}
        .rating-undesirable {{ background: #dc3545; }}
        
        .list-section ul {{
            list-style: none;
        }}
        
        .list-section li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        .list-section li::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        
        .strengths li::before {{ background: #28a745; }}
        .improvements li::before {{ background: #ffc107; }}
        
        .footer {{
            text-align: center;
            margin-top: 2rem;
            opacity: 0.6;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ Call Quality Evaluation Report</h1>
            <div class="score-circle">
                <span class="percentage">{report.percentage_score:.0f}%</span>
                <span class="rating">{report.overall_rating}</span>
            </div>
            <div class="meta-info">
                <span>📋 ID: {report.transcript_id}</span>
                <span>📅 {report.evaluation_timestamp}</span>
                <span>📊 Score: {report.total_score:.2f}/{report.max_possible_score:.2f}</span>
            </div>
        </div>
"""
        
       
        html += '<div class="section"><h2>📊 Detailed Evaluation by Category</h2>'
        
        for agent_name, result in report.agent_results.items():
            agent_pct = (result.total_score / result.max_possible_score * 100) if result.max_possible_score > 0 else 0
            html += f'''
            <div class="agent-card">
                <div class="agent-header">
                    <h3>{result.agent_name}</h3>
                    <span class="agent-score">{result.total_score:.2f}/{result.max_possible_score:.2f} ({agent_pct:.0f}%)</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Criteria</th>
                            <th>Rating</th>
                            <th>Score</th>
                            <th>Evidence</th>
                        </tr>
                    </thead>
                    <tbody>
'''
            for eval in result.evaluations:
                rating_class = f"rating-{eval.rating.lower()}"
                evidence_str = "; ".join(eval.evidence[:2]) if eval.evidence else "N/A"
                if len(evidence_str) > 80:
                    evidence_str = evidence_str[:77] + "..."
                
                html += f'''
                        <tr>
                            <td>{eval.criteria_name}</td>
                            <td><span class="rating-badge {rating_class}">{eval.rating}</span></td>
                            <td>{eval.score:.2f}/{eval.max_score:.2f}</td>
                            <td>{evidence_str}</td>
                        </tr>
'''
            html += '</tbody></table></div>'
        
        html += '</div>'
        
       
        if report.strengths:
            html += '<div class="section list-section strengths"><h2>✅ Strengths</h2><ul>'
            for strength in report.strengths[:5]:
                html += f'<li>{strength}</li>'
            html += '</ul></div>'
        
      
        if report.areas_for_improvement:
            html += '<div class="section list-section improvements"><h2>⚠️ Areas for Improvement</h2><ul>'
            for area in report.areas_for_improvement[:5]:
                html += f'<li>{area}</li>'
            html += '</ul></div>'
        
        html += '''
        <div class="footer">
            <p>✨ This report was generated by AI using AWS Bedrock Claude Sonnet 3.5</p>
        </div>
    </div>
</body>
</html>
'''
        
        return html

