
import json
import logging
import os
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from evaluation_criteria import EVALUATION_CRITERIA, AGENT_ASSIGNMENTS, get_criteria_by_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for default credential/model values
load_dotenv()
DEFAULT_AWS_PROFILE = os.getenv("AWS_PROFILE", "").strip()
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20240620-v1:0"
)

# Optional per-agent model overrides (for cost cutting)
# Examples:
#   EARLY_PHASE_BEDROCK_MODEL_ID=anthropic.claude-3-haiku-...
#   OPENING_GREETING_BEDROCK_MODEL_ID=...
#   VERIFICATION_BEDROCK_MODEL_ID=...
EARLY_PHASE_BEDROCK_MODEL_ID = os.getenv("EARLY_PHASE_BEDROCK_MODEL_ID", "").strip()
OPENING_GREETING_BEDROCK_MODEL_ID = os.getenv("OPENING_GREETING_BEDROCK_MODEL_ID", "").strip()
VERIFICATION_BEDROCK_MODEL_ID = os.getenv("VERIFICATION_BEDROCK_MODEL_ID", "").strip()

# ============================================================================
# AWS Bedrock Configuration
# ============================================================================

class BedrockConfig:
    """Configuration for AWS Bedrock"""
    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        model_id: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.profile_name = profile_name or DEFAULT_AWS_PROFILE
        self.region_name = region_name or DEFAULT_AWS_REGION
        self.model_id = model_id or DEFAULT_BEDROCK_MODEL_ID
        self.aws_access_key_id = (aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", "")).strip()
        self.aws_secret_access_key = (aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")).strip()


def create_bedrock_llm(config: BedrockConfig) -> ChatBedrock:
    """Create a ChatBedrock LLM instance."""
    import boto3
    
    profile = (getattr(config, "profile_name", None) or "").strip() or None
    region = (getattr(config, "region_name", None) or "").strip() or (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "").strip() or None
    access_key = (getattr(config, "aws_access_key_id", None) or "").strip()
    secret_key = (getattr(config, "aws_secret_access_key", None) or "").strip()

    if access_key and secret_key:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    elif profile:
        try:
            session = boto3.Session(profile_name=profile, region_name=region)
        except Exception:
            session = boto3.Session(region_name=region)
    else:
        session = boto3.Session(region_name=region)

    # SSL verification is strongly recommended in AWS.
    # Set AWS_SSL_VERIFY=false only in non-prod environments if you truly need to.
    ssl_verify_env = (os.getenv("AWS_SSL_VERIFY", "true") or "true").strip().lower()
    ssl_verify = ssl_verify_env not in {"0", "false", "no", "n"}

    bedrock_client = session.client("bedrock-runtime", region_name=region, verify=ssl_verify)
    
    return ChatBedrock(
        model_id=config.model_id,
        region_name=region,
        client=bedrock_client,  # Pass custom client with SSL disabled
        model_kwargs={
            # Lower temperature improves repeatability/stability of ratings/scores across runs.
            "temperature": 0.0,
            "max_tokens": 4096
        }
    )


# ============================================================================
# Evaluation Result Models
# ============================================================================

class Rating(str, Enum):
    UNDESIRABLE = "Undesirable"
    BASIC = "Basic"
    EXPECTED = "Expected"
    DESIRABLE = "Desirable"


@dataclass
class CriteriaEvaluation:
    """Evaluation result for a single criterion"""
    criteria_id: str
    criteria_name: str
    rating: str
    score: float
    max_score: float
    weight: float
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class AgentEvaluationResult:
    """Complete evaluation result from an agent"""
    agent_name: str
    evaluations: List[CriteriaEvaluation] = field(default_factory=list)
    total_score: float = 0.0
    max_possible_score: float = 0.0
    
    def calculate_totals(self):
        """Calculate totals using Python math, ensuring accuracy."""
        # Use math.fsum for precise floating point summation if needed, 
        # but built-in sum is generally sufficient for this scale.
        self.total_score = sum(e.score for e in self.evaluations)
        self.max_possible_score = sum(e.max_score for e in self.evaluations)


# ============================================================================
# Pydantic Models for Output Parsing
# ============================================================================

class SingleCriteriaOutput(BaseModel):
    """Output format for a single criteria evaluation"""
    criteria_id: str = Field(description="The ID of the criteria being evaluated")
    rating: str = Field(description="The rating: Undesirable, Basic, Expected, or Desirable")
    evidence: List[str] = Field(description="List of specific quotes or observations from transcript supporting the rating")
    reasoning: str = Field(description="Explanation of why this rating was given")


class AgentOutput(BaseModel):
    """Output format for an agent's evaluation"""
    evaluations: List[SingleCriteriaOutput] = Field(description="List of evaluations for each criteria")


class ImprovementSummaryOutput(BaseModel):
    """Output format for improvement coaching summary"""
    summary: str = Field(description="A concise, intuitive coaching summary for improvement (no per-criterion checklist).")


class CallInsightsOutput(BaseModel):
    """
    Output format for call intent + issues + articulation coaching.
    """
    classification: str = Field(description="One of: Policy, Claim, Product Related, Complaints, Follow Up, Others")
    subject: str = Field(description="Short subject line (<= 12 words) describing what the call is about")
    issues: List[str] = Field(description="Bullet list of the customer's issues/questions raised (3-8 items)")
    articulation_suggestions: Dict[str, str] = Field(
        description=(
            "Coaching suggestions for better articulation per evaluation agent category. Keys must be:\n"
            "opening_greeting, verification, soft_skills, enquiry_resolution, cross_selling, wrap_up.\n"
            "Each value should be 2-4 concise bullet points (as a single string)."
        )
    )

# ============================================================================
# Base Agent Class
# ============================================================================

class BaseEvaluationAgent:
    """Base class for all evaluation agents"""
    
    def __init__(
        self,
        name: str,
        criteria_paths: List[str],
        llm: ChatBedrock
    ):
        self.name = name
        self.criteria_paths = criteria_paths
        self.llm = llm
        self.criteria_details = self._load_criteria_details()
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser(pydantic_object=AgentOutput)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return t

    @staticmethod
    def _extract_json_object(text: str) -> str:
        t = (text or "").strip()
        start = t.find("{")
        end = t.rfind("}") + 1
        if start != -1 and end > start:
            return t[start:end]
        return ""

    @staticmethod
    def _repair_common_json_issues(text: str) -> str:
        """
        Best-effort JSON sanitizer for common LLM formatting issues.
        This won't fix everything, but it improves success rate without extra model calls.
        """
        t = (text or "").strip()
        # Normalize smart quotes
        t = t.replace("“", "\"").replace("”", "\"").replace("’", "'")
        # Remove trailing commas: {...,} or [...,]
        t = re.sub(r",\s*([}\]])", r"\1", t)
        # Python literals -> JSON
        t = re.sub(r"\bTrue\b", "true", t)
        t = re.sub(r"\bFalse\b", "false", t)
        t = re.sub(r"\bNone\b", "null", t)
        return t

    def _json_repair_via_llm(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Last-resort: ask the LLM to rewrite its own output into valid JSON.
        We only call this when local parsing fails, because it costs extra tokens.
        """
        try:
            fixer_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are a JSON repair tool.\n"
                            "Given an attempted JSON output from a call QA evaluator, rewrite it into STRICTLY valid JSON.\n"
                            "Rules:\n"
                            "- Output ONLY JSON (no markdown, no commentary).\n"
                            "- Must be an object with key \"evaluations\".\n"
                            "- \"evaluations\" must be a list of objects each containing: criteria_id, rating, evidence, reasoning.\n"
                            "- Escape any quotes inside strings.\n"
                        ),
                    ),
                    (
                        "human",
                        "Fix this into valid JSON only:\n\n{raw}",
                    ),
                ]
            )
            chain = fixer_prompt | self.llm
            resp = chain.invoke({"raw": raw_text})
            fixed_text = self._strip_code_fences(getattr(resp, "content", str(resp)))
            fixed_json = self._extract_json_object(fixed_text) or fixed_text
            fixed_json = self._repair_common_json_issues(fixed_json)
            parsed = json.loads(fixed_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    
    def _load_criteria_details(self) -> Dict[str, Any]:
        """Load the criteria details for this agent"""
        details = {}
        for path in self.criteria_paths:
            criteria = get_criteria_by_path(path)
            if criteria:
                details[path] = criteria
        return details
    
    def _create_criteria_description(self) -> str:
        """Create a detailed description of criteria for the prompt"""
        descriptions = []
        
        for path, criteria in self.criteria_details.items():
            desc = f"\n### {criteria['name']} (ID: {path})\n"
            desc += f"Weight: {criteria.get('weight', 0)}%\n"
            desc += f"Max Score: {criteria.get('max_score', 0)}\n"
            
            if 'description' in criteria:
                desc += f"Description: {criteria['description']}\n"
            
            desc += "\nRating Levels:\n"
            for rating_name, rating_info in criteria.get('ratings', {}).items():
                desc += f"\n**{rating_name}** (Score: {rating_info['score']}):\n"
                for criterion in rating_info.get('criteria', []):
                    desc += f"  - {criterion}\n"
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the evaluation prompt for this agent"""
        criteria_desc = self._create_criteria_description()
        allowed_ids = ", ".join(f'"{path}"' for path in self.criteria_details.keys())
        
        system_template = f"""✨ You are a specialized call center quality evaluation agent: {self.name}

Your role is to analyze call transcripts and evaluate them based on specific criteria.

## Agent SME Context (important):
You will be given a roster-based context that includes the agent's canonical name and domain skills:
- LI = Life Insurance
- GI = General Insurance
- HI = Health Insurance

Use the SME context ONLY for criteria that depend on product/process knowledge (e.g., Provision of Critical Information, Ownership, Escalation):
- If the customer's primary issue belongs to a domain the agent is skilled in, the agent is expected to answer correctly.
  If they cannot answer / provide incorrect information for in-scope questions, treat it as more severe (often Undesirable).
- If the issue is outside the agent's skills, do NOT penalize for not knowing; instead evaluate whether they handle it properly
  (set expectations, escalate/route, provide correct next steps).

## Your Evaluation Criteria:
{criteria_desc}

## Instructions:
1. Carefully read the transcript provided.
2. For each criterion assigned to you, determine the appropriate rating.
3. Provide specific evidence from the transcript (MUST be direct quotes; do NOT cite the roster context as evidence).
4. Explain your reasoning for each rating.
5. Default rating is "Expected" if the criterion is not applicable or cannot be determined.
6. You MUST evaluate **every** criterion assigned to you exactly once.

## Output Format (very important):
- Respond with **valid JSON**.
- The top-level object must contain a key `"evaluations"`.
- `"evaluations"` must be an array of objects.
- Each object must have the following keys:
  - `"criteria_id"`: the criteria path ID (must be one of: {allowed_ids}).
  - `"rating"`: one of `"Undesirable"`, `"Basic"`, `"Expected"`, or `"Desirable"`.
  - `"evidence"`: a list of short direct quotes or observations from the transcript.
  - `"reasoning"`: a concise explanation of why this rating was assigned.

Do **not** include any extra text outside the JSON.

Important:
- Only evaluate the criteria assigned to you.
- Be objective and fair.
- Use direct quotes from the transcript as evidence.
- If something is not applicable, rate it as "Expected".
"""
        
        human_template = """Please evaluate the following call transcript:

## Agent Context (from roster):
- Extracted Agent Name: {agent_extracted_name}
- Canonical Roster Name: {agent_canonical_name}
- Agent Skills (LI/GI/HI): {agent_skills}

## Transcript:
{transcript}

## Speaker Information:
- The CSO (Customer Service Officer) is the agent being evaluated
- The PH (Policyholder) or Customer is the caller

Provide your evaluation in the required JSON format."""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def evaluate(self, transcript: str, context: Optional[Dict[str, Any]] = None) -> AgentEvaluationResult:
        """Evaluate the transcript and return results"""
        logger.info(f"✨ {self.name} starting evaluation...")
        
        try:
            # Create the chain
            chain = self.prompt | self.llm
            
            ctx = context or {}
            agent_extracted_name = str(ctx.get("agent_extracted_name") or "").strip()
            agent_canonical_name = str(ctx.get("agent_canonical_name") or "").strip()
            agent_skills = ctx.get("agent_skills")
            if isinstance(agent_skills, (list, tuple, set)):
                agent_skills_str = ", ".join([str(s).strip() for s in agent_skills if str(s).strip()])
            else:
                agent_skills_str = str(agent_skills or "").strip()
            if not agent_skills_str:
                agent_skills_str = "Unknown"

            # Invoke the chain
            response = chain.invoke(
                {
                    "transcript": transcript,
                    "agent_extracted_name": agent_extracted_name or "Unknown",
                    "agent_canonical_name": agent_canonical_name or "Unknown",
                    "agent_skills": agent_skills_str,
                }
            )
            
            # Parse the response
            response_text = self._strip_code_fences(getattr(response, "content", str(response)))

            # Extract + parse JSON from response (best-effort + one LLM repair fallback)
            json_str = self._extract_json_object(response_text) or response_text
            json_str = self._repair_common_json_issues(json_str)
            try:
                parsed_output = json.loads(json_str)
            except Exception as e:
                # Try one more local repair on the extracted object
                repaired = self._repair_common_json_issues(json_str)
                try:
                    parsed_output = json.loads(repaired)
                except Exception:
                    logger.warning(
                        "%s: invalid JSON from model (%s). Attempting repair via LLM. Output head=%r",
                        self.name,
                        str(e) or type(e).__name__,
                        (response_text or "")[:400],
                    )
                    repaired_obj = self._json_repair_via_llm(response_text)
                    if repaired_obj is None:
                        logger.error("Could not parse/repair JSON for %s; using defaults.", self.name)
                        return self._create_default_result()
                    parsed_output = repaired_obj

            if not isinstance(parsed_output, dict):
                logger.error("Model output for %s was not a JSON object; using defaults.", self.name)
                return self._create_default_result()
            
            # Convert to AgentEvaluationResult
            result = AgentEvaluationResult(agent_name=self.name)
            seen_ids = set()

            for eval_data in parsed_output.get('evaluations', []):
                criteria_id = eval_data.get('criteria_id', '')

                # Skip unknown IDs to avoid corrupting scoring
                if criteria_id not in self.criteria_details:
                    logger.warning(f"{self.name}: Unknown criteria_id returned by LLM: {criteria_id}")
                    continue

                criteria = self.criteria_details[criteria_id]
                seen_ids.add(criteria_id)
                
                rating = eval_data.get('rating', 'Expected')
                # Guard against invalid ratings (avoid accidentally giving max score)
                if rating not in (criteria.get("ratings") or {}):
                    rating = criteria.get("default_rating", "Expected")
                rating_info = criteria.get('ratings', {}).get(rating, {})
                score = rating_info.get('score', criteria.get('max_score', 0))
                
                evidence = eval_data.get('evidence', [])
                
                evaluation = CriteriaEvaluation(
                    criteria_id=criteria_id,
                    criteria_name=criteria.get('name', criteria_id),
                    rating=rating,
                    score=score,
                    max_score=criteria.get('max_score', 0),
                    weight=criteria.get('weight', 0),
                    evidence=evidence,
                    reasoning=eval_data.get('reasoning', '')
                )
                result.evaluations.append(evaluation)

            # Ensure every criterion gets an evaluation (default Expected)
            for path, criteria in self.criteria_details.items():
                if path in seen_ids:
                    continue
                default_rating = criteria.get("default_rating", "Expected")
                rating_info = criteria.get("ratings", {}).get(default_rating, {})

                evaluation = CriteriaEvaluation(
                    criteria_id=path,
                    criteria_name=criteria.get("name", path),
                    rating=default_rating,
                    score=rating_info.get("score", criteria.get("max_score", 0)),
                    max_score=criteria.get("max_score", 0),
                    weight=criteria.get("weight", 0),
                    evidence=[],
                    reasoning="Default rating applied because the model did not return an explicit evaluation for this criterion.",
                )
                result.evaluations.append(evaluation)
            
            result.calculate_totals()
            logger.info(f"✨ {self.name} completed evaluation. Score: {result.total_score}/{result.max_possible_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return self._create_default_result()
    
    def _create_default_result(self) -> AgentEvaluationResult:
        """Create a default result with Expected ratings"""
        result = AgentEvaluationResult(agent_name=self.name)
        
        for path, criteria in self.criteria_details.items():
            default_rating = criteria.get('default_rating', 'Expected')
            rating_info = criteria.get('ratings', {}).get(default_rating, {})
            
            evaluation = CriteriaEvaluation(
                criteria_id=path,
                criteria_name=criteria.get('name', path),
                rating=default_rating,
                score=rating_info.get('score', criteria.get('max_score', 0)),
                max_score=criteria.get('max_score', 0),
                weight=criteria.get('weight', 0),
                evidence=[],
                reasoning="Default rating applied - evaluation could not be completed"
            )
            result.evaluations.append(evaluation)
        
        result.calculate_totals()
        return result


# ============================================================================
# Improvement Summary Agent (separate AI call)
# ============================================================================

class ImprovementSummaryAgent:
    """
    Generates an intuition-style coaching summary from evaluation findings.

    This is intentionally a separate AI agent call so the 'AI Suggestion for Improvement'
    can be created without hard-coded per-criterion mapping.
    """

    def __init__(self, llm: ChatBedrock):
        self.llm = llm
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser(pydantic_object=ImprovementSummaryOutput)

    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = (
            "You are an expert call coaching assistant.\n\n"
            "You will receive JSON describing a call's evaluation findings (scores/ratings, short evidence quotes "
            "and timestamps, and AI remarks).\n\n"
            "Task:\n"
            "- Write an intuitive 'AI Suggestion for Improvement' summary for the agent.\n"
            "- DO NOT list suggestions per criterion and DO NOT mention criterion IDs.\n"
            "- Group feedback into 3-5 high-level themes (e.g., Verification, Soft Skills, Resolution, Wrap-up).\n"
            "- Use the provided evidence (with timestamps) to ground suggestions (max 1 example per theme).\n"
            "- Be practical: tell the agent what to do differently next time.\n\n"
            "Input notes:\n"
            "- Prefer `issues` if present (biggest gaps), but you may also use `evaluations`.\n"
            "- If there are few/no issues, still provide improvement-oriented coaching by suggesting how to move\n"
            "  from Expected to Desirable and how to improve consistency; do not return an empty summary.\n\n"
            "Style constraints:\n"
            "- 120-180 words total.\n"
            "- Bullet list preferred.\n"
            "- No extra preamble.\n\n"
            "Output format:\n"
            # NOTE: LangChain templates treat `{...}` as variables. Escape braces for literal JSON examples.
            "- Return ONLY valid JSON: {{\"summary\": \"...\"}}\n"
        )

        human_template = "Evaluation findings JSON:\n{findings_json}"

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])

    def generate(self, findings: Dict[str, Any]) -> str:
        """Generate the coaching summary as a single string."""
        def _strip_code_fences(t: str) -> str:
            t = (t or "").strip()
            if t.startswith("```"):
                lines = t.splitlines()
                if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                    return "\n".join(lines[1:-1]).strip()
            return t

        def _best_effort_extract_summary(response_text: str) -> str:
            text = _strip_code_fences(response_text)

            # Try full JSON parse first
            try:
                parsed_full = json.loads(text)
                if isinstance(parsed_full, dict):
                    s = (parsed_full.get("summary") or "").strip()
                    if s:
                        return s
            except Exception:
                pass

            # Try JSON substring between first/last brace
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_candidate = text[json_start:json_end]
                try:
                    parsed = json.loads(json_candidate)
                    if isinstance(parsed, dict):
                        s = (parsed.get("summary") or "").strip()
                        if s:
                            return s
                except Exception:
                    # If JSON is malformed, fall back to raw text below
                    pass

                # Regex extraction of the summary field (handles escaped quotes)
                m = re.search(r"\"summary\"\s*:\s*\"(?P<val>(?:\\.|[^\"\\])*)\"", json_candidate)
                if m:
                    try:
                        return json.loads(f"\"{m.group('val')}\"").strip()
                    except Exception:
                        pass

            # Fall back to plain text (better than empty)
            return text.strip()

        chain = self.prompt | self.llm
        payload = json.dumps(findings, ensure_ascii=False)

        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                response = chain.invoke({"findings_json": payload})
                response_text = getattr(response, "content", str(response))
                summary = _best_effort_extract_summary(response_text)
                return summary
            except Exception as e:
                last_err = e
                msg = str(e) or type(e).__name__
                transient = any(
                    k in msg.lower()
                    for k in (
                        "throttl",
                        "too many requests",
                        "rate exceeded",
                        "timeout",
                        "timed out",
                        "temporarily unavailable",
                        "service unavailable",
                        "connection reset",
                        "endpointconnectionerror",
                    )
                )
                if attempt < 3 and transient:
                    backoff = 1.5 * (2 ** (attempt - 1))
                    logger.warning(f"ImprovementSummaryAgent transient error (attempt {attempt}/3): {msg}. Retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    continue
                logger.exception(f"ImprovementSummaryAgent failed (attempt {attempt}/3): {msg}")
                break

        return "" if last_err is not None else ""


# ============================================================================
# Call Insights Agent (Intent classification + articulation coaching)
# ============================================================================

class CallInsightsAgent:
    """
    Produces:
    - Call intent classification (insurance context)
    - Subject of the call
    - Customer issues raised
    - Per-category articulation coaching for the CSO
    """

    ALLOWED_CLASSIFICATIONS = [
        "Policy",
        "Claim",
        "Product Related",
        "Complaints",
        "Follow Up",
        "Others",
    ]

    REQUIRED_ARTICULATION_KEYS = [
        "opening_greeting",
        "verification",
        "soft_skills",
        "enquiry_resolution",
        "cross_selling",
        "wrap_up",
    ]

    def __init__(self, llm: ChatBedrock):
        self.llm = llm
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        allowed = ", ".join([f'"{x}"' for x in self.ALLOWED_CLASSIFICATIONS])
        keys = ", ".join([f'"{k}"' for k in self.REQUIRED_ARTICULATION_KEYS])

        system_template = (
            "You are an insurance call analytics assistant.\n\n"
            "You will receive:\n"
            "- Roster context for the CSO (agent) including LI/GI/HI skills\n"
            "- A call transcript (may be long)\n"
            "- A compact list of evaluation issues (criteria with Basic/Undesirable)\n\n"
            "Tasks:\n"
            "1) Identify the call intent and classify it into EXACTLY one of: "
            f"{allowed}.\n"
            "2) Provide a short subject line describing what the call is about.\n"
            "3) List the customer's issues/questions raised (3-8 items).\n"
            "4) Provide coaching suggestions for better articulation per evaluation category "
            "(opening greeting, verification, soft skills, enquiry resolution, cross-selling, wrap-up).\n\n"
            "Important constraints:\n"
            "- Ground your issues and coaching in what is actually in the transcript/evaluation issues.\n"
            "- Do NOT invent facts (policy numbers, claim IDs, dates) not present.\n"
            "- Use SME context (LI/GI/HI) only to judge the severity of not knowing product/process info:\n"
            "  * If the issue is in-scope for the agent skills, not knowing should be treated as more severe.\n"
            "  * If out-of-scope, focus coaching on routing/escalation language and expectation setting.\n"
            "- For articulation coaching: focus on wording, clarity, structure, and confidence.\n"
            "- Keep each category's suggestion to 2-4 bullet points.\n\n"
            "Output format:\n"
            "- Return ONLY valid JSON (no markdown).\n"
            "- Required keys:\n"
            "  - classification\n"
            "  - subject\n"
            "  - issues\n"
            "  - articulation_suggestions (object)\n"
            f"- articulation_suggestions MUST contain keys: {keys}\n"
        )

        human_template = (
            "Agent context:\n"
            "- Extracted Agent Name: {agent_extracted_name}\n"
            "- Canonical Roster Name: {agent_canonical_name}\n"
            "- Agent Skills (LI/GI/HI): {agent_skills}\n\n"
            "Evaluation issues JSON (only criteria rated Basic/Undesirable):\n"
            "{evaluation_issues_json}\n\n"
            "Transcript:\n"
            "{transcript}\n"
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", human_template),
            ]
        )

    @staticmethod
    def _canonicalize_classification(value: str) -> str:
        s = str(value or "").strip().lower()
        if not s:
            return "Others"
        if "policy" in s:
            return "Policy"
        if "claim" in s:
            return "Claim"
        if "product" in s or "prod" in s:
            return "Product Related"
        if "complain" in s:
            return "Complaints"
        if "follow" in s:
            return "Follow Up"
        if "other" in s:
            return "Others"
        return "Others"

    def analyze(
        self,
        *,
        transcript: str,
        agent_extracted_name: str,
        agent_canonical_name: str,
        agent_skills: str,
        evaluation_issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns a dict with keys matching CallInsightsOutput.
        Best-effort parsing: returns defaults if anything fails.
        """
        def _strip_code_fences(t: str) -> str:
            t = (t or "").strip()
            if t.startswith("```"):
                lines = t.splitlines()
                if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                    return "\n".join(lines[1:-1]).strip()
            return t

        def _truncate_transcript(t: str, max_chars: int = 12000) -> str:
            t = str(t or "")
            if len(t) <= max_chars:
                return t
            half = max_chars // 2
            return (t[:half] + "\n...\n" + t[-half:]).strip()

        default = {
            "classification": "Others",
            "subject": "",
            "issues": [],
            "articulation_suggestions": {k: "" for k in self.REQUIRED_ARTICULATION_KEYS},
        }

        payload_issues = json.dumps(evaluation_issues or [], ensure_ascii=False)
        chain = self.prompt | self.llm

        try:
            response = chain.invoke(
                {
                    "agent_extracted_name": agent_extracted_name or "Unknown",
                    "agent_canonical_name": agent_canonical_name or "Unknown",
                    "agent_skills": agent_skills or "Unknown",
                    "evaluation_issues_json": payload_issues,
                    "transcript": _truncate_transcript(transcript),
                }
            )
            response_text = _strip_code_fences(getattr(response, "content", str(response)))

            # Parse best-effort JSON
            parsed = None
            try:
                parsed = json.loads(response_text)
            except Exception:
                # Try substring between braces
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    try:
                        parsed = json.loads(response_text[json_start:json_end])
                    except Exception:
                        parsed = None

            if not isinstance(parsed, dict):
                return default

            classification = self._canonicalize_classification(parsed.get("classification"))
            subject = str(parsed.get("subject") or "").strip()
            issues = parsed.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            issues = [str(x).strip() for x in issues if str(x).strip()][:8]

            art = parsed.get("articulation_suggestions") or {}
            if not isinstance(art, dict):
                art = {}
            art_out = {k: str(art.get(k) or "").strip() for k in self.REQUIRED_ARTICULATION_KEYS}

            return {
                "classification": classification,
                "subject": subject,
                "issues": issues,
                "articulation_suggestions": art_out,
            }
        except Exception as e:
            logger.warning(f"CallInsightsAgent failed: {type(e).__name__}: {e}")
            return default

# ============================================================================
# Specialized Agents
# ============================================================================

class OpeningGreetingAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating opening greetings"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Opening Greeting Agent",
            criteria_paths=AGENT_ASSIGNMENTS["opening_greeting_agent"],
            llm=llm
        )


class VerificationAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating verification procedures"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Verification Agent",
            criteria_paths=AGENT_ASSIGNMENTS["verification_agent"],
            llm=llm
        )


class SoftSkillsAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating soft skills"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Soft Skills Agent",
            criteria_paths=AGENT_ASSIGNMENTS["soft_skills_agent"],
            llm=llm
        )


class EnquiryResolutionAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating enquiry resolution"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Enquiry Resolution Agent",
            criteria_paths=AGENT_ASSIGNMENTS["enquiry_resolution_agent"],
            llm=llm
        )


class CrossSellingAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating cross-selling attempts"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Cross-Selling Agent",
            criteria_paths=AGENT_ASSIGNMENTS["cross_selling_agent"],
            llm=llm
        )


class WrapUpAgent(BaseEvaluationAgent):
    """Agent specialized in evaluating call wrap-up"""
    
    def __init__(self, llm: ChatBedrock):
        super().__init__(
            name="Wrap-Up Agent",
            criteria_paths=AGENT_ASSIGNMENTS["wrap_up_agent"],
            llm=llm
        )


# ============================================================================
# Agent Factory
# ============================================================================

def create_all_agents(config: BedrockConfig) -> Dict[str, BaseEvaluationAgent]:
    """Create all evaluation agents"""
    llm = create_bedrock_llm(config)

    # Optional cheaper model for early-phase criteria (opening greeting + verification)
    model_overrides = {
        "opening_greeting": OPENING_GREETING_BEDROCK_MODEL_ID or EARLY_PHASE_BEDROCK_MODEL_ID,
        "verification": VERIFICATION_BEDROCK_MODEL_ID or EARLY_PHASE_BEDROCK_MODEL_ID,
    }

    llm_cache: Dict[str, ChatBedrock] = {}

    def _get_llm_for(agent_key: str) -> ChatBedrock:
        override_model_id = (model_overrides.get(agent_key) or "").strip()
        if not override_model_id or override_model_id == config.model_id:
            return llm
        if override_model_id in llm_cache:
            return llm_cache[override_model_id]

        # Be robust to older/mismatched BedrockConfig definitions that may not accept kwargs.
        # (We have seen environments where `BedrockConfig()` takes no args.)
        try:
            cfg = BedrockConfig(
                profile_name=getattr(config, "profile_name", None),
                region_name=getattr(config, "region_name", None),
                model_id=override_model_id,
            )
        except TypeError as e:
            logger.warning(f"BedrockConfig constructor rejected kwargs in create_all_agents ({e}); using fallback config object.")
            try:
                cfg2 = BedrockConfig()
                try:
                    setattr(cfg2, "profile_name", getattr(config, "profile_name", None))
                    setattr(cfg2, "region_name", getattr(config, "region_name", None))
                    setattr(cfg2, "model_id", override_model_id)
                except Exception:
                    pass
                if not all(hasattr(cfg2, k) for k in ("profile_name", "region_name", "model_id")):
                    raise AttributeError("BedrockConfig instance missing required attributes")
                cfg = cfg2
            except Exception:
                from types import SimpleNamespace
                cfg = SimpleNamespace(
                    profile_name=getattr(config, "profile_name", None),
                    region_name=getattr(config, "region_name", None),
                    model_id=override_model_id,
                )
        llm_cache[override_model_id] = create_bedrock_llm(cfg)
        return llm_cache[override_model_id]
    
    return {
        "opening_greeting": OpeningGreetingAgent(_get_llm_for("opening_greeting")),
        "verification": VerificationAgent(_get_llm_for("verification")),
        "soft_skills": SoftSkillsAgent(llm),
        "enquiry_resolution": EnquiryResolutionAgent(llm),
        "cross_selling": CrossSellingAgent(llm),
        "wrap_up": WrapUpAgent(llm)
    }

