
import logging
import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

import pandas as pd
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
DEFAULT_AWS_PROFILE = os.getenv("AWS_PROFILE", "income-adfs")
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")


@dataclass
class ComprehensiveEvaluationReport:
    """Complete evaluation report combining all agent results"""
    transcript_id: str
    evaluation_timestamp: str
    detected_agent_name: str = ""
    agent_results: Dict[str, AgentEvaluationResult] = field(default_factory=dict)
    total_score: float = 0.0
    max_possible_score: float = 0.0
    percentage_score: float = 0.0
    overall_rating: str = ""
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    
    def calculate_totals(self):
        """Calculate total scores from all agent results using Python math"""
        self.total_score = sum(r.total_score for r in self.agent_results.values())
        self.max_possible_score = sum(r.max_possible_score for r in self.agent_results.values())
        
        if self.max_possible_score > 0:
            self.percentage_score = (self.total_score / self.max_possible_score) * 100
        else:
            self.percentage_score = 0.0
        
      
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
        """Generate a summary of the evaluation"""
        self.summary = f"""
✨ AI-Generated Call Quality Evaluation Summary
================================================
Transcript ID: {self.transcript_id}
Evaluation Date: {self.evaluation_timestamp}

Overall Score: {self.total_score:.2f} / {self.max_possible_score:.2f} ({self.percentage_score:.1f}%)
Overall Rating: {self.overall_rating}

Agent Breakdown:
"""
        for agent_name, result in self.agent_results.items():
            agent_percentage = (result.total_score / result.max_possible_score * 100) if result.max_possible_score > 0 else 0
            self.summary += f"  - {result.agent_name}: {result.total_score:.2f}/{result.max_possible_score:.2f} ({agent_percentage:.1f}%)\n"
        
        if self.strengths:
            self.summary += "\nStrengths:\n"
            for strength in self.strengths[:5]:  # Top 5 strengths
                self.summary += f"  ✓ {strength}\n"
        
        if self.areas_for_improvement:
            self.summary += "\nAreas for Improvement:\n"
            for area in self.areas_for_improvement[:5]:  # Top 5 areas
                self.summary += f"  ⚠ {area}\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "transcript_id": self.transcript_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "detected_agent_name": self.detected_agent_name,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
            "percentage_score": self.percentage_score,
            "overall_rating": self.overall_rating,
            "summary": self.summary,
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
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
            evaluation_timestamp=datetime.now().isoformat()
        )
        
        if parallel:
          
            report.agent_results = self._evaluate_parallel(transcript)
        else:
       
            report.agent_results = self._evaluate_sequential(transcript)
        
    
        report.calculate_totals()
        report.identify_strengths_and_improvements()
        report.generate_summary()

        try:
            report.detected_agent_name = self._extract_agent_name(transcript)
        except Exception as e:
            logger.error(f"Error extracting agent name: {e}")
        
        logger.info(f"✨ Evaluation complete. Score: {report.percentage_score:.1f}%")
        
        return report
    
    def _evaluate_parallel(self, transcript: str) -> Dict[str, AgentEvaluationResult]:
        """Run all agents in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
      
            future_to_agent = {
                executor.submit(agent.evaluate, transcript): agent_name
                for agent_name, agent in self.agents.items()
            }
            
       
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
    
    def _evaluate_sequential(self, transcript: str) -> Dict[str, AgentEvaluationResult]:
        """Run all agents sequentially"""
        results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"✨ Running {agent_name} agent...")
                result = agent.evaluate(transcript)
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
        return self.agents[category].evaluate(transcript)

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

        any_agent = next(iter(self.agents.values()))
        llm = any_agent.llm

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
    model_id: Optional[str] = None
) -> EvaluationOrchestrator:
    """
    Factory function to create an EvaluationOrchestrator
    
    Args:
        profile_name: AWS Profile name (falls back to env AWS_PROFILE or income-adfs)
        region_name: AWS Region (falls back to env AWS_REGION or ap-southeast-1)
        model_id: Bedrock model ID (falls back to env BEDROCK_MODEL_ID)
    
    Returns:
        Configured EvaluationOrchestrator instance
    """
    profile = profile_name or DEFAULT_AWS_PROFILE
    region = region_name or DEFAULT_AWS_REGION or "ap-southeast-1"
    model = model_id or DEFAULT_BEDROCK_MODEL_ID

    config = BedrockConfig(
        profile_name=profile,
        region_name=region,
        model_id=model
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
    def generate_excel_report(report: ComprehensiveEvaluationReport) -> io.BytesIO:
        """Generate an Excel report with criteria, rating, weight and score."""
        rows: List[Dict[str, Any]] = []

        for agent_key, result in report.agent_results.items():
            for eval in result.evaluations:
                rows.append(
                    {
                        "Agent Category": result.agent_name,
                        "Criteria ID": eval.criteria_id,
                        "Criteria Name": eval.criteria_name,
                        "Rating": eval.rating,
                        "Weight (%)": eval.weight,
                        "Score": eval.score,
                        "Max Score": eval.max_score,
                    }
                )

        df = pd.DataFrame(rows)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
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

