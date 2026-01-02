
import json
import logging
import os
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
DEFAULT_AWS_PROFILE = os.getenv("AWS_PROFILE", "income-adfs")
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20240620-v1:0"
)

# ============================================================================
# AWS Bedrock Configuration
# ============================================================================

class BedrockConfig:
    """Configuration for AWS Bedrock"""
    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        self.profile_name = profile_name or DEFAULT_AWS_PROFILE
        self.region_name = region_name or DEFAULT_AWS_REGION
        self.model_id = model_id or DEFAULT_BEDROCK_MODEL_ID


def create_bedrock_llm(config: BedrockConfig) -> ChatBedrock:
    """Create a ChatBedrock LLM instance"""
    import boto3
    
    # Create boto3 session with profile for federated auth
    session = boto3.Session(
        profile_name=config.profile_name,
        region_name=config.region_name
    )
    
    # Get credentials from session
    credentials = session.get_credentials()
    
    return ChatBedrock(
        model_id=config.model_id,
        region_name=config.region_name,
        credentials_profile_name=config.profile_name,
        model_kwargs={
            "temperature": 0.1,
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

## Your Evaluation Criteria:
{criteria_desc}

## Instructions:
1. Carefully read the transcript provided.
2. For each criterion assigned to you, determine the appropriate rating.
3. Provide specific evidence from the transcript (direct quotes when possible).
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
    
    def evaluate(self, transcript: str) -> AgentEvaluationResult:
        """Evaluate the transcript and return results"""
        logger.info(f"✨ {self.name} starting evaluation...")
        
        try:
            # Create the chain
            chain = self.prompt | self.llm
            
            # Invoke the chain
            response = chain.invoke({"transcript": transcript})
            
            # Parse the response
            response_text = response.content
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_output = json.loads(json_str)
            else:
                logger.error(f"Could not find JSON in response: {response_text}")
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
                rating_info = criteria.get('ratings', {}).get(rating, {})
                score = rating_info.get('score', criteria.get('max_score', 0))
                
                evidence = eval_data.get('evidence', [])
                
                # If no evidence is found, force score to 0
                if not evidence:
                    score = 0.0
                
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
    
    return {
        "opening_greeting": OpeningGreetingAgent(llm),
        "verification": VerificationAgent(llm),
        "soft_skills": SoftSkillsAgent(llm),
        "enquiry_resolution": EnquiryResolutionAgent(llm),
        "cross_selling": CrossSellingAgent(llm),
        "wrap_up": WrapUpAgent(llm)
    }

