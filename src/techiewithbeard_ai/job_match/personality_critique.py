"""
Personality-Based Critique Agent

Generates resume critiques in different HR market styles.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.job_match.personality import (
    ResumePersonality,
    get_personality_config,
)
from techiewithbeard_ai.job_match.schemas import Critique
from techiewithbeard_ai.schema.provider import ModelConfig


@traceable
def generate_personality_critique(
    candidate_name: str,
    candidate_skills: list[str],
    experience_summaries: str,
    required_skills: list[str],
    matching_skills: list[str],
    missing_skills: list[str],
    personality: ResumePersonality,
    config: ModelConfig,
) -> Critique:
    """
    Generate a resume critique tailored to a specific HR market personality.
    
    Args:
        candidate_name: Candidate's name
        candidate_skills: Skills from candidate's resume
        experience_summaries: Summary of candidate's experiences
        required_skills: Skills required by the job
        matching_skills: Skills that match (intersection)
        missing_skills: Skills that don't match
        personality: HR market personality (American, European, etc.)
        config: Model configuration
        
    Returns:
        Critique with personality-specific analysis
    """
    
    personality_config = get_personality_config(personality)
    parser = PydanticOutputParser(pydantic_object=Critique)
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a senior HR recruitment specialist at a {personality} tech firm.

Your communication style:
{critique_style}

Your task: Generate a detailed critique of {candidate_name}'s resume for this specific job.

Focus on:
1. STRENGTHS: What makes this candidate strong for this role
   (based on matching skills and experience)
2. WEAKNESSES: Gaps and areas for improvement (missing skills, experience gaps)
3. LEARNING POTENTIAL: How quickly could they learn missing skills?
4. RECOMMENDATIONS: Specific, actionable suggestions for improving resume for this market
5. OVERALL ASSESSMENT: Is this candidate suitable for the role in your market's perspective?

Remember your {personality} perspective when evaluating:
- What matters most to your market
- How to describe strengths and weaknesses
- What recommendations would resonate

{format_instructions}
"""
            ),
            (
                "human",
                """
Candidate: {candidate_name}

Candidate Skills:
{candidate_skills}

Experience Summary:
{experience_summaries}

Job Requirements:
Required Skills: {required_skills}

Match Analysis:
- Matching Skills ({matching_skill_count}): {matching_skills}
- Missing Skills ({missing_skill_count}): {missing_skills}

Please provide a detailed critique from your {personality} perspective.
"""
            ),
        ]).partial(
            format_instructions=parser.get_format_instructions(),
            personality=personality.value,
            critique_style=personality_config.critique_style,
        )
        
        llm = get_chat_model(config)
        chain = prompt | llm | parser
        
        critique = chain.invoke({
            "candidate_name": candidate_name,
            "candidate_skills": ", ".join(candidate_skills) or "None listed",
            "experience_summaries": experience_summaries,
            "required_skills": ", ".join(required_skills),
            "matching_skills": ", ".join(matching_skills) if matching_skills else "None",
            "matching_skill_count": len(matching_skills),
            "missing_skills": ", ".join(missing_skills) if missing_skills else "None",
            "missing_skill_count": len(missing_skills),
        })
        
        print(f"\n✓ Generated {personality.value.upper()} critique for {candidate_name}")
        return critique
        
    except Exception as e:
        print(f"❌ Error generating critique: {str(e)}")
        # Return default critique
        return Critique(
            strengths=["Experience and skill demonstrations"],
            weaknesses=["Some gaps identified"],
            missing_skills=missing_skills,
            learning_potential="Candidate demonstrates learning capability",
            recommendations=["Review job description carefully"],
            overall_assessment="Review candidate profile in detail",
        )
