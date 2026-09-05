"""
Experience-Specific Agents

Each agent handles tailoring for a specific work experience entry.
The main orchestrator decides which skills go to which experience.
"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.job_match.personality import (
    ResumePersonality,
    get_personality_config,
)
from techiewithbeard_ai.job_match.schemas import Experience
from techiewithbeard_ai.schema.provider import ModelConfig


@traceable
def tailor_experience_section(
    experience: Experience,
    required_skills: list[str],
    relevant_skills_for_this_exp: list[str],
    personality: ResumePersonality,
    config: ModelConfig,
) -> Experience:
    """
    Tailor a single experience entry to match job requirements.
    
    This agent:
    1. Analyzes the experience against required skills
    2. Identifies relevant bullets for this job
    3. Reorders bullets by relevance
    4. Enhances bullets (conservative - never adds false info)
    5. Applies personality-based critique style
    
    Args:
        experience: The experience entry to tailor
        required_skills: All skills required by the job
        relevant_skills_for_this_exp: Skills from this experience that match job
        personality: HR market personality style
        config: Model configuration
        
    Returns:
        Tailored experience entry with reordered/enhanced bullets
    """
    
    personality_config = get_personality_config(personality)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert resume tailoring specialist with a {personality} HR tech firm perspective.

{bullet_enhancement_style}

Your task: Tailor work experience to highlight relevance to a specific job.

CRITICAL RULES:
1. NEVER add false information or skills not in the original bullets
2. NEVER invent achievements or responsibilities
3. ONLY rephrase existing bullets to emphasize relevance
4. Reorder bullets to lead with most relevant achievements
5. Keep original meaning - enhance clarity and relevance

Input:
- Experience: {company} at {title} ({dates})
- Relevant skills for this job: {relevant_skills}
- Original bullets: {original_bullets}

Output: List of enhanced, reordered bullets that:
1. Lead with most relevant achievements (matching {relevant_skills})
2. Emphasize impact using the style guide above
3. Maintain complete accuracy to original content
4. Are ordered by job relevance (most relevant first)

Return as a JSON array of strings (bullet points).
"""
        ),
        (
            "human",
            """
Job Skills Required: {required_skills}

Work Experience:
- Company: {company}
- Title: {title}
- Duration: {dates}
- Skills from this role relevant to job: {relevant_skills}

Original Bullets:
{original_bullets}

Please tailor these bullets following the personality guidelines.
Return ONLY a JSON array of strings (enhanced bullet points).
"""
        ),
    ]).partial(
        personality=personality.value,
        bullet_enhancement_style=personality_config.bullet_enhancement_style,
    )
    
    llm = get_chat_model(config)
    
    # Format inputs
    company = experience.company
    title = experience.title
    dates = f"{experience.start_date or 'Unknown'} - {experience.end_date or 'Present'}"
    original_bullets = "\n".join(f"- {b}" for b in (experience.bullets or []))
    relevant_skills_str = (
        ", ".join(relevant_skills_for_this_exp)
        if relevant_skills_for_this_exp
        else "None found"
    )
    required_skills_str = ", ".join(required_skills)
    
    try:
        response = llm.invoke(prompt.format_prompt(
            required_skills=required_skills_str,
            company=company,
            title=title,
            dates=dates,
            relevant_skills=relevant_skills_str,
            original_bullets=original_bullets,
        ))
        
        response_content = response.content
        response_text = (
            response_content
            if isinstance(response_content, str)
            else str(response_content)
        )
        
        # Try to extract JSON array
        try:
            # Look for JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                tailored_bullets = json.loads(json_match.group())
                if isinstance(tailored_bullets, list) and tailored_bullets:
                    experience.bullets = tailored_bullets
                    return experience
        except json.JSONDecodeError:
            pass
        
        # If parsing fails, keep original bullets
        print(f"⚠ Could not parse tailored bullets for {title}, keeping original")
        return experience
        
    except Exception as e:
        print(f"❌ Error tailoring experience {title}: {str(e)}")
        return experience


@traceable
def skill_to_experience_orchestrator(
    candidate_skills: list[str],
    required_skills: list[str],
    experiences: list[Experience],
    personality: ResumePersonality,
    config: ModelConfig,
) -> dict[str, list[str]]:
    """
    Main orchestrator that decides which skills go to which experience.
    
    This agent:
    1. Maps required skills to candidate's experiences
    2. Determines skill-experience relevance
    3. Creates routing map for tailoring agents
    
    Args:
        candidate_skills: All skills from candidate's resume
        required_skills: Skills required by the job
        experiences: List of experiences in resume
        personality: HR market personality
        config: Model configuration
        
    Returns:
        Dict mapping experience index to relevant skills
        Example: {0: ['Angular', 'TypeScript'], 1: ['React', 'JavaScript']}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert in matching candidate skills to work experiences.

Analyze where each skill was likely used based on job titles, companies,
and the skill domains.

CRITICAL: Only use skills that ACTUALLY EXIST in the candidate's resume.
Never invent or guess which skills belong to which experience.

For each required skill, determine which experience(s) likely used it.
A skill might be relevant to multiple experiences.

Return a JSON object:
{{
  "0": ["skill1", "skill2"],  // Skills relevant to experience 1
  "1": ["skill3", "skill4"],  // Skills relevant to experience 2
  ...
}}
"""
        ),
        (
            "human",
            """
Candidate's Skills: {candidate_skills}

Required Job Skills: {required_skills}

Candidate's Experiences:
{experiences}

For each required skill, identify which experience(s) most likely used it.
Return ONLY valid JSON mapping.
"""
        ),
    ])
    
    llm = get_chat_model(config)
    
    # Format experiences for prompt
    exp_text = "\n\n".join(
        [
            (
                f"Experience {i}:\n"
                f"  Title: {exp.title}\n"
                f"  Company: {exp.company}\n"
                f"  Dates: {exp.start_date} - {exp.end_date or 'Present'}\n"
                f"  Bullets: {', '.join(exp.bullets[:2])}..."
            )
            for i, exp in enumerate(experiences)
        ]
    )
    
    try:
        response = llm.invoke(prompt.format_prompt(
            candidate_skills=", ".join(candidate_skills),
            required_skills=", ".join(required_skills),
            experiences=exp_text,
        ))
        
        response_content = response.content
        response_text = (
            response_content
            if isinstance(response_content, str)
            else str(response_content)
        )
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            routing = json.loads(json_match.group())
            # Convert string keys to int
            return {int(k): v for k, v in routing.items()}
    except Exception as e:
        print(f"❌ Orchestrator error: {str(e)}")
    
    # Fallback: distribute skills across all experiences.
    if not experiences:
        return {}

    skill_count = max(1, len(required_skills) // len(experiences))

    return {
        i: required_skills[i*skill_count:(i+1)*skill_count]
        for i in range(len(experiences))
    }
