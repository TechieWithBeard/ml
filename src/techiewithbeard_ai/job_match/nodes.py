import json
from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.job_match.schemas import JobRequirements, ResumeProfile, SkillMatchResult, TransferabilityAnalysis, Transferability, SkillMatch, Critique
from techiewithbeard_ai.job_match.state import JobMatchState
from techiewithbeard_ai.schema.provider import ModelConfig


@traceable
def parse_requirements(
    state: JobMatchState,
) -> dict:

    job_description = state.get("job_description")
    config = state.get("config")

    if not job_description:
        raise ValueError(
            "Job description is missing from graph state."
        )

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a technical job requirement analyzer.

                    Analyze the provided job description.

                    Extract ONLY information explicitly present.

                    Extract:

                    1. Required technical skills
                    2. Required experience
                    3. Job responsibilities

                    Do not invent requirements.

                    If something is not mentioned, return an empty list.
                """,
            ),
            (
                "human",
                """
                JOB DESCRIPTION:

                {job_description}
                """,
            ),
        ]
    )

    llm = get_chat_model(config).with_structured_output(
        JobRequirements
    )

    chain = prompt | llm

    raw_result = chain.invoke(
        {
            "job_description": job_description,
        }
    )

    result = cast(
        JobRequirements,
        raw_result,
    )

    return {
        "required_skills": result.required_skills,
        "required_experience": result.required_experience,
        "responsibilities": result.responsibilities,
    }

@traceable
def parse_resume(state: JobMatchState) -> dict:

    resume_text = state.get("resume_text")
    config = state.get("config")

    if not resume_text:
        raise ValueError(
            "Resume text is missing from graph state."
        )

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    prompt = f"""
You are a resume information extraction system.

Extract information from the resume below.

Return ONLY a JSON object.

The JSON must have EXACTLY these fields:

{{
  "candidate_name": null,
  "skills": [],
  "experience": []
}}

Rules:

- candidate_name: candidate's full name.
- skills: technical skills explicitly mentioned.
- experience: work experience explicitly mentioned.
- Do not invent information.
- Do not infer skills.
- Do not add additional fields.
- Do not use markdown.
- Do not use ```json.
- Make sure the JSON is COMPLETE and properly closed.

RESUME:

{resume_text}
"""

    llm = get_chat_model(config)

    raw_result = llm.invoke(prompt)

    content = raw_result.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    content = str(content).strip()

    print("\n========== RESUME LLM OUTPUT ==========")
    print(content)
    print("=======================================\n")

    # Remove markdown fences if the model adds them.
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        
        data = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Resume parser returned invalid JSON.\n\n"
            f"Raw model output:\n{content}\n\n"
            f"JSON error: {exc}"
        ) from exc

    result = ResumeProfile.model_validate(data)

    return {
        "candidate_name": result.candidate_name,
        "candidate_skills": result.skills,
        "candidate_experience": result.experience,
    }
    
    

@traceable   
def analyze_transferability(
    state: JobMatchState,
) -> dict:

    missing_skills = state.get("missing_skills") or []
    candidate_skills = state.get("candidate_skills") or []
    candidate_experience = state.get("candidate_experience") or []
    config = state.get("config")

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    if not missing_skills:
        return {
            "transferability": []
        }

    prompt = """

    OUTPUT FORMAT:

    Return ONLY valid JSON.

    Do NOT return:
    - Markdown
    - headings
    - tables
    - explanations outside JSON
    - code fences
    - ```json

    The response MUST follow this exact structure:

    {
    "analyses": [
        {
        "missing_skill": "c#",
        "related_skills": [
            "TypeScript",
            "JavaScript"
        ],
        "transferability_score": 65,
        "learning_difficulty": "medium",
        "reasoning": "The candidate has experience with strongly typed languages and therefore has transferable programming concepts."
        }
    ]
    }

    Rules:

    - Return exactly one analysis for every missing skill.
    - `missing_skill` must exactly match a skill from the missing skills list.
    - `related_skills` must only contain skills the candidate actually has.
    - Never claim that a related skill means the candidate already knows the missing skill.
    - `transferability_score` must be between 0 and 100.
    - `learning_difficulty` must be one of:
    "low", "medium", "high".
    - `reasoning` must be concise.
    - Return complete, valid JSON.
    """
    # llm = get_chat_model(config)

    # structured_llm = llm.with_structured_output(
    #     TransferabilityAnalysis
    # )
    # # llm = get_chat_model(config)

    # raw_result = structured_llm.invoke(prompt)
    

    # result = cast(
    #         TransferabilityAnalysis,
    #         raw_result,
    #     )
    
    # results:list[Transferability]= []
    # for missing_skills in result.analyses:
    #     result = cast(
    #         Transferability,
    #         missing_skills
    #     )
    #     results.append(result)
        
    # return {
    #     "transferability": results
    # }
    llm = get_chat_model(config)

    raw_result = llm.invoke(prompt)

    content = raw_result.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    content = str(content).strip()

    print("\n========== TRANSFERABILITY RAW OUTPUT ==========")
    print(content)
    print("===============================================\n")

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        data = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Transferability analyzer returned invalid JSON.\n\n"
            f"Raw model output:\n{content}\n\n"
            f"JSON error: {exc}"
        ) from exc

    result = TransferabilityAnalysis.model_validate(data)

    return {
        "transferability": result.analyses,
    }
    
@traceable  
def calculate_score(
    state: JobMatchState,
) -> dict:

    required_skills = state.get("required_skills") or []
    candidate_skills = state.get("candidate_skills") or []

    skill_matches = state.get("skill_matches") or []

    if required_skills:

        matched_count = sum(
            1
            for match in skill_matches
            if match.matched
        )

        skills_score = (
            matched_count / len(required_skills)
        ) * 100

    else:
        skills_score = 100.0

    # Temporary scores.
    #
    # We should replace these with proper
    # experience/responsibility matching nodes.

    experience_score = 100.0
    responsibility_score = 100.0

    overall_score = (
        skills_score * 0.50
        + experience_score * 0.30
        + responsibility_score * 0.20
    )

    return {
        "skills_score": round(skills_score, 2),
        "experience_score": round(
            experience_score,
            2,
        ),
        "responsibility_score": round(
            responsibility_score,
            2,
        ),
        "overall_score": round(
            overall_score,
            2,
        ),
    }
    
    
@traceable
def match_skills(state: JobMatchState) -> dict:

    required_skills = state.get("required_skills") or []
    candidate_skills = state.get("candidate_skills") or []
    config = state.get("config")

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    if not required_skills:
        return {
            "skill_matches": [],
            "matching_skills": [],
            "missing_skills": [],
        }

    structured_llm = get_chat_model(
        config
    ).with_structured_output(SkillMatchResult)


    prompt = f"""
            You are a technical skill matching agent.

            Compare ALL required skills against the candidate's listed skills.

            CANDIDATE SKILLS:
            {candidate_skills}

            REQUIRED SKILLS:
            {required_skills}

            Rules:

            - Return exactly ONE SkillMatch for every required skill.
            - Match common technology naming variations.

            Examples:
            Angular == angular == Angular.js
            React == React.js
            Node == Node.js
            .NET == .NET Core

            Do not infer unrelated skills.

            Examples:

            Angular != React
            JavaScript != TypeScript
            Azure != AWS

            Only mark `matched=true` when the candidate's listed skills
            provide reasonable evidence.

            If matched:
            - provide concise evidence
            - confidence between 0 and 1

            If not matched:
            - evidence must be null
            - confidence should reflect your confidence that the skill
            is genuinely missing

            Return exactly one SkillMatch.
            """

    result = cast(
            SkillMatchResult,
            structured_llm.invoke(prompt),
        )
    matches = result.matches
    
     # Make sure the model didn't invent requirements.
    required_lookup = {
        skill.lower(): skill
        for skill in required_skills
    }
    
    valid_matches: list[SkillMatch] = []

    for match in matches:

        normalized = match.requirement.lower()

        if normalized not in required_lookup:
            continue

        # Preserve the original requirement from the job.
        match.requirement = required_lookup[normalized]

        valid_matches.append(match)

    matching_skills = [
        match.requirement
        for match in valid_matches
        if match.matched
    ]

    missing_skills = [
        match.requirement
        for match in valid_matches
        if not match.matched
    ]
    
    return {
            "skill_matches": valid_matches,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
        }

@traceable
def generate_critique(state: JobMatchState) -> dict:

    config = state.get("config")

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    candidate_name = state.get("candidate_name") or "Candidate"

    candidate_skills = state.get("candidate_skills") or []
    candidate_experience = (
        state.get("candidate_experience") or []
    )

    required_skills = state.get("required_skills") or []
    required_experience = (
        state.get("required_experience") or []
    )

    responsibilities = (
        state.get("responsibilities") or []
    )

    skill_matches = state.get("skill_matches") or []
    missing_skills = state.get("missing_skills") or []

    transferability = (
        state.get("transferability") or []
    )

    score = state.get("score", 0.0)

    prompt = f"""
        You are a senior technical recruiter and hiring advisor.

        Evaluate the candidate against the job requirements.

        Your assessment must be evidence-based.

        Do NOT invent experience or skills.

        CANDIDATE:
        {candidate_name}

        CANDIDATE SKILLS:
        {candidate_skills}

        CANDIDATE EXPERIENCE:
        {candidate_experience}

        REQUIRED SKILLS:
        {required_skills}

        REQUIRED EXPERIENCE:
        {required_experience}

        JOB RESPONSIBILITIES:
        {responsibilities}

        SKILL MATCHES:
        {skill_matches}

        MISSING SKILLS:
        {missing_skills}

        TRANSFERABILITY ANALYSIS:
        {transferability}

        OVERALL SCORE:
        {score}

        Generate a professional candidate critique.

        Consider:

        1. Candidate strengths
        - Skills that directly match the job
        - Relevant experience
        - Relevant responsibilities

        2. Candidate weaknesses
        - Important gaps
        - Missing required experience
        - Missing technical skills

        3. Missing skills
        - Only skills identified as missing by the matching analysis

        4. Learning potential
        - Assess whether the candidate appears capable of
            learning the missing skills based on related skills
            and experience.
        - Do not assume learning ability without evidence.
        - Use the transferability analysis when available.

        5. Recommendations
        - Should the candidate proceed to interview?
        - What should the interviewer verify?
        - What technical areas should be tested?

        6. Overall assessment
        - Provide a concise hiring-oriented assessment.

        Be balanced.

        Do not reject a candidate solely because of one missing skill
        if the transferability analysis indicates that the skill is
        reasonably learnable.

        Return only the structured critique.
        
        Keep the critique concise.
        Limit the response to approximately 300-500 words.
        """

    llm = get_chat_model(config).with_structured_output(
        Critique
    )

    result = cast(
        Critique,
        llm.invoke(prompt),
    )

    return {
        "critique": result,
    }