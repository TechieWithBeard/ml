import json
from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.job_match.schemas import JobRequirements, ResumeProfile, SkillMatchResult, TransferabilityAnalysis, Transferability, SkillMatch, Critique
from techiewithbeard_ai.job_match.state import JobMatchState
from techiewithbeard_ai.schema.provider import ModelConfig
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser


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

    parser = PydanticOutputParser(
        pydantic_object=JobRequirements,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are a technical job requirements extraction system.

    Extract information ONLY from the job description.

    Rules:

    - required_skills: technical skills, technologies, frameworks,
    programming languages, tools, platforms, databases, and other
    technical requirements explicitly mentioned.
    - required_experience: experience requirements explicitly stated.
    - responsibilities: responsibilities explicitly stated.
    - Do NOT infer or invent information.
    - If a category is not present, return an empty list.
    - Keep extracted items concise.
    - Return complete and valid JSON.
    - The response must start with {{ and end with }}.

    {format_instructions}
    """,
        ),
        (
            "human",
            """
    JOB DESCRIPTION:

    {job_description}
    """,
        ),
    ]).partial(
        format_instructions=parser.get_format_instructions(),
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = cast(
        JobRequirements,
        chain.invoke(
            {
                "job_description": job_description,
            }
        ),
    )

    print("\n========== REQUIREMENTS OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "required_skills": result.required_skills,
        "required_experience": result.required_experience,
        "responsibilities": result.responsibilities,
    }
    
    
    
@traceable
def parse_resume(
    state: JobMatchState,
) -> dict:

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

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a resume information extraction system.

Extract information ONLY from the resume provided by the user.

Return ONLY a valid JSON object.

The JSON MUST have exactly these fields:

{{
  "candidate_name": null,
  "skills": [],
  "experience": []
}}

Rules:

- candidate_name must be the candidate's full name if explicitly present.
- Use null if the name is not explicitly present.
- skills must contain only technical skills explicitly mentioned in the resume.
- experience must be a list of structured work experience objects.
- Do NOT infer skills.
- Do NOT invent experience.
- Do NOT add fields that are not part of the schema.
- Use [] when no skills or experience are found.
- Use null when candidate_name is unavailable.
- Do NOT use Markdown.
- Do NOT use ```json.
- Do NOT add explanations before or after the JSON.
- Return complete and valid JSON.
- The response must start with {{ and end with }}.

Example:

{{
  "candidate_name": "John Doe",
  "skills": [
    "Python",
    "FastAPI",
    "React"
  ],
  "experience": [
    {{
      "title": "Senior Software Engineer",
      "company": "ABC",
      "start_date": "2022",
      "end_date": "Present"
    }},
    {{
      "title": "Software Engineer",
      "company": "XYZ",
      "start_date": "2020",
      "end_date": "2022"
    }}
  ]
}}
""",
            ),
            (
                "human",
                """
RESUME:

{resume_text}
""",
            ),
        ]
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    try:
        data = chain.invoke(
            {
                "resume_text": resume_text,
            }
        )

    except Exception as exc:

        print("\n========== RESUME PARSING ERROR ==========")
        print(repr(exc))
        print("==========================================\n")

        raise ValueError(
            "The model did not return valid JSON "
            "for resume extraction."
        ) from exc

    print("\n========== RESUME JSON OUTPUT ==========")
    print(data)
    print("========================================\n")

    try:
        result = ResumeProfile.model_validate(data)

    except Exception as exc:

        print("\n========== RESUME VALIDATION ERROR ==========")
        print(repr(exc))
        print("=============================================\n")

        raise ValueError(
            "Resume JSON does not match the ResumeProfile schema."
        ) from exc

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
    config = state.get("config")

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    if not missing_skills:
        return {
            "transferability": []
        }

    parser = PydanticOutputParser(
        pydantic_object=TransferabilityAnalysis,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a technical skills transferability analyzer.

Your job is to analyze how transferable the candidate's existing
skills are to each missing skill.

IMPORTANT:

You MUST return ALL fields defined in the output schema.

Every analysis object MUST contain EXACTLY these fields:

- missing_skill
- related_skills
- transferability_score
- learning_difficulty
- reasoning

Never omit any of these fields.

Candidate skills:

{candidate_skills}

Missing skills:

{missing_skills}

Rules:

1. Return exactly one analysis for every missing skill.

2. `missing_skill` must exactly match one of the provided missing skills.

3. `related_skills` may ONLY contain skills from the candidate's
   existing skills list.

4. Never claim that a related skill means the candidate already knows
   the missing skill.

5. `transferability_score` must ALWAYS be an integer from 0 to 100.

6. `learning_difficulty` must ALWAYS be one of:
   "low"
   "medium"
   "high"

7. `reasoning` must ALWAYS be present and concise.

8. If there are no related skills, return:

"related_skills": []

9. Do not invent candidate skills.

10. Do not omit fields.

11. Do not add fields.

For example, if the missing skill is C++ and the candidate has
TypeScript and JavaScript, the response MUST look like:

{{
  "analyses": [
    {{
      "missing_skill": "C++",
      "related_skills": [
        "TypeScript",
        "JavaScript"
      ],
      "transferability_score": 65,
      "learning_difficulty": "medium",
      "reasoning": "The candidate has experience with programming languages that provide transferable programming concepts."
    }}
  ]
}}

{format_instructions}
""",
            ),
            (
                "human",
                """
Candidate skills:

{candidate_skills}

Missing skills:

{missing_skills}

Return the complete JSON response now.
""",
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions(),
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = cast(
        TransferabilityAnalysis,
        chain.invoke(
            {
                "candidate_skills": candidate_skills,
                "missing_skills": missing_skills,
            }
        ),
    )

    print("\n========== TRANSFERABILITY OUTPUT ==========")
    print(result)
    print("============================================\n")

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

    parser = PydanticOutputParser(
        pydantic_object=SkillMatchResult,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a technical skill matching agent.

Compare ALL required skills against the candidate's listed skills.

Candidate Skills:
{candidate_skills}

Required Skills:
{required_skills}

Rules:

- Return exactly ONE SkillMatch for every required skill.
- The `requirement` field must correspond to a skill from the
  required skills list.
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
- confidence must be between 0 and 1

If not matched:

- evidence must be null
- confidence should reflect your confidence that the skill
  is genuinely missing

Do not invent candidate skills.

Do not invent required skills.

Return exactly one SkillMatch for every required skill.

{format_instructions}
""",
            ),
            (
                "human",
                """
Candidate skills:

{candidate_skills}

Required skills:

{required_skills}
""",
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions(),
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = cast(
        SkillMatchResult,
        chain.invoke(
            {
                "candidate_skills": candidate_skills,
                "required_skills": required_skills,
            }
        ),
    )

    print("\n========== SKILL MATCH OUTPUT ==========")
    print(result)
    print("========================================\n")

    matches = result.matches

    # ---------------------------------------------------------
    # Validate model output against the original requirements.
    # ---------------------------------------------------------

    required_lookup = {
        skill.strip().lower(): skill
        for skill in required_skills
    }

    valid_matches: list[SkillMatch] = []

    for match in matches:

        normalized = match.requirement.strip().lower()

        if normalized not in required_lookup:
            continue

        # Preserve the exact requirement from the job description.
        match.requirement = required_lookup[normalized]

        valid_matches.append(match)

    # ---------------------------------------------------------
    # Build matching / missing skill lists.
    # ---------------------------------------------------------

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
    candidate_experience = state.get("candidate_experience") or []
    required_skills = state.get("required_skills") or []
    required_experience = state.get("required_experience") or []
    responsibilities = state.get("responsibilities") or []
    skill_matches = state.get("skill_matches") or []
    missing_skills = state.get("missing_skills") or []
    transferability = state.get("transferability") or []
    score = state.get("overall_score", 0.0)

    parser = PydanticOutputParser(
        pydantic_object=Critique,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a senior technical recruiter.

Evaluate the candidate against the job requirements.

Use ONLY the information provided below.
Do not invent skills or experience.

Candidate:
{candidate_name}

Candidate Skills:
{candidate_skills}

Candidate Experience:
{candidate_experience}

Required Skills:
{required_skills}

Required Experience:
{required_experience}

Responsibilities:
{responsibilities}

Skill Matches:
{skill_matches}

Missing Skills:
{missing_skills}

Transferability:
{transferability}

Overall Score:
{score}

{format_instructions}

Do not use markdown.
Do not use ```json.
Do not add any text before or after the JSON.
""",
            )
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "candidate_name": candidate_name,
            "candidate_skills": candidate_skills,
            "candidate_experience": candidate_experience,
            "required_skills": required_skills,
            "required_experience": required_experience,
            "responsibilities": responsibilities,
            "skill_matches": skill_matches,
            "missing_skills": missing_skills,
            "transferability": transferability,
            "score": score,
        }
    )

    print("\n========== CRITIQUE OUTPUT ==========")
    print(result)
    print("=====================================\n")

    return {
        "critique": result,
    }