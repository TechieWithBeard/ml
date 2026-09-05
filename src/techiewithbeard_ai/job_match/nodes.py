from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.job_match.schemas import JobRequirements, ResumeDocument, ResumeTailoring, SkillMatchResult, TransferabilityAnalysis, Transferability, SkillMatch, Critique
from techiewithbeard_ai.job_match.state import JobMatchState
from techiewithbeard_ai.schema.provider import ModelConfig
from langchain_core.output_parsers import PydanticOutputParser

SKILL_MATCH_BATCH_SIZE = 5

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
    
    
    
# @traceable
# def parse_resume(
#     state: JobMatchState,
# ) -> dict:

#     resume_text = state.get("resume_text")
#     config = state.get("config")

#     if not resume_text:
#         raise ValueError(
#             "Resume text is missing from graph state."
#         )

#     if config is None:
#         raise ValueError(
#             "Model configuration is missing from graph state."
#         )

#     parser = JsonOutputParser()

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
# You are a resume information extraction system.

# Extract information ONLY from the resume provided by the user.

# Return ONLY a valid JSON object.

# The JSON MUST have exactly these fields:

# {{
#   "candidate_name": null,
#   "skills": [],
#   "experience": []
# }}

# Rules:

# - candidate_name must be the candidate's full name if explicitly present.
# - Use null if the name is not explicitly present.
# - skills must contain only technical skills explicitly mentioned in the resume.
# - experience must be a list of structured work experience objects.
# - Do NOT infer skills.
# - Do NOT invent experience.
# - Do NOT add fields that are not part of the schema.
# - Use [] when no skills or experience are found.
# - Use null when candidate_name is unavailable.
# - Do NOT use Markdown.
# - Do NOT use ```json.
# - Do NOT add explanations before or after the JSON.
# - Return complete and valid JSON.
# - The response must start with {{ and end with }}.

# Example:

# {{
#   "candidate_name": "John Doe",
#   "skills": [
#     "Python",
#     "FastAPI",
#     "React"
#   ],
#   "experience": [
#     {{
#       "title": "Senior Software Engineer",
#       "company": "ABC",
#       "start_date": "2022",
#       "end_date": "Present"
#     }},
#     {{
#       "title": "Software Engineer",
#       "company": "XYZ",
#       "start_date": "2020",
#       "end_date": "2022"
#     }}
#   ]
# }}
# """,
#             ),
#             (
#                 "human",
#                 """
# RESUME:

# {resume_text}
# """,
#             ),
#         ]
#     )

#     llm = get_chat_model(config)

#     chain = prompt | llm | parser

#     try:
#         data = chain.invoke(
#             {
#                 "resume_text": resume_text,
#             }
#         )

#     except Exception as exc:

#         print("\n========== RESUME PARSING ERROR ==========")
#         print(repr(exc))
#         print("==========================================\n")

#         raise ValueError(
#             "The model did not return valid JSON "
#             "for resume extraction."
#         ) from exc

#     print("\n========== RESUME JSON OUTPUT ==========")
#     print(data)
#     print("========================================\n")

#     try:
#         result = ResumeProfile.model_validate(data)

#     except Exception as exc:

#         print("\n========== RESUME VALIDATION ERROR ==========")
#         print(repr(exc))
#         print("=============================================\n")

#         raise ValueError(
#             "Resume JSON does not match the ResumeProfile schema."
#         ) from exc

#     return {
#         "candidate_name": result.candidate_name,
#         "candidate_skills": result.skills,
#         "candidate_experience": result.experience,
#     }
    
    
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

    parser = PydanticOutputParser(
        pydantic_object=ResumeDocument
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a resume extraction system.

Convert the provided resume text into the
ResumeDocument schema.

IMPORTANT:

The resume is the ONLY source of truth.

Extract information exactly from the resume.

Do NOT:
- invent information
- infer missing skills
- invent dates
- invent achievements
- invent companies
- invent contact information
- add technologies not present

Preserve the original meaning and wording.

If information does not exist:
- use null for optional scalar fields
- use [] for lists

{format_instructions}
"""
            ),
            (
                "human",
                """
RESUME:

{resume_text}
"""
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "resume_text": resume_text,
        }
    )

    print("\n========== RESUME DOCUMENT ==========")
    print(result)
    print("=====================================\n")

    return {
        "resume_document": result,

        # Keep these because your existing
        # job-match nodes already depend on them.
        "candidate_name": result.candidate_name,
        "candidate_skills": [
            skill
            for group in result.skills
            for skill in group.skills
        ],
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
        "skill_score": round(skills_score, 2),
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
    
 
 
 
def match_skill_batch(
    llm,
    candidate_skills: list[str],
    required_skills: list[str],
) -> SkillMatchResult:

    parser = PydanticOutputParser(
        pydantic_object=SkillMatchResult,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a technical skill matching agent.

Compare the candidate's skills against the required skills.

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
- confidence must be between 0 and 1

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

    return result

   
   
   
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

    llm = get_chat_model(config)

    # ---------------------------------------------------------
    # Split required skills into batches.
    # ---------------------------------------------------------

    batches = [
        required_skills[
            i:i + SKILL_MATCH_BATCH_SIZE
        ]
        for i in range(
            0,
            len(required_skills),
            SKILL_MATCH_BATCH_SIZE,
        )
    ]

    print("\n========== SKILL MATCH BATCHING ==========")
    print(
        f"Required skills: {len(required_skills)}"
    )
    print(
        f"Batch size: {SKILL_MATCH_BATCH_SIZE}"
    )
    print(
        f"Number of batches: {len(batches)}"
    )
    print("==========================================\n")

    all_matches: list[SkillMatch] = []

    # ---------------------------------------------------------
    # Process each batch.
    # ---------------------------------------------------------

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"\n========== SKILL MATCH BATCH "
            f"{batch_number}/{len(batches)} =========="
        )

        print("Required skills:")
        for skill in batch:
            print(f"  - {skill}")

        try:

            result = match_skill_batch(
                llm=llm,
                candidate_skills=candidate_skills,
                required_skills=batch,
            )

        except Exception as exc:

            print(
                f"\n========== BATCH {batch_number} FAILED =========="
            )
            print(repr(exc))
            print("===============================================\n")

            raise ValueError(
                f"Skill matching failed for batch "
                f"{batch_number}/{len(batches)}."
            ) from exc

        print("\nBatch result:")
        print(result)

        all_matches.extend(result.matches)

    # ---------------------------------------------------------
    # Validate model output.
    # ---------------------------------------------------------

    required_lookup = {
        skill.strip().lower(): skill
        for skill in required_skills
    }

    valid_matches: list[SkillMatch] = []

    for match in all_matches:

        normalized = (
            match.requirement.strip().lower()
        )

        if normalized not in required_lookup:
            continue

        # Preserve exact requirement from job description.
        match.requirement = (
            required_lookup[normalized]
        )

        valid_matches.append(match)

    # ---------------------------------------------------------
    # Make sure the model returned every requirement.
    # ---------------------------------------------------------

    returned_requirements = {
        match.requirement.strip().lower()
        for match in valid_matches
    }

    missing_from_model = [
        skill
        for skill in required_skills
        if skill.strip().lower()
        not in returned_requirements
    ]

    if missing_from_model:

        raise ValueError(
            "Model did not return matches for: "
            + ", ".join(missing_from_model)
        )

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

    # ---------------------------------------------------------
    # Final output.
    # ---------------------------------------------------------

    print("\n========== FINAL SKILL MATCH OUTPUT ==========")
    print(
        f"Total matches: {len(valid_matches)}"
    )
    print(
        f"Matching skills: {matching_skills}"
    )
    print(
        f"Missing skills: {missing_skills}"
    )
    print("==============================================\n")

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
    
    

@traceable
def tailor_resume(
    state: JobMatchState,
) -> dict:

    resume_document = state.get("resume_document")
    job_description = state.get("job_description")
    required_skills = state.get("required_skills") or []
    required_experience = state.get("required_experience") or []
    responsibilities = state.get("responsibilities") or []
    skill_matches = state.get("skill_matches") or []
    missing_skills = state.get("missing_skills") or []
    critique = state.get("critique")
    config = state.get("config")

    if resume_document is None:
        raise ValueError(
            "ResumeDocument is missing from graph state."
        )

    if not job_description:
        raise ValueError(
            "Job description is missing from graph state."
        )

    if config is None:
        raise ValueError(
            "Model configuration is missing from graph state."
        )

    parser = PydanticOutputParser(
        pydantic_object=ResumeTailoring
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a professional resume tailoring specialist.

Your task is to identify ONLY the changes that would
improve an existing resume for a target job.

IMPORTANT:

The original ResumeDocument is the SINGLE SOURCE
OF TRUTH.

You are NOT creating a new resume.

You are returning ONLY proposed changes.

==================================================
ALLOWED CHANGES
==================================================

You may:

1. Improve the professional headline.
2. Improve the professional summary.
3. Rewrite existing experience bullets.
4. Reorder existing experience bullets when relevant.
5. Reorder existing skill groups to emphasize relevant
   groups for the target job.

==================================================
STRICT FACTUAL RULES
==================================================

DO NOT invent:

- Skills
- Technologies
- Frameworks
- Companies
- Job titles
- Dates
- Responsibilities
- Achievements
- Certifications
- Projects
- Education

DO NOT add missing skills.

DO NOT create new experience.

DO NOT remove experience.

DO NOT create new bullets.

The tailored resume must never contain a
new factual claim that is not supported by
the original ResumeDocument.

==================================================
EXACT BULLET MATCHING
==================================================

For every BulletChange:

1. `original` MUST be copied EXACTLY from the
   original ResumeDocument.
2. `original` MUST be a complete existing bullet.
3. Do NOT truncate the original bullet.
4. Do NOT summarize the original bullet.
5. Do NOT paraphrase the original field.
6. Do NOT change punctuation in the original field.
7. Do NOT change numbers or percentages.
8. The original value MUST exactly match one
   existing bullet in the source resume.

The `revised` value may improve wording, clarity,
keyword alignment, and emphasis, but must preserve
the facts contained in the original bullet.

==================================================
HEADLINE RULES
==================================================

The headline may be improved to better position
the candidate for the target role.

However:

- Do not invent seniority.
- Do not claim expertise not supported by the resume.
- Do not introduce technologies not present.
- Do not change the candidate into a different role.

==================================================
SUMMARY RULES
==================================================

The summary may be rewritten to better emphasize
experience relevant to the job.

It must remain fully supported by the original
ResumeDocument.

Do not introduce new experience or skills.

==================================================
SKILL GROUP ORDER
==================================================

`skill_groups_order` must contain ONLY exact
category names already present in:

ResumeDocument.skills[].category

Do not return individual skill names.

Do not create new categories.

Do not add skills.

Do not remove skills.

Only recommend the ordering of existing skill groups.

==================================================
MISSING SKILLS
==================================================

If a required skill is missing from the original
resume, DO NOT add it.

Example:

Original:
Angular
TypeScript
JavaScript

Job requirement:
Electron

DO NOT add Electron to the resume.

==================================================
OUTPUT RULES
==================================================

Return ONLY the proposed changes.

Do NOT return a complete ResumeDocument.

Do NOT return unchanged sections.

If no change is required:

- headline = null
- summary = null
- experience = []
- skill_groups_order = []

{format_instructions}
"""
            ),
            (
                "human",
                """
ORIGINAL RESUME:

{resume_document}

JOB DESCRIPTION:

{job_description}

REQUIRED SKILLS:

{required_skills}

REQUIRED EXPERIENCE:

{required_experience}

RESPONSIBILITIES:

{responsibilities}

SKILL MATCH ANALYSIS:

{skill_matches}

MISSING SKILLS:

{missing_skills}

RECRUITER CRITIQUE:

{critique}

Return ONLY the proposed resume changes.
"""
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    llm = get_chat_model(config)

    chain = prompt | llm | parser

    result = cast(
        ResumeTailoring,
        chain.invoke(
            {
                "resume_document": resume_document.model_dump(),
                "job_description": job_description,
                "required_skills": required_skills,
                "required_experience": required_experience,
                "responsibilities": responsibilities,
                "skill_matches": skill_matches,
                "missing_skills": missing_skills,
                "critique": critique,
            }
        ),
    )

    print("\n========== RESUME TAILORING ==========")
    print(result)
    print("======================================\n")

    return {
        "resume_tailoring": result,
    }


def apply_resume_tailoring(
    resume: ResumeDocument,
    tailoring: ResumeTailoring,
) -> ResumeDocument:

    updated = resume.model_copy(deep=True)

    # =========================================================
    # Headline
    # =========================================================

    if tailoring.headline:
        updated.headline = tailoring.headline.strip()

    # =========================================================
    # Summary
    # =========================================================

    if tailoring.summary:
        updated.summary = tailoring.summary.strip()

    # =========================================================
    # Experience bullet changes
    # =========================================================

    for experience_change in tailoring.experience:

        matched_experience = None

        for experience in updated.experience:

            if (
                experience.company.strip().lower()
                == experience_change.company.strip().lower()
            ):
                matched_experience = experience
                break

        if matched_experience is None:

            print(
                "\n⚠️ TAILORING WARNING"
            )
            print(
                "Experience company was not found:"
            )
            print(
                experience_change.company
            )

            continue

        for bullet_change in experience_change.bullet_changes:

            original_bullet = bullet_change.original.strip()
            revised_bullet = bullet_change.revised.strip()

            found = False

            for index, existing_bullet in enumerate(
                matched_experience.bullets
            ):

                if existing_bullet.strip() == original_bullet:

                    matched_experience.bullets[index] = (
                        revised_bullet
                    )

                    found = True
                    break

            if not found:

                print(
                    "\n⚠️ TAILORING WARNING"
                )
                print(
                    f"Company: {matched_experience.company}"
                )
                print(
                    "Original bullet was not found:"
                )
                print(
                    repr(bullet_change.original)
                )

    # =========================================================
    # Skill group ordering
    # =========================================================

    if tailoring.skill_groups_order:

        existing_groups = {
            group.category.strip().lower(): group
            for group in updated.skills
        }

        reordered_groups = []

        # Add AI-requested groups first.
        for category in tailoring.skill_groups_order:

            normalized_category = (
                category.strip().lower()
            )

            group = existing_groups.get(
                normalized_category
            )

            if group is None:
                print(
                    "\n⚠️ TAILORING WARNING"
                )
                print(
                    "Skill group was not found:"
                )
                print(
                    category
                )
                continue

            if group not in reordered_groups:
                reordered_groups.append(group)

        # Preserve groups AI did not mention.
        for group in updated.skills:

            if group not in reordered_groups:
                reordered_groups.append(group)

        updated.skills = reordered_groups

    return updated


def validate_resume_preservation(
    original: ResumeDocument,
    tailored: ResumeDocument,
) -> None:

    if len(original.experience) != len(
        tailored.experience
    ):
        raise ValueError(
            "Resume tailoring changed the number "
            "of experience entries."
        )

    if len(original.education) != len(
        tailored.education
    ):
        raise ValueError(
            "Resume tailoring changed education."
        )

    if len(original.projects) != len(
        tailored.projects
    ):
        raise ValueError(
            "Resume tailoring changed projects."
        )

    if len(original.certifications) != len(
        tailored.certifications
    ):
        raise ValueError(
            "Resume tailoring changed certifications."
        )
        
 
def validate_resume_document(
    resume: ResumeDocument,
) -> None:

    if not resume.candidate_name:
        raise ValueError(
            "Resume has no candidate name."
        )

    if not resume.experience:
        raise ValueError(
            "Resume contains no experience."
        )

    for index, job in enumerate(
        resume.experience,
        start=1,
    ):

        if not job.company:
            raise ValueError(
                f"Experience #{index} has no company."
            )

        if not job.title:
            raise ValueError(
                f"Experience #{index} has no title."
            )

        if not job.bullets:
            print(
                f"⚠️ Experience #{index} has no bullets: "
                f"{job.company}"
            )       
        

@traceable
def apply_tailoring(
    state: JobMatchState,
) -> dict:

    resume_document = state.get("resume_document")
    resume_tailoring = state.get("resume_tailoring")

    if resume_document is None:
        raise ValueError(
            "ResumeDocument is missing."
        )

    if resume_tailoring is None:
        raise ValueError(
            "ResumeTailoring is missing."
        )

    tailored_resume = apply_resume_tailoring(
        resume_document,
        resume_tailoring,
    )

    validate_resume_preservation(
        resume_document,
        tailored_resume,
    )

    print("\n========== TAILORED RESUME ==========")
    print(tailored_resume)
    print("=====================================\n")

    return {
        "tailored_resume": tailored_resume,
    }