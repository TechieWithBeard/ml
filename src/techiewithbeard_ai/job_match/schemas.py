from pydantic import BaseModel, Field

class Experience(BaseModel):
    title: str | None = Field(
        default=None,
        description="Job title or position held."
    )
    company: str | None = Field(
        default=None,
        description="Company or organization name."
    )
    duration: str | None = Field(
        default=None,
        description="Duration of employment (e.g., 'Jan 2020 - Dec 2021')."
    )
        
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Summry of responsibilities and tasks performed in the role."
    )


class ResumeProfile(BaseModel):
    candidate_name: str | None = Field(
        default=None,
        description="Candidate's full name exactly as stated in the resume."
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills explicitly mentioned in the resume."
    )

    experience: list[Experience] = Field(
        default_factory=list,
        description="Brief summaries of each relevant work experience. Keep each item concise."
    )


class JobRequirements(BaseModel):
    required_skills: list[str] = Field(
        default_factory=list,
        description="Technical skills explicitly required by the job."
    )

    required_experience: list[str] = Field(
        default_factory=list,
        description="Experience requirements mentioned in the job description."
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="Responsibilities expected from the candidate."
    )
    
    
class SkillMatch(BaseModel):

    requirement: str = Field(
        description="The job requirement being evaluated."
    )

    matched: bool = Field(
        description="Whether the candidate has this skill."
    )

    evidence: str | None = Field(
        default=None,
        description="Evidence from the candidate's listed skills."
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the matching decision."
    )


class SkillMatchResult(BaseModel):
    matches: list[SkillMatch] = Field(
        default_factory=list
    )


class Transferability(BaseModel):
    missing_skill: str

    related_skills: list[str] = Field(
        default_factory=list
    )

    transferability_score: float = Field(
        ge=0.0,
        le=100.0,
        description="How easily the candidate can transfer existing skills to the missing skill, from 0 to 100."
    )

    learning_difficulty: str

    reasoning: str
       
class TransferabilityAnalysis(BaseModel):
    analyses: list[Transferability] = Field(
        default_factory=list
    )



class CandidateScore(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    skills_score: float = Field(ge=0, le=100)
    experience_score: float = Field(ge=0, le=100)
    responsibility_score: float = Field(ge=0, le=100)
    reasoning: str
    
class Critique(BaseModel):
    strengths: list[str] = Field(
        default_factory=list,
        description="Candidate strengths relevant to the job."
    )

    weaknesses: list[str] = Field(
        default_factory=list,
        description="Candidate weaknesses or gaps relevant to the job."
    )

    missing_skills: list[str] = Field(
        default_factory=list,
        description="Required skills that the candidate does not currently demonstrate."
    )

    learning_potential: str = Field(
        description="Assessment of how realistically the candidate can learn the missing skills."
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for the recruiter or hiring manager."
    )

    overall_assessment: str = Field(
        description="Concise overall assessment of the candidate's suitability."
    )