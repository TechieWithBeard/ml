from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    candidate_name: str | None = Field(
        default=None,
        description="Candidate's full name exactly as stated in the resume."
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills explicitly mentioned in the resume."
    )

    experience: list["Experience"] = Field(
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
        default_factory=list,
        description="List of skill match results for each job requirement."
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
        description="Short list of relevant candidate strengths."
    )

    weaknesses: list[str] = Field(
        default_factory=list,
        description="Short list of relevant candidate weaknesses or gaps."
    )

    missing_skills: list[str] = Field(
        default_factory=list,
        description="Required skills that are missing from the candidate."
    )

    learning_potential: str = Field(
        description="Short assessment of the candidate's ability to learn missing skills."
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Short practical recommendations for the hiring manager."
    )

    overall_assessment: str = Field(
        description="Short overall assessment of candidate suitability."
    )
    
    
    
    

class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None


class SkillGroup(BaseModel):
    category: str
    skills: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: str
    title: str
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str
    institution: str
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    details: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    url: str | None = None


class ResumeDocument(BaseModel):

    candidate_name: str

    headline: str | None = None

    contact: ContactInfo = Field(
        default_factory=ContactInfo
    )

    summary: str | None = None

    skills: list[SkillGroup] = Field(
        default_factory=list
    )

    experience: list[Experience] = Field(
        default_factory=list
    )

    education: list[Education] = Field(
        default_factory=list
    )

    certifications: list[str] = Field(
        default_factory=list
    )

    projects: list[Project] = Field(
        default_factory=list
    )
    
class BulletChange(BaseModel):
    original: str
    revised: str
    reason: str


class ExperienceTailoring(BaseModel):
    company: str

    bullet_changes: list[BulletChange] = Field(
        default_factory=list
    )


class ResumeTailoring(BaseModel):
    headline: str | None = None

    summary: str | None = None

    experience: list[ExperienceTailoring] = Field(
        default_factory=list
    )

    skill_groups_order: list[str] = Field(
        default_factory=list
    )