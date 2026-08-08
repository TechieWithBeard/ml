from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class LinkedInExperience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None


class LinkedInEducation(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field: str | None = None


class LinkedInAnalysis(BaseModel):
    profile_url: str

    name: str | None = None
    headline: str | None = None
    location: str | None = None

    about: str | None = None

    current_role: str | None = None
    current_company: str | None = None

    experience: list[LinkedInExperience] = Field(default_factory=list)
    education: list[LinkedInEducation] = Field(default_factory=list)

    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    profile_summary: str | None = None


class LinkedInAnalyzer:
    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://localhost:11434",
    ):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
        ).with_structured_output(LinkedInAnalysis)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are an expert LinkedIn profile analyzer.

                        Analyze ONLY the LinkedIn profile content provided by the user.

                        Extract:
                        - candidate name
                        - headline
                        - location
                        - about section
                        - current role
                        - current company
                        - previous work experience
                        - education
                        - skills
                        - certifications
                        - an overall profile summary

                        Rules:

                        1. Do not invent information.
                        2. Do not infer information that is not explicitly present.
                        3. If information is missing, return null.
                        4. If a list has no data, return an empty list.
                        5. Preserve company names, job titles and technologies accurately.
                        6. The profile_summary must be based only on the supplied content.
                        """,
                                        ),
                                        (
                                            "human",
                                            """
                        LinkedIn Profile URL:
                        {profile_url}

                        LinkedIn Profile Content:
                        {profile_content}
                    """,
                ),
            ]
        )

        self.chain = self.prompt | self.llm

    def analyze(
        self,
        profile_url: str,
        profile_content: str,
    ) -> LinkedInAnalysis:

        result = self.chain.invoke(
            {
                "profile_url": profile_url,
                "profile_content": profile_content,
            }
        )

        if isinstance(result, LinkedInAnalysis):
            print(result)
            return result

        return LinkedInAnalysis.model_validate(result)