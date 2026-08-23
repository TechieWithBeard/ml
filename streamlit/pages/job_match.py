import os
import tempfile

import streamlit as st
from pypdf import PdfReader
from streamlit_mermaid import st_mermaid

from techiewithbeard_ai.job_match.graph import build_job_match_graph
from techiewithbeard_ai.schema.provider import ModelConfig


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Job Match Analyzer",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# Session state
# =========================================================

DEFAULT_SESSION_STATE = {
    "provider": "Ollama",
    "chat_model": "gemma4:e4b",
    "embedding_model": "embeddinggemma:latest",
    "ollama_url": "http://localhost:11434",
    "hf_token": None,
}


for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# Model configuration
# =========================================================

def get_model_config() -> ModelConfig:
    return ModelConfig(
        provider=st.session_state.provider,
        chat_model=(
            st.session_state.chat_model
            or "gemma4:e4b"
        ),
        embedding_model=(
            st.session_state.embedding_model
            or "embeddinggemma:latest"
        ),
        ollama_url=(
            st.session_state.ollama_url
            or "http://localhost:11434"
        ),
        temperature=0.0,
        max_new_tokens=768,
    )


# =========================================================
# Resume extraction
# =========================================================

def extract_resume_text(uploaded_file) -> str:

    temp_pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            temp_pdf_path = tmp.name

        reader = PdfReader(temp_pdf_path)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    finally:

        if (
            temp_pdf_path
            and os.path.exists(temp_pdf_path)
        ):
            os.remove(temp_pdf_path)


# =========================================================
# Header
# =========================================================

st.title("🎯 AI Job Match Analyzer")

st.markdown(
    """
Compare a candidate's resume against a job description
using a **LangGraph-powered analysis workflow**.

The analyzer evaluates:

- Technical skill match
- Experience relevance
- Job responsibilities
- Missing skills
- Transferable skills
- Learning potential
- Overall candidate fit
"""
)

st.divider()


# =========================================================
# Input section
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# Resume
# ---------------------------------------------------------

with left:

    st.subheader("📄 Candidate Resume")

    resume_file = st.file_uploader(
        "Upload candidate resume",
        type=["pdf"],
        help="Upload the candidate's PDF resume.",
    )

    if resume_file:

        st.success(
            f"Selected: {resume_file.name}"
        )

        st.caption(
            f"{resume_file.size / 1024:.1f} KB"
        )


# ---------------------------------------------------------
# Job description
# ---------------------------------------------------------

with right:

    st.subheader("💼 Job Requirements")

    job_description = st.text_area(
        "Paste the job description",
        height=300,
        placeholder="""
Example:

We are looking for a Senior Angular Developer
with 5+ years of experience.

Required:
- Angular
- TypeScript
- RxJS
- NgRx
- REST APIs

Nice to have:
- AWS
- Docker
- Kubernetes
""",
    )


st.divider()


# =========================================================
# Analyze button
# =========================================================

analyze = st.button(
    "🚀 Analyze Candidate",
    type="primary",
    use_container_width=True,
)


if analyze:

    # =====================================================
    # Validate input
    # =====================================================

    if resume_file is None:

        st.error(
            "Please upload a candidate resume."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "Please enter the job description."
        )

        st.stop()


    # =====================================================
    # Extract resume
    # =====================================================

    with st.spinner(
        "📄 Reading candidate resume..."
    ):

        try:

            resume_text = extract_resume_text(
                resume_file
            )

        except Exception as exc:

            st.error(
                "Unable to read the resume."
            )

            st.exception(exc)

            st.stop()


    if not resume_text.strip():

        st.error(
            "No text could be extracted from the PDF."
        )

        st.stop()


    # =====================================================
    # Model configuration
    # =====================================================

    config = get_model_config()


    # =====================================================
    # Run LangGraph
    # =====================================================

    with st.spinner(
        "🤖 Running candidate analysis..."
    ):

        try:

            graph = build_job_match_graph()

            # -------------------------------------------------
            # Render graph
            # -------------------------------------------------

            mermaid = (
                graph
                .get_graph()
                .draw_mermaid()
            )

            with st.expander(
                "🔍 View Analysis Workflow"
            ):

                st_mermaid(mermaid)

            # -------------------------------------------------
            # Invoke graph
            # -------------------------------------------------

            result = graph.invoke(
                {
                    "resume_text": resume_text,
                    "job_description": job_description,
                    "config": config,
                }
            )

        except Exception as exc:

            st.error(
                "Unable to analyze the candidate."
            )

            st.exception(exc)

            st.stop()


    # =====================================================
    # Analysis completed
    # =====================================================

    st.success(
        "✅ Candidate analysis completed."
    )

    st.divider()


    # =====================================================
    # Candidate
    # =====================================================

    st.subheader("👤 Candidate")

    candidate_name = result.get(
        "candidate_name"
    )

    st.markdown(
        f"### {candidate_name or 'Unknown'}"
    )


    # =====================================================
    # Candidate skills
    # =====================================================

    st.subheader("🛠️ Candidate Skills")

    candidate_skills = result.get(
        "candidate_skills",
        [],
    )

    if candidate_skills:

        st.write(
            ", ".join(candidate_skills)
        )

    else:

        st.info(
            "No skills extracted."
        )


    # =====================================================
    # Required skills
    # =====================================================

    st.subheader("📋 Required Skills")

    required_skills = result.get(
        "required_skills",
        [],
    )

    if required_skills:

        st.write(
            ", ".join(required_skills)
        )

    else:

        st.info(
            "No technical skills found in the job description."
        )


    # =====================================================
    # Skill matching
    # =====================================================

    st.divider()

    st.subheader("🎯 Skill Analysis")


    skill_matches = result.get(
        "skill_matches",
        [],
    )


    matching_skills = [
        match.requirement
        for match in skill_matches
        if match.matched
    ]


    missing_skills = [
        match.requirement
        for match in skill_matches
        if not match.matched
    ]


    col1, col2 = st.columns(2)


    # ---------------------------------------------------------
    # Matching
    # ---------------------------------------------------------

    with col1:

        st.markdown(
            "### ✅ Matching Skills"
        )

        if matching_skills:

            for skill in matching_skills:

                st.success(
                    skill
                )

        else:

            st.info(
                "No matching skills found."
            )


    # ---------------------------------------------------------
    # Missing
    # ---------------------------------------------------------

    with col2:

        st.markdown(
            "### ⚠️ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:

                st.warning(
                    skill
                )

        else:

            st.success(
                "No missing skills!"
            )


    # =====================================================
    # Detailed skill evidence
    # =====================================================

    if skill_matches:

        st.divider()

        st.subheader(
            "🔎 Skill Match Evidence"
        )

        for match in skill_matches:

            with st.expander(
                f"{'✅' if match.matched else '⚠️'} "
                f"{match.requirement}"
            ):

                st.write(
                    f"**Matched:** "
                    f"{'Yes' if match.matched else 'No'}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{match.confidence:.0%}"
                )

                if match.evidence:

                    st.write(
                        f"**Evidence:** "
                        f"{match.evidence}"
                    )


    # =====================================================
    # Transferability
    # =====================================================

    transferability = result.get(
        "transferability",
        [],
    )

    if transferability:

        st.divider()

        st.subheader(
            "🔄 Transferability Analysis"
        )

        for item in transferability:

            with st.expander(
                f"🧠 {item.missing_skill}"
            ):

                st.write(
                    f"**Transferability:** "
                    f"{item.transferability_score:.0%}"
                )

                st.write(
                    f"**Learning difficulty:** "
                    f"{item.learning_difficulty}"
                )

                if item.related_skills:

                    st.write(
                        "**Related skills:** "
                        + ", ".join(
                            item.related_skills
                        )
                    )

                st.write(
                    f"**Reasoning:** "
                    f"{item.reasoning}"
                )


    # =====================================================
    # Scores
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Candidate Score"
    )

    score_col1, score_col2, score_col3, score_col4 = (
        st.columns(4)
    )


    with score_col1:

        st.metric(
            "Overall",
            f"{result.get('overall_score', 0):.0f}%",
        )


    with score_col2:

        st.metric(
            "Skills",
            f"{result.get('skill_score', 0):.0f}%",
        )


    with score_col3:

        st.metric(
            "Experience",
            f"{result.get('experience_score', 0):.0f}%",
        )


    with score_col4:

        st.metric(
            "Responsibilities",
            f"{result.get('responsibility_score', 0):.0f}%",
        )


    # =====================================================
    # Critique
    # =====================================================

    critique = result.get(
        "critique"
    )

    if critique:

        st.divider()

        st.subheader(
            "🧠 AI Candidate Critique"
        )

        # Pydantic object
        if hasattr(
            critique,
            "strengths",
        ):

            st.markdown(
                "### 💪 Strengths"
            )

            for item in critique.strengths:

                st.success(item)


            st.markdown(
                "### ⚠️ Weaknesses"
            )

            for item in critique.weaknesses:

                st.warning(item)


            st.markdown(
                "### 📚 Learning Potential"
            )

            st.write(
                critique.learning_potential
            )


            st.markdown(
                "### 💡 Recommendations"
            )

            for item in critique.recommendations:

                st.info(item)


    # =====================================================
    # Debug information
    # =====================================================

    with st.expander(
        "🛠️ Debug: LangGraph State"
    ):

        st.json(
            {
                key: (
                    value.model_dump()
                    if hasattr(value, "model_dump")
                    else value
                )
                for key, value in result.items()
                if key not in {
                    "resume_text",
                    "job_description",
                    "config",
                }
            }
        )