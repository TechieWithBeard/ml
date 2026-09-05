import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from techiewithbeard_ai.job_match.schemas import ResumeDocument

TEMPLATE_DIR = Path(__file__).resolve().parent
TEMPLATE_NAME = "TalentLens_Resume_Template.html"


def _get_template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_cv_html(
    resume: ResumeDocument,
) -> str:
    """
    Render a ResumeDocument into the download-ready TalentLens A4 HTML CV.

    The template controls presentation.
    ResumeDocument controls content.
    """

    template = _get_template_environment().get_template(TEMPLATE_NAME)

    return template.render(
        resume=resume,
    )


def render_cv_preview_html(
    resume: ResumeDocument,
) -> str:
    """
    Render a ResumeDocument into HTML optimized for Gradio preview.
    """

    html_content = render_cv_html(resume)

    # Wrap with preview styles for better Gradio rendering
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                "Helvetica Neue", Arial, sans-serif;
        }}
        .resume-preview-container {{
            transform: scale(0.8);
            transform-origin: top left;
            width: 125%;
        }}
    </style>
</head>
<body>
    <div class="resume-preview-container">
        {html_content}
    </div>
</body>
</html>
"""


def render_resume_html(
    resume: ResumeDocument,
) -> str:
    """
    Backward-compatible alias for the Gradio preview renderer.
    """

    return render_cv_preview_html(resume)


def write_cv_html(
    resume: ResumeDocument,
) -> str:
    """
    Write the download-ready CV HTML to a temporary file.
    """

    html_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        prefix="talentlens_cv_",
        delete=False,
        encoding="utf-8",
    )

    html_file.write(render_cv_html(resume))
    html_file.close()

    return html_file.name


def write_cv_pdf(
    resume: ResumeDocument,
) -> str | None:
    """
    Write the CV as a PDF when WeasyPrint is installed.

    Returns None when the optional PDF dependency is unavailable.
    """

    try:
        from weasyprint import HTML
    except ImportError:
        return None

    pdf_file = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        prefix="talentlens_cv_",
        delete=False,
    )
    pdf_file.close()

    HTML(string=render_cv_html(resume), base_url=str(TEMPLATE_DIR)).write_pdf(
        pdf_file.name
    )

    return pdf_file.name
