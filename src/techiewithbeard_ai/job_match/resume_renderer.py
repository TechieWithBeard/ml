from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from techiewithbeard_ai.job_match.schemas import ResumeDocument


TEMPLATE_DIR = Path(__file__).resolve().parent
TEMPLATE_NAME = "TalentLens_Resume_Template.html"


def render_resume_html(
    resume: ResumeDocument,
) -> str:
    """
    Render a ResumeDocument into the TalentLens A4 HTML template.

    The template controls presentation.
    ResumeDocument controls content.
    """

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(TEMPLATE_NAME)

    html_content = template.render(
        resume=resume,
    )
    
    # Wrap with preview styles for better Gradio rendering
    wrapped_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
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
    
    return wrapped_html
