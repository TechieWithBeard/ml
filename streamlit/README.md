 Streamlit UI 🖥️
This folder contains the Streamlit frontend for the AI/ML applications in this project.
The UI provides a simple interface for interacting with the RAG applications and AI-powered resume/job analysis workflows.
🚀 Running the Application
From the project root:
python -m streamlit run streamlit/Home.py
​
Streamlit will start a local development server, typically available at:
<http://localhost:8501>
​
📁 Structure
streamlit/
├── Home.py
├── pages/
│   └── job_match.py
└── README.md
​
Home.py
The main Streamlit entry point for the application.
It provides access to the different AI/ML applications available in the project.
pages/job_match.py
The AI Job Match Analyzer interface.
It allows users to:
📄 Upload a candidate resume
💼 Enter a job description
🧠 Run the LangGraph analysis workflow
🔎 Compare candidate skills with job requirements
⚠️ Identify missing skills
🔄 Analyze skill transferability
📊 Calculate candidate scores
📝 Generate a candidate critique
🕸️ Visualize the LangGraph workflow
🧠 Application Flow
                    Streamlit
                        │
                        ▼
                 ┌─────────────┐
                 │   Home.py   │
                 └──────┬──────┘
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Resume Analyzer       Job Match Analyzer
             │                     │
             │                     ▼
             │                 LangGraph
             │                     │
             │        ┌────────────┴────────────┐
             │        │                         │
             │        ▼                         ▼
             │   Resume Parser          Requirements Parser
             │        │                         │
             │        └──────────┬──────────────┘
             │                   ▼
             │              Skill Matching
             │                   │
             │          ┌────────┴────────┐
             │          │                 │
             │          ▼                 ▼
             │   Transferability       Scoring
             │          │                 │
             │          └────────┬────────┘
             │                   ▼
             │              Critique
             │                   │
             └───────────────────┴──────► Result
​
⚙️ Requirements
Make sure the Python environment is activated before running Streamlit.
Example:
source .venv/bin/activate
​
The project dependencies are managed through pyproject.toml and PDM.
If using PDM:
pdm install
​
🤖 Local LLM
The application currently supports local LLM execution using Ollama.
Make sure Ollama is running before launching the application.
Example:
ollama serve
​
The default Ollama endpoint is:
<http://localhost:11434>
​
The current configuration uses:
Chat Model:
gemma4:e4b

Embedding Model:
embeddinggemma:latest
​
🔬 Development
The Streamlit UI is primarily intended as an experimentation and demonstration layer for the AI components under:
src/techiewithbeard_ai/
​
The application is expected to evolve as new AI workflows, agents, RAG pipelines, and experiments are added.
📌 Notes
The Job Match Analyzer currently runs locally using Ollama + Gemma.
The complete analysis workflow can take several minutes depending on the local hardware and model configuration.
Performance profiling and optimization are currently ongoing, with a focus on identifying bottlenecks within the LangGraph workflow and individual LLM calls.


 python -m streamlit run streamlit/Home.py