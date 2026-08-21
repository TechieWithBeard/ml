Techie With Beard AI Lab 🧠
A hands-on AI/ML experimentation workspace focused on building practical, production-oriented applications with Python, LLMs, LangChain, LangGraph, RAG, local models, vector databases, Pydantic, and Streamlit.
This repository documents my journey of experimenting with AI application architecture, structured LLM workflows, local model execution, evaluation, observability, and performance optimization.
🚀 Current Projects
1. Resume Analyzer & RAG Application
A Streamlit-based Resume Analyzer that allows users to:
📄 Upload a PDF resume
🔍 Extract and chunk document content
🧠 Generate embeddings using Ollama
🗄️ Store embeddings in Chroma
🔎 Retrieve relevant resume sections using semantic similarity
🤖 Ask questions about the uploaded resume
📦 Generate structured responses using Pydantic models
🔗 Work with external profile information as the project evolves
Resume Analyzer Architecture
flowchart LR
    A[PDF Resume] --> B[PDF Loader]
    B --> C[Text Splitting]
    C --> D[Ollama Embeddings]
    D --> E[Chroma]
    E --> F[RAG Chain]
    F --> G[Gemma]
    G --> H[Pydantic Output]
PDF Resume

PDF Loader

Text Splitting

Ollama Embeddings

Chroma

RAG Chain

Gemma

Pydantic Output

​
2. AI Job Match Analyzer
A LangGraph-powered workflow that compares a candidate's resume against a job description.
The workflow currently:
📄 Parses the candidate's resume
💼 Extracts job requirements
🛠️ Matches candidate skills against required skills
⚠️ Identifies missing skills
🔄 Analyzes transferable skills
📊 Calculates candidate fit scores
📝 Generates an AI-powered critique
🧩 Uses conditional LangGraph routing based on missing skills
📦 Uses Pydantic models for structured LLM output
Job Match Workflow
flowchart TD
    A[Resume] --> C[Parse Resume]
    B[Job Description] --> D[Parse Requirements]

    C --> E[Match Skills]
    D --> E

    E --> F{Missing Skills?}

    F -->|Yes| G[Transferability]
    F -->|No| H[Calculate Score]

    G --> H
    H --> I[Generate Critique]
    I --> J[Final Report]
Resume

Parse Resume

Job Description

Parse Requirements

Match Skills

Missing Skills?

Transferability

Calculate Score

Generate Critique

Final Report

Yes

No

​
🤖 Models
The project currently focuses on running models locally using Ollama.
Purpose
Model
Chat / LLM
gemma4:e4b
Embeddings
embeddinggemma:latest
Runtime
Ollama
Running models locally makes experimentation inexpensive and keeps the workflow under local control.
🧰 Technology Stack
AI / LLM
Python
LangChain
LangGraph
Ollama
Gemma
Pydantic
RAG
Vector Search
Chroma
Application
Streamlit
Development
PDM
Pytest
Git
📂 Project Structure
ml/
├── data/
├── artifacts/
├── chroma_db/
├── notebooks/
├── tests/
├── streamlit/
│   ├── Home.py
│   └── pages/
│       └── job_match.py
├── src/
│   └── techiewithbeard_ai/
│       ├── agents/
│       ├── chains/
│       ├── job_match/
│       ├── loaders/
│       └── schema/
├── pyproject.toml
├── pdm.lock
└── README.md
​
▶️ Running the Streamlit Application
1. Install dependencies
This project uses PDM for dependency management.
pdm install
​
Activate the virtual environment if required:
source .venv/bin/activate
​
2. Start Ollama
Make sure Ollama is running locally and the required models are available.
ollama serve
​
Pull the models if they are not already installed:
ollama pull gemma4:e4b
ollama pull embeddinggemma:latest
​
3. Start Streamlit
From the project root:
python -m streamlit run streamlit/Home.py
​
The application will be available at:
<http://localhost:8501>
​
🧠 What I'm Learning
This repository is not just about building individual AI applications. It is also an exploration of the engineering challenges involved in making LLM applications reliable and production-ready.
Current areas of exploration include:
LLM application architecture
RAG pipelines
LangGraph workflows
Structured LLM output
Pydantic validation
Local LLM inference
Prompt engineering
Agentic workflows
Vector search
Evaluation
Observability
Performance optimization
⚡ Performance Investigation
The AI Job Match Analyzer currently runs entirely using a local Gemma model through Ollama.
The complete workflow currently takes approximately 5 minutes on my local setup.
This makes performance optimization the next major focus.
I'm currently looking at:
⏱️ Identifying slow nodes in the LangGraph workflow
🔎 Understanding model inference latency
🔁 Reducing unnecessary LLM calls
🧩 Improving structured output generation
📦 Optimizing prompt size
⚡ Exploring parallel execution where possible
📊 Adding tracing and observability
🧪 Evaluating output quality alongside performance
Tools and approaches I'm exploring include LangSmith, evaluation frameworks, and more advanced agentic workflows.
🔭 Roadmap

PDF resume extraction

Resume RAG pipeline

Local embeddings with Ollama

Vector search with Chroma

Structured outputs with Pydantic

Resume parsing

Job requirement extraction

Skill matching

Missing skill detection

Transferability analysis

Candidate scoring

AI-generated critique

Conditional LangGraph workflow

Improve workflow performance

Add LangSmith tracing

Add automated evaluation

Improve evaluation metrics

Experiment with parallel LangGraph nodes

Explore agentic workflows

Improve UI and visualizations

Add LinkedIn profile analysis

Deploy the application
📈 Development Philosophy
Build → Measure → Understand → Optimize → Repeat
The goal is to learn by building real applications rather than only experimenting with isolated models or tutorials.
🔗 Project
The project is continuously evolving as I experiment with different AI application patterns, models, frameworks, and architectures.
More experiments and improvements will be added over time.
👨‍💻 Author
Vishnu Thankappan
Senior Frontend Engineer | Angular | Nx | AI Interfaces
⭐ If you find the experiments useful, feel free to explore the repository and follow along with the progress.