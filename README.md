Techie With Beard AI Lab 🧠
A hands-on AI/ML experimentation workspace focused on building practical LLM-powered applications using Python, LangChain, LangGraph, RAG, local models, vector databases, Pydantic, and Streamlit.
The goal is to explore how AI concepts can be turned into practical, production-oriented applications.
🚀 Current Projects
1. Resume Analyzer & RAG Application
A Streamlit-based Resume Analyzer that allows users to:
📄 Upload a PDF resume
🔍 Extract and chunk document content
🧠 Generate embeddings using Ollama
🗄️ Store embeddings in Chroma Cloud
🔎 Retrieve relevant resume sections using semantic similarity search
🤖 Ask questions about the uploaded resume
📦 Generate structured responses using Pydantic models
🔗 Analyze external profile links as part of the next stage
2. AI Job Match Analyzer
A LangGraph-powered workflow that analyzes a candidate's resume against a job description.
The workflow currently:
📄 Parses the candidate's resume
💼 Extracts technical requirements from the job description
🔎 Matches candidate skills against required skills
⚠️ Identifies missing skills
🔄 Analyzes skill transferability
📊 Calculates an overall candidate score
📝 Generates a candidate critique
🕸️ Visualizes the workflow using Mermaid
The application currently runs locally using Ollama + Gemma, allowing the entire analysis pipeline to run without relying on a hosted LLM API.
🧠 What I'm Learning
This project is also a learning lab for exploring:
LangChain
LangGraph
RAG architectures
Prompt engineering
Structured LLM output
Pydantic validation
Local LLMs with Ollama
Embedding models
Vector databases
Semantic search
Agentic workflows
Streamlit
AI application architecture
LLM performance optimization
🏗️ Current RAG Architecture
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    │                     │
                    │  PDF Upload         │
                    │  Resume Q&A         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PDF Loader       │
                    │                     │
                    │ PyPDFLoader         │
                    │ Text Splitting      │
                    │ File Hashing        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ollama Embeddings   │
                    │                     │
                    │ embeddinggemma      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Chroma Cloud      │
                    │                     │
                    │ Vector Storage      │
                    │ Semantic Search     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     RAG Chain       │
                    │                     │
                    │ Retrieve Context    │
                    │ Build Prompt        │
                    │ Generate Answer     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Ollama LLM      │
                    │                     │
                    │       Gemma         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Structured Output  │
                    │                     │
                    │  Pydantic Models    │
                    │  ResumeAnalyzer     │
                    └─────────────────────┘
​
🔄 Job Match Architecture
                    ┌─────────────────────┐
                    │     Resume PDF      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Parse Resume     │
                    │                     │
                    │ Candidate Name      │
                    │ Skills              │
                    │ Experience          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Parse Requirements  │
                    │                     │
                    │ Required Skills     │
                    │ Experience          │
                    │ Responsibilities    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Match Skills     │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
              Missing Skills?            │
                     │                   │
                    Yes                  No
                     │                   │
                     ▼                   │
          ┌─────────────────────┐        │
          │ Transferability     │        │
          │ Analysis            │        │
          └──────────┬──────────┘        │
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Calculate Score     │
                    │                     │
                    │ Skill Score         │
                    │ Experience Score    │
                    │ Responsibility Score│
                    │ Overall Score       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Generate Critique    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Final Report     │
                    └─────────────────────┘
​
⚡ Performance Work in Progress
The Job Match Analyzer currently runs completely locally using Ollama + Gemma.
The current end-to-end workflow takes around 5 minutes on my local setup.
The next focus is improving performance and identifying bottlenecks:
⏱️ Identify the slowest nodes
🔍 Profile individual LLM calls
📉 Reduce unnecessary model invocations
🔄 Explore parallel LangGraph execution
🧩 Optimize prompts and context size
🧠 Evaluate model configuration and token limits
⚡ Improve structured-output performance
📊 Measure latency across the complete workflow
The goal is to understand where the bottlenecks are and progressively improve the architecture rather than simply switching to a larger or faster hosted model.
🛠️ Tech Stack
Python
LangChain
LangGraph
Ollama
Gemma
Pydantic
Chroma
Streamlit
PyPDF
PDM
📁 Project Structure
ml/
├── data/
├── notebooks/
├── streamlit/
│   ├── app.py
│   └── pages/
│       └── job_match.py
├── src/
│   └── techiewithbeard_ai/
│       ├── agents/
│       ├── chains/
│       ├── job_match/
│       │   ├── graph.py
│       │   ├── nodes.py
│       │   ├── schemas.py
│       │   └── state.py
│       ├── loaders/
│       └── schema/
├── tests/
├── pyproject.toml
└── README.md
​
🚧 What's Next?
Improve Job Match Analyzer performance
Add better experience and responsibility matching
Improve transferability analysis
Add explainable scoring
Experiment with parallel LangGraph nodes
Add evaluation and benchmarking
Explore different local models
Experiment with hosted LLMs
Expand RAG capabilities
Integrate the projects into my developer portfolio
🔗 Project
GitHub: [Add your GitHub repository link here]
This repository is an ongoing AI/ML learning and experimentation lab.
The focus is not just on building AI features, but understanding the architecture, trade-offs, performance, and engineering challenges behind them.