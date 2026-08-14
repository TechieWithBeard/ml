# Techie With Beard AI Lab 🧠

A hands-on AI/ML experimentation workspace focused on building practical **LLM-powered applications with Python, LangChain, RAG, local models, vector databases, and Streamlit**.

This project is part of my exploration into building production-oriented AI features that can eventually be integrated into my developer portfolio.

## 🚀 Current Project

### Resume Analyzer & RAG Application

The current application is a Streamlit-based Resume Analyzer that allows users to:

- 📄 Upload a PDF resume
- 🔍 Extract and chunk document content
- 🧠 Generate embeddings using Ollama
- 🗄️ Store embeddings in Chroma Cloud
- 🔎 Retrieve relevant resume sections using semantic similarity search
- 🤖 Ask questions about the uploaded resume
- 📦 Return structured responses using Pydantic models
- 🔗 Analyze external profile links as part of the next stage of the project

### Current Architecture

```text
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
                    │      Gemma          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Structured Output   │
                    │                     │
                    │ Pydantic Models     │
                    │ ResumeAnalyser      │
                    └─────────────────────┘