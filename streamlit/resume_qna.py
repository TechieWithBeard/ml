import os
import tempfile

import streamlit as st

from techiewithbeard_ai.loaders.pdf_loader import pdf_loader
from techiewithbeard_ai.chains.rag_chain import build_rag_chain
from techiewithbeard_ai.schema.provider import ModelConfig


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Techie With Beard AI Lab",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "chain" not in st.session_state:
    st.session_state.chain = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "provider" not in st.session_state:
    st.session_state.provider = "Ollama"

if "chat_model" not in st.session_state:
    st.session_state.chat_model = "gemma4:e4b"

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "embeddinggemma:latest"

if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"

if "chroma_mode" not in st.session_state:
    st.session_state.chroma_mode = "local"

if "chain" not in st.session_state:
    st.session_state.chain = None
    
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"

with st.sidebar:

    st.title("🤖 AI Configuration")

    provider = st.selectbox(
        "Model Provider",
        options=[
            "Ollama",
            "Hugging Face",
        ],
    )

    # Always initialize these variables
    chat_model = "gemma4:e4b"
    embedding_model = "embeddinggemma:latest"
    ollama_url = DEFAULT_OLLAMA_URL

    if provider == "Ollama":

        st.subheader("🦙 Ollama")

        ollama_url = st.text_input(
            "Ollama URL",
            value=DEFAULT_OLLAMA_URL,
        )

        chat_model = st.text_input(
            "Chat Model",
            value="gemma4:e4b",
        )

        embedding_model = st.text_input(
            "Embedding Model",
            value="embeddinggemma:latest",
        )

    else:

        st.subheader("🤗 Hugging Face")

        chat_model = st.text_input(
            "Chat Model",
            value="Qwen/Qwen3-8B",
        )
        hf_token = st.sidebar.text_input(
            "Hugging Face Token",
            type="password",
            help="Your Hugging Face access token. It is used only for this session."
        )
        embedding_model = st.text_input(
            "Embedding Model",
            value="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Ollama isn't used for HF
        ollama_url = ""

    if st.button(
        "⚙️ Apply Configuration",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.provider = provider
        st.session_state.chat_model = chat_model
        st.session_state.embedding_model = embedding_model
        st.session_state.ollama_url = ollama_url

        st.session_state.chain = None

        st.success("Configuration updated.")
        
    st.divider()

    # ==============================================
    # VECTOR STORE CONFIGURATION
    # ==============================================

    st.subheader("🗄️ Vector Store")

    chroma_mode = st.selectbox(
        "Chroma Storage",
        ["Local", "Cloud"],
        index=(
            0
            if st.session_state.chroma_mode == "local"
            else 1
        ),
        help=(
            "Local stores vectors on this machine. "
            "Cloud uses Chroma Cloud."
        ),
    )

    if chroma_mode == "Local":

        st.info(
            "📁 Vectors will be stored locally in "
            "`./data/chroma`."
        )

        chroma_mode_value = "local"

    else:

        st.info(
            "☁️ Using Chroma Cloud. "
            "Credentials are loaded from environment secrets."
        )

        chroma_mode_value = "cloud"

    apply_chroma = st.button(
        "Apply Chroma Configuration",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📄 RAG Document Assistant")

st.markdown(
    "Upload a PDF and ask questions about its content using "
    "your configured AI model."
)


# ---------------------------------------------------------
# Current configuration
# ---------------------------------------------------------

with st.expander("🔧 Current AI Configuration"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Provider",
            st.session_state.provider,
        )

    with col2:
        st.metric(
            "Chat Model",
            st.session_state.chat_model,
        )

    with col3:
        st.metric(
            "Embedding Model",
            st.session_state.embedding_model,
        )


# ---------------------------------------------------------
# Build RAG chain
# ---------------------------------------------------------

def get_chain():

    provider = st.session_state.provider
    config = ModelConfig(
        provider=provider,
        chat_model=st.session_state.chat_model or "gemma4:e4b",
        embedding_model=st.session_state.embedding_model or "embeddinggemma:latest",
        ollama_url=st.session_state.ollama_url or"http://localhost:11434",
    )
    
    #  { 
    #         provider=provider,
    #         chat_model=st.session_state.chat_model,
    #         embedding_model=st.session_state.embedding_model,
    #         ollama_url=st.session_state.ollama_url}
    # Pass configuration to your chain builder.
    #
    # You will update build_rag_chain() to accept these.
    return build_rag_chain(
      config
    )


if st.session_state.chain is None:

    with st.spinner(
        f"Initializing {st.session_state.provider} AI models..."
    ):
        try:
            st.session_state.chain = get_chain()

        except Exception as e:
            st.error(
                "Unable to initialize the AI model."
            )
            st.exception(e)

            st.stop()


chain = st.session_state.chain


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

st.subheader("📄 Document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a PDF to add it to the RAG knowledge base.",
)


if uploaded_file is not None:

    st.info(
        f"📎 **{uploaded_file.name}** "
        f"({uploaded_file.size / 1024:.1f} KB)"
    )

    if st.button(
        "🚀 Process Document",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Processing PDF and creating embeddings..."
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_pdf_path = tmp.name

            try:

                provider = st.session_state.provider
                
                config = ModelConfig(
                        provider=provider,
                        chat_model=st.session_state.chat_model or "gemma4:e4b",
                        embedding_model=st.session_state.embedding_model or "embeddinggemma:latest",
                        ollama_url=st.session_state.ollama_url or"http://localhost:11434",
                    )
                result = pdf_loader(config,
                    temp_pdf_path
                )

                if result.status == "already_exists":

                    st.info(result.message)

                else:

                    st.success(result.message)

                st.session_state.document_processed = True
                st.session_state.document_name = (
                    uploaded_file.name
                )

                # Clear old conversation when a new document
                # is processed.
                st.session_state.messages = []

            except Exception as e:

                st.error(
                    "Failed to process the PDF."
                )

                st.exception(e)

            finally:

                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)


# ---------------------------------------------------------
# Document status
# ---------------------------------------------------------

if st.session_state.document_processed:

    st.success(
        f"✅ Ready to answer questions about "
        f"**{st.session_state.document_name}**"
    )

else:

    st.warning(
        "Upload and process a PDF before asking questions."
    )


st.divider()


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

prompt = st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.document_processed,
)


if prompt:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            try:

                response = chain.invoke(
                    {
                        "query": prompt,
                    }
                )

                answer = response["answer"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.error(
                    "Something went wrong while answering your question."
                )

                st.exception(e)