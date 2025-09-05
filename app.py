"""
Streamlit frontend for RAG PDF Q&A Assistant.
"""
import streamlit as st
from pdf_processor import extract_text_from_pdf, chunk_text
from embedding_manager import embed_chunks, store_embeddings, clear_collection, reset_database
from rag_pipeline import answer_question
import os
from typing import List
from loguru import logger
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

# --- Constants ---
DATA_DIR = "data"
COLLECTION_NAME = "pdf_chunks"
EXAMPLE_QUESTIONS = [
    "What is the main topic of the document?",
    "Summarize the key findings.",
    "List the steps described in the manual.",
    "What are the limitations mentioned?",
    "Who are the authors?"
]

# --- Session State ---
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "loaded_pdfs" not in st.session_state:
        st.session_state.loaded_pdfs = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "error" not in st.session_state:
        st.session_state.error = None

# --- UI Styling ---
st.set_page_config(page_title="RAG PDF Q&A Assistant", layout="wide")
st.markdown("""
    <style>
    .stChatMessage {
        background: #f5f7fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        color: #222;
    }
    .stChatUser {
        background: #e3e9f7;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        color: #222;
    }
    .stSourceBox {
        background: #f0f0f0;
        border-radius: 6px;
        padding: 6px;
        font-size: 0.9em;
        color: #222;
    }
    .stConfidence {
        font-size: 0.95em;
        color: #2b7a0b;
    }
    body[data-theme='dark'] .stChatMessage,
    body[data-theme='dark'] .stChatUser,
    body[data-theme='dark'] .stSourceBox {
        background: #23272e !important;
        color: #f1f1f1 !important;
    }
    body[data-theme='dark'] .stConfidence {
        color: #8be78b !important;
    }
    body[data-theme='dark'] .stChatMessage *,
    body[data-theme='dark'] .stChatUser *,
    body[data-theme='dark'] .stSourceBox * {
        color: #f1f1f1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def display_error(msg: str):
    st.error(f"❌ {msg}")

def display_success(msg: str):
    st.success(f"✅ {msg}")

def show_sources(sources: List[dict]):
    for idx, src in enumerate(sources):
        with stylable_container(key=f"src_{idx}", css_styles="margin-bottom:4px;"):
            st.markdown(f"<div class='stSourceBox'>PDF: <b>{src['source_pdf']}</b> | Chunk: {src['chunk_index']} | Score: {src['score']:.2f}</div>", unsafe_allow_html=True)

# --- Main App ---
def main():
    init_session_state()
    st.title("📄 RAG PDF Q&A Assistant")
    st.caption("Upload PDFs, ask questions, and get intelligent answers with local AI!")
    add_vertical_space(1)

    # --- Sidebar: Document Management ---
    with st.sidebar:
        st.header("📚 Document Management")
        if st.button("Clear All Data", help="Remove all PDFs and reset database"):
            reset_database()
            st.session_state.loaded_pdfs = []
            st.session_state.chat_history = []
            display_success("Database cleared.")
        st.markdown("**Loaded PDFs:**")
        if st.session_state.loaded_pdfs:
            for pdf in st.session_state.loaded_pdfs:
                st.markdown(f"- {os.path.basename(pdf)}")
        else:
            st.info("No PDFs loaded yet.")
        add_vertical_space(2)
        st.markdown("---")
        st.markdown("**Example Questions:**")
        for q in EXAMPLE_QUESTIONS:
            st.markdown(f"- _{q}_")
        add_vertical_space(2)
        st.markdown("---")
        st.markdown("<small>Powered by open-source AI. All processing is local.</small>", unsafe_allow_html=True)

    # --- Main Area: PDF Upload ---
    st.subheader("1. Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are processed locally."
    )
    if uploaded_files:
        st.session_state.processing = True
        progress = st.progress(0, text="Processing PDFs...")
        for idx, uploaded_file in enumerate(uploaded_files):
            pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            text = extract_text_from_pdf(pdf_path)
            if not text:
                display_error(f"Failed to extract text from {uploaded_file.name}.")
                continue
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            store_embeddings(embeddings, chunks, uploaded_file.name)
            if uploaded_file.name not in st.session_state.loaded_pdfs:
                st.session_state.loaded_pdfs.append(uploaded_file.name)
            progress.progress((idx + 1) / len(uploaded_files), text=f"Processed {uploaded_file.name}")
        progress.empty()
        st.session_state.processing = False
        display_success("All PDFs processed and indexed!")
        add_vertical_space(1)

    # --- Main Area: Chat Interface ---
    st.subheader("2. Ask Questions")
    if not st.session_state.loaded_pdfs:
        st.info("Please upload at least one PDF to begin.")
        return

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='stChatUser'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='stChatMessage'><b>Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                show_sources(msg["sources"])
            if msg.get("confidence") is not None:
                st.markdown(f"<span class='stConfidence'>Confidence: {msg['confidence']:.2f}</span>", unsafe_allow_html=True)

    # User input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Type your question about the PDFs:", placeholder="e.g. What is the main topic?")
        submit = st.form_submit_button("Ask")

    if submit and user_question:
        with st.spinner("Thinking..."):
            response = answer_question(user_question)
        if response["error"]:
            display_error(response["error"])
        else:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"] or "Sorry, I couldn't find an answer.",
                "sources": response["sources"],
                "confidence": response["confidence"]
            })
            st.rerun()

    # --- Limitations & Info ---
    add_vertical_space(2)
    with stylable_container(key="limitations", css_styles="background:#fffbe6;padding:12px;border-radius:8px;"):
        st.markdown("**Limitations:**\n- Only answers questions based on uploaded PDFs.\n- Local models may not match GPT-4 quality.\n- Large PDFs may take time to process.\n- No internet access required after model download.")

if __name__ == "__main__":
    main()