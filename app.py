import os

import streamlit as st
from dotenv import load_dotenv

from src.documents import load_documents
from src.llm import answer_question
from src.retrieval import KnowledgeBase

load_dotenv()

st.set_page_config(page_title="Ask My Notes", page_icon="📚", layout="wide")

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_knowledge_base(uploaded_files):
    progress = st.progress(0, text="Reading documents…")
    chunks = load_documents(uploaded_files)
    if not chunks:
        raise ValueError("No readable text was found in the uploaded files.")
    progress.progress(40, text="Generating embeddings…")
    kb = KnowledgeBase()
    kb.build(chunks)
    progress.progress(100, text="Your notes are ready to query.")
    progress.empty()
    return kb


with st.sidebar:
    st.title("📚 Ask My Notes")
    st.caption("Private, source-grounded answers from your documents.")

    provider = st.selectbox("LLM provider", ["OpenAI", "Anthropic"])
    model = (
        st.text_input("Model", value="gpt-4o-mini")
        if provider == "OpenAI"
        else st.text_input("Model", value="claude-3-5-haiku-latest")
    )
    uploaded_files = st.file_uploader(
        "Upload notes", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Index my notes", type="primary", disabled=not uploaded_files):
        try:
            st.session_state.knowledge_base = build_knowledge_base(uploaded_files)
            st.session_state.messages = []
            st.success("Notes indexed successfully.")
        except Exception as error:
            st.error(f"Could not index the notes: {error}")

    if st.session_state.knowledge_base:
        stats = st.session_state.knowledge_base.stats
        st.divider()
        st.caption(f"{stats['documents']} document(s) • {stats['chunks']} chunks")
        if st.button("Clear session"):
            st.session_state.knowledge_base = None
            st.session_state.messages = []
            st.rerun()

st.title("Ask questions about your notes")
st.write("Upload PDFs, text files, or Markdown notes, then receive answers grounded in retrieved passages.")

if not st.session_state.knowledge_base:
    st.info("Start by uploading one or more documents from the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['label']}** — similarity: {source['score']:.2f}")
                    st.caption(source["preview"])

question = st.chat_input(
    "Ask a question about your uploaded notes…",
    disabled=st.session_state.knowledge_base is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding relevant passages and drafting an answer…"):
            try:
                results = st.session_state.knowledge_base.search(question, k=5)
                answer = answer_question(question, results, provider.lower(), model)
                st.markdown(answer)
                with st.expander("Sources used"):
                    for source in results:
                        st.markdown(f"- **{source['label']}** — similarity: {source['score']:.2f}")
                        st.caption(source["preview"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": results}
                )
            except Exception as error:
                st.error(str(error))
