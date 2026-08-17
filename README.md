# Ask My Notes

A personal Retrieval-Augmented Generation (RAG) application that answers questions from your PDFs and notes. It retrieves the most relevant passages with FAISS, then instructs an OpenAI or Anthropic model to answer only from that context and cite its sources.

## Features

- Upload PDF, TXT, and Markdown notes
- Page-aware text chunking with overlap
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS cosine-similarity retrieval
- OpenAI or Anthropic response generation
- Source labels, retrieved-passage previews, and similarity scores
- Guardrail prompt that asks the model to say when the notes do not contain an answer

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Add either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env`, choose that provider in the sidebar, upload your notes, and select **Index my notes**.

## Architecture

```text
Documents → extraction → overlapping chunks → Sentence-Transformer embeddings
          → FAISS similarity search → retrieved context → cited LLM answer
```

## Notes

Uploaded documents and the FAISS index live only in the running application session. They are not saved to disk by this project.
