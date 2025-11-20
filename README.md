# AetherQuery

**AetherQuery** is a lightweight, modular Retrieval-Augmented Generation (RAG) system built with LangChain, Hugging Face Transformers, and ChromaDB.

It demonstrates how to:
- Load and chunk documents (PDF supported)
- Create embeddings with sentence-transformers
- Store vectors locally with Chroma
- Query documents using open-source LLMs (Flan-T5, Falcon, etc.) in 8-bit or full precision

Perfect as a starting template for private, offline document Q&A applications.

## Installation

```bash
git clone https://github.com/yourorg/aetherquery.git
cd aetherquery
pip install -r <(pip freeze | grep -E "langchain|chromadb|sentence-transformers|transformers|accelerate|pdfplumber|torch")
