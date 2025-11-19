from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from typing import List
from .embeddings import create_embeddings
from .config import AppConfig

def build_vector_store(
    texts: List[str],
    persist_directory: str | None = AppConfig["persist_directory"],
    embedding_model: str = AppConfig["embedding_model"],
) -> Chroma:
    """
    Creates (or loads) a Chroma vector store from raw text chunks.

    Args:
        texts: List of text chunks.
        persist_directory: Directory to persist Chroma collection (None = in-memory).
        embedding_model: HuggingFace model name.

    Returns:
        Chroma vectorstore instance.
    """
    embedding = create_embeddings(embedding_model)

    documents = [Document(page_content=t, metadata={}) for t in texts]

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
    )
    if persist_directory:
        vectordb.persist()
    return vectordb