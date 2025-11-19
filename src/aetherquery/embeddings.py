import torch
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Literal
from .config import EMB_MPNET_BASE, EMB_INSTRUCTOR_XL

def create_embeddings(model_name: str = EMB_MPNET_BASE) -> HuggingFaceEmbeddings:
    """
    Factory function that returns a LangChain-compatible embedding object.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings instance with automatic device placement.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )