from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from typing import List
from .llm import create_flan_t5_pipeline
from .vector_store import build_vector_store
from .config import AppConfig, LLM_FLAN_T5_BASE, LLM_FLAN_T5_LARGE, LLM_FLAN_T5_SMALL

# Default prompt optimised for Flan-T5 models
FLAN_T5_PROMPT_TEMPLATE = """Context: {context}

Question: {question}

Answer:"""

FLAN_PROMPT = PromptTemplate(
    template=FLAN_T5_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def get_qa_chain(
    texts: List[str] | None = None,
    vectorstore = None,
    use_8bit: bool = AppConfig["load_in_8bit"],
    model_name: str = AppConfig["llm_model"],
) -> RetrievalQA:
    """
    Convenience function that returns a ready-to-use RetrievalQA chain.

    Args:
        texts: Optional list of text chunks (will build vector store if provided).
        vectorstore: Pre-built Chroma instance (overrides texts).
        use_8bit: Load LLM in 8-bit quantization.
        model_name: Which LLM to use.

    Returns:
        Configured RetrievalQA object.
    """
    if vectorstore is None:
        if texts is None:
            raise ValueError("Either `texts` or `vectorstore` must be provided.")
        vectorstore = build_vector_store(texts)

    # Currently only Flan-T5 family is battle-tested with the simple prompt above
    if model_name in {LLM_FLAN_T5_SMALL, LLM_FLAN_T5_BASE, LLM_FLAN_T5_LARGE}:
        hf_pipe = create_flan_t5_pipeline(model_name=model_name, load_in_8bit=use_8bit)
    else:
        raise NotImplementedError(f"Model {model_name} not yet wrapped.")

    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )

    # Apply custom prompt for Flan-T5 models
    qa_chain.combine_documents_chain.llm_chain.prompt = FLAN_PROMPT

    return qa_chain