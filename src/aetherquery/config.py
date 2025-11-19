from typing import Literal

# Available embedding models (anonymized identifiers)
EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"

# Available LLMs
LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"

AppConfig = {
    "persist_directory": None,               # Set path str to persist Chroma DB
    "load_in_8bit": False,
    "embedding_model": EMB_MPNET_BASE,
    "llm_model": LLM_FLAN_T5_BASE,
}