from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
from typing import Literal
from .config import (
    LLM_FLAN_T5_BASE,
    LLM_FLAN_T5_LARGE,
    LLM_FLAN_T5_SMALL,
    LLM_FALCON_7B_INSTRUCT,
)

def create_flan_t5_pipeline(
    model_name: str = LLM_FLAN_T5_BASE,
    load_in_8bit: bool = False,
) -> pipeline:
    """Creates a text2text-generation pipeline for any Flan-T5 variant."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return pipeline(
        task="text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.0,
        model_kwargs={
            "device_map": "auto",
            "load_in_8bit": load_in_8bit,
            "torch_dtype": torch.bfloat16 if not load_in_8bit else torch.float16,
        },
    )

def create_falcon_instruct_pipeline(load_in_8bit: bool = False) -> pipeline:
    """Creates a text-generation pipeline for Falcon-7B-Instruct."""
    model = LLM_FALCON_7B_INSTRUCT
    tokenizer = AutoTokenizer.from_pretrained(model)

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        max_new_tokens=200,
        temperature=0.01,
        model_kwargs={
            "device_map": "auto",
            "load_in_8bit": load_in_8bit,
            "torch_dtype": torch.bfloat16,
        },
    )