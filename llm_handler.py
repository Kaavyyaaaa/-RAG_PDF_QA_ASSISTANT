"""
LLM loading and inference.

Supports multiple HuggingFace LLMs (default: google/flan-t5-base) for local Q&A generation.
"""
from typing import Any, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from functools import lru_cache
from loguru import logger

DEFAULT_MODEL = "google/flan-t5-base"  # Use base for speed; can switch to large if needed

@lru_cache(maxsize=2)
def load_llm(model_name: str = DEFAULT_MODEL) -> Any:
    """
    Loads and caches a HuggingFace LLM and tokenizer for inference.
    Args:
        model_name (str): Model name or path.
    Returns:
        pipeline: HuggingFace pipeline for text2text-generation.
    """
    try:
        logger.info(f"Loading LLM: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.info(f"LLM loaded: {model_name}")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load LLM {model_name}: {e}")
        raise

def format_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say 'I don't know.' "
        "Be as detailed as possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

def generate_answer(
    question: str,
    context: str,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 256
) -> Optional[str]:
    """
    Generates an answer using the specified LLM and context.
    Args:
        question (str): User question.
        context (str): Retrieved context.
        model_name (str): LLM to use.
        max_new_tokens (int): Max tokens to generate.
    Returns:
        Optional[str]: Generated answer, or None if failed.
    """
    try:
        pipe = load_llm(model_name)
        prompt = format_prompt(question, context)
        result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1
        )
        answer = result[0]["generated_text"].strip()
        return answer
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return None