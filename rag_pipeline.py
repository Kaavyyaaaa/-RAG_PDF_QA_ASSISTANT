"""
RAG pipeline: retrieval and generation logic.

Orchestrates retrieval of relevant PDF chunks and answer generation using a local LLM.
"""
from typing import List, Dict, Any, Optional
from embedding_manager import query_similar_chunks
from llm_handler import generate_answer
from loguru import logger

def answer_question(
    question: str,
    top_k: int = 5,
    collection_name: str = "pdf_chunks",
    model_name: str = "google/flan-t5-base"
) -> Dict[str, Any]:
    """
    Main RAG function: retrieves relevant chunks, generates answer, and formats response.
    Args:
        question (str): User's question.
        top_k (int): Number of chunks to retrieve.
        collection_name (str): ChromaDB collection name.
        model_name (str): LLM to use.
    Returns:
        Dict[str, Any]: Contains answer, sources, confidence, and retrieval info.
    """
    try:
        # Retrieve top-k relevant chunks
        retrieved = query_similar_chunks(question, top_k=top_k, collection_name=collection_name)
        if not retrieved:
            return {
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "chunks": [],
                "error": "No relevant context found."
            }
        # Assemble context (concatenate top chunks)
        context = "\n\n".join([item["chunk"] for item in retrieved])
        # Generate answer
        answer = generate_answer(question, context, model_name=model_name)
        # Compute simple confidence (average similarity score)
        scores = [item["score"] for item in retrieved]
        confidence = float(sum(scores)) / len(scores) if scores else 0.0
        # Format sources
        sources = [
            {
                "source_pdf": item["metadata"].get("source_pdf", "unknown"),
                "chunk_index": item["metadata"].get("chunk_index", -1),
                "score": item["score"]
            }
            for item in retrieved
        ]
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks": retrieved,
            "error": None
        }
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        return {
            "answer": None,
            "sources": [],
            "confidence": 0.0,
            "chunks": [],
            "error": str(e)
        }
