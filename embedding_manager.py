"""
Embeddings and ChromaDB management.

Handles embedding generation, storage, retrieval, and collection management for RAG PDF Q&A Assistant.
"""
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from loguru import logger
from functools import lru_cache

# Constants
CHROMA_DB_DIR = os.path.join("data", "chromadb")
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure ChromaDB directory exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the sentence-transformers model.
    Returns:
        SentenceTransformer: Loaded model.
    """
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

def get_chroma_client() -> chromadb.Client:
    """
    Initializes and returns a persistent ChromaDB client.
    Returns:
        chromadb.Client: ChromaDB client instance.
    """
    return chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))

def get_or_create_collection(collection_name: str = COLLECTION_NAME):
    """
    Gets or creates a ChromaDB collection.
    Args:
        collection_name (str): Name of the collection.
    Returns:
        Collection: ChromaDB collection.
    """
    client = get_chroma_client()
    try:
        return client.get_or_create_collection(collection_name)
    except Exception as e:
        logger.error(f"Error getting/creating collection: {e}")
        raise

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.
    Args:
        chunks (List[str]): List of text chunks.
    Returns:
        List[List[float]]: Embeddings for each chunk.
    """
    model = load_embedding_model()
    try:
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).tolist()
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def store_embeddings(
    embeddings: List[List[float]],
    chunks: List[str],
    source_pdf: str,
    collection_name: str = COLLECTION_NAME
) -> None:
    """
    Stores embeddings and metadata in ChromaDB.
    Args:
        embeddings (List[List[float]]): Embeddings to store.
        chunks (List[str]): Corresponding text chunks.
        source_pdf (str): Source PDF filename.
        collection_name (str): ChromaDB collection name.
    """
    collection = get_or_create_collection(collection_name)
    try:
        ids = [f"{source_pdf}_{i}" for i in range(len(chunks))]
        metadatas = [{"source_pdf": source_pdf, "chunk_index": i} for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(chunks)} embeddings for {source_pdf} in {collection_name}")
    except Exception as e:
        logger.error(f"Failed to store embeddings: {e}")
        raise

def query_similar_chunks(
    query: str,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """
    Retrieves top-k most similar chunks for a query.
    Args:
        query (str): User question or query.
        top_k (int): Number of results to return.
        collection_name (str): ChromaDB collection name.
    Returns:
        List[Dict[str, Any]]: List of dicts with chunk, score, and metadata.
    """
    collection = get_or_create_collection(collection_name)
    model = load_embedding_model()
    try:
        query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        # Format results
        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            output.append({
                "chunk": doc,
                "score": 1.0 - dist,  # ChromaDB returns distance; convert to similarity
                "metadata": meta
            })
        return output
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

def clear_collection(collection_name: str = COLLECTION_NAME) -> None:
    """
    Clears all data from a ChromaDB collection.
    Args:
        collection_name (str): Name of the collection to clear.
    """
    collection = get_or_create_collection(collection_name)
    try:
        collection.delete()
        logger.info(f"Cleared collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to clear collection: {e}")

def reset_database() -> None:
    """
    Deletes all collections and resets the ChromaDB database.
    """
    try:
        client = get_chroma_client()
        for name in client.list_collections():
            client.delete_collection(name)
        logger.info("ChromaDB database reset.")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")