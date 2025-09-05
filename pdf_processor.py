"""
PDF processing: text extraction and chunking.

This module provides functions to extract text from PDF files and split the text into overlapping chunks for embedding.
"""
from typing import List, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def is_valid_pdf(pdf_path: str) -> bool:
    """
    Checks if the file at pdf_path is a valid PDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        bool: True if valid, False otherwise.
    """
    return os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf")

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts all text from a PDF file using pypdf.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        Optional[str]: Extracted text, or None if extraction fails.
    """
    if not is_valid_pdf(pdf_path):
        return None
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text if text.strip() else None
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[str]:
    """
    Splits text into overlapping chunks using RecursiveCharacterTextSplitter.
    Args:
        text (str): The text to split.
        chunk_size (int): Max characters per chunk.
        overlap (int): Overlap between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return splitter.split_text(text)
