"""
Text extraction and chunking.

Chunking strategy: fixed-size character windows with overlap. This is the
simplest RAG chunking approach and is deliberately chosen over sentence- or
token-aware splitting for transparency and zero extra dependencies — it's
easy to explain in an interview and easy to swap out later (mention this as
a future improvement: semantic/recursive chunking).
"""
from io import BytesIO
from typing import List
from pypdf import PdfReader

from app.config import settings


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF's bytes, page by page."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dispatch extraction based on file extension. Add new formats here."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split text into overlapping fixed-size chunks.

    Overlap preserves context across chunk boundaries so an answer that
    straddles two chunks isn't lost during retrieval.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # step forward, keeping `overlap` chars of context
    return chunks
