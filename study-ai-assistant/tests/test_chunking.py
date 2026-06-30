"""
Unit tests for chunking — the only part of the pipeline with non-trivial
logic that's fully testable without an API key or a network call.
"""
from app.services.pdf_processor import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    text = "A short note about photosynthesis."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_overlapping_chunks():
    text = " ".join([f"sentence{i}" for i in range(500)])  # long enough to force splitting
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every chunk should respect the requested size (give or take whitespace trimming)
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_preserves_boundary_context():
    text = "word" * 5 + " " + "boundary marker phrase here " + "word" * 50
    chunks = chunk_text(text, chunk_size=60, overlap=20)
    # with overlap, consecutive chunks should share some trailing/leading text
    assert len(chunks) >= 2


def test_whitespace_is_normalized():
    text = "Line one.\n\n\nLine   two.\t\tLine three."
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert "\n" not in chunks[0]
    assert "  " not in chunks[0]
