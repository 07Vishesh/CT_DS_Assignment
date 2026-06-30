"""
Document ingestion: upload -> extract -> chunk -> embed -> index.

This router is the entry point of the RAG pipeline. Everything downstream
(Q&A, quiz, flashcards, summarize) reads from what gets stored here.
"""
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.config import settings
from app.services.pdf_processor import extract_text, chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=schemas.DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(400, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.")

    try:
        text = extract_text(file.filename, raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not text.strip():
        raise HTTPException(400, "No extractable text found in this file.")

    pieces = chunk_text(text)
    if not pieces:
        raise HTTPException(400, "Document produced no usable chunks.")

    # 1. Create the document row first to get its id
    doc = models.Document(
        filename=file.filename,
        subject=subject,
        full_text=text,
        num_chunks=len(pieces),
    )
    db.add(doc)
    db.flush()  # assigns doc.id without committing yet

    # 2. Embed all chunks in one batch call, push into FAISS, get back vector ids
    vectors = embed_texts(pieces)
    vector_ids = get_vector_store().add(vectors)

    # 3. Persist chunk rows, each pointing at its FAISS vector id
    for idx, (piece, vid) in enumerate(zip(pieces, vector_ids)):
        db.add(models.Chunk(
            document_id=doc.id,
            chunk_index=idx,
            text=piece,
            vector_id=vid,
        ))

    db.commit()
    db.refresh(doc)
    return doc


@router.get("/", response_model=List[schemas.DocumentOut])
def list_documents(subject: str = None, db: Session = Depends(get_db)):
    query = db.query(models.Document)
    if subject:
        query = query.filter(models.Document.subject == subject)
    return query.order_by(models.Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=schemas.DocumentDetail)
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return doc


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")

    vector_ids = [c.vector_id for c in doc.chunks if c.vector_id is not None]
    get_vector_store().remove(vector_ids)

    db.delete(doc)  # cascades to chunks, quizzes, flashcards
    db.commit()
    return {"detail": "Document deleted."}
