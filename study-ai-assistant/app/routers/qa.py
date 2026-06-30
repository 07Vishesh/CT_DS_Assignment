"""
Retrieval-Augmented Q&A.

Pipeline: embed the question -> FAISS similarity search -> filter retrieved
chunks down to the requested document/subject scope -> stuff into a prompt
-> LLM generates a grounded answer -> return answer + cited source chunks.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.config import settings
from app.services.embeddings import embed_query
from app.services.vector_store import get_vector_store
from app.services.llm_client import generate, LLMError
from app.services.prompts import QA_SYSTEM_PROMPT, qa_user_prompt

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=schemas.AskResponse)
def ask_question(req: schemas.AskRequest, db: Session = Depends(get_db)):
    if not req.document_id and not req.subject:
        raise HTTPException(400, "Provide either document_id or subject to scope the search.")

    # Over-fetch from FAISS, then filter to scope, since the global index
    # doesn't natively support per-subject filtering (see vector_store.py).
    query_vec = embed_query(req.question)
    raw_hits = get_vector_store().search(query_vec, top_k=settings.TOP_K_RETRIEVAL * 5)
    if not raw_hits:
        raise HTTPException(400, "No documents have been indexed yet. Upload notes first.")

    vector_id_to_score = {vid: score for vid, score in raw_hits}
    chunk_query = db.query(models.Chunk).filter(
        models.Chunk.vector_id.in_(vector_id_to_score.keys())
    )

    if req.document_id:
        chunk_query = chunk_query.filter(models.Chunk.document_id == req.document_id)
    elif req.subject:
        chunk_query = chunk_query.join(models.Document).filter(
            models.Document.subject == req.subject
        )

    candidates = chunk_query.all()
    candidates.sort(key=lambda c: vector_id_to_score.get(c.vector_id, 0), reverse=True)
    top_chunks = candidates[: settings.TOP_K_RETRIEVAL]

    if not top_chunks:
        raise HTTPException(404, "No relevant notes found in the selected scope.")

    context_texts = [c.text for c in top_chunks]
    try:
        answer = generate(
            system=QA_SYSTEM_PROMPT,
            user=qa_user_prompt(req.question, context_texts),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))

    sources = [
        schemas.SourceChunk(
            document_id=c.document_id,
            document_filename=c.document.filename,
            chunk_text=c.text[:300] + ("..." if len(c.text) > 300 else ""),
            relevance_score=round(vector_id_to_score.get(c.vector_id, 0), 3),
        )
        for c in top_chunks
    ]
    return schemas.AskResponse(answer=answer, sources=sources)
