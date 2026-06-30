"""
Summarization. Accepts either a stored document_id or ad-hoc raw_text, so
the feature is usable both from the library (summarize what I uploaded)
and as a quick scratchpad tool (paste anything, summarize it).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services.llm_client import generate, LLMError
from app.services.prompts import SUMMARIZE_SYSTEM_PROMPT, summarize_user_prompt

router = APIRouter(prefix="/summarize", tags=["summarize"])

MAX_CONTENT_CHARS = 12000


@router.post("/", response_model=schemas.SummarizeResponse)
def summarize(req: schemas.SummarizeRequest, db: Session = Depends(get_db)):
    if req.document_id:
        doc = db.get(models.Document, req.document_id)
        if not doc:
            raise HTTPException(404, "Document not found.")
        content = doc.full_text
    elif req.raw_text:
        content = req.raw_text
    else:
        raise HTTPException(400, "Provide either document_id or raw_text.")

    content = content[:MAX_CONTENT_CHARS]
    if not content.strip():
        raise HTTPException(400, "Nothing to summarize.")

    try:
        summary = generate(
            system=SUMMARIZE_SYSTEM_PROMPT,
            user=summarize_user_prompt(content, req.level, req.format),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))

    return schemas.SummarizeResponse(summary=summary)
