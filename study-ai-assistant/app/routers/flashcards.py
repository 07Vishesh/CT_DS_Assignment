"""
Flashcard generation and review tracking.

Status transitions (new -> learning -> known) are driven entirely by the
user marking cards during review, deliberately not by a timer/algorithm —
true spaced-repetition scheduling (e.g. SM-2) is flagged as a future
improvement in the README rather than half-implemented here.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services.llm_client import generate_json, LLMError
from app.services.prompts import FLASHCARD_SYSTEM_PROMPT, flashcard_user_prompt

router = APIRouter(prefix="/flashcards", tags=["flashcards"])

MAX_CONTENT_CHARS = 12000


@router.post("/generate", response_model=List[schemas.FlashcardOut])
def generate_flashcards(req: schemas.FlashcardGenerateRequest, db: Session = Depends(get_db)):
    doc = db.get(models.Document, req.document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")

    content = doc.full_text[:MAX_CONTENT_CHARS]
    try:
        payload = generate_json(
            system=FLASHCARD_SYSTEM_PROMPT,
            user=flashcard_user_prompt(content, req.num_cards),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))

    raw_cards = payload.get("flashcards", [])
    if not raw_cards:
        raise HTTPException(502, "Model returned no flashcards. Try again.")

    created = []
    for c in raw_cards:
        front, back = c.get("front", "").strip(), c.get("back", "").strip()
        if not front or not back:
            continue
        card = models.Flashcard(
            document_id=doc.id, subject=doc.subject, front=front, back=back
        )
        db.add(card)
        created.append(card)

    db.commit()
    for c in created:
        db.refresh(c)
    return created


@router.get("/", response_model=List[schemas.FlashcardOut])
def list_flashcards(
    subject: str = None,
    document_id: int = None,
    status: str = None,
    db: Session = Depends(get_db),
):
    query = db.query(models.Flashcard)
    if subject:
        query = query.filter(models.Flashcard.subject == subject)
    if document_id:
        query = query.filter(models.Flashcard.document_id == document_id)
    if status:
        query = query.filter(models.Flashcard.status == status)
    return query.order_by(models.Flashcard.created_at.desc()).all()


@router.patch("/{card_id}", response_model=schemas.FlashcardOut)
def update_flashcard(card_id: int, req: schemas.FlashcardUpdateRequest, db: Session = Depends(get_db)):
    import datetime as dt
    card = db.get(models.Flashcard, card_id)
    if not card:
        raise HTTPException(404, "Flashcard not found.")
    card.status = req.status
    card.times_reviewed += 1
    card.last_reviewed = dt.datetime.utcnow()
    db.commit()
    db.refresh(card)
    return card
