"""
Quiz generation and grading.

Generation reads the full document text (rather than retrieved chunks,
since the goal is broad coverage of the material, not answering one
specific question) and asks the LLM for structured JSON, which is
validated and persisted as individual question rows.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.services.llm_client import generate_json, LLMError
from app.services.prompts import QUIZ_SYSTEM_PROMPT, quiz_user_prompt

router = APIRouter(prefix="/quiz", tags=["quiz"])

# Cap how much raw text we feed the model per generation call, to keep
# prompts (and cost) bounded for very long documents.
MAX_CONTENT_CHARS = 12000


@router.post("/generate", response_model=schemas.QuizOut)
def generate_quiz(req: schemas.QuizGenerateRequest, db: Session = Depends(get_db)):
    doc = db.get(models.Document, req.document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")

    content = doc.full_text[:MAX_CONTENT_CHARS]
    try:
        payload = generate_json(
            system=QUIZ_SYSTEM_PROMPT,
            user=quiz_user_prompt(content, req.num_questions, req.difficulty),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))

    raw_questions = payload.get("questions", [])
    if not raw_questions:
        raise HTTPException(502, "Model returned no questions. Try again.")

    quiz = models.Quiz(
        document_id=doc.id,
        title=f"{doc.filename} Quiz",
        subject=doc.subject,
        difficulty=req.difficulty,
    )
    db.add(quiz)
    db.flush()

    for q in raw_questions:
        options = q.get("options", [])
        correct_index = q.get("correct_index", 0)
        if len(options) < 2 or not (0 <= correct_index < len(options)):
            continue  # skip malformed entries rather than failing the whole batch
        db.add(models.QuizQuestion(
            quiz_id=quiz.id,
            question=q.get("question", "").strip(),
            options=options,
            correct_index=correct_index,
            explanation=q.get("explanation", ""),
        ))

    db.commit()
    db.refresh(quiz)
    if not quiz.questions:
        db.delete(quiz)
        db.commit()
        raise HTTPException(502, "All generated questions were malformed. Try again.")
    return quiz


@router.get("/{quiz_id}", response_model=schemas.QuizOut)
def get_quiz(quiz_id: int, db: Session = Depends(get_db)):
    quiz = db.get(models.Quiz, quiz_id)
    if not quiz:
        raise HTTPException(404, "Quiz not found.")
    return quiz


@router.post("/{quiz_id}/submit", response_model=schemas.QuizResult)
def submit_quiz(quiz_id: int, req: schemas.QuizSubmitRequest, db: Session = Depends(get_db)):
    quiz = db.get(models.Quiz, quiz_id)
    if not quiz:
        raise HTTPException(404, "Quiz not found.")
    if len(req.answers) != len(quiz.questions):
        raise HTTPException(400, f"Expected {len(quiz.questions)} answers, got {len(req.answers)}.")

    score = 0
    review = []
    for question, selected in zip(quiz.questions, req.answers):
        is_correct = selected == question.correct_index
        score += int(is_correct)
        review.append(schemas.QuestionWithAnswer(
            id=question.id,
            question=question.question,
            options=question.options,
            correct_index=question.correct_index,
            explanation=question.explanation,
        ))

    attempt = models.QuizAttempt(
        quiz_id=quiz.id, score=score, total=len(quiz.questions), answers=req.answers
    )
    db.add(attempt)
    db.commit()

    total = len(quiz.questions)
    return schemas.QuizResult(
        score=score,
        total=total,
        percentage=round(100 * score / total, 1) if total else 0.0,
        review=review,
    )
