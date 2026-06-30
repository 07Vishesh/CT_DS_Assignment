"""
Pydantic schemas (the API's request/response contracts).

Kept separate from the ORM models on purpose: ORM models describe storage,
schemas describe the wire format. This indirection is what lets the DB
evolve without breaking API consumers (and vice versa).
"""
import datetime as dt
from typing import Optional, List
from pydantic import BaseModel, Field


# ---------- Documents ----------

class DocumentOut(BaseModel):
    id: int
    filename: str
    subject: str
    num_chunks: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class DocumentDetail(DocumentOut):
    full_text: str


# ---------- Q&A ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    document_id: Optional[int] = None
    subject: Optional[str] = None  # if set instead of document_id, searches across the subject


class SourceChunk(BaseModel):
    document_id: int
    document_filename: str
    chunk_text: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


# ---------- Quiz ----------

class QuizGenerateRequest(BaseModel):
    document_id: int
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("medium", pattern="^(easy|medium|hard)$")


class QuestionOut(BaseModel):
    id: int
    question: str
    options: List[str]

    class Config:
        from_attributes = True


class QuestionWithAnswer(QuestionOut):
    correct_index: int
    explanation: str


class QuizOut(BaseModel):
    id: int
    title: str
    subject: str
    difficulty: str
    questions: List[QuestionOut]

    class Config:
        from_attributes = True


class QuizSubmitRequest(BaseModel):
    answers: List[int]  # selected option index per question, in question order


class QuizResult(BaseModel):
    score: int
    total: int
    percentage: float
    review: List[QuestionWithAnswer]


# ---------- Flashcards ----------

class FlashcardGenerateRequest(BaseModel):
    document_id: int
    num_cards: int = Field(8, ge=1, le=30)


class FlashcardOut(BaseModel):
    id: int
    front: str
    back: str
    status: str
    times_reviewed: int

    class Config:
        from_attributes = True


class FlashcardUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(new|learning|known)$")


# ---------- Summarize ----------

class SummarizeRequest(BaseModel):
    document_id: Optional[int] = None
    raw_text: Optional[str] = None
    level: str = Field("medium", pattern="^(brief|medium|detailed)$")
    format: str = Field("paragraph", pattern="^(paragraph|bullets)$")


class SummarizeResponse(BaseModel):
    summary: str


# ---------- Planner ----------

class TopicCreate(BaseModel):
    subject: str
    title: str
    priority: int = Field(2, ge=1, le=3)
    due_date: Optional[dt.datetime] = None
    notes: str = ""


class TopicUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern="^(not_started|in_progress|completed)$")
    priority: Optional[int] = Field(None, ge=1, le=3)
    due_date: Optional[dt.datetime] = None
    notes: Optional[str] = None
    progress_pct: Optional[float] = Field(None, ge=0, le=100)


class TopicOut(BaseModel):
    id: int
    subject: str
    title: str
    status: str
    priority: int
    due_date: Optional[dt.datetime]
    notes: str
    progress_pct: float
    updated_at: dt.datetime

    class Config:
        from_attributes = True


class SubjectProgress(BaseModel):
    subject: str
    total_topics: int
    completed: int
    in_progress: int
    not_started: int
    avg_progress_pct: float
