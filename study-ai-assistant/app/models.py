"""
ORM models.

Entity overview
----------------
Document        - an uploaded PDF/text file (the source material)
Chunk            - a slice of a document's text; the unit that gets embedded
                   and retrieved during RAG. The actual vector lives in FAISS;
                   this row is the "payload" FAISS points back to.
Quiz / Question  - an auto-generated quiz tied to a document, with attempts
Flashcard        - a generated front/back card with a spaced-repetition-ish
                   status field
StudyTopic       - a manually or auto-created planner entry with progress
"""
import datetime as dt
import enum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Float, Enum, JSON
)
from sqlalchemy.orm import relationship

from app.database import Base


def now():
    return dt.datetime.utcnow()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    subject = Column(String, index=True, default="General")
    full_text = Column(Text, nullable=False)
    num_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=now)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="document", cascade="all, delete-orphan")
    flashcards = relationship("Flashcard", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    # position of this chunk's vector inside the FAISS index (faiss uses int64 ids)
    vector_id = Column(Integer, unique=True, index=True)

    document = relationship("Document", back_populates="chunks")


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    title = Column(String, nullable=False)
    subject = Column(String, index=True, default="General")
    difficulty = Column(String, default="medium")
    created_at = Column(DateTime, default=now)

    document = relationship("Document", back_populates="quizzes")
    questions = relationship("QuizQuestion", back_populates="quiz", cascade="all, delete-orphan")
    attempts = relationship("QuizAttempt", back_populates="quiz", cascade="all, delete-orphan")


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    question = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)         # ["opt A", "opt B", "opt C", "opt D"]
    correct_index = Column(Integer, nullable=False)  # index into options
    explanation = Column(Text, default="")

    quiz = relationship("Quiz", back_populates="questions")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    score = Column(Integer, nullable=False)
    total = Column(Integer, nullable=False)
    answers = Column(JSON, default=list)  # list of selected indices, parallel to questions
    taken_at = Column(DateTime, default=now)

    quiz = relationship("Quiz", back_populates="attempts")


class FlashcardStatus(str, enum.Enum):
    NEW = "new"
    LEARNING = "learning"
    KNOWN = "known"


class Flashcard(Base):
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    subject = Column(String, index=True, default="General")
    front = Column(Text, nullable=False)
    back = Column(Text, nullable=False)
    status = Column(Enum(FlashcardStatus), default=FlashcardStatus.NEW)
    times_reviewed = Column(Integer, default=0)
    last_reviewed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=now)

    document = relationship("Document", back_populates="flashcards")


class TopicStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class StudyTopic(Base):
    __tablename__ = "study_topics"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    status = Column(Enum(TopicStatus), default=TopicStatus.NOT_STARTED)
    priority = Column(Integer, default=2)  # 1=high, 2=medium, 3=low
    due_date = Column(DateTime, nullable=True)
    notes = Column(Text, default="")
    progress_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)
