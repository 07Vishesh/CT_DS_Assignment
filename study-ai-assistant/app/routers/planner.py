"""
Study planner: plain CRUD over topics plus an aggregated progress view per
subject. No LLM involved here on purpose — this is the "boring but
necessary" persistence-and-state-management part of the app that balances
out the AI-heavy features and shows ordinary backend competence too.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app import models, schemas

router = APIRouter(prefix="/planner", tags=["planner"])


@router.post("/topics", response_model=schemas.TopicOut)
def create_topic(req: schemas.TopicCreate, db: Session = Depends(get_db)):
    topic = models.StudyTopic(**req.dict())
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return topic


@router.get("/topics", response_model=List[schemas.TopicOut])
def list_topics(subject: str = None, status: str = None, db: Session = Depends(get_db)):
    query = db.query(models.StudyTopic)
    if subject:
        query = query.filter(models.StudyTopic.subject == subject)
    if status:
        query = query.filter(models.StudyTopic.status == status)
    return query.order_by(
        models.StudyTopic.priority.asc(), models.StudyTopic.due_date.asc()
    ).all()


@router.patch("/topics/{topic_id}", response_model=schemas.TopicOut)
def update_topic(topic_id: int, req: schemas.TopicUpdate, db: Session = Depends(get_db)):
    topic = db.get(models.StudyTopic, topic_id)
    if not topic:
        raise HTTPException(404, "Topic not found.")

    updates = req.dict(exclude_unset=True)
    for field, value in updates.items():
        setattr(topic, field, value)

    # Marking a topic completed implies 100% progress, and vice versa,
    # keeping the two fields from silently drifting apart.
    if updates.get("status") == "completed":
        topic.progress_pct = 100.0
    elif "progress_pct" in updates and updates["progress_pct"] >= 100:
        topic.status = models.TopicStatus.COMPLETED

    db.commit()
    db.refresh(topic)
    return topic


@router.delete("/topics/{topic_id}")
def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = db.get(models.StudyTopic, topic_id)
    if not topic:
        raise HTTPException(404, "Topic not found.")
    db.delete(topic)
    db.commit()
    return {"detail": "Topic deleted."}


@router.get("/progress", response_model=List[schemas.SubjectProgress])
def progress_by_subject(db: Session = Depends(get_db)):
    subjects = db.query(models.StudyTopic.subject).distinct().all()
    results = []
    for (subject,) in subjects:
        topics = db.query(models.StudyTopic).filter(models.StudyTopic.subject == subject).all()
        total = len(topics)
        completed = sum(1 for t in topics if t.status == models.TopicStatus.COMPLETED)
        in_progress = sum(1 for t in topics if t.status == models.TopicStatus.IN_PROGRESS)
        not_started = total - completed - in_progress
        avg_progress = sum(t.progress_pct for t in topics) / total if total else 0.0
        results.append(schemas.SubjectProgress(
            subject=subject,
            total_topics=total,
            completed=completed,
            in_progress=in_progress,
            not_started=not_started,
            avg_progress_pct=round(avg_progress, 1),
        ))
    return results
