"""
End-to-end tests for the planner router, run against an isolated in-memory
SQLite DB (via dependency override) so they don't touch the real
storage/study_assistant.db file or require any LLM/API key.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


def test_create_and_list_topic():
    resp = client.post("/planner/topics", json={
        "subject": "Data Structures",
        "title": "Binary Search Trees",
        "priority": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Binary Search Trees"
    assert body["status"] == "not_started"

    resp = client.get("/planner/topics", params={"subject": "Data Structures"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_update_topic_status_to_completed_sets_full_progress():
    created = client.post("/planner/topics", json={
        "subject": "OS", "title": "Deadlocks",
    }).json()

    resp = client.patch(f"/planner/topics/{created['id']}", json={"status": "completed"})
    assert resp.status_code == 200
    assert resp.json()["progress_pct"] == 100.0


def test_progress_aggregation_across_subjects():
    client.post("/planner/topics", json={"subject": "DBMS", "title": "Normalization"})
    t2 = client.post("/planner/topics", json={"subject": "DBMS", "title": "Indexing"}).json()
    client.patch(f"/planner/topics/{t2['id']}", json={"status": "completed"})

    resp = client.get("/planner/progress")
    assert resp.status_code == 200
    dbms = next(p for p in resp.json() if p["subject"] == "DBMS")
    assert dbms["total_topics"] == 2
    assert dbms["completed"] == 1


def test_delete_topic():
    created = client.post("/planner/topics", json={"subject": "CN", "title": "TCP/IP"}).json()
    resp = client.delete(f"/planner/topics/{created['id']}")
    assert resp.status_code == 200

    resp = client.get("/planner/topics", params={"subject": "CN"})
    assert resp.json() == []
