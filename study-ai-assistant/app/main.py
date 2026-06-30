"""
Application entry point. Run with:  uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings, BASE_DIR
from app.database import init_db
from app.routers import documents, qa, quiz, flashcards, summarize, planner

app = FastAPI(
    title="StudyAI — AI-Powered Study Assistant",
    description=(
        "Upload notes, ask grounded questions (RAG), auto-generate quizzes "
        "and flashcards, summarize material, and track study progress."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(quiz.router)
app.include_router(flashcards.router)
app.include_router(summarize.router)
app.include_router(planner.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# --- Serve the frontend (single-page vanilla app) ---
frontend_dir = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(frontend_dir / "index.html"))
