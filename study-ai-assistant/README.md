# StudyAI — AI-Powered Study Assistant

A full-stack study tool that turns uploaded notes into a working study system: ask grounded questions about them (RAG), auto-generate quizzes and flashcards, get summaries at different depths, and track topic-by-topic progress across subjects.

Built as a portfolio project to demonstrate applied RAG, not just an LLM API wrapper: local embeddings, a persistent vector index, structured-output prompting, and a normal relational schema sitting underneath all of it.

## Why this exists

Most "AI study assistant" projects are a chat box bolted onto an LLM call. This one is built the way a real retrieval system is built: notes are chunked and embedded locally, stored in a persistent FAISS index, and every answer is retrieved from and grounded in the student's own material rather than the model's general knowledge. The LLM is used for exactly two things it's actually good at — generating fluent text from given context, and producing structured quiz/flashcard data — and nothing is asked of it that the architecture should be doing instead.

## Features

- **Library** — upload PDF or text notes; each file is chunked, embedded, and indexed automatically.
- **Ask (RAG Q&A)** — ask a question scoped to one document or an entire subject; the answer is generated only from the most relevant retrieved chunks, with similarity-scored sources shown alongside it.
- **Quiz generator** — auto-generates multiple-choice questions at a chosen difficulty, grades submissions, and persists every attempt.
- **Flashcards** — auto-generates front/back cards from a document; cards are flipped and marked new / still learning / known, with review counts tracked over time.
- **Summarize** — summarizes a stored document or pasted text, with adjustable length and format (paragraph or bullets).
- **Planner** — plain CRUD for study topics per subject, with priority, due dates, and an aggregated progress view.

## Architecture

```
Upload (PDF/txt)
      │
      ▼
extract_text → chunk_text (overlapping windows)
      │
      ▼
sentence-transformers (local) → embeddings
      │
      ▼
FAISS index  ◄────────────┐
      │                    │  similarity search
      ▼                    │
SQLite (documents,         │
chunks, quizzes,           │
flashcards, topics) ───────┘
      │
      ▼
FastAPI routers → LLM client (Anthropic/OpenAI, swappable) → JSON/text response
      │
      ▼
Vanilla JS frontend (single page, no build step)
```

**Why these choices, not just what they are:**

- **Local embeddings (sentence-transformers), hosted LLM for generation** — embeddings run on every chunk and every query, so keeping them local keeps the project free to run end-to-end; the LLM call is reserved for the one part that genuinely needs a strong model (reasoning over retrieved context, generating well-formed quiz JSON).
- **FAISS flat index, not IVF/HNSW** — at the scale of one student's notes (hundreds to low thousands of chunks), brute-force search is exact and still fast. Swapping to an approximate index is a one-line change if this ever needed to scale to a multi-user product, noted here rather than over-engineered upfront.
- **A single global vector index with post-search filtering by document/subject**, instead of one index per subject — far simpler to manage, and the filtering cost is negligible at this scale.
- **SQLite, not Postgres** — this is a single-user local tool; SQLAlchemy's ORM layer means swapping to Postgres later is a connection-string change, not a rewrite.
- **Pluggable LLM provider** — `LLM_PROVIDER=anthropic|openai` in `.env` switches providers without touching application code, since the prompt/response contract is identical either way.

## Tech stack

`FastAPI` · `SQLAlchemy` · `SQLite` · `sentence-transformers` (all-MiniLM-L6-v2) · `FAISS` · `Anthropic API` / `OpenAI API` · `pypdf` · vanilla `HTML/CSS/JS` frontend · `pytest`

## Project structure

```
study-ai-assistant/
├── app/
│   ├── main.py              # FastAPI app, router registration, static frontend mount
│   ├── config.py            # env-driven settings
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # ORM models (Document, Chunk, Quiz, Flashcard, StudyTopic, ...)
│   ├── schemas.py           # Pydantic request/response contracts
│   ├── routers/
│   │   ├── documents.py     # upload → extract → chunk → embed → index
│   │   ├── qa.py             # RAG retrieval + grounded answer generation
│   │   ├── quiz.py           # quiz generation + grading
│   │   ├── flashcards.py     # flashcard generation + review status
│   │   ├── summarize.py      # document or raw-text summarization
│   │   └── planner.py        # topic CRUD + progress aggregation
│   └── services/
│       ├── pdf_processor.py  # text extraction + chunking
│       ├── embeddings.py     # sentence-transformers wrapper
│       ├── vector_store.py   # FAISS persistence layer
│       ├── llm_client.py     # Anthropic/OpenAI abstraction
│       └── prompts.py        # prompt templates
├── frontend/index.html       # single-page vanilla JS UI (no build step)
├── tests/                    # pytest — chunking logic + planner API, no API key needed
├── storage/                  # SQLite DB, FAISS index, uploaded files (gitignored)
├── requirements.txt
├── .env.example
└── run.py
```

## Setup

**1. Clone and create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure your LLM provider**

```bash
cp .env.example .env
```

Edit `.env` and add an API key for whichever provider you'll use:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

(or set `LLM_PROVIDER=openai` and fill in `OPENAI_API_KEY` instead). Embeddings and vector search don't need any key — they run locally via sentence-transformers.

**3. Run it**

```bash
python run.py
```

Open `http://localhost:8000` — the frontend is served directly by FastAPI. Interactive API docs are at `http://localhost:8000/docs`.

**4. Run the tests**

```bash
pytest
```

Chunking and planner tests run with zero configuration (no API key, no network) since they don't touch the LLM.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/documents/upload` | Upload + chunk + embed + index a file |
| GET | `/documents/` | List documents |
| DELETE | `/documents/{id}` | Delete a document and its vectors |
| POST | `/qa/ask` | Ask a grounded question (RAG) |
| POST | `/quiz/generate` | Generate a quiz from a document |
| POST | `/quiz/{id}/submit` | Submit answers, get score + review |
| POST | `/flashcards/generate` | Generate flashcards from a document |
| PATCH | `/flashcards/{id}` | Update a card's review status |
| POST | `/summarize/` | Summarize a document or raw text |
| POST | `/planner/topics` | Create a study topic |
| PATCH | `/planner/topics/{id}` | Update status/progress/priority |
| GET | `/planner/progress` | Aggregated progress per subject |

Full request/response schemas are auto-documented at `/docs` (Swagger UI) once the server is running.

## Known limitations / future improvements

Documenting these honestly rather than pretending the project is "done" — this is exactly what comes up in interviews:

- **Chunking is fixed-size, not semantic.** Splitting on sentence or paragraph boundaries (or recursive chunking) would preserve meaning across boundaries better than a fixed character window.
- **No spaced-repetition scheduling.** Flashcard status is manually set (new/learning/known); a real SM-2-style algorithm would schedule reviews automatically.
- **Flat FAISS index.** Fine at single-user scale; would move to IVF/HNSW or a managed vector DB (pgvector, Pinecone) for multi-user scale.
- **No auth.** Single-user local tool by design; adding accounts would mean scoping every query by `user_id`.
- **No streaming responses.** Answers and summaries return as a single block; streaming would improve perceived latency for longer generations.

## License

MIT — built as a portfolio project.
