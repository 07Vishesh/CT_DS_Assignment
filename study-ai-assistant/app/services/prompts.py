"""
Prompt templates, kept separate from the LLM client so they can be tuned
or A/B tested without touching API plumbing.

Quiz and flashcard prompts request strict JSON output, since both features
parse the model's response programmatically.
"""

QA_SYSTEM_PROMPT = """You are a focused study assistant. Answer the student's \
question using ONLY the provided context from their notes. If the context \
doesn't contain the answer, say so plainly instead of guessing. Keep answers \
clear and exam-relevant; use short paragraphs or bullet points where helpful."""


def qa_user_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"
    return f"""Context from the student's notes:

{context}

Question: {question}

Answer based only on the context above."""


QUIZ_SYSTEM_PROMPT = """You are a quiz generator for a study app. Generate \
multiple-choice questions strictly from the given study material. \
Respond with ONLY valid JSON, no markdown fences, no commentary, matching \
this exact schema:

{
  "questions": [
    {
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "correct_index": 0,
      "explanation": "string, 1-2 sentences on why the answer is correct"
    }
  ]
}"""


def quiz_user_prompt(content: str, num_questions: int, difficulty: str) -> str:
    return f"""Study material:

{content}

Generate exactly {num_questions} multiple-choice questions at {difficulty} \
difficulty, each with 4 options and exactly one correct answer. Cover \
distinct concepts from the material rather than repeating the same idea."""


FLASHCARD_SYSTEM_PROMPT = """You are generating spaced-repetition flashcards \
from study material. Each card should test one discrete fact, definition, or \
concept. Respond with ONLY valid JSON, no markdown fences, no commentary, \
matching this exact schema:

{
  "flashcards": [
    {"front": "string (question or term)", "back": "string (answer or definition)"}
  ]
}"""


def flashcard_user_prompt(content: str, num_cards: int) -> str:
    return f"""Study material:

{content}

Generate exactly {num_cards} flashcards covering the most important, \
distinct concepts in this material. Keep the front concise (a question or \
term) and the back precise (a direct answer, 1-3 sentences max)."""


SUMMARIZE_SYSTEM_PROMPT = """You are a study assistant that writes clear, \
exam-focused summaries. Preserve key terms, definitions, and relationships \
between concepts. Do not introduce information that isn't in the source text."""


def summarize_user_prompt(content: str, level: str, fmt: str) -> str:
    length_map = {
        "brief": "3-4 sentences",
        "medium": "1-2 short paragraphs",
        "detailed": "a thorough multi-paragraph summary covering all major sub-topics",
    }
    format_instruction = (
        "Format the summary as bullet points." if fmt == "bullets"
        else "Format the summary as flowing prose paragraphs."
    )
    return f"""Text to summarize:

{content}

Write a summary of length: {length_map[level]}. {format_instruction}"""
