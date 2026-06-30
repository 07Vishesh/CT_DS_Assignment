"""
LLM client abstraction.

Swapping providers (Anthropic <-> OpenAI) is a one-line .env change
(LLM_PROVIDER=anthropic|openai) rather than a code change — the rest of the
app only ever calls `generate()` / `generate_json()` from this module.
"""
import json
import re
from app.config import settings


class LLMError(RuntimeError):
    """Raised when the LLM call fails or the configured provider has no API key."""


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    import anthropic

    if not settings.ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def _call_openai(system: str, user: str, max_tokens: int) -> str:
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def generate(system: str, user: str, max_tokens: int = 1024) -> str:
    """Send a system+user prompt to the configured provider, return raw text."""
    if settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    elif settings.LLM_PROVIDER == "openai":
        return _call_openai(system, user, max_tokens)
    raise LLMError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def _strip_json_fences(text: str) -> str:
    """Strip ```json ... ``` fences some models wrap JSON responses in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_json(system: str, user: str, max_tokens: int = 2048) -> dict:
    """
    Generate and parse a strict-JSON response.

    Raises LLMError with the raw response included if parsing fails, since
    silently swallowing a malformed quiz/flashcard payload would surface as
    a confusing empty list in the UI instead of an actionable error.
    """
    raw = generate(system, user, max_tokens)
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {e}\nRaw response: {raw[:500]}")
