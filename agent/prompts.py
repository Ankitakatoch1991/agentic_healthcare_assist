"""Prompt templates for the brief: planner, EHR summarizer, RAG answer, final response, and memory extraction."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def require_api_key() -> str:
    """Load OPENAI_API_KEY from the environment or a local .env file."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-your"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set OPENAI_API_KEY."
        )
    return key


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Shared chat model. Fails with a clear message when the API key is absent."""
    key = require_api_key()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return ChatOpenAI(model=model, temperature=temperature, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    """Embeddings used for the patient FAISS index and medical-search RAG."""
    key = require_api_key()
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)


def explain_openai_error(exc: Exception) -> str:
    """Turn a raw OpenAI error into a short message for the UI and the final reply."""
    text = str(exc)
    if "insufficient_quota" in text or "credit_balance_exhausted" in text:
        return (
            "OPENAI_API_KEY was found, but the account has no credits left. "
            "Add credits, or put a different key in .env, then try again."
        )
    if "invalid_api_key" in text or "Incorrect API key" in text:
        return "OPENAI_API_KEY was rejected. Check the key in .env."
    if "OPENAI_API_KEY is missing" in text:
        return text
    return f"The OpenAI request failed: {exc}"


def message_text(message) -> str:
    """Normalize ChatOpenAI content, which may be a string or a list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def fill(template: str, **kwargs) -> str:
    """Replace {placeholders} without interpreting braces inside user or web text."""
    text = template
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text


PLANNER_PROMPT = """You are the planner for a student healthcare assistant. Break the user query into an ordered list of tool steps.

Tools (use these exact names):
- ehr_history: look up a patient's chart. input is the patient name.
- find_slots: list open doctor slots. input is a specialty: Nephrology, Cardiology, General Medicine, or Endocrinology.
- book_appointment: book one slot. input format: patient: NAME; doctor: first available; slot: first
- medical_search: search MedlinePlus and WHO, then summarize. input is the medical topic.
- update_record: append a note to the chart. input format: patient: NAME; text: NOTE
- none: no tool. input is a short note for the final answer.

Rules:
- Return JSON only. No markdown.
- When the user wants an appointment, include ehr_history, then find_slots, then book_appointment.
- book_appointment should use slot "first" unless the user named a time.
- Include medical_search only when the user asks for treatment, symptoms, or general medical information.
- Do not book an appointment unless the user asked to book or schedule one.
- Use the patient name from the context when the user says "my father" or similar.

Patient context:
{patient_context}

Recent conversation:
{history}

User query:
{query}

Return a JSON list:
[{"step": 1, "goal": "why this step", "tool": "ehr_history", "input": "patient name"}]
"""

SUMMARIZER_PROMPT = """Summarize this patient chart for a caregiver.
Organize the answer into three short sections: past diagnoses, treatments, and alerts (follow-ups, abnormal vitals, or missing information).
Use only the chart. If a section is not documented, say so.

Chart:
{chart}
"""

RAG_PROMPT = """Answer the question using only the retrieved pages from MedlinePlus or WHO.
Write a short summary of the latest general treatment or care information.
Cite the source URLs you actually use. If the pages do not contain the answer, say so.
This is general information, not personal medical advice.

Question:
{question}

Retrieved pages:
{context}
"""

FINAL_PROMPT = """Write one friendly reply that combines every step result below.
Include the patient name, the history summary, any booked appointment (doctor and time), and any treatment summary.
If a step failed, say what failed in plain language.
End with one sentence: this is an educational project and not medical advice.

Patient context:
{patient_context}

User query:
{query}

Step results:
{steps}
"""

MEMORY_PROMPT = """Extract new facts about this patient from the conversation.
Return JSON only, shaped as {"facts": [{"key": "age", "value": "70"}]}.
Useful keys: age, gender, condition, relationship, preference.
Do not repeat facts that are already stored. If nothing is new, return {"facts": []}.

Known facts:
{facts}

User:
{query}

Assistant:
{answer}
"""
