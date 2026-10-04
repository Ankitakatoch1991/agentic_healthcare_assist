"""Long-term memory for the brief: one JSON file per patient, plus a FAISS index of summaries.

Short-term conversation history lives in Streamlit session state, not here.
data/memory/ is gitignored, so the demo father is recreated on startup if the file is missing.
"""

import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agent.prompts import MEMORY_PROMPT, fill, get_embeddings, get_llm, message_text, require_api_key
from tools.ehr_tool import get_patient, load_patients, seed_father_if_missing

ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "data" / "memory"
FAISS_DIR = ROOT / "data" / "faiss_patients"

FATHER_FACTS = [
    {"key": "age", "value": "70"},
    {"key": "gender", "value": "Male"},
    {"key": "condition", "value": "chronic kidney disease"},
    {"key": "relationship", "value": "father of the user"},
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "patient"


def _path_for(name: str) -> Path:
    return MEMORY_DIR / f"{_slug(name)}.json"


def _empty(name: str) -> dict:
    return {"name": name, "facts": [], "interactions": []}


def load_memory(name: str):
    path = _path_for(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_memory(memory: dict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _path_for(memory["name"]).write_text(json.dumps(memory, indent=2), encoding="utf-8")


def all_memories() -> list:
    if not MEMORY_DIR.exists():
        return []
    memories = []
    for path in sorted(MEMORY_DIR.glob("*.json")):
        memories.append(json.loads(path.read_text(encoding="utf-8")))
    return memories


def ensure_seed() -> None:
    """Create the demo father in chart updates and long-term memory when he is missing."""
    seed_father_if_missing()
    name = "Arun Sharma"
    memory = load_memory(name) or _empty(name)
    existing = {(fact["key"], fact["value"]) for fact in memory["facts"]}
    for fact in FATHER_FACTS:
        if (fact["key"], fact["value"]) not in existing:
            memory["facts"].append(fact)
    save_memory(memory)


def context_text(name: str | None) -> str:
    """Text block the planner sees for the resolved patient."""
    if not name:
        return "No patient identified yet."
    lines = [f"Name: {name}"]
    memory = load_memory(name)
    if memory:
        lines.extend(f"{fact['key']}: {fact['value']}" for fact in memory.get("facts", []))
    patient = get_patient(name)
    if patient:
        lines.append(f"Age on chart: {patient['age'] or 'not recorded'}")
        lines.append(f"Gender on chart: {patient['gender'] or 'not recorded'}")
        lines.append(f"Conditions mentioned in chart notes: {_snippet(patient)}")
    return "\n".join(lines)


def _snippet(patient: dict) -> str:
    chunks = [record.get("text", "")[:400] for record in patient.get("records", [])[:2]]
    text = " ".join(chunks).strip()
    return text or "none"


def _documents() -> list:
    by_name = {}
    for patient in load_patients().values():
        by_name[patient["name"]] = (
            f"{patient['name']}. Age {patient['age']}. Gender {patient['gender']}. "
            f"Address {patient['address']}. {_snippet(patient)}"
        )
    for memory in all_memories():
        facts = "; ".join(f"{fact['key']}: {fact['value']}" for fact in memory.get("facts", []))
        prior = by_name.get(memory["name"], memory["name"])
        by_name[memory["name"]] = f"{prior}. Long-term memory: {facts}"
    return [
        Document(page_content=text, metadata={"name": name})
        for name, text in by_name.items()
    ]


def rebuild_index():
    """Embed patient summaries into data/faiss_patients/. Returns None when the key is missing."""
    try:
        require_api_key()
    except RuntimeError:
        return None
    documents = _documents()
    if not documents:
        return None
    try:
        store = FAISS.from_documents(documents, get_embeddings())
    except Exception:
        return None
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_DIR))
    return store


def _load_index():
    if not (FAISS_DIR / "index.faiss").exists():
        return rebuild_index()
    try:
        require_api_key()
        return FAISS.load_local(
            str(FAISS_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def _keyword_lookup(query: str):
    lowered = query.lower()
    if "father" in lowered:
        for memory in all_memories():
            blob = json.dumps(memory).lower()
            if "father" in blob:
                return memory
    for memory in all_memories():
        if memory["name"].lower() in lowered:
            return memory
    return None


def lookup_patient(query: str):
    """Find a patient from a phrase such as 'my father' using the FAISS summary index."""
    faiss_hit = None
    store = _load_index()
    if store is not None:
        try:
            hits = store.similarity_search(query, k=1)
        except Exception:
            hits = []
        if hits:
            name = hits[0].metadata.get("name")
            faiss_hit = load_memory(name) if name else None
            if faiss_hit is None and name:
                faiss_hit = _empty(name)
    # If the question is about "my father" and FAISS pointed at someone else, use the stored relationship.
    if "father" in query.lower():
        for memory in all_memories():
            if "father" in json.dumps(memory).lower():
                if faiss_hit and faiss_hit.get("name") == memory.get("name"):
                    return faiss_hit
                if not faiss_hit or "father" not in json.dumps(faiss_hit).lower():
                    return memory
    return faiss_hit or _keyword_lookup(query)


def remember(name: str, query: str, answer: str) -> dict:
    """Ask the LLM for new facts and merge them into the patient's JSON file."""
    if not name:
        return {}
    memory = load_memory(name) or _empty(name)
    known = json.dumps(memory.get("facts", []))
    try:
        raw = message_text(
            get_llm().invoke(
                fill(MEMORY_PROMPT, facts=known, query=query[:2000], answer=answer[:2000])
            )
        )
        parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        new_facts = parsed.get("facts", [])
    except Exception:
        new_facts = []
    existing = {(fact["key"].casefold(), fact["value"].casefold()) for fact in memory["facts"]}
    for fact in new_facts:
        key = str(fact.get("key", "")).strip()
        value = str(fact.get("value", "")).strip()
        if key and value and (key.casefold(), value.casefold()) not in existing:
            memory["facts"].append({"key": key, "value": value})
            existing.add((key.casefold(), value.casefold()))
    memory["interactions"].append(
        {"query": query[:500], "answer": answer[:500]}
    )
    memory["interactions"] = memory["interactions"][-20:]
    save_memory(memory)
    try:
        rebuild_index()
    except Exception:
        pass
    return memory
