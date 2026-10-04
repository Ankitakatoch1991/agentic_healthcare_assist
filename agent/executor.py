"""Executor for the brief: run each planned step, log it, and compose one final reply.

Returns {plan, step_results, final_answer, memory_used, log_entries} for the Streamlit trace.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.memory import context_text, ensure_seed, lookup_patient, remember
from agent.planner import make_plan
from agent.prompts import FINAL_PROMPT, explain_openai_error, fill, get_llm, message_text
from tools.appointment_tool import book_appointment, canonicalize_specialty, find_slots
from tools.ehr_tool import add_or_update_record, get_history, patient_names, summarize_history
from tools.search_tool import medical_search

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "agent_log.jsonl"
SLOT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def write_log(patient, tool, tool_input, success, duration, error) -> dict:
    """Append one JSON line per tool call."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "patient": patient,
        "tool": tool,
        "input": str(tool_input)[:2000],
        "success": bool(success),
        "duration": round(float(duration), 3),
        "error": error,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def read_logs() -> list:
    if not LOG_PATH.exists():
        return []
    rows = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tool_metrics(rows: list | None = None) -> list:
    """Per-tool call count, success rate, and average duration."""
    grouped = {}
    for row in rows if rows is not None else read_logs():
        slot = grouped.setdefault(row["tool"], {"calls": 0, "successes": 0, "duration": 0.0})
        slot["calls"] += 1
        slot["successes"] += 1 if row.get("success") else 0
        slot["duration"] += float(row.get("duration") or 0)
    metrics = []
    for tool, slot in sorted(grouped.items()):
        calls = slot["calls"]
        metrics.append(
            {
                "tool": tool,
                "calls": calls,
                "success_rate": round(slot["successes"] / calls, 3) if calls else 0,
                "avg_duration_sec": round(slot["duration"] / calls, 3) if calls else 0,
            }
        )
    return metrics


def _fields(text: str) -> dict:
    found = {}
    for part in re.split(r"[;\n]", text or ""):
        if ":" in part:
            key, value = part.split(":", 1)
            found[key.strip().lower()] = value.strip()
    return found


def _named_patient(text: str):
    """Return a chart name only when that full name appears in the text."""
    for name in sorted(patient_names(), key=len, reverse=True):
        if name.lower() in (text or "").lower():
            return name
    return None


def identify_patient(query: str, selected: str | None = None):
    """Prefer the UI choice, then a name in the query, then FAISS for phrases like 'my father'."""
    if selected:
        return selected
    named = _named_patient(query)
    if named:
        return named
    personal = ("father", "mother", "dad", "mom", "my ", "patient")
    if not any(word in query.lower() for word in personal):
        return None
    memory = lookup_patient(query)
    if memory:
        return memory.get("name")
    return None


def _history_text(messages: list | None) -> str:
    lines = []
    for message in (messages or [])[-6:]:
        lines.append(f"{message.get('role', 'user')}: {message.get('content', '')[:400]}")
    return "\n".join(lines)


def complete_plan(steps: list, query: str, patient: str | None) -> list:
    """Add a missing tool when the query clearly needs it, so the sample scenario still runs."""
    lowered = query.lower()
    tools = [step["tool"] for step in steps]
    ordered = list(steps)

    def add(index, tool, goal, tool_input):
        ordered.insert(index, {"step": 0, "goal": goal, "tool": tool, "input": tool_input})

    wants_booking = any(word in lowered for word in ("book", "appointment", "schedule"))
    wants_search = any(word in lowered for word in ("treatment", "latest", "symptom", "what is", "what are"))
    if patient and "ehr_history" not in tools and (wants_booking or "history" in lowered or "chart" in lowered or "summarize" in lowered):
        if "treatment" in lowered and "history" not in lowered and "chart" not in lowered and not wants_booking:
            pass
        else:
            add(0, "ehr_history", "Retrieve the patient chart", patient)
    tools = [step["tool"] for step in ordered]
    if wants_booking and "find_slots" not in tools:
        specialty = canonicalize_specialty(query)
        insert_at = tools.index("book_appointment") if "book_appointment" in tools else len(ordered)
        add(insert_at, "find_slots", "Find an open specialty slot", specialty)
    tools = [step["tool"] for step in ordered]
    if wants_booking and "book_appointment" not in tools:
        insert_at = tools.index("find_slots") + 1 if "find_slots" in tools else len(ordered)
        add(insert_at, "book_appointment", "Book the first open slot", f"patient: {patient or ''}; slot: first")
    if wants_search and "medical_search" not in [step["tool"] for step in ordered]:
        ordered.append(
            {
                "step": 0,
                "goal": "Summarize medical information",
                "tool": "medical_search",
                "input": query,
            }
        )
    for index, step in enumerate(ordered, start=1):
        step["step"] = index
    return ordered


def _first_opening(prior: str):
    doctor = ""
    for line in prior.splitlines():
        stripped = line.strip()
        if stripped.startswith("Dr. "):
            doctor = stripped.split("(")[0].strip()
        match = SLOT_RE.search(stripped)
        if doctor and match:
            return doctor, match.group(0)
    return None


def _run_tool(step: dict, query: str, patient: str | None, prior: str):
    tool = step["tool"]
    raw_input = step.get("input") or ""
    fields = _fields(raw_input)
    if tool == "ehr_history":
        candidate = fields.get("patient") or raw_input.strip()
        name = _named_patient(candidate) or patient
        return name, summarize_history(name or "")
    if tool == "find_slots":
        specialty = canonicalize_specialty(fields.get("specialty") or raw_input or query)
        return patient, find_slots(specialty)
    if tool == "book_appointment":
        name = _named_patient(fields.get("patient", "")) or patient or ""
        doctor = fields.get("doctor", "")
        slot = fields.get("slot", "")
        opening = _first_opening(prior)
        if opening and (not SLOT_RE.search(slot) or "first" in doctor.lower() or not doctor):
            doctor, slot = opening
        return name, book_appointment(name, doctor, slot)
    if tool == "medical_search":
        topic = fields.get("topic") or raw_input or query
        result = medical_search(topic)
        return patient, result
    if tool == "update_record":
        name = fields.get("patient") or patient or ""
        note = fields.get("text") or raw_input
        return name, add_or_update_record(name, note)
    return patient, raw_input or step.get("goal") or ""


def run_agent(query: str, selected_patient: str | None = None, messages: list | None = None) -> dict:
    """Plan, execute, update long-term memory, and return the trace for the UI."""
    ensure_seed()
    patient = identify_patient(query, selected_patient)
    patient_context = context_text(patient)
    try:
        plan = make_plan(query, patient_context, _history_text(messages))
    except Exception as exc:
        return {
            "plan": [],
            "step_results": [],
            "final_answer": explain_openai_error(exc),
            "memory_used": {},
            "log_entries": [],
            "patient": patient,
            "medical_info": None,
        }
    plan = complete_plan(plan, query, patient)
    step_results = []
    log_entries = []
    medical_info = None
    prior_chunks = []
    for step in plan:
        started = time.perf_counter()
        try:
            used_patient, output = _run_tool(step, query, patient, "\n".join(prior_chunks))
            success, error = True, None
        except Exception as exc:
            used_patient, output = patient, f"{type(exc).__name__}: {exc}"
            success, error = False, str(exc)
        duration = time.perf_counter() - started
        if isinstance(output, dict) and step["tool"] == "medical_search":
            medical_info = output
            shown = output.get("answer", "")
            links = [item.get("link") for item in output.get("sources", []) if item.get("link")]
            if links:
                shown += "\nSources: " + ", ".join(links)
            output = shown
        if step["tool"] != "none":
            log_entries.append(
                write_log(used_patient or patient, step["tool"], step.get("input"), success, duration, error)
            )
        step_results.append(
            {
                "step": step["step"],
                "goal": step["goal"],
                "tool": step["tool"],
                "input": step.get("input", ""),
                "output": output,
                "success": success,
            }
        )
        prior_chunks.append(f"Step {step['step']} ({step['tool']}):\n{output}")
    steps_text = "\n\n".join(prior_chunks) or "No steps were run."
    try:
        final_answer = message_text(
            get_llm().invoke(
                fill(FINAL_PROMPT, patient_context=patient_context, query=query, steps=steps_text[:12000])
            )
        )
    except Exception as exc:
        final_answer = explain_openai_error(exc) + "\n\n" + steps_text
    memory_used = {}
    try:
        if patient:
            memory_used = remember(patient, query, final_answer)
    except Exception:
        memory_used = {}
    return {
        "plan": plan,
        "step_results": step_results,
        "final_answer": final_answer,
        "memory_used": memory_used,
        "log_entries": log_entries,
        "patient": patient,
        "medical_info": medical_info,
    }
