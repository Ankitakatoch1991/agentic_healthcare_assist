"""LLM planner for the brief: one chat call turns a query into an ordered JSON tool list."""

import json
import re

from agent.prompts import PLANNER_PROMPT, fill, get_llm, message_text

TOOLS = {
    "ehr_history",
    "find_slots",
    "book_appointment",
    "medical_search",
    "update_record",
    "none",
}


def _parse_steps(text: str) -> list:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start_list = cleaned.find("[")
    start_obj = cleaned.find("{")
    if start_list == -1 or (start_obj != -1 and start_obj < start_list):
        start, end = start_obj, cleaned.rfind("}")
    else:
        start, end = start_list, cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Planner did not return JSON.")
    data = json.loads(cleaned[start : end + 1])
    if isinstance(data, dict):
        data = data.get("steps") or data.get("plan") or [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Planner JSON was empty.")
    steps = []
    for index, step in enumerate(data, start=1):
        tool = str(step.get("tool", "none")).strip()
        if tool not in TOOLS:
            tool = "none"
        steps.append(
            {
                "step": int(step.get("step", index)),
                "goal": str(step.get("goal", "")),
                "tool": tool,
                "input": str(step.get("input", "")),
            }
        )
    return steps


def _fallback(query: str) -> list:
    return [
        {
            "step": 1,
            "goal": "Respond directly to the user",
            "tool": "none",
            "input": query,
        }
    ]


def make_plan(query: str, patient_context: str, history: str) -> list:
    """Call the planner once, retry once if the JSON is invalid, then fall back."""
    llm = get_llm()
    prompt = fill(
        PLANNER_PROMPT,
        patient_context=patient_context or "None",
        history=history or "None",
        query=query,
    )
    first = message_text(llm.invoke(prompt))
    try:
        return _parse_steps(first)
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    second = message_text(
        llm.invoke(prompt + "\n\nYour previous reply was not valid JSON. Return only the JSON list.")
    )
    try:
        return _parse_steps(second)
    except (ValueError, json.JSONDecodeError, TypeError):
        return _fallback(query)
