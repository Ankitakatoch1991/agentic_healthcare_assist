"""EHR loader for the brief: retrieve histories and add or update records.

records.xlsx columns (Sheet1): Phone_number, Email, Name, Age, Gender, Address, Summary.
The patient identifier is Name. PDFs join on that same name via a "Patient:" line,
or, in sample_patient.pdf, the first line plus "Patient #:".
Original xlsx and pdf files are never modified. New notes go to data/ehr_updates.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from agent.prompts import SUMMARIZER_PROMPT, fill, get_llm, message_text

ROOT = Path(__file__).resolve().parents[1]
EHR_DIR = ROOT / "data" / "ehr"
UPDATES_PATH = ROOT / "data" / "ehr_updates.json"

# Added because the supplied EHR has no 70-year-old father with kidney disease.
DEMO_FATHER = {
    "name": "Arun Sharma",
    "phone": "+91-98000-10070",
    "email": "",
    "age": "70",
    "gender": "Male",
    "address": "14 Palm Grove, Delhi",
    "text": (
        "70-year-old male. Relationship: father of the user. "
        "Condition: chronic kidney disease. "
        "This note was added so the sample scenario can retrieve a history. "
        "The original xlsx and pdf files were not modified."
    ),
}

_PATIENTS = None


def _clean(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        return text[:-2]
    return text


def _blank(name: str) -> dict:
    return {
        "name": name,
        "phone": "",
        "email": "",
        "age": "",
        "gender": "",
        "address": "",
        "patient_number": "",
        "records": [],
    }


def _find_key(patients: dict, name: str):
    for key in patients:
        if key.casefold() == name.casefold():
            return key
    return None


def _read_updates() -> dict:
    if not UPDATES_PATH.exists():
        return {}
    return json.loads(UPDATES_PATH.read_text(encoding="utf-8"))


def _write_updates(data: dict) -> None:
    UPDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPDATES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_xlsx(patients: dict) -> None:
    for path in sorted(EHR_DIR.glob("*.xlsx")):
        frame = pd.read_excel(path, engine="openpyxl")
        seen = set()
        for row in frame.to_dict(orient="records"):
            name = _clean(row.get("Name"))
            if not name:
                continue
            fingerprint = tuple(_clean(row.get(col)) for col in frame.columns)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            key = _find_key(patients, name) or name
            patient = patients.setdefault(key, _blank(key))
            patient["phone"] = patient["phone"] or _clean(row.get("Phone_number"))
            patient["email"] = patient["email"] or _clean(row.get("Email"))
            patient["age"] = patient["age"] or _clean(row.get("Age"))
            patient["gender"] = patient["gender"] or _clean(row.get("Gender"))
            patient["address"] = patient["address"] or _clean(row.get("Address"))
            summary = _clean(row.get("Summary"))
            if summary:
                patient["records"].append({"source": path.name, "text": summary})


def _pdf_name(text: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("patient:"):
            return line.split(":", 1)[1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and "patient #" in text.lower() and len(lines[0]) < 80:
        return lines[0]
    return ""


def _load_pdfs(patients: dict) -> None:
    for path in sorted(EHR_DIR.glob("*.pdf")):
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        name = _pdf_name(text)
        if not name:
            continue
        key = _find_key(patients, name) or name
        patient = patients.setdefault(key, _blank(key))
        if "patient #" in text.lower():
            for line in text.splitlines():
                if "patient #" in line.lower():
                    patient["patient_number"] = line.split(":", 1)[-1].strip()
                    break
        patient["records"].append({"source": path.name, "text": text})


def _apply_updates(patients: dict) -> None:
    for name, notes in _read_updates().items():
        key = _find_key(patients, name) or name
        patient = patients.setdefault(key, _blank(key))
        for note in notes:
            patient["phone"] = patient["phone"] or _clean(note.get("phone"))
            patient["email"] = patient["email"] or _clean(note.get("email"))
            patient["age"] = patient["age"] or _clean(note.get("age"))
            patient["gender"] = patient["gender"] or _clean(note.get("gender"))
            patient["address"] = patient["address"] or _clean(note.get("address"))
            patient["records"].append(
                {
                    "source": "ehr_updates.json",
                    "timestamp": note.get("timestamp", ""),
                    "text": note.get("text", ""),
                }
            )


def load_patients(force: bool = False) -> dict:
    """Load every xlsx and pdf under data/ehr/, then overlay saved updates."""
    global _PATIENTS
    if _PATIENTS is not None and not force:
        return _PATIENTS
    patients = {}
    if EHR_DIR.exists():
        _load_xlsx(patients)
        _load_pdfs(patients)
    _apply_updates(patients)
    _PATIENTS = patients
    return patients


def patient_names() -> list:
    return sorted(load_patients(), key=str.casefold)


def list_patient_rows() -> list:
    """Flat rows for the Patients table in the Streamlit UI."""
    rows = []
    for patient in load_patients().values():
        rows.append(
            {
                "Name": patient["name"],
                "Age": patient["age"],
                "Gender": patient["gender"],
                "Phone": patient["phone"],
                "Email": patient["email"],
                "Address": patient["address"],
                "Records": len(patient["records"]),
            }
        )
    return rows


def get_patient(name: str):
    patients = load_patients()
    key = _find_key(patients, name) if name else None
    return patients.get(key) if key else None


def get_history(patient: str) -> str:
    """Return the raw chart. The summarizer prompt turns this into diagnoses, treatments, and alerts."""
    found = get_patient(patient)
    if not found:
        known = ", ".join(patient_names()) or "none"
        return f"No EHR record found for '{patient}'. Known patients: {known}."
    lines = [
        f"Patient: {found['name']}",
        f"Age: {found['age'] or 'not recorded'}",
        f"Gender: {found['gender'] or 'not recorded'}",
        f"Phone: {found['phone'] or 'not recorded'}",
        f"Email: {found['email'] or 'not recorded'}",
        f"Address: {found['address'] or 'not recorded'}",
    ]
    if found["patient_number"]:
        lines.append(f"Patient number: {found['patient_number']}")
    if not found["records"]:
        lines.append("\nNo visit notes are on file.")
        return "\n".join(lines)
    for record in found["records"]:
        stamp = f" {record['timestamp']}" if record.get("timestamp") else ""
        lines.append(f"\n[{record['source']}{stamp}]\n{record['text']}")
    return "\n".join(lines)


def summarize_history(patient: str) -> str:
    """LLM summary of get_history: past diagnoses, treatments, and alerts."""
    chart = get_history(patient)
    if chart.startswith("No EHR record"):
        return chart
    llm = get_llm()
    message = llm.invoke(fill(SUMMARIZER_PROMPT, chart=chart[:8000]))
    return message_text(message)


def add_or_update_record(patient: str, text: str, demographics: dict | None = None) -> str:
    """Append free text for a patient. Does not change the original EHR files."""
    name = (patient or "").strip()
    note = (text or "").strip()
    if not name or not note:
        raise ValueError("Both a patient name and note text are required.")
    updates = _read_updates()
    existing = None
    for key in updates:
        if key.casefold() == name.casefold():
            existing = key
            break
    key = existing or name
    entry = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "text": note,
    }
    if demographics:
        entry.update({k: v for k, v in demographics.items() if k != "text"})
    updates.setdefault(key, []).append(entry)
    _write_updates(updates)
    load_patients(force=True)
    return f"Saved a note for {key}."


def seed_father_if_missing() -> None:
    """Add the demo father chart note when the original files do not contain him."""
    if get_patient(DEMO_FATHER["name"]):
        return
    demo = DEMO_FATHER
    add_or_update_record(
        demo["name"],
        demo["text"],
        {
            "phone": demo["phone"],
            "email": demo["email"],
            "age": demo["age"],
            "gender": demo["gender"],
            "address": demo["address"],
        },
    )
