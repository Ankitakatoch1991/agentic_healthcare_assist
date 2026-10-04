"""Mock doctor schedule for the brief: find slots and book an appointment.

Reads data/doctors.json. A booking removes the slot and appends data/appointments.json.
There is no external scheduling API.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCTORS_PATH = ROOT / "data" / "doctors.json"
APPOINTMENTS_PATH = ROOT / "data" / "appointments.json"

SPECIALTIES = (
    ("nephro", "Nephrology"),
    ("cardio", "Cardiology"),
    ("endocrin", "Endocrinology"),
    ("general", "General Medicine"),
    ("primary", "General Medicine"),
)


def canonicalize_specialty(text: str) -> str:
    lowered = (text or "").lower()
    for needle, name in SPECIALTIES:
        if needle in lowered:
            return name
    cleaned = (text or "").strip()
    return cleaned or "General Medicine"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_doctors() -> list:
    """Return doctors. If every slot is already in the past, refresh the next 7 days."""
    doctors = _read_json(DOCTORS_PATH, [])
    if doctors and _slots_are_stale(doctors):
        doctors = _refresh_slots(doctors)
        _write_json(DOCTORS_PATH, doctors)
    return doctors


def load_appointments() -> list:
    return _read_json(APPOINTMENTS_PATH, [])


def _slots_are_stale(doctors: list) -> bool:
    stamps = []
    for doctor in doctors:
        for slot in doctor.get("slots", []):
            try:
                stamps.append(datetime.fromisoformat(slot))
            except ValueError:
                continue
    if not stamps:
        return True
    return max(stamps) < datetime.now()


def _refresh_slots(doctors: list) -> list:
    start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    for index, doctor in enumerate(doctors):
        first = start + timedelta(days=index + 1)
        second = (start + timedelta(days=index + 3)).replace(hour=14, minute=0)
        doctor["slots"] = [
            first.isoformat(timespec="seconds"),
            second.isoformat(timespec="seconds"),
        ]
    return doctors


def find_slots(specialty: str) -> str:
    """List open slots for a specialty. Returns a readable message, including when none match."""
    wanted = canonicalize_specialty(specialty)
    doctors = load_doctors()
    matches = [doctor for doctor in doctors if doctor.get("specialty", "").lower() == wanted.lower()]
    if not matches:
        available = sorted({doctor.get("specialty", "") for doctor in doctors})
        return f"No doctors found for '{specialty}'. Specialties on file: {', '.join(available)}."
    lines = [f"Available slots for {wanted}:"]
    for doctor in matches:
        lines.append(f"{doctor['name']} ({doctor['specialty']})")
        slots = doctor.get("slots") or []
        lines.extend(f"  {slot}" for slot in slots)
        if not slots:
            lines.append("  (no open slots)")
    return "\n".join(lines)


def _match_doctor(doctors: list, doctor_name: str):
    needle = doctor_name.lower().replace("dr.", "").strip()
    for doctor in doctors:
        label = doctor["name"].lower().replace("dr.", "").strip()
        if needle and (needle in label or label in needle):
            return doctor
    return None


def book_appointment(patient: str, doctor: str, slot: str) -> str:
    """Remove the slot from the doctor and append the booking. Raises ValueError on failure."""
    patient_name = (patient or "").strip()
    slot = (slot or "").strip()
    if not patient_name:
        raise ValueError("A patient name is required to book.")
    doctors = load_doctors()
    match = _match_doctor(doctors, doctor)
    if not match:
        names = ", ".join(item["name"] for item in doctors)
        raise ValueError(f"No doctor matching '{doctor}'. Doctors on file: {names}.")
    if slot not in match.get("slots", []):
        open_slots = ", ".join(match.get("slots") or []) or "none"
        raise ValueError(f"{match['name']} does not have {slot} open. Open slots: {open_slots}.")
    match["slots"].remove(slot)
    _write_json(DOCTORS_PATH, doctors)
    appointments = load_appointments()
    appointments.append(
        {
            "patient": patient_name,
            "doctor": match["name"],
            "specialty": match["specialty"],
            "slot": slot,
            "booked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    _write_json(APPOINTMENTS_PATH, appointments)
    return f"Booked {patient_name} with {match['name']} ({match['specialty']}) at {slot}."
