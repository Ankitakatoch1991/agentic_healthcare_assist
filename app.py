"""Streamlit dashboard for the brief: assistant, patients, schedule, medical info, evaluation, and logs."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.executor import read_logs, run_agent, tool_metrics
from agent.memory import all_memories, ensure_seed
from agent.prompts import explain_openai_error, require_api_key
from tools.appointment_tool import load_appointments, load_doctors
from tools.ehr_tool import (
    add_or_update_record,
    get_patient,
    list_patient_rows,
    summarize_history,
)

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🩺", layout="wide")

ROOT = Path(__file__).resolve().parent
EVAL_RESULTS = ROOT / "logs" / "eval_results.json"

EXAMPLES = [
    (
        "Father / CKD booking",
        "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
        "📅",
    ),
    (
        "Summarize history",
        "Summarize the medical history of Ramesh Kulkarni.",
        "📋",
    ),
    (
        "Find a cardiologist slot",
        "Find an open cardiologist slot for David Thompson.",
        "📅",
    ),
    (
        "Search a disease",
        "What are the latest treatment methods for type 2 diabetes?",
        "🔍",
    ),
]

TOOL_ICONS = {
    "ehr_history": "📋",
    "update_record": "📋",
    "find_slots": "📅",
    "book_appointment": "📅",
    "medical_search": "🔍",
}

TOOL_LABELS = {
    "ehr_history": "Chart history",
    "update_record": "Update record",
    "find_slots": "Find slots",
    "book_appointment": "Book appointment",
    "medical_search": "Medical search",
}

EMPTY_HINTS = (
    "no doctors found",
    "no open slots",
    "(no open slots)",
    "no ehr record found",
    "no visit notes are on file",
    "no medlineplus or who pages were found",
    "no patient was named",
)

DATETIME_FORMAT = "YYYY-MM-DD HH:mm"


def _key_error():
    try:
        require_api_key()
        return None
    except RuntimeError as exc:
        return str(exc)


def _output_text(output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2)


def _step_kind(step: dict) -> str:
    """success, warning for an empty or partial result, or error when the tool failed."""
    if not step.get("success"):
        return "error"
    text = _output_text(step.get("output")).strip()
    if not text:
        return "warning"
    lowered = text.lower()
    if any(hint in lowered for hint in EMPTY_HINTS):
        return "warning"
    return "success"


def _status_icon(kind: str) -> str:
    return {"success": "✅", "warning": "⚠️", "error": "❌"}[kind]


def _tool_icon(tool: str) -> str:
    return TOOL_ICONS.get(tool, "")


def _format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    return f"{float(seconds):.3f} s"


def _step_durations(result: dict) -> list:
    """Pair each executed step with the duration stored on its log line."""
    logs = list(result.get("log_entries") or [])
    durations = []
    cursor = 0
    for step in result.get("step_results") or []:
        if step.get("tool") == "none":
            durations.append(None)
            continue
        if cursor < len(logs):
            durations.append(logs[cursor].get("duration"))
            cursor += 1
        else:
            durations.append(None)
    return durations


def _callout(kind: str, text: str) -> None:
    body = text.strip() or "No output was returned."
    if kind == "error":
        st.error(body)
    elif kind == "warning":
        st.warning(body)
    else:
        st.success(body)


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _show_table(frame: pd.DataFrame, column_config: dict) -> None:
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def _datetime_column(label: str) -> st.column_config.DatetimeColumn:
    return st.column_config.DatetimeColumn(label, format=DATETIME_FORMAT, timezone="UTC")


def _render_plan(steps: list) -> None:
    if not steps:
        return
    lines = []
    for index, step in enumerate(steps, start=1):
        kind = _step_kind(step)
        icon = _tool_icon(step.get("tool", ""))
        goal = step.get("goal") or "Step"
        prefix = f"{_status_icon(kind)} {icon}".strip()
        lines.append(f"{index}. {prefix} {goal}")
    st.markdown("\n".join(lines))


def _render_trace(result: dict) -> None:
    with st.expander("Plan & execution trace", expanded=False):
        st.markdown(f"Patient used: **{result.get('patient') or 'none'}**")
        st.markdown("**Planner JSON**")
        st.json(result.get("plan") or [])
        durations = _step_durations(result)
        for step, duration in zip(result.get("step_results") or [], durations):
            kind = _step_kind(step)
            icon = _tool_icon(step.get("tool", ""))
            tool = step.get("tool") or "unknown"
            st.markdown(f"**{_status_icon(kind)} {icon} Step {step.get('step')} — {tool}**".replace("  ", " "))
            if step.get("goal"):
                st.caption(step["goal"])
            st.markdown("**Input**")
            st.code(str(step.get("input") or ""), language=None)
            st.markdown("**Output**")
            _callout(kind, _output_text(step.get("output")))
            st.markdown(f"Success: `{bool(step.get('success'))}`")
            st.markdown(f"Duration: `{_format_duration(duration)}`")


def _render_assistant_body(answer: str, trace: dict) -> None:
    _render_plan(trace.get("step_results") or [])
    if answer and answer.strip():
        st.markdown(answer)
    else:
        st.warning("The assistant did not return an answer.")
    _render_trace(trace)


def _render_message(message: dict) -> None:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🩺"):
            trace = message.get("trace")
            if trace:
                _render_assistant_body(message.get("content") or "", trace)
            else:
                st.markdown(message.get("content") or "")
        return
    with st.chat_message("user"):
        st.markdown(message.get("content") or "")


def _assistant_tab(selected_patient: str | None) -> None:
    st.subheader("Assistant")
    st.caption("Send a request to book a visit, review a chart, or search a condition.")
    example_columns = st.columns(4)
    for index, (column, (label, query, icon)) in enumerate(zip(example_columns, EXAMPLES)):
        if column.button(
            label,
            key=f"example_{index}",
            help=query,
            icon=icon,
            width="stretch",
            type="primary" if index == 0 else "secondary",
        ):
            st.session_state.pending_query = query
    if not st.session_state.messages:
        st.info("Choose an example above, or type a request below.")
    for message in st.session_state.messages:
        _render_message(message)
    typed = st.chat_input("Ask about a patient, an appointment, or a condition")
    if typed:
        st.session_state.pending_query = typed
    pending = st.session_state.pending_query
    if not pending:
        return
    st.session_state.pending_query = None
    key_error = _key_error()
    if key_error:
        st.error(key_error)
        return
    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user"):
        st.markdown(pending)
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Planning and running tools..."):
            try:
                result = run_agent(
                    pending,
                    selected_patient=selected_patient,
                    messages=st.session_state.messages,
                )
            except Exception as exc:
                result = {
                    "final_answer": explain_openai_error(exc),
                    "plan": [],
                    "step_results": [],
                    "log_entries": [],
                    "patient": selected_patient,
                    "medical_info": None,
                }
    if result.get("medical_info"):
        st.session_state.last_medical = result["medical_info"]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("final_answer") or "",
            "trace": result,
        }
    )
    st.rerun()


def _history_frame(patient: dict) -> pd.DataFrame:
    rows = []
    for record in patient.get("records") or []:
        rows.append(
            {
                "source": record.get("source") or "",
                "recorded": record.get("timestamp") or None,
                "note": record.get("text") or "",
            }
        )
    frame = pd.DataFrame(rows, columns=["source", "recorded", "note"])
    frame["recorded"] = _as_datetime(frame["recorded"])
    return frame


def _patients_tab(selected_patient: str | None) -> None:
    st.subheader("📋 Patients")
    st.caption("View the selected patient's details, raw chart notes, and a generated summary.")
    if st.session_state.get("saved_note") == selected_patient and selected_patient:
        st.success(f"Saved a note for {selected_patient}.")
        st.session_state.saved_note = None
    if not selected_patient:
        st.info("Choose a patient in the sidebar to open a chart.")
        return
    patient = get_patient(selected_patient)
    if not patient:
        st.info(f"No chart found for {selected_patient}.")
        return
    details, history = st.columns([1, 3])
    with details:
        st.markdown(f"**{patient['name']}**")
        st.write(f"Age: {patient['age'] or 'Not recorded'}")
        st.write(f"Gender: {patient['gender'] or 'Not recorded'}")
        st.write(f"Phone: {patient['phone'] or 'Not recorded'}")
        st.write(f"Email: {patient['email'] or 'Not recorded'}")
        st.write(f"Address: {patient['address'] or 'Not recorded'}")
        if patient.get("patient_number"):
            st.write(f"Patient number: {patient['patient_number']}")
        st.write(f"Records on file: {len(patient['records'])}")
    with history:
        st.markdown("**📋 Raw chart notes**")
        frame = _history_frame(patient)
        if frame.empty:
            st.info("No visit notes are on file.")
        else:
            _show_table(
                frame,
                {
                    "source": st.column_config.TextColumn("Source"),
                    "recorded": _datetime_column("Recorded"),
                    "note": st.column_config.TextColumn("Note", width="large"),
                },
            )
        with st.container(border=True):
            st.markdown("**📋 Chart summary**")
            if st.button("Summarize this chart", key="summarize_chart"):
                key_error = _key_error()
                if key_error:
                    st.error(key_error)
                else:
                    with st.spinner("Summarizing..."):
                        try:
                            summary = summarize_history(selected_patient)
                        except Exception as exc:
                            st.error(explain_openai_error(exc))
                        else:
                            st.session_state.chart_summary = {
                                "patient": selected_patient,
                                "text": summary,
                            }
            cached = st.session_state.get("chart_summary") or {}
            if cached.get("patient") == selected_patient and cached.get("text"):
                st.markdown(cached["text"])
            else:
                st.caption("Generate a summary of diagnoses, treatments, and alerts.")
    with st.expander("Add / update record", expanded=False):
        with st.form("add_record"):
            note = st.text_area("Note")
            if st.form_submit_button("Save note"):
                try:
                    add_or_update_record(selected_patient, note)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.saved_note = selected_patient
                    st.rerun()


def _doctor_frame(doctors: list) -> pd.DataFrame:
    rows = []
    for doctor in doctors:
        slots = doctor.get("slots") or []
        if not slots:
            rows.append(
                {
                    "doctor": doctor.get("name") or "",
                    "specialty": doctor.get("specialty") or "",
                    "slot": None,
                }
            )
            continue
        for slot in slots:
            rows.append(
                {
                    "doctor": doctor.get("name") or "",
                    "specialty": doctor.get("specialty") or "",
                    "slot": slot,
                }
            )
    frame = pd.DataFrame(rows, columns=["doctor", "specialty", "slot"])
    frame["slot"] = _as_datetime(frame["slot"])
    return frame


def _appointment_frame(appointments: list) -> pd.DataFrame:
    frame = pd.DataFrame(appointments)
    for column in ("patient", "doctor", "specialty", "slot", "booked_at"):
        if column not in frame.columns:
            frame[column] = None
    frame = frame[["patient", "doctor", "specialty", "slot", "booked_at"]]
    frame["slot"] = _as_datetime(frame["slot"])
    frame["booked_at"] = _as_datetime(frame["booked_at"])
    return frame


def _schedule_tab() -> None:
    st.subheader("📅 Doctors & Appointments")
    st.caption("See which doctors have open slots and which appointments are already booked.")
    # Read the schedule files on every rerun so a booking from Assistant shows up immediately.
    doctors = load_doctors()
    appointments = load_appointments()
    open_slots = sum(len(doctor.get("slots") or []) for doctor in doctors)
    available, open_count, booked = st.columns(3)
    available.metric("Doctors available", len(doctors))
    open_count.metric("Open slots", open_slots)
    booked.metric("Appointments booked", len(appointments))
    slots_column, booked_column = st.columns(2)
    with slots_column:
        st.markdown("**📅 Available doctors and slots**")
        if doctors:
            _show_table(
                _doctor_frame(doctors),
                {
                    "doctor": st.column_config.TextColumn("Doctor"),
                    "specialty": st.column_config.TextColumn("Specialty"),
                    "slot": _datetime_column("Open slot"),
                },
            )
        else:
            st.info("No doctors are on file.")
    with booked_column:
        st.markdown("**📅 Booked appointments**")
        if appointments:
            _show_table(
                _appointment_frame(appointments),
                {
                    "patient": st.column_config.TextColumn("Patient"),
                    "doctor": st.column_config.TextColumn("Doctor"),
                    "specialty": st.column_config.TextColumn("Specialty"),
                    "slot": _datetime_column("Appointment time"),
                    "booked_at": _datetime_column("Booked at"),
                },
            )
        else:
            st.info("No appointments booked yet.")


def _source_lines(sources: list) -> list:
    lines = []
    for source in sources:
        title = (source.get("title") or "Source").replace("[", "(").replace("]", ")")
        link = source.get("link") or ""
        if link:
            lines.append(f"- [{title}]({link})")
        else:
            lines.append(f"- {title}")
    return lines


def _medical_tab() -> None:
    st.subheader("🔍 Medical Info")
    st.caption("Shows the latest search summary and the sources it came from.")
    info = st.session_state.last_medical
    if not info:
        st.info("Run a medical search from the Assistant tab to see results here.")
        return
    with st.container(border=True):
        st.subheader(info.get("query") or "Medical search")
        st.markdown(info.get("answer") or "")
        st.markdown("**Sources**")
        lines = _source_lines(info.get("sources") or [])
        if lines:
            st.markdown("\n".join(lines))
        else:
            st.caption("No source links were returned.")


def _graded_correct(row: dict) -> bool:
    grade = row.get("grade")
    if grade == 1:
        return True
    return str(grade).upper() in {"CORRECT", "Y", "1"}


def _qa_label(payload) -> str:
    rows = (payload or {}).get("results") or []
    if not rows:
        return "—"
    correct = sum(1 for row in rows if _graded_correct(row))
    return f"{correct / len(rows):.0%}"


def _rate_for(metrics: list, tool: str):
    for item in metrics:
        if item.get("tool") == tool:
            return item.get("success_rate")
    return None


def _percent_label(rate) -> str:
    if rate is None:
        return "—"
    return f"{float(rate):.0%}"


def _average_duration(metrics: list):
    calls = sum(item.get("calls") or 0 for item in metrics)
    if not calls:
        return None
    total = sum((item.get("avg_duration_sec") or 0) * (item.get("calls") or 0) for item in metrics)
    return total / calls


def _evaluation_tab() -> None:
    st.subheader("📊 Evaluation")
    st.caption("Grade the fixed dataset and review how often each tool succeeds.")
    if st.button("Run evaluation"):
        key_error = _key_error()
        if key_error:
            st.error(key_error)
        else:
            from evaluation.evaluate import run_evaluation

            with st.spinner("Running the dataset. This makes several API calls."):
                try:
                    st.session_state.eval_payload = run_evaluation()
                except Exception as exc:
                    st.error(explain_openai_error(exc))
    payload = st.session_state.eval_payload
    if payload is None and EVAL_RESULTS.exists():
        payload = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    metrics = tool_metrics()
    score, booking, search, average = st.columns(4)
    score.metric("QAEvalChain score", _qa_label(payload))
    booking.metric("Booking success rate", _percent_label(_rate_for(metrics, "book_appointment")))
    search.metric("Search success rate", _percent_label(_rate_for(metrics, "medical_search")))
    average_seconds = _average_duration(metrics)
    average.metric(
        "Average response time",
        "—" if average_seconds is None else f"{average_seconds:.2f} s",
    )
    if metrics:
        chart = pd.DataFrame(
            {
                "Tool": [TOOL_LABELS.get(item["tool"], item["tool"]) for item in metrics],
                "Success rate (%)": [round((item.get("success_rate") or 0) * 100, 1) for item in metrics],
            }
        )
        st.bar_chart(
            chart,
            x="Tool",
            y="Success rate (%)",
            x_label="Tool",
            y_label="Success rate (%)",
        )
    else:
        st.info("No tool log yet. Ask the assistant something, or run evaluation.")
    st.markdown("**Detailed results**")
    rows = (payload or {}).get("results") or []
    if not rows:
        st.info("No evaluation results yet. Run evaluation to grade the fixed dataset.")
        return
    frame = pd.DataFrame(rows)
    for column in ("query", "reference", "prediction", "tool", "grade", "reasoning"):
        if column not in frame.columns:
            frame[column] = ""
    _show_table(
        frame[["query", "reference", "prediction", "tool", "grade", "reasoning"]],
        {
            "query": st.column_config.TextColumn("Question", width="medium"),
            "reference": st.column_config.TextColumn("Reference answer", width="large"),
            "prediction": st.column_config.TextColumn("Prediction", width="large"),
            "tool": st.column_config.TextColumn("Tool"),
            "grade": st.column_config.TextColumn("Grade"),
            "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )


def _log_frame(logs: list) -> pd.DataFrame:
    frame = pd.DataFrame(logs)
    frame["Success"] = frame["success"].map(lambda ok: "✅" if ok else "❌")
    frame["timestamp"] = _as_datetime(frame["timestamp"])
    columns = ["timestamp", "patient", "tool", "Success", "duration", "input", "error"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _memory_tab(selected_patient: str | None) -> None:
    st.subheader("🧠 Memory & Logs")
    st.caption("Read one patient's saved memory and filter the tool-call log.")
    memory_column, log_column = st.columns([1, 2])
    with memory_column:
        st.markdown("**🧠 Long-term memory**")
        if not selected_patient:
            st.info("Choose a patient in the sidebar to view long-term memory.")
        else:
            memory = next(
                (
                    item
                    for item in all_memories()
                    if str(item.get("name", "")).casefold() == selected_patient.casefold()
                ),
                None,
            )
            if memory:
                st.json(memory)
            else:
                st.info(f"No long-term memory stored for {selected_patient} yet.")
    with log_column:
        st.markdown("**Tool log**")
        logs = read_logs()
        status = st.selectbox("Status", ["All", "Success", "Failed"], key="log_status")
        tool_names = sorted({row.get("tool") or "unknown" for row in logs})
        tool_name = st.selectbox("Tool", ["All", *tool_names], key="log_tool")
        filtered = logs
        if status == "Success":
            filtered = [row for row in filtered if row.get("success")]
        elif status == "Failed":
            filtered = [row for row in filtered if not row.get("success")]
        if tool_name != "All":
            filtered = [row for row in filtered if (row.get("tool") or "unknown") == tool_name]
        if not filtered:
            st.info("No log rows for that filter.")
            return
        _show_table(
            _log_frame(filtered),
            {
                "timestamp": _datetime_column("When"),
                "patient": st.column_config.TextColumn("Patient"),
                "tool": st.column_config.TextColumn("Tool"),
                "Success": st.column_config.TextColumn("Success"),
                "duration": st.column_config.NumberColumn("Duration (seconds)", format="%.3f"),
                "input": st.column_config.TextColumn("Input", width="large"),
                "error": st.column_config.TextColumn("Error", width="medium"),
            },
        )


def _sidebar(rows: list) -> str | None:
    names = ["Auto (from the message)"] + sorted((row["Name"] for row in rows), key=str.casefold)
    selected = st.sidebar.selectbox(
        "Patient",
        names,
        help="Used by every tab. Auto lets the assistant resolve a name from the message.",
    )
    if st.sidebar.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()
    st.sidebar.metric("API key loaded", "✅" if _key_error() is None else "❌")
    st.sidebar.metric("Patients loaded", len(rows))
    st.sidebar.metric("Appointments booked", len(load_appointments()))
    st.sidebar.caption("Educational project — not medical advice. Uses sample/mock data.")
    if selected.startswith("Auto"):
        return None
    return selected


def main():
    ensure_seed()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "last_medical" not in st.session_state:
        st.session_state.last_medical = None
    if "eval_payload" not in st.session_state:
        st.session_state.eval_payload = None
    st.title("🩺 Agentic Healthcare Assistant")
    st.caption(
        "Books appointments · Manages records · Summarizes histories · Searches trusted medical sources"
    )
    key_error = _key_error()
    if key_error:
        st.error(key_error)
    selected_patient = _sidebar(list_patient_rows())
    assistant, patients, schedule, medical, evaluation, memory = st.tabs(
        [
            "Assistant",
            "Patients",
            "Doctors & Appointments",
            "Medical Info",
            "Evaluation",
            "Memory & Logs",
        ]
    )
    with assistant:
        _assistant_tab(selected_patient)
    with patients:
        _patients_tab(selected_patient)
    with schedule:
        _schedule_tab()
    with medical:
        _medical_tab()
    with evaluation:
        _evaluation_tab()
    with memory:
        _memory_tab(selected_patient)


main()
