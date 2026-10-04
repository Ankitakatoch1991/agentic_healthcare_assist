# Agentic Healthcare Assistant for Medical Task Automation

This project is a student capstone: a small agent that books appointments, manages medical records, retrieves patient histories, and searches medical information. A planner turns a request into an ordered list of tool steps. An executor runs those steps, writes a log line for each tool call, and asks the model for one combined reply. Patient facts are stored as JSON and in a FAISS index. The Streamlit app is the dashboard for the chat, charts, schedule, search results, evaluation, and logs.

## Architecture

```
Streamlit UI (app.py)
    -> Planner (agent/planner.py)          one LLM call -> JSON steps
    -> Executor (agent/executor.py)        run steps, pass outputs forward, write logs
         -> EHR tool (tools/ehr_tool.py)
         -> Appointment tool (tools/appointment_tool.py)
         -> Medical search (tools/search_tool.py)   DuckDuckGo + FAISS RAG
         -> Memory (agent/memory.py)                per-patient JSON + FAISS summaries
    -> Final LLM reply
```

| Requirement from the brief | File |
| --- | --- |
| Planner: query to ordered JSON steps, retry once, then fall back | `agent/planner.py` |
| All prompt templates (planner, summarizer, RAG, final reply, memory) | `agent/prompts.py` |
| Executor, step chaining, tool logs, final answer | `agent/executor.py` |
| Long-term JSON memory and FAISS patient lookup ("my father") | `agent/memory.py` |
| Load xlsx/pdf charts, history summary, append a note | `tools/ehr_tool.py` |
| Find specialty slots and book one | `tools/appointment_tool.py` |
| DuckDuckGo search on MedlinePlus / WHO, then RAG | `tools/search_tool.py` |
| 10 question / reference pairs | `evaluation/eval_dataset.json` |
| QAEvalChain grades plus per-tool success metrics | `evaluation/evaluate.py` |
| Streamlit tabs | `app.py` |
| Mock doctors | `data/doctors.json` |
| Booked appointments | `data/appointments.json` |
| Notes added by the agent (original EHR files stay unchanged) | `data/ehr_updates.json` |
| Tool-call log | `logs/agent_log.jsonl` |
| Evaluation output | `logs/eval_results.json` |

Short-term chat history is kept in Streamlit `st.session_state`.

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key

## Setup

```bash
git clone <repo-url>
cd agentic-healthcare-assistant
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# macOS/Linux: cp .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`. Optional: `OPENAI_MODEL` (default `gpt-4o-mini`).

If the key is missing, the app and the evaluator stop with: `OPENAI_API_KEY is missing. Copy .env.example to .env and set OPENAI_API_KEY.`

## EHR files

Place `.xlsx` and `.pdf` files in `data/ehr/`. The copies used for this submission are already there.

`records.xlsx` has one sheet, `Sheet1`, with these columns:

`Phone_number`, `Email`, `Name`, `Age`, `Gender`, `Address`, `Summary`

The patient identifier is **Name**. Duplicate spreadsheet rows for the same person are collapsed. A row with an empty `Summary` still supplies demographics.

PDFs are matched to that same name:

- `sample_report_ramesh.pdf`, `sample_report_anjali.pdf`, and `sample_report_david.pdf` contain a `Patient:` line.
- `sample_patient.pdf` is a longer note whose first line is the patient name and which includes `Patient #:`.

Patients in the supplied files: Rahul Negi, Rebeca Nagle, Ramesh Kulkarni, Anjali Mehra, and David Thompson.

Those files do **not** include a 70-year-old father with chronic kidney disease. On startup the app adds a demo patient, **Arun Sharma** (age 70, chronic kidney disease, relationship "father of the user"). The chart note is stored in `data/ehr_updates.json`. Long-term memory is recreated in `data/memory/` because that folder is gitignored. The original xlsx and pdf files are not modified.

## Run

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

```bash
python -m evaluation.evaluate
```

The dashboard tabs are Assistant, Patients, Doctors & Appointments, Medical Info, Evaluation, and Memory & Logs.

## UI overview

The sidebar selects the patient for every tab, clears the chat, and shows whether the API key is loaded, how many patients are on file, and how many appointments are booked. Screenshot paths below are placeholders.

**Assistant** — Chat from an example query or your own request, then read the plan, the answer, and the execution trace.

![Assistant tab](docs/screenshots/assistant.png)

**Patients** — View the selected patient's details, raw chart, and summary, and add a note.

![Patients tab](docs/screenshots/patients.png)

**Doctors & Appointments** — See how many doctors and open slots are available, and which appointments are already booked.

![Doctors and appointments tab](docs/screenshots/doctors.png)

**Medical Info** — Read the latest medical-search summary and its source links.

![Medical info tab](docs/screenshots/medical-info.png)

**Evaluation** — Check the QAEvalChain score, booking and search success, average response time, and per-question results.

![Evaluation tab](docs/screenshots/evaluation.png)

**Memory & Logs** — Inspect one patient's saved memory and filter the tool-call log by result and tool name.

![Memory and logs tab](docs/screenshots/memory.png)

## Sample scenario

In the Assistant tab, choose **Auto (from the message)** and use the button **Father / CKD booking**, or paste this query:

`My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?`

What you should see:

1. The assistant resolves "my father" to **Arun Sharma** (FAISS summary index, with the stored relationship as a backup).
2. It summarizes his chart: age 70 and chronic kidney disease.
3. It lists nephrology slots and books the first open one (Dr. Meera Iyer or Dr. Arvind Rao).
4. It searches MedlinePlus and WHO and summarizes general CKD treatment, with source links.
5. One combined reply appears in the chat, with the plan listed above it. Open **Plan & execution trace** to see the planner JSON and each step's tool, input, output, success flag, and duration.
6. **Doctors & Appointments** shows the new row from `data/appointments.json`.
7. **Medical Info** shows the treatment summary and links.

Other example buttons cover Ramesh Kulkarni's history, an open cardiology slot for David Thompson, and type 2 diabetes treatment.

Doctor slots in `data/doctors.json` cover the week of 4–10 October 2026. If every slot is already in the past when you run the app, the appointment tool refreshes them across the next 7 days.

## Evaluation and logs

`evaluation/eval_dataset.json` has 10 pairs. Six ask for a chart summary. Four ask for general medical information.

`python -m evaluation.evaluate` (or the **Run evaluation** button) does three things:

- Answers history questions with the EHR summarizer and medical questions with the search tool.
- Grades each prediction with QAEvalChain. On current LangChain, that class is imported from `langchain.evaluation` when it exists, and otherwise from `langchain_classic.evaluation` (pulled in by `langchain-community`). The project does not use LangGraph.
- Writes `logs/eval_results.json` with the grade, the prediction, and per-tool metrics.

Every tool call is appended to `logs/agent_log.jsonl` with timestamp, patient, tool, input, success, duration, and error. The Evaluation tab shows the QAEvalChain score, booking and search success rates, average response time, a bar chart of success rate by tool, and the grade table. The Memory & Logs tab shows each patient's saved memory and the same log with a success/failed filter and a tool filter.

## Project structure

```
agentic-healthcare-assistant/
├── .streamlit/
│   └── config.toml
├── app.py
├── agent/
│   ├── planner.py
│   ├── executor.py
│   ├── prompts.py
│   └── memory.py
├── tools/
│   ├── appointment_tool.py
│   ├── ehr_tool.py
│   └── search_tool.py
├── evaluation/
│   ├── eval_dataset.json
│   └── evaluate.py
├── data/
│   ├── ehr/
│   ├── doctors.json
│   ├── appointments.json
│   ├── ehr_updates.json
│   └── memory/
├── logs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Submission

Submit the GitHub repository link. `.env` is listed in `.gitignore` and must not be committed. The grader copies `.env.example` to `.env` and adds their own `OPENAI_API_KEY`. `logs/*.jsonl`, `__pycache__/`, `data/memory/*`, and FAISS index files are also ignored.

## Disclaimer

This is an educational project. It is not medical advice and it is not for clinical use. Charts in `data/ehr/` are sample files. Doctor availability and Arun Sharma's record are mock data created for the assignment.
