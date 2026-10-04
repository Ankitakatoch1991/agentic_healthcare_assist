"""QAEvalChain grading for the brief, plus per-tool metrics from logs/agent_log.jsonl.

Run from the project root: python -m evaluation.evaluate
On LangChain 1.x, QAEvalChain lives in langchain_classic (installed with langchain-community).
The import below tries langchain.evaluation first, which is what the brief names.
"""

import json
from pathlib import Path

from agent.executor import tool_metrics, write_log
from agent.prompts import explain_openai_error, get_llm, require_api_key
from tools.ehr_tool import patient_names, summarize_history
from tools.search_tool import medical_search

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = ROOT / "logs" / "eval_results.json"


def _eval_chain():
    try:
        from langchain.evaluation import QAEvalChain
    except ImportError:
        from langchain_classic.evaluation import QAEvalChain
    return QAEvalChain.from_llm(get_llm())


def _route(query: str):
    """History questions go to the chart summarizer. Everything else goes to medical search."""
    lowered = query.lower()
    for name in sorted(patient_names(), key=len, reverse=True):
        if name.lower() in lowered:
            return "ehr_history", name
    if "history" in lowered or "chart" in lowered:
        return "ehr_history", None
    return "medical_search", None


def _predict(item: dict) -> tuple:
    import time

    query = item["query"]
    tool, patient = _route(query)
    started = time.perf_counter()
    error = None
    try:
        if tool == "ehr_history":
            if not patient:
                prediction = "No patient was named in the question."
            else:
                prediction = summarize_history(patient)
        else:
            prediction = medical_search(query)["answer"]
        success = True
    except Exception as exc:
        prediction = f"{type(exc).__name__}: {exc}"
        success, error = False, str(exc)
    write_log(patient, tool, query, success, time.perf_counter() - started, error)
    return prediction, tool


def run_evaluation() -> dict:
    """Grade the dataset and write logs/eval_results.json."""
    require_api_key()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    predictions = []
    graded_rows = []
    for item in dataset:
        prediction, tool = _predict(item)
        predictions.append({"result": prediction})
        graded_rows.append(
            {
                "query": item["query"],
                "reference": item["reference"],
                "prediction": prediction,
                "tool": tool,
            }
        )
    chain = _eval_chain()
    graded = chain.evaluate(
        [{"query": item["query"], "answer": item["reference"]} for item in dataset],
        predictions,
        question_key="query",
        answer_key="answer",
        prediction_key="result",
    )
    for row, grade in zip(graded_rows, graded):
        row["grade"] = grade.get("results") or grade.get("value") or grade.get("score")
        row["reasoning"] = grade.get("text") or grade.get("reasoning") or ""
    payload = {"results": graded_rows, "tool_metrics": tool_metrics()}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    try:
        payload = run_evaluation()
    except Exception as exc:
        print(explain_openai_error(exc))
        raise SystemExit(1)
    correct = 0
    for row in payload["results"]:
        grade = str(row.get("grade", "")).upper()
        if grade in {"CORRECT", "Y", "1"} or row.get("grade") == 1:
            correct += 1
        print(f"[{row.get('grade')}] {row['query']}")
    print(f"\n{correct}/{len(payload['results'])} graded correct")
    print(f"Wrote {RESULTS_PATH}")
    for metric in payload["tool_metrics"]:
        print(
            f"{metric['tool']}: {metric['calls']} calls, "
            f"success {metric['success_rate']}, avg {metric['avg_duration_sec']}s"
        )


if __name__ == "__main__":
    main()
