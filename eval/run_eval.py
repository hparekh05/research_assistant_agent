"""Baseline evaluation for the research assistant on the real-PDF question set.

Runs each question in eval/questions.csv (or the file given with --dataset) through the real orchestrator (the same code the
app uses) with that question's PDF uploaded, then scores:

  retrieval_hit     did the PDF search return at least one gold page?         (answerable questions)
  retrieval_recall  what share of the gold pages did the search return?       (answerable)
  cites_gold        does the answer text cite at least one gold page?         (answerable)
  correct           LLM judge: matches the reference, or correctly says "not stated"
  faithful          LLM judge: every claim is supported by the retrieved text
  latency           seconds per question

Nothing here changes pass/fail behavior. It records a baseline.
Results are written to eval/results/baseline_<timestamp>.json (full answers) and .md (summary table).

Usage (from the repo root):
    python eval/run_eval.py                    # all questions
    python eval/run_eval.py --only a03,a10     # specific question ids
    python eval/run_eval.py --limit 5          # first N questions
    python eval/run_eval.py --no-judge         # skip the LLM judge (retrieval metrics only)
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

JUDGE_PROMPT = """You are grading an AI research assistant that answers questions about a PDF.

Question:
{question}

Text the assistant retrieved from the PDF:
{contexts}

Assistant's answer:
{answer}

Reference answer written by a human:
{reference}

Judge two things:
1. faithful: is EVERY factual claim in the answer supported by the retrieved text above?
   (Say true if the answer only says the information is not available.)
2. correct: {correct_rule}

Respond with JSON only: {{"faithful": true|false, "correct": true|false, "reason": "<one sentence>"}}"""

CORRECT_RULE_ANSWERABLE = (
    "does the answer agree with the reference answer? If the reference has several parts, "
    "all parts must be present. Extra detail is fine as long as it does not contradict the reference."
)
CORRECT_RULE_UNANSWERABLE = (
    "the PDF does not contain the answer. The answer is correct ONLY if it clearly says the "
    "information is not stated or not available, and does not invent a figure or fact."
)

PAGE_RE = re.compile(r"\b(?:pages?|pp?\.)\s*((?:\b\d{1,3}\b\s*(?:,|and|&|to|-|\u2013)?\s*)+)", re.I)


class LocalUpload:
    """Mimics Streamlit's UploadedFile (the orchestrator only uses .name and .getvalue())."""

    def __init__(self, path):
        self.name = Path(path).name
        self._data = Path(path).read_bytes()

    def getvalue(self):
        return self._data


def answer_text(answer):
    """Claude can return a list of content blocks instead of a plain string."""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
    return str(answer or "")


def pages_in_text(text):
    pages = set()
    for m in PAGE_RE.finditer(text or ""):
        pages.update(int(n) for n in re.findall(r"\b\d{1,3}\b", m.group(1)))
    return pages


def retrieved_pages(pdf_sources, pdf_name):
    pages = set()
    for s in pdf_sources or []:
        m = re.match(r"(.+), page (\d+)$", s)
        if m and m.group(1) == pdf_name:
            pages.add(int(m.group(2)))
    return pages


def load_questions(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gold = [int(p) for p in r["gold_pages_viewer"].split(";") if p.strip()]
            rows.append({**r, "gold": gold})
    return rows


def judge(client, item, answer, contexts):
    unanswerable = item["type"] == "unanswerable"
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        contexts=("\n---\n".join(contexts))[:120000] or "(nothing retrieved)",
        answer=answer or "(empty answer)",
        reference=item["reference_answer"],
        correct_rule=CORRECT_RULE_UNANSWERABLE if unanswerable else CORRECT_RULE_ANSWERABLE,
    )
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return bool(data.get("faithful")), bool(data.get("correct")), data.get("reason", "")


def rate(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def summarize(rows):
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": len(rows),
        "retrieval_hit": rate(rows, "retrieval_hit"),
        "retrieval_recall": rate(rows, "retrieval_recall"),
        "cites_gold": rate(rows, "cites_gold"),
        "correct": rate(rows, "correct"),
        "faithful": rate(rows, "faithful"),
        "empty_answer": rate(rows, "empty_answer"),
        "avg_web_calls": rate(rows, "web_calls"),
        "crashed": sum(1 for r in rows if r["error"]),
        "latency_mean_s": round(statistics.mean(lat), 1),
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
    }


def markdown_table(summaries):
    cols = ["n", "retrieval_hit", "retrieval_recall", "cites_gold", "correct", "faithful",
            "empty_answer", "avg_web_calls", "crashed", "latency_mean_s", "latency_p95_s"]
    out = ["| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        out.append(f"| {name} | " + " | ".join("-" if s[c] is None else str(s[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "questions.csv"))
    ap.add_argument("--docs", default=str(ROOT / "eval" / "docs"))
    ap.add_argument("--outdir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="run every question N times (results vary run to run)")
    args = ap.parse_args()

    items = load_questions(args.dataset)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit("No questions selected.")
    items = [dict(i, run=r + 1) for r in range(max(1, args.repeats)) for i in items]

    missing = sorted({i["pdf"] for i in items if not (Path(args.docs) / i["pdf"]).is_file()})
    if missing:
        sys.exit("Missing PDFs in eval/docs/: " + ", ".join(missing) +
                 "\nPut them in eval/docs/ with exactly these names (see eval/SOURCES.md for the real-PDF set).")

    from src.orchestrator import orchestrator  # imported late so errors above show first

    client = None
    if not args.no_judge:
        import anthropic
        client = anthropic.Anthropic()

    rows, judge_error_shown = [], False
    for n, item in enumerate(items, 1):
        pdf_path = Path(args.docs) / item["pdf"]
        unanswerable = item["type"] == "unanswerable"

        start = time.perf_counter()
        try:
            out = orchestrator(query=item["question"], uploaded_files=[LocalUpload(pdf_path)], chat_history=[])
            error = None
        except Exception as e:  # a crash is recorded as a failed case, not a stopped run
            out, error = {}, f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - start

        answer = answer_text(out.get("answer"))
        calls = out.get("tool_calls", [])
        contexts = out.get("contexts", [])
        got = retrieved_pages(out.get("pdf_sources", []), item["pdf"])
        cited = pages_in_text(answer)
        gold = set(item["gold"])

        faithful = correct = None
        reason = "skipped"
        if not error and not answer.strip():
            faithful, correct, reason = False, False, "empty answer"
        elif client and not error:
            try:
                faithful, correct, reason = judge(client, item, answer, contexts)
            except Exception as e:
                reason = f"judge error: {type(e).__name__}: {e}"
                if not judge_error_shown:
                    print(f"\n!! The LLM judge failed ({reason}).\n"
                          f"!! Check ANTHROPIC_API_KEY, or set JUDGE_MODEL in .env to a model you can use.\n")
                    judge_error_shown = True

        row = {
            "id": item["id"], "type": item["type"], "pdf": item["pdf"], "question": item["question"],
            "reference_answer": item["reference_answer"], "answer": answer,
            "gold_pages": sorted(gold), "retrieved_pages": sorted(got), "pages_cited_in_answer": sorted(cited),
            "retrieval_hit": None if unanswerable else bool(gold & got),
            "retrieval_recall": None if unanswerable else round(len(gold & got) / len(gold), 2),
            "cites_gold": None if unanswerable else bool(gold & cited),
            "called_pdf_tool": any(c["name"] == "search_pdfs" for c in calls),
            "web_calls": sum(1 for c in calls if c["name"] == "search_web"),
            "run": item["run"],
            "tools_called": [c["name"] for c in calls],
            "empty_answer": not answer.strip(),
            "faithful": faithful, "correct": correct, "judge_reason": reason,
            "latency_s": round(latency, 1), "error": error,
        }
        rows.append(row)
        print(f"[{n}/{len(items)}] {item['id']} run{item['run']} ({item['type']}) "
              f"retrieved={sorted(got)} gold={sorted(gold)} hit={row['retrieval_hit']} "
              f"correct={correct} faithful={faithful} {latency:.0f}s" + (f"  ERROR {error}" if error else ""))

    summaries = {"ALL": summarize(rows)}
    for t in ["single-fact", "multi-page", "no-keyword-overlap", "unanswerable"]:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            summaries[t] = summarize(sub)
    table = markdown_table(summaries)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (outdir / f"baseline_{stamp}.json").write_text(json.dumps({"summary": summaries, "rows": rows}, indent=2))
    (outdir / f"baseline_{stamp}.md").write_text(
        f"# Baseline eval ({stamp})\n\nJudge model: {JUDGE_MODEL}\n\n{table}\n")
    print("\n" + table)
    print(f"\nSaved to {outdir}/baseline_{stamp}.json and .md")


if __name__ == "__main__":
    main()
