"""Phase 5 evaluation CLI.

    python scripts/run_evaluation.py ground_truth   # verifier precision/recall
    python scripts/run_evaluation.py matrix          # baseline vs treatment
    python scripts/run_evaluation.py retry           # clear failed matrix runs, then rerun
    python scripts/run_evaluation.py report          # regenerate report from checkpoint

Both ground_truth and matrix are resumable -- re-running either skips
whatever's already recorded in data/evaluation/. Groq's free tier has a
tokens-per-day cap, not just tokens-per-minute, and a TPD wall doesn't
clear with a short backoff -- so a run that dies partway through must be
resumable from where it stopped, not restarted from item 1.
"""

import json
import sys
from pathlib import Path

from paper_review.evaluation.ground_truth import GROUND_TRUTH
from paper_review.evaluation.harness import ResultStore, run_matrix
from paper_review.evaluation.metrics import confusion_matrix, precision_recall_f1
from paper_review.evaluation.report import build_report
from paper_review.verification.pipeline import verify_claim

GROUND_TRUTH_CHECKPOINT = Path("data/evaluation/ground_truth_results.jsonl")


def cmd_ground_truth():
    GROUND_TRUTH_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    done_claims = set()
    if GROUND_TRUTH_CHECKPOINT.exists():
        with open(GROUND_TRUTH_CHECKPOINT) as f:
            done_claims = {json.loads(line)["claim"] for line in f}

    for item in GROUND_TRUTH:
        if item.claim in done_claims:
            continue
        print(f"verifying: {item.claim[:70]}...")
        verified = verify_claim(item.claim, [item.paper_id])
        row = {
            "claim": item.claim,
            "expected": item.expected,
            "predicted": verified.verdict.verdict,
            "correct": item.expected == verified.verdict.verdict,
            "reasoning": verified.verdict.reasoning,
            "note": item.note,
        }
        with open(GROUND_TRUTH_CHECKPOINT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  expected={item.expected} got={verified.verdict.verdict} {'OK' if row['correct'] else 'MISS'}")

    with open(GROUND_TRUTH_CHECKPOINT) as f:
        rows = [json.loads(line) for line in f]

    y_true = [r["expected"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    labels = ["supported", "contradicted", "unsupported"]
    accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0

    print(f"\n{len(rows)}/{len(GROUND_TRUTH)} claims verified. accuracy: {round(accuracy, 3)}\n")
    for score in precision_recall_f1(y_true, y_pred, labels):
        print(
            f"  {score.label:12s} precision={score.precision:.3f} "
            f"recall={score.recall:.3f} f1={score.f1:.3f} support={score.support}"
        )
    print(f"\nconfusion matrix (true, pred) -> count: {dict(confusion_matrix(y_true, y_pred))}")
    print("\nmisclassified:")
    for row in rows:
        if not row["correct"]:
            print(f"  expected={row['expected']:12s} got={row['predicted']:12s} claim={row['claim']}")


def cmd_matrix():
    run_matrix()
    cmd_report()


def cmd_retry():
    dropped = ResultStore().clear_failed()
    print(f"cleared {dropped} failed run(s), retrying...")
    cmd_matrix()


def cmd_report():
    results = ResultStore().all_results()
    report = build_report(results)
    print(report)
    with open("data/evaluation/report.md", "w") as f:
        f.write(report)
    print("\nwritten to data/evaluation/report.md")


COMMANDS = {
    "ground_truth": cmd_ground_truth,
    "matrix": cmd_matrix,
    "retry": cmd_retry,
    "report": cmd_report,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
