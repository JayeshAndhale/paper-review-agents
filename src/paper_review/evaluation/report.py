"""Aggregates checkpointed harness results into a before/after markdown
report: baseline (no verification loop) vs treatment (verification loop
wired in), both scored with the same rubric."""

from collections import defaultdict


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_report(results: list[dict]) -> str:
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in completed:
        by_condition[r["condition"]].append(r)

    lines = ["# Evaluation Report: Baseline vs Treatment", ""]
    lines.append(f"{len(completed)} completed runs, {len(failed)} failed.")
    lines.append("")
    lines.append("| Condition | Runs | Avg grounded rate | Fully grounded | Avg reviewer revisions | Avg verifier revisions |")
    lines.append("|---|---|---|---|---|---|")

    for condition in ("baseline", "treatment"):
        runs = by_condition.get(condition, [])
        if not runs:
            lines.append(f"| {condition} | 0 | - | - | - | - |")
            continue
        grounded_rates = [r["scoring"]["grounded_rate"] for r in runs]
        fully_grounded = sum(1 for r in runs if r["scoring"]["fully_grounded"])
        reviewer_revs = [r["reviewer_revision_count"] for r in runs]
        verifier_revs = [r["verifier_revision_count"] for r in runs]
        lines.append(
            f"| {condition} | {len(runs)} | {_avg(grounded_rates)} "
            f"| {fully_grounded}/{len(runs)} | {_avg(reviewer_revs)} | {_avg(verifier_revs)} |"
        )

    lines.append("")
    lines.append("## Per-topic breakdown")
    lines.append("")
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in completed:
        by_topic[r["topic_name"]].append(r)

    for topic_name, runs in by_topic.items():
        lines.append(f"### {topic_name}")
        for r in sorted(runs, key=lambda r: (r["condition"], r["run_number"])):
            s = r["scoring"]
            lines.append(
                f"- **{r['condition']}** run {r['run_number']}: "
                f"{s['supported']}/{s['total_claims']} supported "
                f"(grounded_rate={s['grounded_rate']}, fully_grounded={s['fully_grounded']}), "
                f"{r['reviewer_revision_count']} reviewer revisions, "
                f"{r['verifier_revision_count']} verifier revisions"
            )
        lines.append("")

    if failed:
        lines.append("## Failed runs")
        for r in failed:
            lines.append(f"- {r['topic_name']} / {r['condition']} / run {r['run_number']}: {r['error']}")

    return "\n".join(lines)
