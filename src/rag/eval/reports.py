"""Save EvalReport to JSON + markdown summary."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag.eval.harness import EvalReport


def save_report(report: EvalReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = report.config_name

    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=_default)

    md = f"""# Eval Report: {name}

- **N queries:** {report.n_queries}
- **Precision@5 (mean):** {report.precision_at_5_mean:.4f}
- **Recall@5 (mean):** {report.recall_at_5_mean:.4f}
- **NDCG@5 (mean):** {report.ndcg_at_5_mean:.4f}
- **Latency p50 (warm):** {report.latency_p50_ms:.1f} ms
- **Latency p95 (warm):** {report.latency_p95_ms:.1f} ms
- **Latency p99 (warm):** {report.latency_p99_ms:.1f} ms
- **First-query (cold):** {report.first_query_latency_ms:.1f} ms
"""
    with open(output_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(md)


def _default(o):
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Cannot serialize {type(o)}")


def combine_reports_table(reports: list[EvalReport]) -> str:
    lines = [
        "| Config | P@5 | R@5 | NDCG@5 | p50 ms | p95 ms | p99 ms | cold ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.config_name} "
            f"| {r.precision_at_5_mean:.4f} "
            f"| {r.recall_at_5_mean:.4f} "
            f"| {r.ndcg_at_5_mean:.4f} "
            f"| {r.latency_p50_ms:.1f} "
            f"| {r.latency_p95_ms:.1f} "
            f"| {r.latency_p99_ms:.1f} "
            f"| {r.first_query_latency_ms:.1f} |"
        )
    return "\n".join(lines)
