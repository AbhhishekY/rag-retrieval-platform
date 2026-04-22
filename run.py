"""
run.py — One-shot launcher for the full RAG pipeline.

Runs every step in order:
  1. Preflight   — download models + corpus
  2. Ingest      — chunk + embed + build indices
  3. Tests       — verify 22/22 pass
  4. Eval x4     — semantic_only, bm25_only, hybrid, hybrid+rerank
  5. Summary     — results table printed to terminal

Before running, install dependencies:
  pip install -e ".[dev]"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ──────────────────────────────────────────────
# Dependency check — fast fail with clear message
# ──────────────────────────────────────────────
def _check_deps() -> None:
    missing = []
    for pkg in ("fastembed", "faiss", "rank_bm25", "fastapi", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n\033[91m✗  Dependencies not installed.\033[0m")
        print("   Run this first:\n")
        print("     pip install -e \".[dev]\"\n")
        print(f"   Missing: {', '.join(missing)}\n")
        sys.exit(1)

_check_deps()


# ──────────────────────────────────────────────
# ANSI colour palette (no extra deps)
# ──────────────────────────────────────────────
R  = "\033[0m"          # reset
BOLD = "\033[1m"
DIM  = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"

BOX_H  = "─"
BOX_V  = "│"
BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_LM = "├"
BOX_RM = "┤"

WIDTH = 72


def _line(char: str = BOX_H) -> str:
    return char * WIDTH


def banner() -> None:
    lines = [
        "",
        f"{CYAN}{BOLD}{BOX_TL}{_line()}{BOX_TR}{R}",
        f"{CYAN}{BOLD}{BOX_V}{R}{'RAG Retrieval Platform — Full Pipeline Runner':^{WIDTH}}{CYAN}{BOLD}{BOX_V}{R}",
        f"{CYAN}{BOLD}{BOX_V}{R}{'hybrid BM25 + semantic + reranker | eval harness':^{WIDTH}}{CYAN}{BOLD}{BOX_V}{R}",
        f"{CYAN}{BOLD}{BOX_BL}{_line()}{BOX_BR}{R}",
        "",
    ]
    print("\n".join(lines))


def section(title: str) -> None:
    pad = WIDTH - 2 - len(title)
    left = pad // 2
    right = pad - left
    print(f"\n{BLUE}{BOLD}{BOX_TL}{BOX_H * left} {title} {BOX_H * right}{BOX_TR}{R}")


def step_start(n: int, total: int, label: str) -> None:
    tag = f"[{n}/{total}]"
    print(f"\n{YELLOW}{BOLD}  {tag}{R} {WHITE}{label}{R}")
    print(f"  {DIM}{'─' * (WIDTH - 4)}{R}")


def step_ok(label: str, elapsed: float) -> None:
    print(f"  {GREEN}{BOLD}✓  {label}{R}  {DIM}({elapsed:.1f}s){R}")


def step_fail(label: str, elapsed: float) -> None:
    print(f"  {RED}{BOLD}✗  {label}{R}  {DIM}({elapsed:.1f}s){R}")


def info(msg: str) -> None:
    for line in textwrap.wrap(msg, WIDTH - 6):
        print(f"  {DIM}→ {line}{R}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠  {msg}{R}")


def result_row(label: str, value: str, colour: str = WHITE) -> None:
    dots = "." * (WIDTH - len(label) - len(value) - 6)
    print(f"  {BOLD}{label}{R}  {DIM}{dots}{R}  {colour}{BOLD}{value}{R}")


# ──────────────────────────────────────────────
# Subprocess runner
# ──────────────────────────────────────────────

def run_cmd(
    cmd: list[str],
    label: str,
    *,
    verbose: bool = False,
    timeout: int = 1800,
) -> tuple[bool, str, float]:
    """Run a command, stream output if verbose, return (ok, combined_output, elapsed)."""
    t0 = time.monotonic()
    lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            lines.append(line)
            if verbose:
                print(f"  {DIM}│ {line}{R}")

        proc.wait(timeout=timeout)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        ok = False
        lines.append(f"[TIMEOUT after {timeout}s]")
    except Exception as exc:
        ok = False
        lines.append(str(exc))

    elapsed = time.monotonic() - t0
    output = "\n".join(lines)
    return ok, output, elapsed


# ──────────────────────────────────────────────
# Eval result parser
# ──────────────────────────────────────────────

def parse_eval_output(output: str) -> dict | None:
    """Extract P@5, R@5, NDCG@5, p95 from run_eval.py stdout."""
    for line in output.splitlines():
        line = line.strip()
        if "NDCG@5=" in line:
            try:
                parts = dict(p.split("=") for p in line.split())
                return {
                    "P@5":    float(parts.get("P@5",    0)),
                    "R@5":    float(parts.get("R@5",    0)),
                    "NDCG@5": float(parts.get("NDCG@5", 0)),
                    "p95ms":  float(parts.get("p95",    "0").rstrip("ms")),
                }
            except Exception:
                pass
    # fallback: try to read json report
    return None


def load_report_json(config: str, output_dir: Path) -> dict | None:
    p = output_dir / "runs" / f"{config}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return {
            "P@5":    data.get("precision_at_5_mean", 0),
            "R@5":    data.get("recall_at_5_mean",    0),
            "NDCG@5": data.get("ndcg_at_5_mean",      0),
            "p95ms":  data.get("latency_p95_ms",       0),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

def summary_table(results: dict[str, dict | None], step_log: list[tuple]) -> None:
    section("FINAL RESULTS")

    # Step status table
    print(f"\n  {BOLD}{'Step':<28}{'Status':>10}{'Time':>10}{R}")
    print(f"  {DIM}{'─'*48}{R}")
    for name, ok, elapsed in step_log:
        status = f"{GREEN}PASSED{R}" if ok else f"{RED}FAILED{R}"
        print(f"  {name:<28}{status:>18}  {DIM}{elapsed:>5.1f}s{R}")

    # Eval metrics table
    if any(v is not None for v in results.values()):
        print(f"\n  {BOLD}{'Config':<22}{'P@5':>8}{'R@5':>8}{'NDCG@5':>10}{'p95 (ms)':>12}{R}")
        print(f"  {DIM}{'─'*60}{R}")

        best_ndcg = max((v["NDCG@5"] for v in results.values() if v), default=0)

        for config, v in results.items():
            if v is None:
                print(f"  {config:<22}  {DIM}(not run){R}")
                continue
            ndcg = v["NDCG@5"]
            star = f" {YELLOW}★{R}" if ndcg == best_ndcg else "  "
            ndcg_col = f"{GREEN}{BOLD}{ndcg:.4f}{R}" if ndcg == best_ndcg else f"{ndcg:.4f}"
            p95_col = f"{RED}{v['p95ms']:.0f}{R}" if v["p95ms"] > 500 else f"{v['p95ms']:.0f}"
            print(
                f"  {config:<22}"
                f"  {v['P@5']:.4f}"
                f"  {v['R@5']:.4f}"
                f"  {ndcg_col:>16}"
                f"  {p95_col:>14}"
                f"{star}"
            )

    print(f"\n  {CYAN}{BOLD}{'─'*WIDTH}{R}")
    print(f"  {DIM}Reports saved to outputs/runs/  |  Indices in indices/default/{R}")
    print(f"  {DIM}Start API: uvicorn rag.api.app:app --reload{R}\n")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

EVAL_CONFIGS = ["semantic_only", "bm25_only", "hybrid", "hybrid+rerank"]
TOTAL_STEPS = 3 + len(EVAL_CONFIGS)   # preflight + ingest + tests + evals

INDEX_DIR = Path("indices/default")
OUTPUT_DIR = Path("outputs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full RAG pipeline end-to-end.")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of eval queries per config (default: 50 for speed; use 200 for full eval)",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip model/corpus download (use if already done)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip ingest (use if indices already exist)",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip pytest",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream subprocess output in real time",
    )
    parser.add_argument(
        "--configs", nargs="+", default=EVAL_CONFIGS,
        choices=EVAL_CONFIGS,
        help="Which eval configs to run",
    )
    args = parser.parse_args()

    banner()

    py = sys.executable
    step_log: list[tuple[str, bool, float]] = []
    eval_results: dict[str, dict | None] = {}
    step_n = 0

    total = (
        (0 if args.skip_preflight else 1)
        + (0 if args.skip_ingest   else 1)
        + (0 if args.skip_tests    else 1)
        + len(args.configs)
    )

    # ── STEP: PREFLIGHT ────────────────────────────────────────
    if not args.skip_preflight:
        step_n += 1
        step_start(step_n, total, "Preflight: downloading models + corpus")
        info("Downloads: FastEmbed embedder, cross-encoder, MultiHop-RAG dataset")
        info("This takes 5–10 min on first run; instant if already cached")
        ok, out, elapsed = run_cmd(
            [py, "scripts/preflight.py"],
            "Preflight",
            verbose=args.verbose,
        )
        step_log.append(("Preflight", ok, elapsed))
        if ok:
            step_ok("Models + corpus ready", elapsed)
        else:
            step_fail("Preflight failed", elapsed)
            if not args.verbose:
                print(f"\n{RED}--- last 20 lines of output ---{R}")
                for ln in out.splitlines()[-20:]:
                    print(f"  {DIM}{ln}{R}")
            warn("Continuing — if indices already exist eval may still work")
    else:
        warn("Preflight skipped")

    # ── STEP: INGEST ───────────────────────────────────────────
    if not args.skip_ingest:
        step_n += 1
        step_start(step_n, total, "Ingest: chunk + embed + build BM25 & FAISS indices")
        info("Reads 609 MultiHop-RAG articles → ~47K chunks")
        info("Unchanged docs are skipped via SHA256 manifest (fast on re-runs)")
        ok, out, elapsed = run_cmd(
            [py, "scripts/ingest.py"],
            "Ingest",
            verbose=args.verbose,
            timeout=900,
        )
        step_log.append(("Ingest", ok, elapsed))
        if ok:
            # try to extract stats
            for line in out.splitlines():
                if any(kw in line for kw in ("chunks", "docs", "indexed", "skipped", "saved")):
                    info(line.strip())
            step_ok("Indices built at indices/default/", elapsed)
        else:
            step_fail("Ingest failed", elapsed)
            if not args.verbose:
                for ln in out.splitlines()[-20:]:
                    print(f"  {DIM}{ln}{R}")
    else:
        warn("Ingest skipped")

    # ── STEP: TESTS ────────────────────────────────────────────
    if not args.skip_tests:
        step_n += 1
        step_start(step_n, total, "Tests: pytest (22 unit + integration tests)")
        ok, out, elapsed = run_cmd(
            [py, "-m", "pytest", "tests/", "-v", "--tb=short"],
            "Tests",
            verbose=args.verbose,
        )
        step_log.append(("Tests", ok, elapsed))
        # extract summary line
        for line in reversed(out.splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                col = GREEN if ok else RED
                print(f"  {col}{BOLD}  {line.strip()}{R}")
                break
        if ok:
            step_ok("All tests passed", elapsed)
        else:
            step_fail("Tests failed — check output above", elapsed)
            if not args.verbose:
                for ln in out.splitlines()[-30:]:
                    print(f"  {DIM}{ln}{R}")
    else:
        warn("Tests skipped")

    # ── STEP: EVAL CONFIGS ─────────────────────────────────────
    section(f"Evaluation  ({args.limit} queries per config)")
    print(f"\n  {DIM}Metric guide:{R}")
    print(f"  {DIM}  NDCG@5  = quality of top-5 ranking (1.0 = perfect){R}")
    print(f"  {DIM}  P@5     = precision: fraction of top-5 that are relevant{R}")
    print(f"  {DIM}  R@5     = recall: fraction of all relevant docs found{R}")
    print(f"  {DIM}  p95     = 95th-percentile latency in ms (budget: 500ms){R}")

    for config in args.configs:
        step_n += 1
        step_start(step_n, total, f"Eval: {config}")
        if config == "semantic_only":
            info("FAISS dense search only — no BM25, no rerank")
        elif config == "bm25_only":
            info("BM25 keyword search only — no embeddings")
        elif config == "hybrid":
            info("BM25 + dense fused via RRF — the shipping config")
        elif config == "hybrid+rerank":
            info("Hybrid + cross-encoder rerank on top-20 → top-5")

        ok, out, elapsed = run_cmd(
            [py, "scripts/run_eval.py", "--config", config, "--limit", str(args.limit)],
            f"Eval:{config}",
            verbose=args.verbose,
            timeout=600,
        )
        step_log.append((f"Eval: {config}", ok, elapsed))

        # parse metrics
        metrics = parse_eval_output(out)
        if metrics is None and ok:
            metrics = load_report_json(config, OUTPUT_DIR)
        eval_results[config] = metrics

        if ok and metrics:
            ndcg = metrics["NDCG@5"]
            p95  = metrics["p95ms"]
            ndcg_col = GREEN if ndcg >= 0.55 else (YELLOW if ndcg >= 0.45 else RED)
            p95_col  = RED if p95 > 500 else GREEN
            print(
                f"  {BOLD}NDCG@5={ndcg_col}{BOLD}{ndcg:.4f}{R}  "
                f"P@5={metrics['P@5']:.4f}  "
                f"R@5={metrics['R@5']:.4f}  "
                f"p95={p95_col}{BOLD}{p95:.0f}ms{R}"
            )
            step_ok(f"{config} eval complete", elapsed)
        elif ok:
            step_ok(f"{config} eval complete (no metrics parsed)", elapsed)
        else:
            step_fail(f"{config} eval failed", elapsed)
            eval_results[config] = None
            if not args.verbose:
                for ln in out.splitlines()[-20:]:
                    print(f"  {DIM}{ln}{R}")

    # ── FINAL SUMMARY ──────────────────────────────────────────
    summary_table(eval_results, step_log)

    any_failed = any(not ok for _, ok, _ in step_log)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
