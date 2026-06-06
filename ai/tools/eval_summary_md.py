"""Render an evaluation summary CSV as a GitHub-flavoured markdown table.

A tiny, dependency-free post-processing step (stdlib `csv` only). It does not
recompute anything — it reads a summary CSV already written by an evaluator
(e.g. `eval_ablation_summary.csv` from `ai.tools.evaluate_ablation`) and emits an
equivalent markdown table for the report / README. Cell values are copied through
verbatim so the table cannot disagree with the CSV; only a trailing ``_mean`` in
column headers is stripped for readability.

Run from the repo root, e.g. for the Phase 6.3 v2 eval:

    python -m ai.tools.eval_summary_md \
        --csv results/v2_eval/eval_ablation_summary.csv \
        --out results/v2_eval/eval_ablation_summary.md \
        --title "Dataset v2 unified evaluation (test split)"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _format_table(header: list[str], rows: list[list[str]], title: str | None) -> str:
    # Strip a trailing "_mean" from headers for a cleaner table; values untouched.
    cols = [h[:-5] if h.endswith("_mean") else h for h in header]
    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        # Pad/truncate defensively so a ragged row can't break the table.
        cells = (r + [""] * len(cols))[: len(cols)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path, help="summary CSV to read")
    ap.add_argument("--out", required=True, type=Path, help="markdown file to write")
    ap.add_argument("--title", default=None, help="optional H1 title")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"eval_summary_md FAILED: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    with args.csv.open(newline="") as f:
        reader = list(csv.reader(f))
    if not reader:
        print(f"eval_summary_md FAILED: empty CSV: {args.csv}", file=sys.stderr)
        return 1

    header, rows = reader[0], reader[1:]
    md = _format_table(header, rows, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"OK: wrote {len(rows)} row(s) to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
