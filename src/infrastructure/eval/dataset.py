from __future__ import annotations

import csv
from pathlib import Path

from src.infrastructure.eval.pipeline import EvalCase


def load_eval_cases_from_csv(csv_path: str | Path) -> list[EvalCase]:
    """CSV вида question,answer — эталонный (held-out) набор, отдельный от
    того, что уже проиндексировано в базе (см. data/external/README.md)."""
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return [
        EvalCase(question=(row.get("question") or "").strip(), ideal_answer=(row.get("answer") or "").strip())
        for row in rows
        if (row.get("question") or "").strip() and (row.get("answer") or "").strip()
    ]
