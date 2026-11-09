"""Persist one record per evaluation run, with its predictions, for later analysis.

Every evaluated system appends a line to ``results/runs.jsonl`` and writes its per-item
predictions to a separate file, so that paired comparisons can be computed after the fact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from shona_news.data import Example
from shona_news.metrics import accuracy, macro_f1, weighted_f1

RUNS_FILE = Path("results/runs.jsonl")


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    task: str
    train_variant: str
    eval_variant: str
    split: str
    seed: int | None
    n: int
    macro_f1: float
    accuracy: float
    weighted_f1: float
    predictions_file: str
    details: dict[str, float | int | str] = {}

    @property
    def key(self) -> tuple[str, str, str, str, str, int | None]:
        return (
            self.system,
            self.task,
            self.train_variant,
            self.eval_variant,
            self.split,
            self.seed,
        )


def record_run(
    *,
    system: str,
    task: str,
    train_variant: str,
    eval_variant: str,
    split: str,
    seed: int | None,
    examples: Sequence[Example],
    predictions: Sequence[str],
    details: dict[str, float | int | str] | None = None,
    root: Path = Path("results"),
) -> RunRecord:
    gold = [example.label for example in examples]
    seed_part = f"_seed{seed}" if seed is not None else ""
    name = f"{task}_{train_variant}_to_{eval_variant}_{split}{seed_part}.jsonl"
    path = root / "predictions" / system / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example, predicted in zip(examples, predictions, strict=True):
            row = {"id": example.id, "gold": example.label, "pred": predicted}
            handle.write(json.dumps(row) + "\n")

    record = RunRecord(
        system=system,
        task=task,
        train_variant=train_variant,
        eval_variant=eval_variant,
        split=split,
        seed=seed,
        n=len(gold),
        macro_f1=macro_f1(gold, predictions),
        accuracy=accuracy(gold, predictions),
        weighted_f1=weighted_f1(gold, predictions),
        predictions_file=str(path),
        details=details or {},
    )
    runs = root / RUNS_FILE.name
    with runs.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return record


def load_runs(root: Path = Path("results")) -> list[RunRecord]:
    path = root / RUNS_FILE.name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [RunRecord.model_validate_json(line) for line in handle if line.strip()]
