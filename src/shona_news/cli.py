"""Command-line entry point: ``shona-news <command>``."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Annotated, cast, get_args

import typer
from transformers import AutoTokenizer

from shona_news import classical
from shona_news.config import Variant, load_classical, load_encoders, load_tasks
from shona_news.data import TaskData, load_variant
from shona_news.encoders import fine_tune
from shona_news.fertility import fertility
from shona_news.runs import load_runs, record_run

app = typer.Typer(no_args_is_help=True, help="Shona and English news topic classification.")

RESULTS = Path("results")


def _variants_for(task: str) -> list[Variant]:
    return ["sna", "eng", "eng_matched"] if task == "masakhanews" else ["sna", "eng"]


@app.command("data-summary")
def data_summary() -> None:
    """Print split sizes and label counts for every task and variant."""
    tasks = load_tasks()
    summary: dict[str, dict[str, int]] = {}
    for task in tasks.tasks:
        for variant in _variants_for(task):
            data = load_variant(tasks, task, variant)
            summary[f"{task}/{variant}"] = {
                split: len(data.split(split)) for split in ("train", "validation", "test")
            }
            typer.echo(f"{task:12s} {variant:12s} {summary[f'{task}/{variant}']}")
    path = RESULTS / "data_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@app.command("fertility")
def fertility_command() -> None:
    """Subword tokens per word on each training split, for every encoder tokenizer."""
    tasks, encoders = load_tasks(), load_encoders()
    table: dict[str, dict[str, float]] = {}
    for name, model_id in encoders.models.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for task in tasks.tasks:
            for variant in ("sna", "eng"):
                data = load_variant(tasks, task, variant)
                value = fertility(tokenizer, (e.text for e in data.train))
                table.setdefault(name, {})[f"{task}/{variant}"] = round(value, 3)
                typer.echo(f"{name:18s} {task:12s} {variant}: {value:.2f}")
    (RESULTS / "fertility.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


@app.command("classical")
def classical_command() -> None:
    """Majority-class and TF-IDF baselines for every task and variant."""
    tasks, config = load_tasks(), load_classical()
    for task in tasks.tasks:
        for variant in _variants_for(task):
            data = load_variant(tasks, task, variant)
            record_run(
                system="majority",
                task=task,
                train_variant=variant,
                eval_variant=variant,
                split="test",
                seed=None,
                examples=data.test,
                predictions=classical.majority(data.train, data.test),
            )
            fitted = classical.fit_select(data.train, data.validation, config)
            run = record_run(
                system="tfidf_char",
                task=task,
                train_variant=variant,
                eval_variant=variant,
                split="test",
                seed=None,
                examples=data.test,
                predictions=classical.predict(fitted.pipeline, data.test),
                details={"best_c": fitted.best_c},
            )
            typer.echo(
                f"{task:12s} {variant:12s} tfidf macro-F1 {run.macro_f1:.3f} C={fitted.best_c}"
            )


def _cross_lingual_target(task: str, variant: Variant) -> Variant | None:
    config = load_encoders().cross_lingual
    if task == config.task and variant == config.train_variant:
        return config.eval_variant
    return None


@app.command("finetune")
def finetune(
    model: Annotated[str, typer.Option(help="Key in configs/encoders.yaml models.")],
    task: Annotated[str, typer.Option()],
    variant: Annotated[str, typer.Option()],
    seed: Annotated[int, typer.Option()],
) -> None:
    """Fine-tune one encoder on one task variant with one seed."""
    if variant not in get_args(Variant):
        raise typer.BadParameter(f"variant must be one of {get_args(Variant)}")
    train_variant = cast(Variant, variant)
    tasks, encoders = load_tasks(), load_encoders()
    data = load_variant(tasks, task, train_variant)
    target = _cross_lingual_target(task, train_variant)
    target_data: TaskData | None = load_variant(tasks, task, target) if target else None
    extra = {"cross_lingual": target_data.test} if target_data else None

    result = fine_tune(
        encoders.models[model],
        data,
        encoders.training,
        seed=seed,
        work_dir=encoders.output_dir / "work" / f"{model}_{task}_{variant}_{seed}",
        extra_evaluations=extra,
    )
    details: dict[str, float | int | str] = {
        "best_validation_macro_f1": round(result.best_validation_macro_f1, 4),
        "epochs": result.epochs_trained,
        "seconds": round(result.seconds, 1),
    }
    run = record_run(
        system=model,
        task=task,
        train_variant=variant,
        eval_variant=variant,
        split="test",
        seed=seed,
        examples=data.test,
        predictions=result.predictions["test"],
        details=details,
    )
    typer.echo(f"{model} {task} {variant} seed {seed}: macro-F1 {run.macro_f1:.3f}")
    if target_data is not None and target is not None:
        transfer = record_run(
            system=model,
            task=task,
            train_variant=variant,
            eval_variant=target,
            split="test",
            seed=seed,
            examples=target_data.test,
            predictions=result.predictions["cross_lingual"],
            details=details,
        )
        typer.echo(f"  zero-shot to {target}: macro-F1 {transfer.macro_f1:.3f}")


@app.command("finetune-grid")
def finetune_grid(
    models: Annotated[list[str] | None, typer.Option(help="Subset of model keys.")] = None,
) -> None:
    """Run every configured fine-tuning job that has not yet been recorded."""
    encoders = load_encoders()
    done = {(r.system, r.task, r.train_variant, r.seed) for r in load_runs()}
    for name in models or list(encoders.models):
        for group in encoders.runs:
            for variant in group.variants:
                for seed in encoders.seeds:
                    if (name, group.task, variant, seed) in done:
                        continue
                    finetune(model=name, task=group.task, variant=variant, seed=seed)


@app.command("summary")
def summary() -> None:
    """Aggregate recorded runs into a Markdown table of test macro-F1."""
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for run in load_runs():
        groups[(run.task, run.system, run.train_variant, run.eval_variant)].append(run.macro_f1)

    lines = [
        "| Task | System | Train | Test | Runs | Macro-F1 mean | SD |",
        "|---|---|---|---|---|---|---|",
    ]
    for (task, system, train, test), values in sorted(groups.items()):
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"| {task} | {system} | {train} | {test} | {len(values)} "
            f"| {statistics.mean(values):.3f} | {sd:.3f} |"
        )
    table = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
