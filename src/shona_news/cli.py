"""Command-line entry point: ``shona-news <command>``."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, cast, get_args

import anthropic
import typer
from dotenv import load_dotenv
from transformers import AutoTokenizer

from shona_news import classical, llm
from shona_news.config import Variant, load_classical, load_encoders, load_llm, load_tasks
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


@app.command("llm-submit")
def llm_submit(
    yes: Annotated[bool, typer.Option("--yes", help="Submit without confirmation.")] = False,
) -> None:
    """Submit one batch covering every configured model, task, variant and shot count."""
    tasks, config = load_tasks(), load_llm()
    if config.batch_record.exists():
        raise typer.BadParameter(f"{config.batch_record} exists; a batch was already submitted")

    keyed = []
    for group in config.evaluations:
        for variant in group.variants:
            data = load_variant(tasks, group.task, variant)
            keyed += [
                (key, request, data.labels) for key, request in llm.build_requests(data, config)
            ]
    if not yes:
        typer.confirm(f"Submit {len(keyed)} requests to {', '.join(config.models)}?", abort=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for key, _, labels in keyed:
            handle.write(json.dumps({**key.model_dump(), "labels": list(labels)}) + "\n")

    load_dotenv()
    batch = anthropic.Anthropic().messages.batches.create(requests=[r for _, r, _ in keyed])
    config.batch_record.write_text(
        json.dumps({"batch_id": batch.id, "n_requests": len(keyed)}, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Submitted {batch.id} ({len(keyed)} requests)")


@app.command("llm-collect")
def llm_collect(poll_seconds: Annotated[int, typer.Option(min=10)] = 60) -> None:
    """Wait for the batch, then record one run per model, task, variant and shot count."""
    tasks, config = load_tasks(), load_llm()
    batch_id = json.loads(config.batch_record.read_text(encoding="utf-8"))["batch_id"]
    load_dotenv()
    client = anthropic.Anthropic()
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll_seconds)

    with (config.output_dir / "requests.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    keys = {
        llm.RequestKey(**{k: row[k] for k in llm.RequestKey.model_fields}): row["labels"]
        for row in rows
    }
    labels_of = {key.custom_id(): labels for key, labels in keys.items()}
    outcome = llm.collect(client.messages.batches.results(batch_id), labels_of)

    runs: dict[tuple[str, str, str, int], list[llm.RequestKey]] = defaultdict(list)
    for key in keys:
        runs[(key.model, key.task, key.variant, key.shots)].append(key)
    for (model, task, variant, shots), run_keys in sorted(runs.items()):
        data = load_variant(tasks, task, cast(Variant, variant))
        ordered = sorted(run_keys, key=lambda k: k.index)
        predictions = [outcome.predictions.get(k.custom_id(), llm.INVALID) for k in ordered]
        record = record_run(
            system=f"{model}_k{shots}",
            task=task,
            train_variant=variant,
            eval_variant=variant,
            split="test",
            seed=None,
            examples=[data.test[k.index] for k in ordered],
            predictions=predictions,
            details={"invalid": predictions.count(llm.INVALID)},
        )
        typer.echo(f"{model} k={shots} {task} {variant}: macro-F1 {record.macro_f1:.3f}")
    usage = {"input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens}
    (config.output_dir / "usage.json").write_text(
        json.dumps(usage, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"usage {usage}")
