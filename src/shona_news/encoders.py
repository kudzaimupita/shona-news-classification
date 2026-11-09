"""Fine-tune multilingual encoders for sequence classification.

Runs on Apple-silicon GPUs through PyTorch MPS in full precision. The best epoch is chosen by
validation macro-F1 with early stopping; checkpoints are deleted after prediction so that a
full grid fits on a laptop disk.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from shona_news.config import TrainingSettings
from shona_news.data import Example, TaskData
from shona_news.metrics import macro_f1


class FineTuneResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    best_validation_macro_f1: float
    epochs_trained: float
    seconds: float
    predictions: dict[str, list[str]]


def _dataset(examples: Sequence[Example], label2id: Mapping[str, int]) -> Dataset:
    return Dataset.from_dict(
        {"text": [e.text for e in examples], "label": [label2id[e.label] for e in examples]}
    )


def fine_tune(
    model_id: str,
    data: TaskData,
    settings: TrainingSettings,
    *,
    seed: int,
    work_dir: Path,
    extra_evaluations: Mapping[str, Sequence[Example]] | None = None,
) -> FineTuneResult:
    """Train on ``data.train``, select on ``data.validation`` and predict on ``data.test``.

    ``extra_evaluations`` are additional labelled sets to predict with the selected model,
    for example the Shona test set for a model trained on English.
    """
    set_seed(seed)
    label2id = {label: index for index, label in enumerate(data.labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    def tokenize(batch: dict[str, Any]) -> Any:
        return tokenizer(batch["text"], truncation=True, max_length=settings.max_length)

    def encode(examples: Sequence[Example]) -> Dataset:
        return _dataset(examples, label2id).map(tokenize, batched=True, remove_columns=["text"])

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        logits = prediction.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        gold = [id2label[int(i)] for i in np.asarray(prediction.label_ids)]
        predicted = [id2label[int(i)] for i in np.argmax(logits, axis=-1)]
        return {"macro_f1": macro_f1(gold, predicted)}

    arguments = TrainingArguments(
        output_dir=str(work_dir),
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=settings.epochs,
        warmup_steps=settings.warmup_fraction,
        weight_decay=settings.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=encode(data.train),
        eval_dataset=encode(data.validation),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings.early_stopping_patience)],
    )

    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started

    def predict(examples: Sequence[Example]) -> list[str]:
        output = trainer.predict(encode(examples))
        logits = (
            output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        )
        return [id2label[int(i)] for i in np.argmax(logits, axis=-1)]

    predictions = {"test": predict(data.test)}
    for name, examples in (extra_evaluations or {}).items():
        predictions[name] = predict(examples)

    best = trainer.state.best_metric
    epochs = trainer.state.epoch or 0.0
    shutil.rmtree(work_dir, ignore_errors=True)
    return FineTuneResult(
        best_validation_macro_f1=float(best) if best is not None else float("nan"),
        epochs_trained=float(epochs),
        seconds=seconds,
        predictions=predictions,
    )
