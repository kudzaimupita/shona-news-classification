"""Load task data for each language variant.

Variants:
- ``sna`` and ``eng``: the official splits for that language, restricted to the task labels.
- ``eng_matched``: English with the training split subsampled per label to the Shona training
  counts of the same task; validation and test are the full English splits.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict

from shona_news.config import TasksConfig, TaskSpec, Variant

SPLITS = ("train", "validation", "test")


class Example(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    label: str


class TaskData(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    variant: Variant
    labels: tuple[str, ...]
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]

    def split(self, name: str) -> tuple[Example, ...]:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}")
        examples: tuple[Example, ...] = getattr(self, name)
        return examples


def to_examples(rows: Iterable[Mapping[str, Any]], spec: TaskSpec) -> tuple[Example, ...]:
    """Convert raw rows to examples, keeping only the task labels and non-empty text."""
    examples = []
    for row in rows:
        label = row[spec.label_field]
        text = "\n".join(str(row[field]).strip() for field in spec.text_fields if row[field])
        if label in spec.labels and text:
            examples.append(Example(id=str(row[spec.id_field]), text=text, label=label))
    return tuple(examples)


def load_language(task: str, spec: TaskSpec, language: str) -> TaskData:
    dataset = load_dataset(spec.dataset, spec.languages[language])
    splits = {name: to_examples(dataset[name], spec) for name in SPLITS}
    variant: Variant = "sna" if language == "sna" else "eng"
    return TaskData(task=task, variant=variant, labels=spec.labels, **splits)


def size_matched(source: TaskData, reference: TaskData, *, seed: int) -> TaskData:
    """Subsample ``source.train`` per label to the label counts of ``reference.train``."""
    target = Counter(example.label for example in reference.train)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in sorted(source.train, key=lambda e: e.id):
        by_label[example.label].append(example)

    rng = random.Random(seed)
    sampled: list[Example] = []
    for label in source.labels:
        pool = by_label[label]
        if len(pool) < target[label]:
            raise ValueError(f"{label}: need {target[label]} examples, source has {len(pool)}")
        sampled += rng.sample(pool, target[label])
    return source.model_copy(
        update={"variant": "eng_matched", "train": tuple(sorted(sampled, key=lambda e: e.id))}
    )


def load_variant(config: TasksConfig, task: str, variant: Variant) -> TaskData:
    spec = config.tasks[task]
    if variant == "eng_matched":
        return size_matched(
            load_language(task, spec, "eng"),
            load_language(task, spec, "sna"),
            seed=config.size_matched_seed,
        )
    return load_language(task, spec, variant)
