"""Experiment settings loaded from ``configs/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

Variant = Literal["sna", "eng", "eng_matched"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TaskSpec(_Frozen):
    dataset: str
    label_field: str
    text_fields: tuple[str, ...]
    id_field: str
    labels: tuple[str, ...]
    languages: dict[str, str]


class TasksConfig(_Frozen):
    tasks: dict[str, TaskSpec]
    size_matched_seed: int


class ClassicalConfig(_Frozen):
    c_grid: tuple[float, ...]
    ngram_range: tuple[int, int]
    min_df: PositiveInt
    seed: int


class TrainingSettings(_Frozen):
    learning_rate: float = Field(gt=0)
    batch_size: PositiveInt
    epochs: PositiveInt
    max_length: PositiveInt
    warmup_fraction: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    early_stopping_patience: PositiveInt


class RunGroup(_Frozen):
    task: str
    variants: tuple[Variant, ...]


class CrossLingual(_Frozen):
    task: str
    train_variant: Variant
    eval_variant: Variant


class EncodersConfig(_Frozen):
    models: dict[str, str]
    runs: tuple[RunGroup, ...]
    seeds: tuple[int, ...]
    training: TrainingSettings
    cross_lingual: CrossLingual
    output_dir: Path


class LlmConfig(_Frozen):
    models: tuple[str, ...]
    effort: Effort
    max_tokens: PositiveInt
    shots: tuple[int, ...]
    shot_seed: int
    evaluations: tuple[RunGroup, ...]
    output_dir: Path
    batch_record: Path


def _load[T: BaseModel](path: Path, model: type[T]) -> T:
    with path.open(encoding="utf-8") as handle:
        return model.model_validate(yaml.safe_load(handle))


def load_tasks(path: Path = Path("configs/tasks.yaml")) -> TasksConfig:
    return _load(path, TasksConfig)


def load_classical(path: Path = Path("configs/classical.yaml")) -> ClassicalConfig:
    return _load(path, ClassicalConfig)


def load_encoders(path: Path = Path("configs/encoders.yaml")) -> EncodersConfig:
    return _load(path, EncodersConfig)


def load_llm(path: Path = Path("configs/llm.yaml")) -> LlmConfig:
    return _load(path, LlmConfig)
