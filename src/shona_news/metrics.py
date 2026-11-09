"""Metrics, bootstrap intervals and paired significance tests."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, f1_score

Metric = Callable[[Sequence[str], Sequence[str]], float]


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    low: float
    high: float


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(accuracy_score(y_true, y_pred))


def _percentiles(samples: Sequence[float], confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric = macro_f1,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    samples = [
        metric(truth[idx].tolist(), pred[idx].tolist())
        for idx in rng.integers(0, len(truth), size=(n_resamples, len(truth)))
    ]
    low, high = _percentiles(samples, confidence)
    return Interval(estimate=metric(y_true, y_pred), low=low, high=high)


def paired_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Metric = macro_f1,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval for ``metric(a) - metric(b)`` with shared resample indices."""
    truth, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    samples = []
    for idx in rng.integers(0, len(truth), size=(n_resamples, len(truth))):
        t = truth[idx].tolist()
        samples.append(metric(t, a[idx].tolist()) - metric(t, b[idx].tolist()))
    low, high = _percentiles(samples, confidence)
    return Interval(estimate=metric(y_true, pred_a) - metric(y_true, pred_b), low=low, high=high)


def mcnemar_exact(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs."""
    only_a = sum(a == t != b for t, a, b in zip(y_true, pred_a, pred_b, strict=True))
    only_b = sum(b == t != a for t, a, b in zip(y_true, pred_a, pred_b, strict=True))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
