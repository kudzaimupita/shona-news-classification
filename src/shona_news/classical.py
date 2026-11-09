"""Character n-gram TF-IDF with logistic regression, and a majority-class baseline.

Character n-grams within word boundaries suit an agglutinative language such as Shona, where
noun-class prefixes and verb affixes carry meaning that whole-word features would miss.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from shona_news.config import ClassicalConfig
from shona_news.data import Example
from shona_news.metrics import macro_f1


class ClassicalFit(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pipeline: Pipeline
    best_c: float
    validation_macro_f1: dict[float, float]


def build_pipeline(c: float, config: ClassicalConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c, class_weight="balanced", max_iter=3000, random_state=config.seed
                ),
            ),
        ]
    )


def fit_select(
    train: Sequence[Example], validation: Sequence[Example], config: ClassicalConfig
) -> ClassicalFit:
    """Fit one model per C and keep the one with the best validation macro-F1."""
    texts, labels = [e.text for e in train], [e.label for e in train]
    gold = [e.label for e in validation]
    scores: dict[float, float] = {}
    best: tuple[float, Pipeline] | None = None
    for c in config.c_grid:
        pipeline = build_pipeline(c, config).fit(texts, labels)
        scores[c] = macro_f1(gold, predict(pipeline, validation))
        if best is None or scores[c] > scores[best[0]]:
            best = (c, pipeline)
    assert best is not None
    return ClassicalFit(pipeline=best[1], best_c=best[0], validation_macro_f1=scores)


def predict(pipeline: Pipeline, examples: Sequence[Example]) -> list[str]:
    return [str(label) for label in pipeline.predict([e.text for e in examples])]


def majority(train: Sequence[Example], examples: Sequence[Example]) -> list[str]:
    label = Counter(e.label for e in train).most_common(1)[0][0]
    return [label] * len(examples)
