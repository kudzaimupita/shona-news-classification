from collections import Counter

import pytest

from shona_news.config import TaskSpec
from shona_news.data import size_matched, to_examples
from tests.conftest import make_data


def test_to_examples_joins_fields_and_drops_other_labels(spec: TaskSpec) -> None:
    rows = [
        {"category": "sports", "headline": " Bhora ", "text": "Mutambo", "url": "u1"},
        {"category": "technology", "headline": "x", "text": "y", "url": "u2"},
        {"category": "health", "headline": "", "text": "", "url": "u3"},
    ]
    examples = to_examples(rows, spec)
    assert [(e.id, e.text, e.label) for e in examples] == [("u1", "Bhora\nMutambo", "sports")]


def test_size_matched_copies_reference_label_counts() -> None:
    reference = make_data("sna", {"business": 3, "health": 2, "politics": 4, "sports": 1}, "s")
    source = make_data("eng", {"business": 9, "health": 9, "politics": 9, "sports": 9}, "e")

    matched = size_matched(source, reference, seed=1)

    assert matched.variant == "eng_matched"
    assert Counter(e.label for e in matched.train) == Counter(e.label for e in reference.train)
    assert matched.test == source.test
    assert size_matched(source, reference, seed=1) == matched


def test_size_matched_fails_when_source_is_too_small() -> None:
    reference = make_data("sna", {"business": 5, "health": 1, "politics": 1, "sports": 1}, "s")
    source = make_data("eng", {"business": 2, "health": 2, "politics": 2, "sports": 2}, "e")
    with pytest.raises(ValueError, match="business"):
        size_matched(source, reference, seed=0)
