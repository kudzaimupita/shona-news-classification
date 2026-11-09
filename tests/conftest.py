import pytest

from shona_news.config import TaskSpec
from shona_news.data import Example, TaskData

LABELS = ("business", "health", "politics", "sports")


@pytest.fixture
def spec() -> TaskSpec:
    return TaskSpec(
        dataset="local/test",
        label_field="category",
        text_fields=("headline", "text"),
        id_field="url",
        labels=LABELS,
        languages={"sna": "sna", "eng": "eng"},
    )


def make_data(variant: str, per_label: dict[str, int], prefix: str) -> TaskData:
    train = tuple(
        Example(id=f"{prefix}-{label}-{i:03d}", text=f"{label} story {i}", label=label)
        for label, count in per_label.items()
        for i in range(count)
    )
    held_out = tuple(
        Example(id=f"{prefix}-v-{lab}", text=f"{lab} item", label=lab) for lab in LABELS
    )
    return TaskData(
        task="masakhanews",
        variant=variant,  # type: ignore[arg-type]
        labels=LABELS,
        train=train,
        validation=held_out,
        test=held_out,
    )
