from pathlib import Path

from shona_news.data import Example
from shona_news.runs import load_runs, record_run


def test_record_run_writes_predictions_and_appends_record(tmp_path: Path) -> None:
    examples = [Example(id=str(i), text="t", label=label) for i, label in enumerate("aabb")]
    record = record_run(
        system="toy",
        task="sib200",
        train_variant="eng",
        eval_variant="sna",
        split="test",
        seed=7,
        examples=examples,
        predictions=["a", "b", "b", "b"],
        root=tmp_path,
    )

    assert record.accuracy == 0.75
    assert Path(record.predictions_file).read_text(encoding="utf-8").count("\n") == 4
    assert load_runs(tmp_path) == [record]
