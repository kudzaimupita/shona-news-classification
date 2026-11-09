import json
import re

from anthropic.types import Message

from shona_news import llm
from shona_news.config import LlmConfig
from tests.conftest import LABELS, make_data

CONFIG = LlmConfig(
    models=("claude-opus-5-5", "claude-haiku-5-5"),
    effort="low",
    max_tokens=2048,
    shots=(0, 1),
    shot_seed=3,
    evaluations=(),
    output_dir="results/llm",  # type: ignore[arg-type]
    batch_record="results/llm/batch.json",  # type: ignore[arg-type]
)


def _message(text: str, stop_reason: str = "end_turn") -> Message:
    return Message.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def test_custom_ids_are_valid_and_unique() -> None:
    data = make_data("sna", dict.fromkeys(LABELS, 3), "s")
    requests = llm.build_requests(data, CONFIG)
    ids = [request["custom_id"] for _, request in requests]
    assert len(requests) == len(CONFIG.models) * len(CONFIG.shots) * len(data.test)
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", i) for i in ids)


def test_shots_take_k_per_label_from_train() -> None:
    data = make_data("sna", dict.fromkeys(LABELS, 5), "s")
    shots = llm.sample_shots(data.train, LABELS, 1, seed=0)
    assert sorted(shot.label for shot in shots) == sorted(LABELS)
    assert {shot.id for shot in shots} <= {e.id for e in data.train}


def test_examples_live_in_the_cached_system_prompt() -> None:
    data = make_data("sna", dict.fromkeys(LABELS, 2), "s")
    shots = llm.sample_shots(data.train, LABELS, 1, seed=0)
    assert "Examples" not in llm.system_prompt(LABELS, "Shona", [])
    assert all(shot.text in llm.system_prompt(LABELS, "Shona", shots) for shot in shots)
    assert "Examples" not in llm.user_prompt("Mutambo webhora")


def test_truncate_keeps_first_words() -> None:
    assert llm.truncate("a b c d", max_words=2) == "a b ..."
    assert llm.truncate("a b", max_words=2) == "a b"


def test_parse_label_accepts_only_known_labels() -> None:
    assert llm.parse_label(_message(json.dumps({"label": "sports"})), LABELS) == "sports"
    assert llm.parse_label(_message(json.dumps({"label": "weather"})), LABELS) == llm.INVALID
    assert llm.parse_label(_message("", stop_reason="refusal"), LABELS) == llm.INVALID
    assert llm.parse_label(_message("not json"), LABELS) == llm.INVALID
