"""Large language model baselines through the Message Batches API.

Each request classifies one text into the task's label set with a closed JSON schema, so the
answer is always one of the labels. Long news articles are cut to their first words to keep
the comparison with the 256-token encoders fair and the cost low. With ``k`` shots the prompt
contains ``k`` training examples per label, drawn once with a fixed seed and shared by every
request.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

from anthropic.types import Message as ApiMessage
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ConfigDict

from shona_news.config import LlmConfig
from shona_news.data import Example, TaskData

MAX_WORDS = 150
INVALID = "<invalid>"
_LANGUAGE_NAMES = {"sna": "Shona", "eng": "English", "eng_matched": "English"}


class RequestKey(BaseModel):
    """Everything needed to route a batch result back to its run and item."""

    model_config = ConfigDict(frozen=True)

    model: str
    task: str
    variant: str
    shots: int
    index: int

    def custom_id(self) -> str:
        model = re.sub(r"[^a-z0-9]", "", self.model.removeprefix("claude-"))
        return f"{model}-{self.task}-{self.variant}-k{self.shots}-{self.index:04d}"


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def output_schema(labels: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": list(labels)}},
        "required": ["label"],
        "additionalProperties": False,
    }


def sample_shots(
    train: Sequence[Example], labels: Sequence[str], k: int, seed: int
) -> list[Example]:
    rng = random.Random(seed)
    shots: list[Example] = []
    for label in labels:
        pool = sorted((e for e in train if e.label == label), key=lambda e: e.id)
        shots += rng.sample(pool, k)
    rng.shuffle(shots)
    return shots


def system_prompt(labels: Sequence[str], language: str, shots: Sequence[Example]) -> str:
    """Instructions plus any examples. Shared by every request of a run, so it is cached."""
    instructions = (
        f"You classify {language} news text by topic. Choose exactly one label from: "
        f"{', '.join(labels)}. Judge by the main subject of the text."
    )
    if not shots:
        return instructions
    parts = [f"Text:\n{truncate(shot.text)}\nLabel: {shot.label}" for shot in shots]
    return instructions + "\n\nExamples:\n\n" + "\n\n".join(parts)


def user_prompt(text: str) -> str:
    return f"Classify this text.\n\nText:\n{truncate(text)}"


def build_requests(data: TaskData, config: LlmConfig) -> list[tuple[RequestKey, Request]]:
    requests = []
    language = _LANGUAGE_NAMES[data.variant]
    for shots_per_label in config.shots:
        shots = sample_shots(data.train, data.labels, shots_per_label, config.shot_seed)
        for model in config.models:
            for index, example in enumerate(data.test):
                key = RequestKey(
                    model=model,
                    task=data.task,
                    variant=data.variant,
                    shots=shots_per_label,
                    index=index,
                )
                params = MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=config.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt(data.labels, language, shots),
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt(example.text)}],
                    output_config={
                        "effort": config.effort,
                        "format": {"type": "json_schema", "schema": output_schema(data.labels)},
                    },
                )
                requests.append((key, Request(custom_id=key.custom_id(), params=params)))
    return requests


def parse_label(message: ApiMessage, labels: Sequence[str]) -> str:
    """Return the predicted label, or ``INVALID`` for refusals, truncation or bad output."""
    if message.stop_reason in {"refusal", "max_tokens"}:
        return INVALID
    text = next((block.text for block in message.content if block.type == "text"), "")
    try:
        label = json.loads(text).get("label")
    except (json.JSONDecodeError, AttributeError):
        return INVALID
    return label if label in labels else INVALID


class LlmPredictions(BaseModel):
    predictions: dict[str, str]
    input_tokens: int
    output_tokens: int


def collect(
    results: Iterable[MessageBatchIndividualResponse], labels_of: dict[str, Sequence[str]]
) -> LlmPredictions:
    """Map each ``custom_id`` to a label. ``labels_of`` gives the label set per ``custom_id``."""
    predictions: dict[str, str] = {}
    input_tokens = output_tokens = 0
    for item in results:
        if item.result.type != "succeeded":
            predictions[item.custom_id] = INVALID
            continue
        message = item.result.message
        input_tokens += message.usage.input_tokens
        output_tokens += message.usage.output_tokens
        predictions[item.custom_id] = parse_label(message, labels_of[item.custom_id])
    return LlmPredictions(
        predictions=predictions, input_tokens=input_tokens, output_tokens=output_tokens
    )
