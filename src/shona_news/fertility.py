"""Tokenizer fertility: subword tokens per whitespace word.

Higher fertility means a language is split into more, smaller pieces, which shortens the
effective context and usually signals that the language was rare in pre-training.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


def fertility(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    words = pieces = 0
    for text in texts:
        for word in text.split():
            words += 1
            pieces += len(tokenizer.tokenize(word))
    return pieces / words if words else 0.0


def fragmented_words(tokenizer: Tokenizer, text: str, *, min_pieces: int = 3) -> list[str]:
    """Words split into at least ``min_pieces`` subwords; used in error analysis."""
    return [word for word in text.split() if len(tokenizer.tokenize(word)) >= min_pieces]
