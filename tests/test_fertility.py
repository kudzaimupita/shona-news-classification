from shona_news.fertility import fertility, fragmented_words


class SplitEveryTwo:
    def tokenize(self, text: str) -> list[str]:
        return [text[i : i + 2] for i in range(0, len(text), 2)]


def test_fertility_is_pieces_per_word() -> None:
    # "ab" -> 1 piece, "abcd" -> 2, "abcdef" -> 3
    assert fertility(SplitEveryTwo(), ["ab abcd", "abcdef"]) == 2.0


def test_fragmented_words_uses_threshold() -> None:
    assert fragmented_words(SplitEveryTwo(), "ab abcdef abcdefgh", min_pieces=3) == [
        "abcdef",
        "abcdefgh",
    ]
