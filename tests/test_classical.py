from shona_news import classical
from shona_news.config import ClassicalConfig
from shona_news.data import Example

WORDS = {
    "sports": "bhora mutambo chikwata",
    "politics": "hurumende sarudzo bato",
    "health": "chirwere chipatara mushonga",
    "business": "mari bhizimisi musika",
}


def test_tfidf_selects_c_on_validation_and_predicts() -> None:
    train = [
        Example(id=f"{label}{i}", text=f"{words} {i}", label=label)
        for label, words in WORDS.items()
        for i in range(8)
    ]
    validation = [
        Example(id=f"v{label}", text=words, label=label) for label, words in WORDS.items()
    ]
    config = ClassicalConfig(c_grid=(1.0, 10.0), ngram_range=(2, 5), min_df=1, seed=0)

    fitted = classical.fit_select(train, validation, config)

    assert set(fitted.validation_macro_f1) == {1.0, 10.0}
    assert classical.predict(fitted.pipeline, validation) == list(WORDS)
    assert classical.majority(train, validation) == ["sports"] * 4
