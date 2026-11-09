# Shona and English news topic classification

How much does topic classification degrade from English to Shona, and how much of the gap do
an Africa-adapted encoder, more Shona training data and large language models close?

Shona (chiShona) has about 12 million speakers and very little public text; it is absent from
the CC-100 corpus used to pre-train XLM-RoBERTa. This project compares the same systems on
Shona and English under two designs, fixed in advance in
[`docs/preregistration.md`](docs/preregistration.md):

- **Controlled:** SIB-200, where the Shona and English sentences are translations of each other,
  so content and labels are identical across languages.
- **Natural:** MasakhaNEWS, native news in each language, restricted to the four topics present
  in the Shona data. English is also run with its training set cut to Shona's size, so that
  language is not confounded with data quantity.

## Systems

| System | Notes |
|---|---|
| Majority class | Floor |
| TF-IDF over character 2–5-grams + logistic regression | Suits Shona's agglutinative morphology |
| XLM-RoBERTa-base | No Shona in pre-training |
| AfroXLMR-base | XLM-RoBERTa-base with continued pre-training on 17 African languages including Shona |
| Zero-shot cross-lingual transfer | Encoders trained on English, tested on Shona (SIB-200) |
| Claude Opus 5.5, Sonnet 5.5, Haiku 5.5 | Zero-shot and one example per label, answers constrained to the label set |

## Results so far

Tokenizer fertility, in subword tokens per whitespace word on the training splits:

| Tokenizer | SIB-200 Shona | SIB-200 English | MasakhaNEWS Shona | MasakhaNEWS English |
|---|---|---|---|---|
| XLM-RoBERTa-base | 2.85 | 1.39 | 2.68 | 1.39 |
| AfroXLMR-base | 2.85 | 1.39 | 2.68 | 1.39 |

Shona words are split into about twice as many pieces as English words. AfroXLMR keeps the
XLM-RoBERTa vocabulary, so any advantage it shows cannot come from tokenization.

Test macro-F1 of the classical baseline:

| Task | Shona | English | English, size-matched |
|---|---|---|---|
| SIB-200 (7 labels, 204 test sentences) | 0.711 | 0.705 | n/a |
| MasakhaNEWS (4 labels) | 0.933 | 0.942 | 0.937 |

The character n-gram model shows no meaningful English–Shona gap on either task. Fine-tuned
encoder and language-model results, with bootstrap intervals and significance tests, follow.

## Repository layout

```
configs/         tasks, baseline, encoder grid and language-model settings
docs/            pre-registration, data card, error-analysis protocol
src/shona_news/
  data.py        task variants, including size-matched English
  classical.py   majority and character n-gram baselines
  encoders.py    fine-tuning on Apple-silicon GPUs (PyTorch MPS)
  llm.py         language-model baselines through the Message Batches API
  fertility.py   tokenizer fertility
  metrics.py     bootstrap intervals, McNemar's exact test, Holm correction
  runs.py        one record and one prediction file per evaluation
  cli.py         command-line entry point
tests/           unit tests
```

## Reproducing

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Datasets and models download from
the Hugging Face Hub on first use.

```bash
make install
make check
make baselines   # data summary, fertility, majority and TF-IDF
make encoders    # 50 fine-tuning runs; several hours on an M-series Mac
make summary     # aggregate results/runs.jsonl into results/summary.md
```

The language-model baselines need `ANTHROPIC_API_KEY` in `.env` (see `.env.example`):

```bash
make llm-submit
make llm-collect
```

## Data and licensing

Code is released under the Apache License 2.0. No data is redistributed: SIB-200 is CC BY-SA
4.0 and MasakhaNEWS is non-commercial research data. See [`docs/data-card.md`](docs/data-card.md).
