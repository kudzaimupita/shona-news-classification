# Pre-registration

Written 9 November 2026, after inspecting dataset sizes and label sets and before training or
scoring any model. Changes after this date are listed at the end with their reason.

## Research question

How much does topic classification degrade from English to Shona, a Bantu language with about
12 million speakers and little public text, and how much of that gap is closed by an encoder
adapted to African languages, by more Shona training data, and by large language models?

## Tasks

| Task | Data | Labels | Why |
|---|---|---|---|
| Controlled | SIB-200 `sna_Latn` and `eng_Latn` | 7 | The two languages are translations of the same sentences, so content and label distribution are identical. |
| Natural | MasakhaNEWS `sna` and `eng` | 4 shared: business, health, politics, sports | Native news text in each language. Shona has 4 of the 6 English labels; English is restricted to those 4. |

Official train, validation and test splits are used unchanged. For the natural task English is
run twice: full training set, and subsampled per label to Shona's training counts so that
language is not confounded with data size.

## Systems

- Majority class.
- TF-IDF over character n-grams (2 to 5, within word boundaries) with logistic regression,
  regularisation selected on validation.
- XLM-RoBERTa-base, whose pre-training data contains no Shona.
- AfroXLMR-base: the same architecture after continued pre-training on 17 African languages
  including Shona.
- Zero-shot cross-lingual transfer: encoders trained on English, evaluated on Shona (controlled
  task only, where test sentences are parallel).
- Large language models, zero-shot and with one example per label, instructions in English.

## Hypotheses

- **H1.** Every system scores lower on Shona than on English in the controlled task.
- **H2.** AfroXLMR-base narrows the Shona gap relative to XLM-RoBERTa-base by at least 5
  macro-F1 points on both tasks.
- **H3.** Zero-shot transfer from English loses more on Shona with XLM-RoBERTa-base than with
  AfroXLMR-base.
- **H4.** Large language models show a larger English–Shona gap than the fine-tuned
  AfroXLMR-base.

## Metrics and analysis

- Primary: macro-F1 on the official test split. Secondary: accuracy, weighted F1 (for
  comparison with published MasakhaNEWS results), per-class F1.
- Fine-tuned encoders: 5 seeds; mean and standard deviation across seeds reported separately
  from test-set uncertainty.
- Test-set uncertainty: 1,000-resample bootstrap intervals; paired bootstrap intervals for
  differences between systems on the same items; McNemar's exact test for paired accuracy;
  Holm correction within each family of comparisons.
- Diagnostics: tokenizer fertility (subword tokens per word) for each model and language.
- Model selection uses the validation split only. Test scores are computed once per final
  configuration.

## Power

SIB-200 has 204 test sentences per language, giving 95% intervals of roughly ±5 to 7 macro-F1
points. Differences inside that band are reported as inconclusive.

## Error analysis

Pre-registered taxonomy and a two-annotator protocol with native Shona speakers are in
[`error-analysis-protocol.md`](error-analysis-protocol.md).

## Changes after registration

None yet.
