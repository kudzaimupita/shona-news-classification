# Data card

No data is redistributed in this repository. Datasets are downloaded from the Hugging Face Hub
at run time.

| Dataset | Hub id and configs | Splits used (train / val / test) | Labels | Licence |
|---|---|---|---|---|
| SIB-200 | `Davlan/sib200`: `sna_Latn`, `eng_Latn` | 701 / 99 / 204 per language | entertainment, geography, health, politics, science/technology, sports, travel | CC BY-SA 4.0 |
| MasakhaNEWS, Shona | `masakhane/masakhanews`: `sna` | 1,288 / 185 / 369 | business, health, politics, sports | CC BY-NC 4.0 per the dataset card (Hub tag: afl-3.0) |
| MasakhaNEWS, English | `masakhane/masakhanews`: `eng` | 3,309 / 472 / 948 before restriction | 6 labels; restricted to the 4 Shona labels | as above |

## Notes

- **SIB-200** is built on FLORES-200: the Shona and English sentences are professional
  translations of the same English source, so the controlled comparison holds content fixed.
  Shona here is translated text, which may differ from native Shona prose.
- **MasakhaNEWS Shona** comes from Kwayedza and VOA Shona. Its label set is smaller than the
  English one, and its training set is less than 40% of the English one.
- **Licence conflict.** The MasakhaNEWS Hub metadata says `afl-3.0` while the dataset card says
  the data is CC 4.0 Non-Commercial. This project treats it as non-commercial research data
  and does not redistribute it.
- **Input text.** For MasakhaNEWS the headline and body are joined and truncated by each
  model's tokenizer. For SIB-200 the single sentence is used.

## Citations

- Adelani et al. 2023. MasakhaNEWS: News Topic Classification for African languages.
- Adelani et al. 2024. SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic
  Classification in 200+ Languages and Dialects.
- Alabi et al. 2022. Adapting Pre-trained Language Models to African Languages via
  Multilingual Adaptive Fine-Tuning (AfroXLMR).
- Conneau et al. 2020. Unsupervised Cross-lingual Representation Learning at Scale
  (XLM-RoBERTa).
