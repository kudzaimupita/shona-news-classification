# Error analysis protocol

Fixed before any predictions were inspected.

## Sample

- Every Shona test error of the best seed of the best encoder on each task.
- Every Shona test error of the best large language model on each task.
- A random sample of English errors of the same systems, equal in number to the Shona errors
  (capped at 60), for comparison.

## Taxonomy

Each error receives one primary category.

| Code | Category | Description |
|---|---|---|
| A | Label ambiguity | The text fits two topics, for example a health budget story (health or politics). |
| B | Gold-label error | The reference label is wrong. Gold labels are never changed; the count is reported. |
| C | Code-switching or loanword | The decisive words are English or another language inside Shona text. |
| D | Named-entity driven | The topic depends on recognising a person, team, party or organisation. |
| E | Headline-only cue | (MasakhaNEWS) The headline is misleading or the topic only appears late in the body. |
| F | Tokenizer fragmentation | Key words split into three or more subword pieces by the model's tokenizer. |
| G | Translation artefact | (SIB-200) The Shona translation is unnatural, ambiguous or loses a cue present in English. |
| H | Other | Anything else, described in a note. |

Category F is checked mechanically from the tokenizer and then confirmed by hand.

## Annotators

- Annotator A: the author, a native Shona speaker.
- Annotator B: a second native Shona speaker who has not seen the predictions or the author's
  categories.
- Both see the text, gold label and predicted label, and categorise independently.
- Agreement is reported as Cohen's kappa with a bootstrap interval; disagreements are
  adjudicated and the adjudicated category is used in tables.

## Label-noise check

Both annotators also label 100 random Shona test items blind to the gold label. Agreement with
gold estimates how much of the remaining error is label noise. MasakhaNEWS reports a Fleiss
kappa of 0.63 among its original annotators.
