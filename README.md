# NLU Component: Intent Classification & Slot Filling

Intent classification and slot filling on the ATIS (Airline Travel Information Systems) dataset using BERT.
- **Intent Classification** — predicts the goal of a user utterance (e.g. `atis_flight`, `atis_airfare`)
- **Slot Filling** — extracts structured information from the utterance using BIO tagging (e.g. `B-fromloc.city_name`, `B-depart_time.time`)

## Dataset

[ATIS (Airline Travel Information Systems)](https://www.aclweb.org/anthology/H90-1021.pdf). It is a benchmark dataset for NLU in the flight booking domain.

| Split | Examples |
|-------|----------|
| Train | 4,978 |
| Dev   | 500 |
| Test  | 893 |

- 26 intent classes (heavily imbalanced — `atis_flight` accounts for ~74%)
- 128 slot tags (81 unique slot types in BIO format)

## Approach

Two separate BERT-based models fine-tuned on the ATIS dataset:

- `text_classification_model`: fine-tunes `bert-base-uncased` for intent classification using the `[CLS]` token representation
- `sequence_labeling_model`: fine-tunes `bert-base-uncased` for slot filling using per-token representations

## Results

| Task | Metric | Dev | Test |
|------|--------|-----|------|
| Intent Classification | Accuracy | 99.6% | 97.8% |
| Slot Filling | F1 (seqeval) | 99.4% | 95.0% |

## Requirements

```
torch
transformers
seqeval
scikit-learn
pandas
matplotlib
seaborn
spacy
```

Install with:
```bash
pip install torch transformers seqeval scikit-learn pandas matplotlib seaborn spacy
```

## Key Findings

- BERT achieves near state-of-the-art performance on both tasks with minimal feature engineering
- Main weakness is rare and multi-intent classes (e.g. `atis_flight#atis_airfare`) due to data imbalance
- Slot errors are concentrated in semantically similar tags (`depart_*` vs `arrive_*`)
