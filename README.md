# Tri-Level Financial News Classifier

UTS Analytics Capstone (41004). Multi-task transformer that tags financial
news with macro indicators (FRED-MD 8), industry sectors (GICS 11), and
named entities (5 types, BIO).

## Headline result

RoBERTa-base + LoRA, post-AL, calibrated thresholds, **test set (n=922)**:

| Head | Micro F1 |
|---|---|
| Macro | 0.6945 |
| Industry | 0.6277 |
| Entity (token) | 0.6085 |
| Entity (span, IOB2 strict) | 0.2569 |

Baseline (TF-IDF + LogReg) on the same split: macro 0.7962, industry
0.7051, no entity output. The transformer's distinct value is the entity
head; classification heads hit a silver-label noise ceiling.

## Quick start

```
git clone <repo>

# 1. Re-label corpus (optional, only if Ollama + Gemma is reachable)
python data/llm_auto_labeller.py --relabel-master --max-rows 20000

# 2. Compute class weights
python -m src.data_prep.class_weights

# 3. Train
python -m src.training.train --backbone roberta-base --epochs 8 --batch 16

# 4. Active learning (optional)
python -m src.active_learning.loop --backbone roberta-base --max-rounds 2

# 5. Evaluate on test
python scripts/eval_all_test.py
```

## Repo layout

```
data/llm_auto_labeller.py     LLM labelling pipeline (Gemma via Ollama)
src/data_prep/                Filters, dataset, class weights
src/model/                    Architecture + LoRA wrap + losses
src/training/                 Train loop, baseline, splits, metrics
src/active_learning/          Score / mine / relabel / loop
src/eval/                     Span F1, per-class report
src/serve/                    Inference wrapper
scripts/                      Visualisation + eval scripts
reports/                      Final report doc + figures
```

## Data files (not in repo)

`silver_dataset_master_with_text.csv` and `data/active_learning/`
snapshots are too large to track. Get them from the team Drive share.

## Report

`reports/report_sections_4_5_6_8.docx` covers Modelling, Findings,
Recommendations, and Difficulties. Figures live in `reports/figures/`.
