# CODE_DETAILS.md — Veritas Tri-Level Pipeline

End-to-end walkthrough of the training infrastructure for the Macro / Industry / Entity multi-head model. Files are presented **in the order data flows through them**, so reading top-to-bottom mirrors what actually happens when you train.

---

## Pipeline overview

```
 raw sources                       labelling                    preparation
 ┌────────────┐                   ┌─────────────────┐           ┌──────────────────┐
 │ database 1 │                   │ llm_auto_       │           │ join_text.py     │
 │   .db      │──┐                │   labeller.py   │           │                  │
 │            │  │                │                 │           │                  │
 │ silver_    │  ├─►  Ollama  ──► │  gemma4:latest  │  ──────►  │ silver_dataset_  │
 │ dataset_   │  │    192.168..   │                 │           │   master_        │
 │ final(in). │  │                │  silver_dataset_│           │   with_text.csv  │
 │ csv        │──┤                │   master.csv    │           │                  │
 │            │  │                │   master.db     │           │                  │
 │ 77 Webhose │  │                └─────────────────┘           └────────┬─────────┘
 │ .zip files │──┘                                                       │
 └────────────┘                                                          ▼
                                                              ┌──────────────────┐
                                                              │ class_weights.py │──► data/class_weights/*.pt
                                                              └──────────────────┘
                                                                       │
                                                                       ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ filters.py   │────►│ dataset.py   │────►│ losses.py    │◄────│ class_weights│
 │              │     │              │     │              │     │  *.pt        │
 │ parse+filter │     │ TriLevel     │     │ TriLevelLoss │     └──────────────┘
 │ label order  │     │   Dataset    │     │ BCE+BCE+CE   │
 └──────────────┘     └──────┬───────┘     └──────▲───────┘
                             │                    │
                             ▼                    │
                      ┌──────────────┐             │
                      │ architecture │             │
                      │ TriLevel     │─── logits ──┘
                      │   Financial  │
                      │   Model      │
                      └──────▲───────┘
                             │
                     ┌───────┴────────┐
                     │ lora_wrap.py   │
                     │ wrap_with_lora │
                     └────────────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │ smoke tests    │
                     │ (Steps 4 + 5)  │
                     └────────────────┘
```

---

## 0. Source data + labelling (upstream, already done)

Three heterogeneous sources feed the labeller:

| Source | File | Rows ingested | Notes |
|---|---|---|---|
| SQLite DB | `data/database 1.db` | 87 | clean, small |
| Raw CSV | `data/silver_dataset_final(in).csv` | ~8K (capped from 61K) | contained unquoted newlines → 671 phantom `Unnamed:*` columns, cleaned with `^Unnamed` regex filter |
| Webhose zips | `data/Datasets/*.zip` | 1,798 | 77 zip files, English-only filter, ~50/zip sampled |

**`data/llm_auto_labeller.py`** calls `gemma4:latest` on a remote Ollama server (`http://192.168.0.19:11434`) with a single system prompt + article content. Per-row JSON is validated against the FRED-MD 8 macro taxonomy and GICS 11 industry taxonomy. Outputs are written row-by-row (resumable) and then merged into:

- `data/silver_dataset_master.csv` — 9,881 raw rows, columns: `article_id, source, title, url, published_at, source_site, is_financial, macro, industry, entity`
- `data/silver_dataset_master.db` — same data in SQLite for downstream query

**Entity column shape** (scanned across the full master): every entity is a dict `{"name": str, "type": str}` with `type ∈ {ORG, PER, TICKER, PRODUCT, LOC}`. 46,076 entity items across 9,193 rows (93% coverage). Distribution: ORG 62.8%, PER 12.9%, TICKER 9.6%, PRODUCT 7.8%, LOC 7.0%.

---

## 1. `src/data_prep/join_text.py` — rehydrate article bodies

### Why it exists
The labeller's master CSV stores labels + metadata but **not** the article body — bodies still live in the three source stores. Downstream training code would have to open DBs, CSVs, and 77 zips on every epoch. Instead we do the join **once** and write `silver_dataset_master_with_text.csv`.

### How it works
1. Load the master CSV, group `article_id`s by `source` column.
2. For `source=db`: open the SQLite DB read-only, `SELECT id, text FROM articles`, keep rows whose `db_<id>` key is in the needed set.
3. For `source=csv`: re-read the raw CSV, apply the same `^Unnamed` cleaning, filter by `article_id`.
4. For `source=zip`: walk the 77 zips, decode JSON, match by `uuid`. Uses a `remaining` set that shrinks as ids are found, so the scan terminates as soon as everything is matched rather than reading 4M JSON files.
5. `master["text"] = master["article_id"].map(text_map).fillna("")` and write to disk.

### Verified output
```
source=csv: 7996 ids   CSV matched: 7996
source=db:   87 ids    DB matched:  87
source=zip: 1798 ids   ZIPs matched: 1798
Total rows: 9881   rows with empty body: 0
```

100% join across all three sources — no body gaps.

---

## 2. `src/data_prep/filters.py` — single source of truth

### Why it exists
Both `class_weights.py` and `dataset.py` need to apply the exact same filter rule and use the exact same label order. Duplicating the logic invites drift (one file updates, the other doesn't, training silently breaks). This module is imported by both.

### What it exports

**Label orders** (all canonical):
- `MACRO_LABELS`: 8 FRED-MD macro categories, imported directly from `llm_auto_labeller.MACRO_RULES.keys()` so the label order is defined once in the whole repo.
- `INDUSTRY_LABELS`: 11 GICS sectors, same pattern.
- `ENTITY_TYPES`: 5 types locked from the silver-set scan (`ORG, PER, TICKER, PRODUCT, LOC`).
- `ENTITY_TAGS`: 11-tag BIO set `[O, B-ORG, I-ORG, B-PER, I-PER, ..., B-LOC, I-LOC]`.
- Index lookups: `MACRO_LABEL_TO_ID`, `INDUSTRY_LABEL_TO_ID`, `ENTITY_TAG_TO_ID`.

**Parsing helpers**:
- `parse_list_cell(cell)` — handles JSON, Python-literal strings, NaN, and already-parsed lists. Covers every way the list columns might come back from `pd.read_csv`.
- `is_financial_true(x)` — bool / str / NaN safe truthiness check for the `is_financial` column.

**Main function**:
- `load_and_filter_master(csv_path)` — loads the CSV, drops `is_financial != True`, parses the three list columns, then applies the **locked build-plan filter**: drop a row only if `macro_list == [] AND industry_list == [] AND entity_list == []`. Rows with empty macro but populated industry/entity stay in with all-zero macro targets — this is intentional and was confirmed during the build-plan review (it's not a bug, it's the point of multi-label BCE).

### Input / output
```
Input:   silver_dataset_master[_with_text].csv (9,881 rows)
Output:  DataFrame with 9,218 rows, three extra columns (macro_list, industry_list, entity_list)
```

The drop chain: `9,881 raw → 9,368 is_financial=True → 9,218 after dropping all-empty`. The 150 all-empty rows are Gemma correctly saying "this is financial noise with nothing to tag" — not worth training on.

---

## 3. `src/data_prep/class_weights.py` — BCE rebalancing tensors

### Why it exists
Our silver set is heavily imbalanced. Stock Market has 1,648 positives; Consumption has 4. An unweighted BCE loss collapses to predicting all-zeros because that already achieves >99% accuracy. `BCEWithLogitsLoss(pos_weight=w)` multiplies the positive term of each class by `w`, rebalancing gradient contributions.

### Formula
```
pos_weight_c = (N - n_pos_c) / max(n_pos_c, 1)
```
Textbook recipe. For a class with `n_pos=1648` out of `N=9218`: `w = 4.59` (mild boost). For `n_pos=4`: `w = 2303` (huge boost — too huge, see loss module for the cap).

### How it works
1. Call `load_and_filter_master` to get the 9,218-row filtered corpus.
2. Count positives per class across `macro_list` and `industry_list`. Labels outside the canonical sets are tracked separately in `unknown_macro_labels` / `unknown_industry_labels` — a hallucination alarm.
3. Compute `pos_weight` tensors in canonical label order.
4. Save to `data/class_weights/macro_pos_weight.pt` and `industry_pos_weight.pt`.
5. Also write `label_stats.json` with human-readable counts, weights, and any unknown labels found.

### Verified output (from actual run)
```
Macro pos_weight:
  Output and Income                         n_pos=  121  w=   75.18
  Labor Market                              n_pos=   95  w=   96.03
  Housing                                   n_pos=   22  w=  418.00
  Consumption, Orders, and Inventories      n_pos=    4  w= 2303.50
  Money and Credit                          n_pos=  183  w=   49.37
  Interest and Exchange Rates               n_pos=  253  w=   35.43
  Prices                                    n_pos=  145  w=   62.57
  Stock Market                              n_pos= 1648  w=    4.59

Industry pos_weight:
  Energy                                    n_pos=  931  w=    8.90
  Materials                                 n_pos=  742  w=   11.42
  Industrials                               n_pos=  646  w=   13.27
  Consumer Discretionary                    n_pos=  921  w=    9.01
  Consumer Staples                          n_pos=  340  w=   26.11
  Health Care                               n_pos=  734  w=   11.56
  Financials                                n_pos= 2525  w=    2.65
  Information Technology                    n_pos= 1273  w=    6.24
  Communication Services                    n_pos=  116  w=   78.47
  Utilities                                 n_pos=  216  w=   41.68
  Real Estate                               n_pos=  296  w=   30.14
```

No unknown labels → Gemma stayed strictly inside the controlled vocabulary, no prompt re-engineering needed.

**Interpretation**: industry head is in good shape (min n_pos=116). Macro head has two effectively-dead classes (Housing @ 22, Consumption @ 4); these are documented limitations and the active-learning loop is where they get recovered.

---

## 4. `src/data_prep/dataset.py` — `TriLevelDataset`

### Why it exists
Converts each filtered row into a PyTorch training example with **three aligned targets** (macro multi-hot, industry multi-hot, entity BIO sequence), tokenized consistently with the chosen backbone.

### Constructor
```python
TriLevelDataset(csv_path, tokenizer, max_length=512,
                text_column_title="title", text_column_body="text")
```
- Calls `load_and_filter_master` so filtering is shared with `class_weights.py`.
- Falls back to empty-string if `text` column is missing (so the title-only master CSV still works for a degenerate smoke test).

### `__getitem__` output
```python
{
  "input_ids":        LongTensor  [max_length]    # padded to max_length
  "attention_mask":   LongTensor  [max_length]
  "macro_targets":    FloatTensor [8]             # multi-hot
  "industry_targets": FloatTensor [11]            # multi-hot
  "entity_targets":   LongTensor  [max_length]    # BIO tag ids, -100 on ignored positions
}
```

### Macro / Industry targets
Straightforward multi-hot encoding: zero tensor of length 8/11, set `1.0` at each index whose label is in `row["macro_list"]` / `row["industry_list"]`. Empty lists → all-zero target (by design; the BCE loss with pos_weight handles this correctly).

### Entity alignment — the tricky part
Gemma only gave us entity surface strings, not character offsets. The dataset has to locate each entity inside the article text at load time. Algorithm:

1. **Find char spans**: for each `(name, type)` in `entity_list`, run a **word-boundary case-insensitive regex** `\bname\b` against `title + "\n\n" + body` lowercased. First match wins. Word boundaries are critical: single-letter tickers like Ford's `F` would otherwise match inside every word.
2. **Resolve overlaps**: sort spans by `(start, -length)`, walk left-to-right dropping any span whose start is inside the previous span's end. Earlier entity wins — deterministic, no arbitrary choices.
3. **Tokenize once** with `return_offsets_mapping=True, return_special_tokens_mask=True` so we know where every token lives in the original character stream.
4. **Paint BIO tags**:
   - Initialize all real tokens to `O`, all special tokens to `-100`.
   - For each token, check if its char span overlaps any entity span. If it does: the token is `B-<type>` if the previous real token wasn't inside the same entity, otherwise `I-<type>`.
   - For tokens that aren't in any entity but are **subword continuations** (i.e. `token.start == prev_token.end` and prev wasn't special), overwrite with `-100` so CE only scores first-subwords.
5. **Empty-entity rows**: spans list stays empty, all real tokens get `O`, non-first-subwords get `-100`. The row still contributes to macro/industry training.

### Verified output
```
dataset size: 9218
Batch shapes (batch_size=4):
  input_ids            (4, 512)  dtype=torch.int64
  attention_mask       (4, 512)  dtype=torch.int64
  macro_targets        (4, 8)    dtype=torch.float32
  industry_targets     (4, 11)   dtype=torch.float32
  entity_targets       (4, 512)  dtype=torch.int64
```
Sanity check on 20 random samples: entity `non-O count` per sample = `[7, 6, 12, 16, 16, 40, 5, 4, 4, 5, 0, 3, 11, 54, 2, 0, 0, 4, 3, 2]`, mean 9.7 — healthy spread. Zeros are rows where surface strings don't exact-match (e.g. `"U.S."` vs `"United States"`, possessives like `Nvidia's`). Active learning can mine those later if entity recall suffers.

---

## 5. `src/model/architecture.py` — `TriLevelFinancialModel`

### Why it exists
Shared encoder + three task heads. Backbone-agnostic: loads any HuggingFace model via `AutoModel`, so swapping RoBERTa → FinBERT is one string change.

### Structure
```
          ┌─────────────────────────────────────┐
input ──► │         self.encoder (AutoModel)    │
          └──────────────┬──────────────────────┘
                         │ last_hidden_state [B, L, H]
              ┌──────────┼──────────┐
              │          │          │
       [CLS] [B, H]   [CLS] [B, H]  full seq [B, L, H]
              │          │          │
              ▼          ▼          ▼
         macro_head  industry_head  entity_head
         [B, 8]      [B, 11]        [B, L, 11]
```

### Design choices
- **[CLS] token for macro/industry**: sentence-level classification, the pooled representation is what BERT-family models are trained to produce.
- **Full sequence for entity**: NER is token-level, every position needs a classification.
- **No `token_type_ids`**: omitted from the forward call so the same module works for RoBERTa (which doesn't use them) and BERT/FinBERT (which ignore them when all-zero).
- **Dropout before each head**: uses the backbone's own `hidden_dropout_prob` when available, else 0.1.
- **Heads are `Sequential(Dropout, Linear)`**: deliberately simple. The encoder does the heavy lifting; adding MLP layers on top of [CLS] usually just overfits on small datasets like ours.

### Forward signature
```python
model(input_ids, attention_mask) -> {
    "macro_logits":    [B, 8],
    "industry_logits": [B, 11],
    "entity_logits":   [B, L, 11],
}
```

---

## 6. `src/model/losses.py` — `TriLevelLoss`

### Why it exists
Combines three different loss functions into one differentiable scalar, with per-head logging, device-aware `pos_weight` tensors, and the ignore-index handling for entity continuations.

### The formula
```
L = α_macro    · BCE(macro_logits,    macro_targets,    pos_weight=clamp(w_m, 100))
  + α_industry · BCE(industry_logits, industry_targets, pos_weight=clamp(w_i, 100))
  + α_entity   · CE (entity_logits,   entity_targets,   ignore_index=-100)
```
Default alphas are all 1.0. Clamp at `POS_WEIGHT_CAP=100.0`.

### Why the pos_weight cap
From `class_weights.py` output: Consumption has `w=2303`. Without a cap, one positive example in a batch would dominate the entire macro gradient. With only 4 positives in the whole dataset, BCE cannot realistically learn this class from weighting alone — the correct recovery mechanism is active learning, not weighting. Cap at 100 keeps rare classes visible in the loss without letting them hijack optimization.

### Why entity CE is unweighted
`O` dominates token classification (>95% of tokens). A down-weight on `O` boosts entity recall but is easy to tune wrong and hurts precision. First-pass policy: unweighted CE with `ignore_index=-100` handling subword continuations. If Step 6 training shows entity recall collapsing, add an explicit O down-weight at that point.

### Why `pos_weight` tensors are buffers
Registered as `nn.Module` buffers (not parameters). This means:
- `loss_fn.to("cuda")` moves them to GPU automatically.
- They're included in `state_dict` for checkpointing.
- But they don't receive gradients (they're fixed statistics).

`F.binary_cross_entropy_with_logits` is used functionally rather than constructing an `nn.BCEWithLogitsLoss` at init time, because the latter stores `pos_weight` as an internal attribute that doesn't move with `.to(device)`. Functional calls read the current buffer location on every forward → always correct.

### Forward output
```python
{
  "loss":          differentiable scalar (this is what you call .backward() on)
  "loss_macro":    detached scalar (logging only)
  "loss_industry": detached scalar
  "loss_entity":   detached scalar
}
```

The three head losses are detached so logging code can't accidentally hold onto computation graphs.

---

## 7. `src/model/lora_wrap.py` — `wrap_with_lora`

### Why it exists
Parameter-efficient fine-tuning via LoRA. Our silver corpus is only ~9K rows; full fine-tuning of 125M parameters on that is a recipe for overfitting + high VRAM + long wall-clock. LoRA adds low-rank `A·B` updates to specific linear layers, keeping <1% of parameters trainable.

### Key design decision: only wrap the encoder
```python
model.encoder = get_peft_model(model.encoder, lora_cfg)
```
The three task heads (`macro_head`, `industry_head`, `entity_head`) are **not** wrapped. Reason: they're tiny (~23K params total) AND random-init. LoRA adds `r*(in+out)` params on top of each target linear layer — on a random-init layer, that's strictly worse than training the layer directly. Heads stay fully trainable; only the backbone gets the PEFT treatment.

### Auto-detected target modules
Instead of hard-coding `["query", "value"]` (which breaks on non-BERT backbones), the function scans `model.encoder` submodule names for the first matching candidate:
- `("query", "value")` — BERT / RoBERTa / FinBERT
- `("q_proj", "v_proj")` — Llama / Mistral / Gemma decoders
- `("q_lin", "v_lin")` — DistilBERT

This means **the same function works for every backbone in the comparison set** without per-model config.

### Hyperparameters
- `r=8`: LoRA rank. 8 is the community-standard default for sub-1B models.
- `alpha=16`: LoRA scaling. Convention is `alpha = 2*r`.
- `dropout=0.05`: light regularization on the adapter path.
- `bias="none"`: don't train bias vectors separately.

### Verified output (from smoke_test_lora.py)
```
[lora_wrap] auto-detected target modules: ('query', 'value')
[lora_wrap] before: trainable=124,668,702 / total=124,668,702
[lora_wrap] after:  trainable=317,982 / total=124,963,614 (0.254%)
[lora_wrap] head trainable params: 23,070
```
**0.254% trainable**. The ~295K param increase in `total` (124.67M → 124.96M) is the LoRA A/B matrices added on top of the frozen base.

---

## 8. Smoke tests

Both smoke tests live in `src/tests/` and run end-to-end on a single batch. They are the **A2 verification bar** — if they pass, the training infrastructure is correct and the only remaining work is the training loop itself.

### 8.1 `smoke_test.py` — bare model

**Checks**
1. Dataset constructs cleanly on 9,218 rows.
2. Model forward produces the right output shapes.
3. Loss forward returns the right dict keys.
4. `loss.backward()` runs without error.
5. **Gradient flows into the encoder** (`encoder has gradient: True`). This catches frozen-backbone bugs before they can hide in the LoRA wrap step.

**Verified output**
```
dataset size: 9218
Batch shapes:
  input_ids            (4, 512)   dtype=torch.int64
  attention_mask       (4, 512)   dtype=torch.int64
  macro_targets        (4, 8)     dtype=torch.float32
  industry_targets     (4, 11)    dtype=torch.float32
  entity_targets       (4, 512)   dtype=torch.int64
Output shapes:
  macro_logits         (4, 8)
  industry_logits      (4, 11)
  entity_logits        (4, 512, 11)
Losses:
  loss                 4.7772
  loss_macro           0.7795
  loss_industry        1.5890
  loss_entity          2.4086
backward OK
encoder has gradient: True
```

**How to read the numbers**: entity loss at 2.41 ≈ `log(11) = 2.40`, which is exactly the expected random-baseline for an 11-way softmax at initialization. Macro (0.78) is lowest because most rows have all-zero macro targets and an untrained model's `sigmoid(0) = 0.5` is decent against that. Industry (1.59) sits in between. Combined 4.78 sums the three.

### 8.2 `smoke_test_lora.py` — with LoRA

**Additional checks beyond the bare test**
1. LoRA wrap prints sensible before/after parameter counts.
2. After backward, **LoRA adapter gradients are nonzero** (adapters are training).
3. **Head gradients are nonzero** (heads weren't accidentally frozen).
4. **Zero frozen base encoder params have nonzero gradients** (freezing is airtight).

**Verified output**
```
[lora_wrap] auto-detected target modules: ('query', 'value')
[lora_wrap] before: trainable=124,668,702 / total=124,668,702
[lora_wrap] after:  trainable=317,982 / total=124,963,614 (0.254%)
[lora_wrap] head trainable params: 23,070
Losses:
  loss                 4.1466
  loss_macro           0.7550
  loss_industry        0.8257
  loss_entity          2.5659
backward OK
LoRA adapter grad total: 190.1891
Head grad total:         161.2987
Frozen base params with nonzero grad: 0  (expect 0)
```
The third line is the critical one: **zero** frozen parameters received gradients. PEFT freezing is working exactly as advertised.

Losses are similar to the bare test (not identical, because the batch is different — `shuffle=True`) which confirms LoRA init is near-identity: the model behaves indistinguishably from the non-wrapped model on the first forward pass.

---

## 9. Files and entry points reference

| File | Purpose | Entry point |
|---|---|---|
| `data/llm_auto_labeller.py` | Upstream labelling (done) | `python data/llm_auto_labeller.py` |
| `src/data_prep/join_text.py` | Rehydrate article bodies | `python src/data_prep/join_text.py` |
| `src/data_prep/filters.py` | Shared filter / label order | imported, not run directly |
| `src/data_prep/class_weights.py` | Compute pos_weight tensors | `python src/data_prep/class_weights.py` |
| `src/data_prep/dataset.py` | `TriLevelDataset` class | imported |
| `src/model/architecture.py` | `TriLevelFinancialModel` | imported |
| `src/model/losses.py` | `TriLevelLoss` | imported |
| `src/model/lora_wrap.py` | `wrap_with_lora` | imported |
| `src/tests/smoke_test.py` | Step 4 verification | `python -m src.tests.smoke_test` |
| `src/tests/smoke_test_lora.py` | Step 5 verification | `python -m src.tests.smoke_test_lora` |

### Required Python packages
```
torch
transformers
peft
pandas
tqdm
scikit-learn
```
(And `seqeval` will be added at Step 6 for metrics.)

---

## 10. What's next (Step 6 onwards — not required for A2)

| Step | File | Purpose |
|---|---|---|
| 6a | `src/train/metrics.py` | macro-F1, micro-F1, seqeval span-F1 |
| 6b | `src/train/train_one_backbone.py` | actual training loop, CLI args, per-epoch eval logging |
| 7 | `src/active_learning/entropy.py` | per-row entropy scoring on unlabelled / low-confidence rows |
| 8 | `src/active_learning/mine.py` | select top-K highest-entropy rows as the hard sample pool |
| 9 | `src/active_learning/relabel_loop.py` | re-label hard samples via Gemma with a targeted prompt, merge back |
| 10 | `src/serve/extract.py` | produce JSON predictions for downstream consumers |

Steps 1–5 are complete and verified. Everything below is A3 scope.
