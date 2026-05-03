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
| `src/training/metrics.py` | Per-head F1 helpers | imported |
| `src/training/splits.py` | Frozen train/val/test split | imported |
| `src/training/train.py` | Step 6 training loop | `python -m src.training.train ...` |

### Required Python packages
```
torch
transformers
peft
pandas
numpy
tqdm
```
(And `seqeval` will be added later if span-level entity F1 is needed; token-level F1 is in-tree.)

---

## 10. Step 6 — Training loop

Three files. Read in this order.

### 10.1 `src/training/splits.py` — frozen data split
- `make_splits(n_rows, val_frac=0.10, test_frac=0.10, seed=42)` shuffles row indices once and returns `{train, val, test, seed, n_rows}`.
- `load_or_create_splits(...)` writes `data/splits/trilevel_splits.json` on first call and reuses it forever after.
- **Append-only behaviour**: when AL adds rows to the master csv, the saved file is updated by appending the new tail-indices to `train` only. `val` and `test` are never touched. This is the contract that lets us keep an honest evaluation set across rounds.
- Index space is positions in the **filtered** dataframe (`load_and_filter_master`), not raw csv rows.

Verified output on current corpus: `train=7,374`, `val=922`, `test=922`, `n_rows=9,218`, seed `42`.

### 10.2 `src/training/metrics.py` — per-head F1
- `multilabel_metrics(logits, targets, threshold=0.5)` for the macro and industry heads. Sigmoid → threshold → per-class P/R/F1, then micro and macro averages, plus full per-class F1 list (used in the report's per-class breakdown).
- `token_entity_metrics(logits, targets, o_index=0)` for the entity head. Argmax over tag dim, drop `-100`, then compute F1 over the union of "ground truth is non-O" and "prediction is non-O" tokens. This excludes the O-class avalanche so the score reflects actual entity discovery, not background majority class.
- Span-level (seqeval) F1 is intentionally deferred: token-F1 is enough for in-loop monitoring and early stopping, span-F1 is a §6 evaluation deliverable.

### 10.3 `src/training/train.py` — main loop
Single-file orchestrator. CLI flags map 1:1 onto `TrainConfig` so configs are reproducible from `config.json` saved alongside the checkpoint.

Pipeline per run:
1. `AutoTokenizer.from_pretrained(backbone)` + `TriLevelDataset(master_csv, ...)`.
2. `load_or_create_splits` → `Subset` for train and val.
3. Build `TriLevelFinancialModel` and (if `--lora`) wrap encoder via `wrap_with_lora`.
4. `TriLevelLoss` (loads pos_weight tensors and clamps at 100).
5. AdamW with no-decay filter on `bias`, `LayerNorm.weight`. Linear warmup over `warmup_ratio` of total steps.
6. `torch.amp.GradScaler` + autocast (fp16) on CUDA, plain fp32 on CPU.
7. Per epoch: train pass with `clip_grad_norm_`, then `evaluate` over the val loader.
8. Save best checkpoint by **combined F1** = mean of (macro micro-F1, industry micro-F1, entity micro-F1). Early stop after `patience` epochs with no improvement.
9. Append a JSON line per epoch to `log.jsonl` (loss, per-head F1, combined F1, wall time).

Run dir layout (`artifacts/runs/{backbone}_{timestamp}/`):
```
config.json   # full TrainConfig as written
splits.json   # snapshot of indices used (matches data/splits/trilevel_splits.json)
log.jsonl     # one row per epoch, machine-readable
best.pt       # state_dict + config + epoch + val metrics
```

### 10.4 Verified smoke run (CUDA, RoBERTa-base, 1 epoch, 16 train / 8 val)
```
[lora_wrap] auto-detected target modules: ('query', 'value')
[lora_wrap] before: trainable=124,668,702 / total=124,668,702
[lora_wrap] after:  trainable=317,982 / total=124,963,614 (0.254%)
[lora_wrap] head trainable params: 23,070
[train] run_dir=artifacts\runs\roberta-base_20260415_115531
[train] device=cuda | train=16 val=8
  ep1 step 8/8  loss=5.1449  lr=0.00e+00
[ep1] val combined_f1=0.0249 (M=0.000 I=0.054 E=0.021) loss=4.8077 time=0.8s
  [best] combined_f1=0.0249 -> best.pt
[train] done. best combined_f1=0.0249  ckpt=...\best.pt
```
Full pipeline reaches `best.pt` and writes `log.jsonl`. Combined F1 is meaningless on 16 examples — the test confirms wiring, not learning.

### 10.5 Real training commands
RoBERTa baseline:
```
python -m src.training.train --backbone roberta-base --epochs 3 --batch 8 --lora
```
FinBERT:
```
python -m src.training.train --backbone yiyanghkust/finbert-tone --epochs 3 --batch 8 --lora
```
DeBERTa-v3:
```
python -m src.training.train --backbone microsoft/deberta-v3-base --epochs 3 --batch 8 --lora
```

All three use the **same frozen split** (loaded from `data/splits/trilevel_splits.json`) so per-backbone metrics are directly comparable.

### 10.6 Things to know before running on the full set
- **Eval is RAM-heavy.** All val logits are concatenated on CPU before metrics. With 922 val rows × 512 tokens × 11 entity classes × float32 ≈ 20 MB — fine. If you bump max_length or val_frac, watch memory.
- **Mixed precision can NaN early.** If you see `loss=nan` at step 1, re-run with `--no-amp` and report; it usually means the entity logits collapsed because of an empty class somewhere. We have not seen this on RoBERTa-base.
- **AL warm path** is the third-pass append behaviour in `splits.py`. After AL adds N new labelled rows to the master csv (appended at the end), the next training run will see `n_rows > old_n` and silently extend `train` with `range(old_n, n_rows)` — val and test stay frozen.

---

## 11. Step 7 — Active learning loop

Goal: rescue the 75% of training rows where Gemma did not assign any macro label. The loop trains a model, asks it which empty-macro rows it thinks have a missed label, sends those rows back to Gemma for a second pass, then retrains.

Four files, plus the round driver. All artefacts live under `data/active_learning/round_{N}/`.

### 11.1 Pool definition
The AL pool is restricted to **rows in the train split with empty macro_list**. Val and test never enter AL — their labels stay frozen so per-round comparisons are honest. Train-pool size on the current corpus: **5,484** rows (74% of train).

### 11.2 `src/active_learning/score.py`
- Loads the best checkpoint for a backbone, recreates the LoRA wrap from the saved config, loads weights with `strict=False`.
- Iterates the pool through the model. For each row writes `{row_idx, macro_probs[8], macro_entropy}` to `scores.jsonl`.
- `macro_entropy` = sum of per-class binary entropies. Used as a scalar fallback ranker (mining uses per-class probs by default).

Verified: full pool of 5,484 rows scored end-to-end on CUDA in seconds.

### 11.3 `src/active_learning/mine.py`
- Strategy: **stratified per-class top-K with a probability floor**. For each macro class c, take the K pool rows with highest `macro_probs[c]` above `min_prob`. Union, dedup (a row claimed by class A is not double-counted under class B).
- Drops any row already in `relabelled_ids.json` (no double-asking Gemma in later rounds).
- Output: `candidates.jsonl` with `{row_idx, top_class, top_prob, all_probs}`.

Why per-class instead of plain entropy: generic uncertainty biases AL toward ambiguous rows the model already half-handles. Per-class rescue directly targets the report's Finding 2 (macro under-firing on Consumption / Inflation / Money-and-Credit).

Verified: with `k=10, min_prob=0.05` the smoke run picked exactly 10 candidates per class for all 8 macro classes (80 total).

### 11.4 `src/active_learning/relabel.py`
- Re-uses `data/llm_auto_labeller.call_ollama + validate` verbatim — same `SYSTEM_PROMPT`, same `temperature=0.1`, same vocab. Annotator drift across rounds = 0.
- Maps `filtered_idx -> raw_csv_idx` via `_filtered_to_master_index` so writes land on the correct row of the unfiltered csv.
- For each candidate: call Gemma with (title, text). If `is_financial=False`, record and skip (no label overwrite). If `macro` is now non-empty, write back; industry/entity are only overwritten if the new pass produced something non-empty (never erase existing labels).
- Backs up the master csv as `*.bak_round_{N}.csv` before any edit. Idempotent: rerunning the same round will overwrite the same backup but only call Gemma on candidates not yet in `relabelled_ids.json`.
- Logs every decision to `relabel_log.jsonl` (before/after macro, industry, entity count, rescued bool).

### 11.5 `src/active_learning/loop.py`
Per-round flow:
1. **Train** (skipped on round 1 if `--baseline-ckpt` is passed).
2. **Score** the pool with the freshest ckpt.
3. **Mine** stratified candidates.
4. **Relabel** via Gemma; merge in place.
5. **Recompute pos_weight** by invoking `src/data_prep/class_weights.py` so the next training run sees the new class balance.
6. Append per-round summary to `data/active_learning/loop_summary.jsonl`.

Stop conditions (OR-combined; any triggers exit):
- `max_rounds` reached (default 5).
- `n_rescued < min_rescued_per_round` (default 20). Diminishing return.
- Val combined-F1 delta `< min_delta` (default 0.005) for `patience_rounds` (default 2) consecutive rounds.
- Candidate list empty after mining.

### 11.6 Run commands
Standalone (debug each stage individually):
```
python -m src.active_learning.score    --ckpt artifacts/runs/<run>/best.pt --round 1
python -m src.active_learning.mine     --round 1 --k-per-class 50 --min-prob 0.30
python -m src.active_learning.relabel  --round 1
```

Full driver (recommended):
```
python -m src.active_learning.loop --backbone roberta-base --max-rounds 5
```

Resume from existing baseline (skip first train):
```
python -m src.active_learning.loop --backbone roberta-base --max-rounds 5 \
    --skip-first-train --baseline-ckpt artifacts/runs/<run>/best.pt
```

### 11.7 Per-round outputs
```
data/active_learning/
  relabelled_ids.json           # cumulative set of filtered_idx already re-asked
  loop_summary.jsonl            # one row per round (history vector inside)
  round_1/
    scores.jsonl
    candidates.jsonl
    relabel_log.jsonl
    summary.json
  round_2/
    ...
```
Plus a backup of the master csv per round at the same level as the original csv: `silver_dataset_master_with_text.bak_round_N.csv`.

### 11.8 Things to know before running
- **Ollama must be reachable.** `OLLAMA_URL = http://192.168.0.19:11434` is hard-wired in `data/llm_auto_labeller.py`. If you are running off-network, point it at localhost first.
- **Splits do not change.** AL rescues existing rows (rewrites their macro string). It does not append new rows. So `data/splits/trilevel_splits.json` stays valid across all rounds — same val and test set throughout the experiment.
- **No diversity filter yet (v1).** Stratified per-class already gives cross-class diversity. If within-class duplicates become a problem, add a kmeans pass over [CLS] embeddings inside `mine.py` and sample 1/cluster.
- **No human spot-check yet.** Recommended: after each round, manually inspect 10 random rows from `relabel_log.jsonl` where `rescued=True` to catch silent prompt drift.

---

## 12. Step 8 — Evaluation deliverables

Two reporting modules. Both take a checkpoint and a split (`val`/`test`) and emit JSON + (optionally) a markdown table.

### 12.1 `src/eval/span_f1.py`
Strict span-level entity F1. A predicted span counts only if both the boundary AND the type match the gold span exactly (the convention every NER paper uses).

- Decodes BIO predictions to `(start, end, type)` tuples via `bio_to_spans`.
- Drops `-100` ignore positions (subword continuations, special tokens) before scoring.
- Uses `seqeval` (IOB2, strict mode) when installed; falls back to a built-in implementation of the same metric if not.
- Output: `artifacts/eval/<run_name>_span_f1_<split>.json` with overall and per-type P/R/F1.

Run:
```
python -m src.eval.span_f1 --ckpt artifacts/runs/<run>/best.pt --split val
python -m src.eval.span_f1 --ckpt artifacts/runs/<run>/best.pt --split test
```

### 12.2 `src/eval/per_class_report.py`
Per-class precision, recall, F1, and support for the macro and industry heads. Uses sigmoid threshold (default 0.5).

- Output JSON: `artifacts/eval/<run_name>_per_class_<split>.json`.
- Output markdown: `artifacts/eval/<run_name>_per_class_<split>.md` — paste-ready tables for the report.

Run:
```
python -m src.eval.per_class_report --ckpt artifacts/runs/<run>/best.pt --split val
```

Smoke-verified end-to-end on the 16-row training ckpt: span F1 0.0012 (junk numbers as expected for a 1-epoch toy ckpt; pipeline confirmed).

---

## 13. Step 9 — Inference / serve

### 13.1 `src/serve/predict.py`
One-shot inference for a single article. Returns a JSON-serialisable dict with macro / industry labels (above threshold) and entity spans (BIO-decoded, mapped back to character offsets in the original text via the tokenizer's offset_mapping).

Run:
```
python -m src.serve.predict --ckpt artifacts/runs/<run>/best.pt \
    --title "Fed hikes rates by 25bps" \
    --text  "The Federal Reserve raised the benchmark interest rate ..."
```

Smoke-verified: returns macro/industry probability distributions over the full vocab + thresholded label list + entity surface strings with character spans.

---

## 14. Data scope: where labels and unlabelled raw live

For A2 + A3 the training and AL loops both operate exclusively on `data/silver_dataset_master_with_text.csv`. The "AL pool" is **partially-labelled rows**, not a separate unlabelled corpus: rows in the train split with `is_financial=True` AND `macro_list=[]`. Pool size = 5,484 of 7,374 train rows. Re-asking Gemma is cheap because those rows already have body text on disk.

There are unlabelled raw archives under `data/Datasets/` that have not been ingested. They are an A3 fallback: if AL plateaus below the 40% macro coverage target, run `data/llm_auto_labeller.py` on the unread archives, append to the master csv, and the train split will silently extend (val/test stay frozen). This path is wired but not exercised yet.

---

## 15. What's next

| Step | File | Purpose |
|---|---|---|
| 10 | `src/serve/api.py` | thin FastAPI wrapper around `predict.py` for live demos |
| 11 | report figures | generate per-class and per-round PNGs for §6 |
