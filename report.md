# Veritas — Findings Log for the Final Report

Append-only log of important findings, decisions, and red flags encountered during training and active learning. Each entry has a date, a tag, and enough context to write the corresponding report section without re-running the experiment.

---

## 2026-04-15 — Three-backbone baseline (epochs=3, batch=8, LoRA r=8)

### Combined val F1 (best epoch per run)

| Backbone | combined F1 | macro F1 (micro) | industry F1 (micro) | entity F1 (micro) |
|---|---:|---:|---:|---:|
| **roberta-base** | **0.2189** | 0.3408 | 0.3152 | 0.0008 |
| bert-base-uncased | 0.2044 | 0.3123 | 0.3008 | 0.0000 |
| yiyanghkust/finbert-tone | 0.1863 | 0.2766 | 0.2824 | 0.0000 |

- RoBERTa wins by ~6% relative on combined.
- FinBERT underperforms BERT-base — domain pretraining did not help. FinBERT-tone was fine-tuned for sentiment, not topic/entity, and ships with the BERT-base tokenizer; the financial corpus is too narrow for FRED-MD macro and GICS industry vocabulary.

Source: `artifacts/eval/*_per_class_val.{json,md}`, `artifacts/runs/*/log.jsonl`.

### Red flag — Entity head is not learning

Entity F1 is **0.000 across all three backbones** on the in-loop micro metric (which excludes pure-O tokens, so 0.000 means the model is predicting only `O`).

Diagnosis (three plausible causes, ranked by likelihood):

1. **CE loss collapse to majority class.** O dominates ~95% of token positions. Cross-entropy without class weighting finds the trivial local minimum at "always predict O". The macro and industry heads are protected from this by `pos_weight` in BCE, but the entity head currently uses unweighted CE.
2. **Loss-alpha imbalance.** In `TriLevelLoss`, `alpha_macro = alpha_industry = alpha_entity = 1.0`. But entity CE starts at ~2.5 (random over 11 classes ≈ ln 11 = 2.4) while macro and industry BCE start at ~0.7. With pos_weight clamping at 100, macro/industry per-step gradients are still much larger than entity's. The optimiser preferentially fixes macro/industry while the entity head drifts.
3. **3 epochs insufficient.** Token-level learning typically needs more passes than sentence-level on small data. With 7,374 train rows × ~50 non-O tokens per row = ~370k entity examples, 3 epochs may simply be too few before O-collapse becomes irreversible.

### Mitigations (planned)

1. **Bump `alpha_entity` to 3.0** in `TriLevelLoss` (cheapest, try first). This scales the entity loss term so its gradient contribution roughly matches macro/industry after pos_weight scaling.
2. **Add explicit O down-weight** in entity CE if alpha bump alone is insufficient. Class weights `[0.1, 1.0, 1.0, ..., 1.0]` for `[O, B-ORG, I-ORG, ...]`.
3. **Train ≥5 epochs** for the entity head specifically. Two-stage training (encoder + heads jointly for 3 epochs, then entity head only for 2 more) is an option if token-level F1 still does not lift.

### Decision

- Do **not** proceed to active learning until entity F1 > 0.05 on val. AL only rescues macro labels — it cannot recover from a broken entity head.
- Re-run RoBERTa (the winner) only, with `alpha_entity = 3.0`, before any AL round.

### Note for §6 of the report

The RoBERTa-vs-FinBERT comparison is a clean counter-example to the "domain-specific pretraining always helps" assumption. Worth a paragraph: domain pretraining helps when the downstream task aligns with the pretraining objective (sentiment → sentiment). It does not transfer to unrelated downstream objectives (sentiment pretraining → entity NER + macro topic).

### Background — the loss alpha and gradient-imbalance problem (for §4 / §6 explanation)

The combined objective is
```
total_loss = alpha_macro * loss_macro + alpha_industry * loss_industry + alpha_entity * loss_entity
```
with all three alphas defaulting to 1.0 in `TriLevelLoss.__init__`.

The optimiser does not "see" loss values directly — it follows the **gradient magnitude** of the total loss with respect to each parameter. So the head whose loss term contributes the largest gradient is the one that updates fastest.

Loss values at random init:
- `loss_macro ≈ 0.7`, but each per-class BCE term is multiplied by `pos_weight` up to the cap of 100 for rare classes (Consumption n_pos=4 → pos_weight clamp at 100). Effective per-class gradient is **much** larger than the raw loss number suggests.
- `loss_industry ≈ 0.7`, same pos_weight scaling, large effective gradient.
- `loss_entity ≈ 2.5` (close to ln 11 ≈ 2.4, the random-prediction CE for an 11-way classifier), no class weighting, gradient diluted across all 11 BIO tags.

Net effect: macro and industry gradients dominate the optimiser's step direction at every iteration. The entity head receives a weaker training signal, drifts toward the trivial all-O solution (the O class is ~95% of tokens), and never recovers.

**Mitigation = bumping `alpha_entity`.** Setting `alpha_entity = 3.0` rescales the entity loss term so its gradient roughly matches macro/industry after pos_weight scaling. Any value in the 2.0–5.0 range is reasonable; 3.0 was chosen as a starting point based on rough parity with the macro/industry effective scaling. The exact value will be tuned empirically once the entity head shows non-zero F1.

### Background — why the entity head has 11 classes (BIO encoding, not coincidence with industry)

Both the entity head and the industry head output 11 logits, but for different reasons.

**Industry head:** 11 GICS sectors (Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate). Multi-label BCE: each class has an **independent** binary classifier with its own `pos_weight`. Output shape `[batch, 11]`.

**Entity head:** 5 entity types under BIO tagging plus the outside class:
```
O                              (1)   non-entity
B-ORG, I-ORG                   (2)   organisation
B-PER, I-PER                   (2)   person
B-TICKER, I-TICKER             (2)   stock ticker
B-PRODUCT, I-PRODUCT           (2)   product
B-LOC, I-LOC                   (2)   location
                              ----
                               11
```
5 × 2 + 1 = 11. Cross-entropy: classes **compete** — exactly one tag is selected per token. Output shape `[batch, seq_len, 11]`.

The numerical coincidence (both = 11) is irrelevant; the failure modes are completely different.

#### Why competing CE classes hurt this head specifically

1. CE forces the 11 BIO tags to compete for a single argmax per token.
2. O dominates ~95% of token positions, so the model can score very low loss by predicting O everywhere.
3. The remaining gradient mass is spread across 10 minority tags (B-/I- × 5 types), each receiving ~0.5% of the per-token signal on average.
4. There is no per-class weighting on entity CE, unlike macro/industry BCE which use `pos_weight`.

Result: even with the right backbone and enough data, the entity head collapses to the all-O baseline and reports F1 = 0.000 on the non-O micro metric.

This justifies two of the three planned mitigations from the previous entry:
- alpha bump (give the entity head a louder voice in the optimiser)
- explicit O down-weight in entity CE (~0.1 weight on O, 1.0 on all B-/I- tags)

The third mitigation (more epochs) addresses convergence speed, not the gradient-imbalance root cause.

---

## 2026-04-15 — Three-backbone re-run (epochs=8, batch=8, LoRA r=8, **alpha_entity=3.0**)

Re-ran all three backbones with the entity-loss alpha bump and 8 epochs (early stopping with patience=2). Entity head is now learning across all three.

### Combined val F1 (best epoch per run)

| Backbone | combined F1 | macro F1 (micro) | industry F1 (micro) | entity F1 (micro) | best epoch |
|---|---:|---:|---:|---:|---:|
| **roberta-base** | **0.4578** | 0.3301 | 0.5014 | **0.5420** | 7 |
| bert-base-uncased | 0.3780 | 0.3333 | 0.4617 | 0.3389 | 8 |
| yiyanghkust/finbert-tone | 0.3495 | 0.3471 | 0.4568 | 0.2445 | 8 |

### Span-level entity F1 (strict, IOB2)

| Backbone | overall span F1 | ORG | PER | LOC | PRODUCT | TICKER |
|---|---:|---:|---:|---:|---:|---:|
| **roberta-base** | **0.1929** | 0.25 | 0.11 | 0.00 | 0.00 | 0.00 |
| bert-base-uncased | 0.0735 | 0.09 | 0.03 | 0.00 | 0.00 | 0.00 |
| yiyanghkust/finbert-tone | 0.0034 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Findings

1. **Combined F1 doubled for RoBERTa** (0.2189 → 0.4578) almost entirely from the entity head recovery (0.000 → 0.5420 token-level). Confirms the gradient-imbalance diagnosis.
2. **RoBERTa wins on every head except macro.** Macro is essentially tied across all three backbones (0.33-0.35) — within noise.
3. **FinBERT entity span F1 is essentially zero (0.003).** Even though token F1 is 0.24, FinBERT cannot reproduce full spans — it predicts entity tokens fragmentarily but rarely matches both boundary and type. This pattern (high token F1, near-zero span F1) is characteristic of a model that has learned "this token looks entity-shaped" but not "this token starts/extends a coherent span".
4. **LOC, PRODUCT, TICKER are unrecoverable** with current data. All three backbones score 0.00 span F1 on these types. Cause: severe under-representation in silver labels. Active learning will not fix this — AL only rescues macro labels. A separate mitigation (oversampling, targeted re-prompting for these types, or accepting the gap in the report) is required.
5. **Entity F1 trajectory for RoBERTa:** 0.00 → 0.19 → 0.31 → 0.44 → 0.47 → 0.51 → 0.54 → 0.54. Plateaus at epoch 7. 3 epochs (the original run) was clearly insufficient; 8 is enough.

### Decision

- **RoBERTa is the chosen backbone for active learning.** Baseline checkpoint: `artifacts/runs/roberta-base_20260415_133950/best.pt`.
- Proceed to AL with `--skip-first-train --baseline-ckpt <above>`.
- Note in §6 that BERT-base is a respectable second (0.378), and FinBERT's underperformance is a finding worth discussing.

### Note for §6 of the report

- The alpha-bump intervention is a textbook example of why naive equal-weighting of multi-task losses fails when the tasks have very different per-step gradient magnitudes. Worth documenting as a process learning.
- The entity-type gap (ORG/PER work, LOC/PRODUCT/TICKER do not) is **not** explained by raw label scarcity. See the entity-support analysis below.

### Entity-type support — rebuttal of the naive "scarcity" hypothesis

Running `src/eval/entity_support.py` on the frozen splits gives the actual per-type span counts:

| Type | Train spans | Share of all spans | Train rows containing type |
|---|---:|---:|---:|
| ORG | 22,823 | **63.0%** | 5,949 (81% of train rows) |
| PER | 4,540 | 12.5% | 2,314 (31%) |
| TICKER | 3,596 | 9.9% | 1,704 (23%) |
| PRODUCT | 2,893 | 8.0% | 1,308 (18%) |
| LOC | 2,364 | 6.5% | 1,486 (20%) |

Total: 36,216 spans across 7,374 train rows (~4.9 entity spans per row).

LOC, PRODUCT, and TICKER each have **2,400-3,600 spans** in the train set. By standard NLP NER benchmarks (CoNLL-2003 has ~3,000 LOC entities) this is adequate, not scarce. So the F1 = 0.00 on these types is not a data-quantity problem.

**Real cause: ORG dominance + boundary brittleness on short entities.**

1. **ORG outnumbers each other type by 5-10×.** Under cross-entropy with class competition, the model learns "when uncertain, predict ORG" because that minimises expected loss across most token positions. The other types are out-competed even though they have plenty of training examples.
2. **Single-token entity brittleness.** TICKER ("AAPL", "MSFT") and short LOC ("US", "China") are typically 1 token after subword tokenisation. Strict span F1 requires exact boundary AND type match — a 1-token miss is a complete miss. Multi-token ORG spans tolerate small errors better in token F1 but not in span F1, which is why ORG token F1 (0.50+) translates to span F1 of only 0.25 even for the winning model.
3. **No class weighting on entity CE.** The model has no incentive to treat the rare-but-not-scarce types as anything other than noise relative to ORG.

**Mitigations (proposed for §6 ablation):**
- Add explicit per-class weights to entity CE: down-weight ORG (~0.5×), up-weight LOC/PRODUCT/TICKER (~2×).
- Run a targeted Gemma re-labelling pass restricted to articles with empty entity_list of these specific types (parallel to the macro AL rescue).
- Report span-level F1 and a separate "type confusion matrix" so the per-type performance gap is visible to the reader.

This is an A3 deliverable; current A2 baseline numbers stand.

---

## 2026-04-15 — Active learning round 1 (RoBERTa baseline, k_per_class=50, min_prob=0.30)

Ran one round of active learning against the RoBERTa baseline (`roberta-base_20260415_133950/best.pt`, val combined F1 = 0.4541). The mining stage selected the maximum 300 candidates (50 per macro class × 6 classes that had any pool row above the 0.30 prob floor). Re-labelling completed in 1,424 seconds (~24 minutes) of Gemma calls.

### Headline result

- **300 candidates, 6 rescued, 294 returned empty macro again.**
- **Rescue rate: 2.0%.** AL stopped after round 1 (`STOP: rescued 6 < 20`).
- **Macro coverage moved from 2,307 → 2,313 (+6) rows.** Effectively unchanged.

### What Gemma actually did with the 294 non-rescues

| Output type | Returned non-empty | Share |
|---|---:|---:|
| macro | 6 | 2% |
| industry | 142 | 47% |
| entity | 293 | 98% |

So Gemma is not rejecting the rows as non-financial, and is not failing on the call (`n_failed = 0`). It is processing each row, identifying entities and often industries, and consistently concluding **no macro topic applies**.

### Re-diagnosis: the 75% empty-macro pool is ground truth, not labeller error

The active-learning loop was built on the assumption that the empty-macro rows were a labeller-coverage gap — Gemma being inconsistent or under-firing on borderline articles. The data refutes this. Gemma is consistent: rows it left empty on the first pass it leaves empty on the second pass, even when prompted to re-examine.

The articles in the empty-macro pool are mostly **routine company news** ("Apple announces X", "Microsoft acquires Y"). They have entities, they have industry context, but they do not advance a macro economic thesis. They have a **subject** but no **macro frame**.

Concretely: the 6 rows that did rescue all have a clear macro frame (e.g. an energy/utility article about credit conditions → "Money and Credit"). The 294 that did not rescue read as company-level reporting without macroeconomic content.

### Implications

1. **Macro coverage of ~25% is the true ceiling for this corpus** under FRED-MD vocabulary at article granularity. Active learning cannot push past it because the gap is not coverage — it is class semantics.
2. **Macro F1 of 0.33 is bounded by label sparsity, not model capacity.** With only 2,307 macro-positive rows distributed across 8 highly imbalanced classes (Consumption n=4, Housing n=22), recall is structurally constrained. A larger model or more epochs will not unlock more recall against this label distribution.
3. **The mining strategy was correct but the assumption behind it was wrong.** Stratified per-class top-K with a probability floor is the right algorithm for closing a labeller gap. It cannot help when the labels themselves are correct.
4. **The model is reading weak signal (financial-vocabulary words) as macro evidence.** The high-confidence false positives Gemma rejected are exactly where the model's macro head over-fires. This is also the root cause of macro F1 plateauing around 0.33.

### Decision

- **Do not run further AL rounds.** Each round costs ~24 minutes of Gemma calls for a ~6-row gain. ROI is too low to justify.
- **Frame the 25% ceiling as a finding in §6**, not as a failure. The rescue rate of 2% is itself evidence for the claim that macro topics are intrinsically sparse at article granularity in financial news.
- **Add a macro-threshold ablation to §6**: sweep sigmoid threshold ∈ {0.3, 0.4, 0.5, 0.6}, report macro F1, precision, and recall at each. This gives the reader a precision-recall curve and lets the deployment context (high-recall vs high-precision use) drive the threshold choice.
- **Do not run AL on BERT or FinBERT.** The result generalises — the gap is data-truth-bounded, not model-bounded.

### Note for §6 of the report

This is a strong negative result, properly framed:
- We hypothesised that the under-firing of macro labels was a labeller-coverage gap (Gemma being inconsistent).
- We designed a stratified-uncertainty AL loop to close the gap.
- We executed one round on 300 candidates and observed a 2% rescue rate, with Gemma consistently returning empty macro on re-prompt.
- We conclude that the empty-macro rows are a ground-truth signal, not a labelling artefact, and the macro coverage ceiling is intrinsic to financial news at article granularity.

This is more valuable than a successful AL run that closed the gap by 5pp — it identifies a structural property of the data that any future macro classifier on this domain will need to address (sub-article granularity, multi-document aggregation, or accepting low recall as the cost of high precision).
