# CS781 Phase 5: Legal Statute Identification (LSI, multi-label)

> **Run from the repository root** as `bash phase5/run_phase5.sh <command>`, or from inside `phase5/`.
>
> **Recorded:** the CPU baselines S1–S4 (TF-IDF logistic / modified-Huber SVM / kNN / their mean). Best is S4, dev macro-F1 0.404.
>
> **Test scores:** S4 TF-IDF average **0.3968** (submitted best), S1 logistic 0.3908, S2 SVM 0.3844, S3 kNN 0.3459; neural label-attention LoRMA models L1 dense 0.3576, L2 gqa 0.3517, enc-dec 0.2977 (not tuned further before the deadline).


**The task.** Given the **facts** of a criminal case, predict every applicable IPC statute (100 statutes, about 3.8 per case).
- **Data:** ILSI (Paul et al. 2022, via IL-TUR), in `Phase_5_LSI/Train/{train,dev}.json` and `Test/test.json`. Records are `{"id", "text", "labels": [ids]}`.
- **Masking:** statute and act citations in the facts are masked, so the model has to infer the statute from what happened, not read it.
- **Submission:** `prediction.json` = `[{"id": ..., "pred": [ids]}, ...]` in `submission.zip`.
- **Scores:** Macro-F1 (primary), Micro-F1, subset accuracy.
- **Rules carried over:** start from the Phase-1 models; all four architectures; at least 4 substantially different experiments; LoRMA and knowledge distillation compulsory; no outside models or data.

## Design

| part | choice | why |
|---|---|---|
| document → windows | the facts (~2.4k words, ~3.5k tokens) cut into 512-token windows, **all of them used** (8 in training, up to 16 at inference; `--stride` for overlap) | a statute's evidence can be anywhere in the facts; truncating to one window loses it |
| windows → document | **label-wise attention** (`--pool label`): each statute has its own query and attends to the windows that describe its offence. Alternatives: `attn` (shared), `max`, `mean` | 100 statutes with different cues; a shared document vector blurs them |
| loss | **asymmetric loss** (`--loss asl`, γ⁻=4, clip 0.05). Alternatives: `wbce` (per-statute positive weight √(neg/pos)), `bce`, `focal` | each document has ~96 easy negatives; ASL keeps them from swamping rare positives |
| rare statutes | ASL, per-statute thresholds, optional `--rare-repeat N` (documents with a bottom-quartile statute seen N× per epoch) | Macro-F1 weights each statute equally |
| thresholds | candidates: fixed 0.5, one tuned global value, **per-statute** thresholds shrunk toward the global one (w = n⁺/(n⁺+s), s ∈ {3, 10, 30}), each ± a "**min-one**" rule (every case gets its top statute). Chosen on dev by an **honest split-half** score; a tuned scheme must beat 0.5 by 0.002 | Macro-F1 decomposes per statute, so per-label thresholds are exact on the tuning set, but rare statutes have few dev positives; shrinkage and the honest check stop overfitting |
| baseline | TF-IDF (1–2-grams) + one-vs-rest logistic regression, class-balanced | the facts' vocabulary ("dowry", "lathi", "succumbed") is predictive; a strong, different ensemble member |
| KD | teacher = dense LoRMA (L1, 395M); students = gqa 57M LoRMA (K1) and a 4-layer hybrid (K2). Soft targets = σ(teacher logit / T), mixed with gold (`--kd-alpha 0.5`, T=2). Teacher logits are cached in the teacher's folder | the required distillation run; document- and statute-level targets need no shared tokenizer |
| ensemble | average of statute probabilities; members chosen greedily on dev (honest per-label score); thresholds chosen again | different backbones and the TF-IDF model make different errors |
| data hygiene | repeated ids merged; train/dev documents that duplicate a test document (exact or ≥50% shared 8-grams) dropped; `stats` reports them | the Phase 4 dev set contained the whole test set |

All 9 techniques in section 11.2.1 are implemented: full, partial (top-k blocks), QLoRA (NF4), LoRMA, AdaLoRA, Adapters, Prefix, Prompt, and KD.

## Runs (`run_phase5.sh`)

| run | backbone | technique | answers |
|---|---|---|---|
| B0_tfidf | – | TF-IDF + OvR LR | non-neural reference |
| L1_dense_lorma / L2_gqa_lorma / L3_hybrid_lorma / L4_encdec_lorma | the four Phase-1 architectures (domain-adapted dense and gqa if present) | LoRMA | "all four architectures"; is Phase-1 pretraining useful? |
| T_full, T_partial, T_qlora, T_adalora, T_adapter, T_prefix, T_prompt | `$BEST` (default gqa) | the technique comparison | PEFT vs full fine-tuning; which technique suits LSI |
| K1_gqa_lorma_kd, K2_hybrid4_kd (K0 without KD) | gqa 57M, 4-layer hybrid ← L1 | KD | does KD keep performance at lower cost? |
| A_pool_attn / A_pool_max | `$BEST` | LoRMA | long-document representation: label-wise vs shared attention vs max |
| A_loss_bce / A_loss_wbce | `$BEST` | LoRMA | loss and class-balancing |
| A_sel_head / A_sel_tail | `$BEST` | LoRMA, first or last 2 windows only | complete document vs selected sections |
| A_overlap / A_rare | `$BEST` | LoRMA | overlapping windows; rare-statute repetition |
| E1_ensemble | – | probability average | final submission |

Every run's `summary.json` records:
- honest dev Macro-F1, Micro-F1 and subset accuracy;
- the chosen threshold scheme and the full threshold table (per-label vs global, min-one);
- per-statute dev F1 and the bottom-quartile ("rare") statutes' mean F1;
- trainable / total parameters and minutes.

`bash run_phase5.sh summary` prints the comparison table.

## Order of work

```bash
bash run_phase5.sh fetch && bash run_phase5.sh stats
bash run_phase5.sh baseline                     # CPU, start it first (JOBS=8 parallel statute classifiers)
bash run_phase5.sh arch                         # L1 || L2, then L3 || L4
bash run_phase5.sh summary                      # pick BEST backbone and TEACHER
BEST=gqa bash run_phase5.sh techniques          # 7 runs, 2 GPUs
TEACHER=$PWD/../cs781_phase1/runsP5/L1_dense_lorma bash run_phase5.sh kd && bash run_phase5.sh kd0
BEST=gqa bash run_phase5.sh ablations
bash run_phase5.sh ensemble && bash run_phase5.sh summary && bash run_phase5.sh collect
```

**Quick check first:** `MAX_TRAIN=6000 EPOCHS=1 OUT=$PWD/../cs781_phase1/runsP5_quick bash run_phase5.sh one L2_gqa_lorma 0`.

**Knobs** (environment variables):
- `EPOCHS`: default 3.
- `TRAIN_CHUNKS` / `EVAL_CHUNKS`: default 8 / 16.
- `DOC_BATCH`: documents per optimizer step, default 16.
- `DPF`: documents per forward, default 2; set 1 if a run runs out of memory.
- `POOL`, `LOSS`, `MAX_TRAIN`, `BEST`, `TEACHER`.

Full fine-tuning (`T_full`) adds gradient checkpointing and 1 document per forward automatically.

**Analysis hooks (section 10.5):**
- **Techniques:** T_* vs L_* (method, trainable parameters, minutes).
- **Long documents:** A_pool_*, A_sel_*, A_overlap.
- **Rare statutes:** `dev_per_label_f1` and `dev_macro_f1_rare_quartile_fit`, A_rare.
- **Thresholds:** `threshold_table` in every summary.
- **KD:** K1/K2 vs L2/K0 and parameter counts.
- **Phase-1 usefulness:** L_* vs B0.

`dev_per_label_f1` also lists the statutes most often missed.
