# CS781 Phase 4: Court Judgment Prediction (CJPE)

> **Run from the repository root** as `bash phase4/run_phase4.sh <command>`, or from inside `phase4/`.
>
> **Recorded dev macro-F1:**
> - B0_tfidf 0.677
> - F3_densedapt_lora 0.746
> - F5_densedapt_adalora 0.737
> - F4_densedapt_rslora 0.428 (collapsed at r=16; retuned to r=64, α=16)
>
> **Test scores:** F3 LoRA **0.7507** (best), E1 ensemble 0.7440, F5 AdaLoRA 0.7322, B0 TF-IDF 0.6686, K1 gqa KD 0.6662, K2 hybrid KD 0.6368, F2 0.6196 (aggregated 0.6202), F1 LoRMA 0.6083 (aggregated 0.6208), F8 gqa 0.5231, F7 enc-dec 0.5182, F4 rsLoRA 0.4641.


**The task.** Given a Supreme Court judgment with its final decision removed, predict whether the appeal was accepted (1) or rejected (0). The data is ILDC (about 34k judgments), in `Phase_4_CJPE/Train/{train,dev}.json` and `Test/test.json`.
- **Score:** macro-F1 over the two classes.
- **Submission:** `prediction.json` = `[{"id": ..., "pred": 0|1}, ...]` inside `submission.zip`.
- **Rules carried over:** official data only, no outside pretrained models, LoRMA in the main runs, one KD run with a smaller student.

## Data hygiene (done by the loader for every command)

- **Repeated ids.** ILDC repeats records: exact repeats are dropped (train 37,387 -> 32,305, dev 3,505 -> 2,511). A different judgment under a repeated id is kept as `<id>#k`.
- **Test judgments copied into train/dev are dropped.** `stats` found every test judgment duplicated in the labelled data, most likely the dev split.
  - A labelled copy of a test judgment would train on the test answer, or choose the epoch, threshold or ensemble members with it.
  - Train or dev documents with the same normalised text as a test judgment, or sharing >= 50% of sampled 8-grams with one, are removed.
  - `stats` prints how many were removed and how many test judgments they covered.
  - `CJPE_KEEP_TEST_OVERLAP=1` turns this off; don't use it for a submission.

## Pipeline

| stage | command | what it does | why |
|---|---|---|---|
| 0. data | `stats` | sizes, label balance, length percentiles, test documents that near-duplicate a train/dev judgment (8-gram shingles), id overlaps | ILDC has related appeals; leakage and imbalance set how far dev can be trusted |
| 0. baseline | `baseline` | TF-IDF (1-2 grams, sublinear) on the first 300 + last 2000 words, logistic regression; C and the threshold chosen on dev | minutes on CPU, no neural model; a strong floor and a different kind of ensemble member |
| 1. fine-tune | `round1`, `techniques`, `kd` | Phase-1/2 backbone + PEFT on the **last 4 windows of 512 tokens** (+ the first window); every window carries the document label; document probability = sigmoid(mean window logit); a learned position embedding tells the head how far from the end a window is | judgments argue the outcome at the end (the ILDC paper found the last 512 tokens most informative); window-level training fits a 24 GB card |
| 1. KD | `kd` | student (gqa 57M / 4-layer hybrid) learns from the teacher's document probability at temperature T, mixed with gold (alpha 0.5) | the required KD experiment; document-level targets need no shared tokenizer |
| 2. embed | `embed RUN` / `hier` | with a stage-1 model: the last 8 windows of every judgment → mean-pooled vector + stage-1 logit, cached per split | covers 4k tokens per judgment without back-propagating through all of them |
| 3. aggregate | `aggregate RUN` / `hier` | BiGRU + attention + last-window feature over the window sequence; 5 seeds averaged; early stopping on dev | the hierarchical model (ILDC's best family): reads the argument's flow across windows |
| 4. ensemble | `ensemble` | probability average **or** majority vote of finished runs (baseline, fine-tuned, aggregated), members greedy on dev, threshold on dev | different views of the text make different errors |
| 5. submit | `collect`, `check` | every run writes `submission/submission.zip`; `check` repeats the scorer's checks (ids, 0/1 ints, no duplicates) | no submission fails on format |

**Thresholds** (all stages): the macro-F1-optimal threshold is tuned on dev. It is kept only if it wins the half-swap check (tune on half of dev, score the other half, swap) by more than 0.003. Otherwise 0.5 is used.

**Techniques** are shared with Phase 3 (`phase3/peft.py`): LoRA, rsLoRA, QLoRA, AdaLoRA, LoRMA, prefix, prompt, IA³, and full fine-tuning. Backbones are the Phase-1 checkpoints, the Phase-2 domain-adapted ones and the Phase-2 NER model (X24, adapters merged).

## Experiments (four distinct, LoRMA, one KD)

| run | backbone | technique | axis |
|---|---|---|---|
| B0_tfidf | – | TF-IDF + LR | non-neural baseline |
| F1_densedapt_lorma | dense, domain-adapted (best Phase-3 backbone) | LoRMA | main |
| F2_p2x24_lorma | Phase-2 X24 NER model | LoRMA | cross-phase transfer |
| F3–F6 | dense, domain-adapted | LoRA / rsLoRA / AdaLoRA / IA³ | technique comparison |
| F7_encdec_lorma | encoder-decoder (encoder stack, bidirectional) | LoRMA | 4th Phase-1 architecture |
| F8_gqa_lorma | gqa 57M | LoRMA | gqa without KD (control for K1) |
| K1_gqa_lora_kd, K2_hybrid4_kd (K0 without KD) | gqa 57M, hybrid 4 layers | LoRA, full | knowledge distillation |
| A_F1, A_F2 | stage-1 encoders | BiGRU aggregator | window-only vs hierarchical |
| E1 | – | ensemble | final |

## Order of work

```bash
bash run_phase4.sh fetch && bash run_phase4.sh stats
bash run_phase4.sh baseline                      # CPU, start it first
bash run_phase4.sh round1                        # F1 || F2   (about 2 h each with 2 epochs on the full data)
bash run_phase4.sh hier                          # stages 2+3 on F1 and F2
bash run_phase4.sh kd                            # needs F1 (teacher)
bash run_phase4.sh arch                          # F7 encoder-decoder || F8 gqa: all four Phase-1 architectures
bash run_phase4.sh techniques                    # the technique table (LoRA, rsLoRA, AdaLoRA, IA3)
bash run_phase4.sh ensemble && bash run_phase4.sh summary && bash run_phase4.sh collect
```

**Quick pass first:** `MAX_TRAIN=8000 EPOCHS=1 bash run_phase4.sh round1` takes a fraction of the time. It shows whether the window setup learns before you commit the full ~2 hours.

**Knobs** (environment variables):
- `CHUNK`: window length, default 512.
- `TAIL`: stage-1 windows per judgment, default 4.
- `EMBED`: stage-2/3 windows per judgment, default 8.
- `EPOCHS`: default 2.
- `BATCH`: default 16.
- `MAX_TRAIN`: number of training judgments to sample (0 = all).
- `TEACHER`: KD teacher run, default F1.

## Not built yet (ideas, ranked)

1. **Rhetorical-role selection from Phase 3.** Tag each judgment's sentences with the best Phase-3 model and feed the windows built from Ratio / Analysis / argument sentences instead of the raw tail. This is the IL-TUR idea of using RR as an intermediate step. It costs one pass of the Phase-3 tagger over about 34k judgments.
2. **Per-window attention in stage 1** (learned weights instead of the mean of window logits).
3. **Explanations** (the "E" in CJPE): attention weights or the highest-logit windows as the explanation. These aren't scored on Codabench, but are useful for the write-up.
