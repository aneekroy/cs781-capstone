# Every Codabench submission, phase by phase

This file lists every distinct Codabench submission. For each one it gives the run name (the name of its output folder under `runsPN/`), what the system was, its official test score and the launcher command that rebuilds it. Repeated uploads of the same predictions and failed runs are left out. Where several submissions differ only in seed or in one setting, they share a row with their scores separated by "/".

- **Architecture, methodology and implementation** for each phase are in the [main README](README.md) and in each `phaseN/README.md`.
- **Hyperparameters** are in the report (Table 4) and in each launcher's `experiments` block (`bash phaseN/run_phaseN.sh list`).
- **Launchers** are run from the repository root. `one <run> [gpu]` trains and packages a single run. The group commands (`round1`, `techniques`, …) run several runs, two at a time on the two GPUs.
- **Outputs.** Every run writes `runsPN/<run>/submission.zip`, which is the file that was uploaded. `bash phaseN/run_phaseN.sh collect` gathers them into one folder.

**Bold** marks the best submission of each phase, which is the one in the results table of the README and the report.

Abbreviations:
- **DAPT:** continued domain pretraining of a Phase-1 backbone (Phase 2, `adapt`).
- **KD:** knowledge distillation.
- **OOF:** out-of-fold cross-validation score.
- **dense / gqa / hybrid / enc–dec:** the four Phase-1 backbones (`phase1/models.py`).

---

## Phase 1: legal language-model pretraining (BERTScore-F1)

8 valid submissions. Rank 71. Leader 0.8689.

| submission | system | test |
|---|---|---|
| **dense_gpt** | 12-layer GPT decoder (105.0M), 395M training tokens, decoder chosen by the official scorer on 1,000 held-out continuations | **0.8312** |
| recurrent_hybrid | 10 layers alternating LSTM and RoPE attention (101.1M), 78M tokens | 0.8280 |
| gqa_local | 14 layers, RoPE, GQA 12:2, 256-token sliding window with a global layer every 4th (103.8M), 58M tokens | 0.8279 |
| encoder_decoder | 6+6 layers (119.1M), predicts the last quarter of each block, 429M tokens | 0.8216 |
| first-round uploads (4) | the same four models after a shorter first training round | 0.8168–0.8275 |

Reproduce: `notebooks/CS781_Phase1_Jupyter_2x24GB_run.ipynb`, or `python -m phase1.experiments matrix … --execute` (see [phase1/README.md](phase1/README.md)).

## Phase 2: legal NER (strict macro-F1)

55 valid submissions. Rank 5. Leader 0.7005.

Each tagger is a Phase-1 backbone with PEFT and a CRF. Unless stated otherwise it uses rules-based post-processing chosen on 2-fold CV. The second column is the CV score after post-processing.

| submission | system | CV | test |
|---|---|---|---|
| E2_encdec_full / X6_encdec_full_bpeseg | enc–dec, full fine-tuning / with punctuation segmentation | .342 / .330 | .392 / .411 |
| X7_encdec_lorma | enc–dec, LoRMA | .418 | .480 |
| X4_encdec_span | enc–dec, span classifier | .201 | .242 |
| A2_encdec_mlm_full | enc–dec after MLM DAPT, full fine-tuning | .511 | .549 |
| E3_gqa_lora_causal / A3_gqa_clm_lora | gqa LoRA, before / after causal DAPT | .505 / .531 | .538 / .574 |
| E1_denselm_lorma_causal | dense continued-LM checkpoint, LoRMA, causal | .533 | .573 |
| X2_densesuffix_lorma | dense suffix checkpoint, LoRMA | .550 | .554 |
| X3_denselm_lorma_bidir | dense LoRMA with the causal mask removed | .408 | .467 |
| X1_denseorig_lorma | dense LoRMA, punctuation split before BPE | .536 | .596 |
| X1 _s13 / _s21 | X1, other seeds | .533 / .555 | .586 / .573 |
| X1 repost | X1 re-decoded with the re-chosen post-processing rules | – | .588 |
| X10_denseorig_lorma_bpeseg | X1 with plain BPE segmentation | .524 | .575 |
| A1_dense_clm_lorma | dense after causal DAPT, LoRMA | .564 | .551 |
| X9_hybrid2_nokd | hybrid, 2 of 10 layers, no KD (37.1M) | .430 | .464 |
| E4_hybrid_kd / A4_hybrid_clm_kd | hybrid KD student; its unlabelled KD text still held test documents (withdrawn) | leaked | .448 / .451 |
| E4f_hybrid_kd | hybrid KD (37.1M), KD text filtered by 8-gram overlap | leaked* | .572 |
| X8_hybrid_kd4 | hybrid KD (53.6M), filtered | leaked* | .577 |
| X14_dense_selfkd / X21_dense_all_kd | dense self-KD / every change plus self-KD | leaked* | .592 / .589 |
| X17_denseorig_header | X1 plus a 160-token case header in every window | .527 | .575 |
| X18_denseorig_aug | X1 plus entity-swap augmentation (decoded with the O-penalty) | .533 | .437 |
| X19_denseorig_header_aug | X1 plus header and augmentation | .526 | .583 |
| X20_denseorig_obias | X1 plus a tuned O-tag penalty, which merges entities (withdrawn) | .544 | .310 |
| X22_gqa_clm_header_aug | adapted gqa plus header and augmentation | .525 | .495 |
| X23_denseorig_entry (s0 / s7 / s13 / s21) | X1 plus a CV-tuned entity-entry bonus, 4 seeds | .558 / .549 / .573 / .564 | .612 / .614 / .571 / .584 |
| X24_dense_coarse / X24_dense_coarse_s13 | dense LoRMA on merged roles (PARTY / COUNSEL); sides come from the header parser | .630 / – | .627 / .656 |
| v15 | X24 resubmitted through the vote pipeline (single member) | – | .627 |
| X26 | X24 plus silver labels from X24 on SC text | leaked* | .611 |
| X5 / X11_ensemble_top4 / X16 | logit averages: dense / top 4 / mixed architectures | – | .524 / .593 / .594 |
| V1_vote / v4 / v5 | early span votes | – | .588 / .612 / .618 |
| V9 / V9b / V10 / V12 | CV-selected span votes (V9, V9b and V12 add KD runs as test-time voters) | .623 / .639 / .640 / .643 | .636 / .641 / .639 / .640 |
| V14 | 7 members, k = 2, plus 5 KD voters | .644 | .644 |
| v16 / v17 | span votes including the merged-role models | – | .650 / .649 |
| V21 | merged-role vote, threshold k = 1 | – | .655 |
| V22 / V23_greedy | fixed / greedy vote over the merged-role models | .654 / .662 | .660 / .662 |
| **V24_all** | **greedy vote over all 36 non-KD runs (X24, X24_s13, X22_s13, X25, X22_s21), k = 2, then post-processing** | **.662** | **.6626** |

\* The KD teacher saw both folds, so CV is not an honest estimate for these runs. Only the test score is meaningful.

Reproduce the best submission:
```bash
bash phase2/run_phase2.sh adapt-data && bash phase2/run_phase2.sh adapt gqa_clm 0
bash phase2/run_phase2.sh round7                    # X24_dense_coarse || X25_gqa_clm_coarse
bash phase2/run_phase2.sh one X24_dense_coarse_s13 0 & bash phase2/run_phase2.sh one X22_gqa_clm_header_aug_s13 1; wait
bash phase2/run_phase2.sh one X22_gqa_clm_header_aug_s21 0
VOTE_NAME=V24_all VOTE_ARGS="--no-extras --no-rerank --no-per-label" bash phase2/run_phase2.sh vote
```
Any single run: `bash phase2/run_phase2.sh one <run> [gpu]`. The groups are `round1` … `round7`, `adapted`, `seeds`, `selfkd`, `ensemble` and `vote`.

## Phase 3: rhetorical-role prediction (macro-F1)

56 valid submissions. **Rank 1.**

Each tagger has a Phase-1 sentence encoder with PEFT, a BiLSTM context layer and a CRF. "OOF" is the 5-fold tuned score with the Dissent rule. ^mk2 marks a resubmission after the Dissent-rule redecode.

**Single taggers**

| submission | system | OOF | test |
|---|---|---|---|
| P1_dense_lorma_ctx | dense LoRMA, document context | .568 | .477 |
| P2_dense_lorma_sent | dense LoRMA, sentence only (no context) | .438 | .393 |
| P3_adapted_dense_lorma | DAPT dense, LoRMA | .520 | .545 |
| P4_adapted_gqa_lora_dom | DAPT gqa, LoRA, domain heads | .556 | .541 |
| P5_p2ner_lorma | initialized from the Phase-2 X24 tagger | .571 | .509 / .593^mk2 |
| P6_encdec_lorma_tf | enc–dec with a transformer context layer | .338 | .509 (domain variant .346) |
| P7_dense_lorma_wce | P1 plus weighted cross-entropy | .568 | .503 / .585^mk2 |
| P8_dense_lorma_rare | P1 plus rare-role oversampling | .567 | .524 |
| K1_hybrid4_kd / K2_gqa_lora_kd | KD students: 4-layer hybrid / gqa LoRA | .520 / .553 | .472 / .595, .610^mk2 |
| T_lora / T_rslora / T_qlora / T_adalora / T_lorma | technique comparison on one backbone | .561 / .566 / .554 / .561 / .576 | .573 / .565 / .586 / .574 / .539 |
| T_ia3 / T_prefix / T_prompt / T_full | technique comparison, continued | .556 / .539 / .533 / .564 | .603 / .589 / .565 / .562 |
| S_p2_x24 / S_p2_x1 | backbone screen: Phase-2 taggers | .549 / .502 | .570 / .610 |
| S_hybrid / S_hybrid_dapt | backbone screen: hybrid | .518 / .525 | .600 / .599 |
| S_dense_lm / S_dense_suffix | backbone screen: dense | .543 / .542 | .543 / .512 |
| M1_gqa_lora_kd_mix / M2_p2ner_lorma_mix | data mixing (rare-role repeats, same-label splices, domain silver data) | .556 / .560 | .603 / .566 |
| M3_dense_lorma_rare_mix / M4_gqa_dapt_lora_kd_mix | data mixing, without / with the speaker rule | .561 / – | .609, .663 / .626, .677 |

**Ensembles** (posterior averaging or voting; "seq" = Viterbi sequence decoding of the averaged posteriors)

| submission | system | test |
|---|---|---|
| E1_ensemble | 5 members, marginal decoding | .541 |
| e2 / e2_fixed | plain average | .624 / .624 |
| E4 / E5 | larger averages | .588 / .621 |
| v1 / v2 / v3 / v4 / v6_diverse | sentence-level votes | .593 / .620 / .622 / .620 / .634 |
| ST1_stack / ST2 | stacked second-level classifier | .588 / .565 |
| E7_greedy_seq / e_greedy_seq | greedily chosen members, seq | .621 / .628 |
| R1A_gqa_ens_doc / R1B_gqa_ens_doc_2pass | ensemble-teacher KD with whole-document / two-pass context | .621 / .627 |
| E6_div_seq | fixed diverse set, seq | .637 |
| E8_div_lex | E6 plus the TF-IDF member L1_tfidf | .640 |
| R1B + speaker rule / E9_div_lex | speaker-tracking rule added | .670 / .685 |
| **E6_div_mk2** | **E6 diverse set, seq, Dissent rule plus speaker-tracking rule** | **.6877** |

Reproduce:
```bash
bash phase3/run_phase3.sh screen; BEST=<backbone> bash phase3/run_phase3.sh techniques
bash phase3/run_phase3.sh round1 && bash phase3/run_phase3.sh round2 && bash phase3/run_phase3.sh kd
bash phase3/run_phase3.sh mix && bash phase3/run_phase3.sh mix2 && bash phase3/run_phase3.sh r1
bash phase3/run_phase3.sh redecode && bash phase3/run_phase3.sh push    # lexical member, E6/E7/E8, votes, stack -> E6_div_mk2
```

## Phase 4: court judgment prediction (macro-F1)

13 valid submissions. Rank 47. Leader 0.9136.

Stage 1 is a window classifier: a Phase-1 backbone with PEFT, trained on the last four 512-token windows plus the first window. Stage 3 is a BiGRU-attention aggregator over window embeddings. Training and dev copies of test documents are dropped.

| submission | system | dev | test |
|---|---|---|---|
| B0_tfidf | TF-IDF plus logistic regression (C = 8) | .677 | .669 |
| F1_densedapt_lorma | dense DAPT, LoRMA | – | .608 |
| F2_p2x24_lorma | initialized from Phase-2 X24, LoRMA | – | .620 |
| **F3_densedapt_lora** | **dense DAPT, LoRA r16 α32, window classifier** | **.746** | **.7507** |
| F4_densedapt_rslora | dense DAPT, rsLoRA r16 (training collapsed) | .428 | .464 |
| F5_densedapt_adalora | dense DAPT, AdaLoRA | .737 | .732 |
| F7_encdec_lorma / F8_gqa_lorma | enc–dec / gqa LoRMA | – | .518 / .523 |
| K1_gqa_lora_kd / K2_hybrid4_kd | KD students: gqa LoRA / 4-layer hybrid | – | .666 / .637 |
| A_F1 / A_F2 | BiGRU aggregator over F1 / F2 window embeddings | – | .621 / .620 |
| E1_ensemble | dev-selected ensemble | – | .744 |

Reproduce: `bash phase4/run_phase4.sh baseline`, `techniques` (F3–F6), `round1` (F1, F2), `arch` (F7, F8), `kd` (K1, K2), `hier` (A_F1, A_F2), `ensemble`; a single run with `one F3_densedapt_lora 0`.

## Phase 5: legal statute identification (macro-F1)

7 valid submissions. Rank 41. Leader 0.4746.

| submission | system | dev | test |
|---|---|---|---|
| S1 | TF-IDF one-vs-rest logistic regression, per-statute thresholds | .397 | .391 |
| S2 | TF-IDF one-vs-rest modified-Huber SVM | .394 | .384 |
| S3 | TF-IDF kNN | .350 | .346 |
| **S4_ensemble** | **mean of S1–S3 scores, tuned thresholds** | **.404** | **.3968** |
| L1_dense_lorma | dense LoRMA, label-wise attention over windows, asymmetric loss | – | .358 |
| L2_gqa_lorma | gqa LoRMA, same head | – | .352 |
| L4_encdec_lorma | enc–dec LoRMA, same head | – | .298 |

Reproduce: `bash phase5/run_phase5.sh baseline` (writes S1–S3 and the S4 ensemble under `runsP5/B0_tfidf`) and `bash phase5/run_phase5.sh arch` (L1–L4).

## Phase 6: abstractive summarization (BERTScore-F1)

41 valid submissions. Rank 11. Leader 0.8606.

Inputs are sentence-aligned chunks of the judgment, cN = chunk length in tokens. Each Phase-1 decoder is fine-tuned to write the matching part of the headnote, with batched masked decoding.

| submission | system | test |
|---|---|---|
| B0_extractive | supervised extractive baseline | .8508 |
| L1_dense_lorma / L2_gqa_lorma / L3_hybrid_lorma | LoRMA, one per decoder backbone (c1024) | .8507 / .8483 / .8392 |
| L4_encdec_lorma / L5_encdec_fid_lorma | enc–dec LoRMA / fusion-in-decoder | .8152 / .8079 |
| T_full / T_lora / T_qlora / T_adalora | gqa: full fine-tuning / LoRA / QLoRA / AdaLoRA | .8509 / .8478 / .8483 / .8478 |
| T_houlsby / T_pfeiffer / T_prefix / T_prompt | gqa: Houlsby / Pfeiffer adapters / prefix† / prompt tuning | .8472 / .8489 / .8228 / .8183 |
| K1_gqa_kd_logit / K2_gqa_kd_seq | gqa student: logit KD / KD from a weak teacher | .8467 / .8179 |
| K4_dense6s_kd / K5_dense6s_nokd / K6_dense6s_kdseq | 6-layer dense student: word KD / no KD / sequence KD | .8511 / .8510 / .8527 |
| C1_dense_lorma_c512 / C2_gqa_full_c512 / C3_dense_full_c512 | combined best settings, c512 | .8521 / .8516 / .8520 |
| C4_hybrid_full_c512 / C5_encdec_full_c512 | combined best settings, c512 | .8476 / .8210 |
| C6_gqa_lorma_c384 / C7_gqa_lorma_c256 | shorter chunks | .8496 / .8509 |
| C8_dense_lorma_c512_e5 | C1 trained 5 epochs | .8521 |
| **C9_dense_full_c512_e5** | **dense, full fine-tuning, c512, 5 epochs** | **.8533** |
| D1_hybrid_dapt_full_c512 / D2_encdec_dapt_full_c512 | DAPT backbones, full fine-tuning | .8532 / .8516 |
| E_dense / E_gqa | same-architecture checkpoint ensembles | .8517 / .8516 |
| A_chunk512 / A_overlap / A_srclm | ablations: 512 chunks / 128-token overlap / auxiliary source-LM loss | .8519 / .8483 / .8484 |
| A_extract / A_align_pos | ablations: extract-then-abstract (50 % of sentences) / positional alignment | .8416 / .8401 |
| A_nodapt / A_scratch | ablations: no DAPT (gqa) / random initialization | .8235 / .8154 |

† Submitted before a fix to batched prefix decoding (per-row rotary positions).

Reproduce: `bash phase6/run_phase6.sh baseline`, `arch`, `techniques`, `kd`, `kd-spread`, `combine`, `chunks`, `longer` (C8 ‖ C9), `dapt`, `ensembles`, `ablations`; the best with `bash phase6/run_phase6.sh one C9_dense_full_c512_e5 0`.

## Phase 7: prior-case retrieval (micro-F1@7)

34 valid submissions. Rank 40. Leader 0.5263.

Most neural runs were submitted twice: alone ("neural") and fused with BM25 using a z-score weight chosen on dev ("fused").

| submission | system | test |
|---|---|---|
| B0_lexical | hashed word n-gram BM25, tuned on train | .421 |
| **B1_lexical_ctx** | **B0 plus citation-context queries (sentences around each `<CITATION>`), fused with the full query** | **.4400** |
| Z1–Z4 _zeroshot | dense / gqa / hybrid / enc–dec Phase-1 models, zero-shot: neural / fused | .320 / .316 / .314 / .309 ; .419 / .421 / .420 / .421 |
| L1–L4 _lorma | LoRMA passage retriever per backbone: neural / fused | .355 / .360 / .354 / .345 ; .425 / .427 / .427 / .425 |
| R1_ctx2 | retriever with citation-context queries: neural / fused | .352 / .437 |
| T_full / T_partial / T_qlora | technique comparison: neural / fused | .360 / .353 / .353 ; .431 / .425 / .423 |
| A_bidir | ablation, bidirectional attention: neural / fused | .354 / .431 |
| A_chunk400 / A_proj256 | ablations, fused: 400-word chunks / 256-d projection | .431 / .431 |
| A_neg_none / A_neg_random | ablations, negatives: neural / fused | .352 ; .424 / .362 ; .427 |

Reproduce: `bash phase7/run_phase7.sh lexical` (B0), `lexical-context` (B1), `zeroshot`, `arch`, `context`, `techniques`, `ablations`, then `combine NAME RUN …` and `collect`.
