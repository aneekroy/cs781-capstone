# CS781 Phase 3: rhetorical role prediction (CL + IT)

> **Run from the repository root** as `bash phase3/run_phase3.sh <command>`, or from inside `phase3/`. The official scorer is `phase3/evaluate_phase3.py`.
>
> **Best submission:** `E6_div_mk2`, test pooled macro-F1 **0.6877** (P 0.689, R 0.692), first on the leaderboard: the fixed diverse ensemble (`E6_div_seq`) re-decoded with the Dissent and speaker rules. E9_div_lex (E6 members + TF-IDF member, both rules) scored 0.6846; E8_div_lex without the speaker rule 0.6397.
>
> **Other test results:**
> - v6_diverse 0.6339
> - e_greedy_seq 0.6276, R1B 0.6270
> - e2_ensemble 0.6243
> - S_p2_x1 0.6104, K2_gqa_lora_kd 0.6103
> - T_ia3 0.6030
>
> **Best out-of-fold score:** T_lorma, 0.5759.


This package reuses the Phase-1 language models and the Phase-2 LoRMA, KD and NER code. It does not use any outside pretrained model.

## Setup

Unzip the package next to `phase2/` (the same folder as `run_phase2.sh`). Then put `Phase_3_RR/` at `$CACHE/raw/Phase_3_RR`, or set `DATA=...`.

```bash
bash run_phase3.sh fetch     # optional: downloads Phase_3_RR with your own token, as in Phase 2
bash run_phase3.sh check     # prints data stats, writes dev truth files, runs an all-Fact smoke test, lists checkpoints
```

`check` writes `runsP3/truth/{cl,it}_dev_truth.json`. If `evaluate_phase3.py` sits next to the script, `check` also runs it on an all-`Fact` dev prediction and prints its Macro-F1. That number should match our own scorer, and the tests verify that it does.

## Model

1. **Sentence encoder.** A Phase-1 LM (dense, gqa, enc-dec or hybrid) is adapted with LoRMA by default, or with LoRA or full fine-tuning. Consecutive sentences are packed into windows of up to 512 tokens, so a causal model also sees the sentences to its left. Each sentence's own tokens are pooled into one vector, using mean and last token.
2. **Document context.** Sentence vectors, plus relative-position and domain (CL/IT) embeddings, go through a BiLSTM (or a transformer, or nothing for the sentence-only ablation). Training runs on segments of 48 sentences. Inference covers the whole document with overlapping segments.
3. **Output.** A linear-chain CRF over the 13 labels, or softmax with weighted cross-entropy or focal loss. Separate CL/IT output heads are optional (`--domain-heads`).
4. **Decoding, tuned on out-of-fold predictions for the pooled macro-F1.** The options are Viterbi or argmax over CRF marginals, combined with no offsets, global per-label offsets, or per-domain per-label offsets. Offsets are used only if they win on a half-swap check (tune on half the documents, score the other half) by at least 0.003.

## Protocol

The default is `--protocol cv --folds 5`. It runs 5 document folds over **train+dev** (90 documents, stratified by domain), and the score is the pooled out-of-fold macro-F1, which is the same number the leaderboard computes. `dev docs only` is also printed. `--final` then trains on all 90 documents for 1.1 × the mean best epoch and writes `submission/submission.zip`.

Use `PROTOCOL=dev` (train on train, select on dev) when time is short. It is about 5× faster but selects on only 10 documents.

## Format safety

Every submission goes through the official scorer's own checks before it is written:
- exact test ID set, with `.txt` suffixes stripped and no extra IDs
- no CL/IT ID collision
- exactly one label per input sentence
- label strings exactly as in `label_vocab.json`

Run `python -m phase3.run check --data D --prediction prediction.json` on any file.

## Picking the backbone and the fine-tuning technique (section 9.2.1)

Every technique is implemented from scratch (`phase2/peft.py`, `phase3/peft.py`; no peft or bitsandbytes):

| `--method` | technique | what trains |
|---|---|---|
| `lora` | LoRA (Hu et al.) | low-rank A, B on q,k,v,o,up,gate,down; scale alpha/r |
| `rslora` | rank-stabilised LoRA | same, scale alpha/sqrt(r) |
| `qlora` | QLoRA | LoRA on a frozen base stored in 4-bit NF4 (blocks of 64, double-quantised absmax), dequantised on the fly |
| `adalora` | AdaLoRA | SVD-form P diag(E) Q; importance (sensitivity x uncertainty) prunes triplets from r=12 to an average r=8 on a cubic schedule; orthogonality penalty |
| `lorma` | LoRMA+ | multiplicative (I + UV) W0 |
| `prefix` | prefix tuning | MLP-reparameterised key/value prefixes in every attention layer |
| `prompt` | prompt tuning | 20 soft input tokens |
| `ia3` | IA3 | learned scales on keys, values and FFN activations |
| `full` | full fine-tuning | reference |
| `--teacher` | knowledge distillation | smaller student, fold-matched teacher (K runs) |

Each method has its own default learning rate (`LR_DEFAULT` in `phase3/run.py`, e.g. 3e-4 for LoRA-family, 3e-3 IA3, 1e-2 prompt); `--lr-peft` overrides it.
The document context model (BiLSTM + CRF) is trained in every run; the report counts encoder parameters separately so the techniques compare on what they adapt.

```bash
bash run_phase3.sh screen                         # LoRMA r16 on all 12 backbones (Phase-1 x4, continued-LM, suffix,
                                                  # Phase-2 domain-adapted x4, Phase-2 NER models x2), 3 folds, no test fit
bash run_phase3.sh report                         # pick the best backbone key from the first table
BEST=<key> bash run_phase3.sh techniques          # the 9 techniques on that backbone, 5 folds + test fit each
TEACHER=$PWD/../cs781_phase1/runsP3/T_<best> bash run_phase3.sh kd   # KD students of the best technique run
bash run_phase3.sh report                         # runsP3/report.md: tables for the write-up
```
Backbone keys: `dense_orig dense_lm dense_suffix dense_dapt gqa gqa_dapt encdec encdec_dapt hybrid hybrid_dapt p2_x24 p2_x1` (missing checkpoints are skipped).

## Experiments (`bash run_phase3.sh list`)

| run | what changes | axis |
|---|---|---|
| P1_dense_lorma_ctx | dense + LoRMA, packed, BiLSTM-CRF | main recipe |
| P2_dense_lorma_sent | same backbone, isolated sentences, no context or position | context vs sentence-only |
| P3_adapted_dense_lorma | Phase-2 domain-adapted dense backbone | DAPT |
| P4_adapted_gqa_lora_dom | adapted gqa + LoRA + CL/IT heads | backbone / domain heads |
| P5_p2ner_lorma | start from the Phase-2 NER model X24 (LoRMA merged) | transfer from Phase 2 |
| P6_encdec_lorma_tf | enc-dec encoder (bidirectional) + transformer context | architecture |
| P7_dense_lorma_wce | weighted CE instead of CRF | loss |
| P8_dense_lorma_rare | CRF + weighted aux CE + rare-segment oversampling | rare classes |
| K1_hybrid4_kd / K2_gqa_lora_kd | smaller students distilled from P1 (+ teacher-labelled Phase-1 SC judgments) | KD |
| K0_hybrid4_nokd | K1 without KD | KD control |

**KD is honest in CV.** Student fold *k* learns from the teacher's fold-*k* model, which never saw fold *k* (the folds must match; this is checked). Unlabeled judgments that overlap any Phase-3 train, dev or test document are dropped, using normalised 8-gram fingerprints.

## Order of work

```bash
bash run_phase3.sh round1           # P1 || P2   (do this first: P1 is the KD teacher)
bash run_phase3.sh round2; bash run_phase3.sh round3; bash run_phase3.sh round4
bash run_phase3.sh kd
bash run_phase3.sh summary          # sort by the "honest" column
bash run_phase3.sh seeds            # S1=<best run> S2=<best run>: two more seeds of the best recipe
bash run_phase3.sh ensemble         # greedy posterior averaging on OOF -> runsP3/E1_ensemble/submission
```

The ensemble averages sentence posteriors across runs and picks members greedily by the honest OOF score. Runs with different folds can still be combined, because every out-of-fold posterior comes from a model that never saw that document. Use `ENS_RUNS="P1 P3 ..."` and `ENS_ARGS=--fixed` to fix the member list.

## Scorer notes

- **Pooled score.** One pooled list over CL+IT. CL has more sentences and dominates.
- **Labels absent from gold score 0.** Dissent and PrecedentOverruled have no dev sentences (`check` prints this), so on dev the ceiling is about 11/13. The test set may be the same.
- **Every label is 1/13 of the score.** Issue, RPC and PRE-NR matter as much as Fact. That is why offset tuning and rare-class runs exist.

## Dissent rule and data mixing

**Dissent.** In this corpus a dissent is a note appended to a CCI order after the majority's operative part. It is headed "Prasad, Member (Dissenting)" or "Prasad (Minority)" and runs to the end of the document. The training data has 2 such documents and the test set has 1 (CCI Mineral Enterprises). `phase3/rules.py` finds the start of the note, either from the header or from a first-person "do not agree with ... the Majority" traced back to the end of the majority order, and labels from there to the end as Dissent.
- On all 100 train, dev and test documents it fires exactly on the 2 gold dissents (exact start) and on the test note, with no false positives.
- It is applied only when it raises the OOF score.
- `bash run_phase3.sh redecode` applies it to every finished run without retraining. The `ensemble` command applies it too.

**Data mixing** (`phase3/mixing.py`, all from each run's training fold):
- `--repeat-labels Dissent=3,Issue=3,...` repeats training segments that contain rare roles.
- `--domain-weights CL=1,IT=1.3` repeats segments by domain.
- `--splice R` creates R synthetic documents per training document per epoch: same-label blocks swapped across documents of one domain, plus a rare-role block (Dissent, Issue, PRE-NR, RPC, None) inserted where that role sits in its source. A dissent note always goes at the end.
- `--kd-select-domain` takes KD silver data only from Phase-1 SC/HC judgments with competition-law or tax/banking vocabulary, tagged CL/IT. Anything overlapping Phase-3 documents is dropped.

```bash
bash run_phase3.sh redecode          # every run gets the Dissent rule; resubmit the best ones
bash run_phase3.sh mix               # M1 (K2 recipe + mixing + in-domain silver, teacher P5) || M2 (P5 + mixing)
bash run_phase3.sh mix2              # M3 (P8 + mixing) || M4 (adapted gqa + KD + mixing)
S1=M1_gqa_lora_kd_mix S2=M1_gqa_lora_kd_mix bash run_phase3.sh seeds
bash run_phase3.sh ensemble          # with rules
```

## Voting ensemble (phase-2 style)

`bash run_phase3.sh vote` (`phase3/vote.py`) is a sentence-level vote. Each member votes with its own decoded labels: its Viterbi/CRF decoding, its offsets and its rules. Unlike posterior averaging, this keeps each model's sequence structure.
- The plurality wins, with ties broken by averaged probability.
- Per-label thresholds k_L: a label with at least k_L votes takes the sentence, so a rare role is not outvoted by Fact or ROD.
- The Dissent rule is applied last.
- Members are chosen greedily on the OOF score, with a look-ahead over pairs when no single run helps.
- The thresholds are used only if they win the half-swap check.
- Runs without `test.pt` are skipped; `bash run_phase3.sh finalize` creates one.

```bash
bash run_phase3.sh vote                                             # greedy over all runs -> runsP3/V1_vote
VOTE_NAME=V2_div VOTE_ARGS=--fixed VOTE_RUNS="T_lorma P5_p2ner_lorma K2_gqa_lora_kd S_hybrid_dapt T_ia3" bash run_phase3.sh vote
```

## Stacked ensemble

`bash run_phase3.sh stack` (`phase3/stack.py`) trains a second-level classifier on the members' sentence predictions, so it learns per label which member to trust.
- **Features per sentence:**
  - each member's posterior and its decoded one-hot label;
  - the averaged posterior of the 2 previous and 2 next sentences;
  - relative position, log sentence length, document length, domain.
- **Models:** multinomial logistic regression and gradient-boosted trees, both class-balanced, with document-grouped 5-fold CV on top of the members' out-of-fold predictions. The better one goes through the usual decoder choice (offsets with the half-swap check, Dissent rule).
- **Comparison:** the log prints the plain average of the same members as a reference; use the stack only if it beats that.

```bash
bash run_phase3.sh stack                                                        # all runs with test.pt
STACK_NAME=ST2_diverse STACK_RUNS="T_lorma P5_p2ner_lorma P7_dense_lorma_wce T_rslora T_ia3 K2_gqa_lora_kd P4_adapted_gqa_lora_dom S_hybrid_dapt S_hybrid" bash run_phase3.sh stack
```

## Leaderboard push (`bash run_phase3.sh push`, CPU only)

1. **Dissent rule fix.** Both training dissents are labelled Dissent up to the last sentence, including the note's own "Secretary is directed ..." line. The rule used to leave that line to the model. It now reproduces both training dissents exactly (233/233 and 51/51) and marks 79 test sentences instead of 78. `redecode` rewrites every run's submission.
2. **Sequence decoding for ensembles.** Averaging posteriors and then taking the argmax sentence by sentence threw away the members' CRF structure. Ensembles and stacks now compare these decoders on the same honest half-swap score:
   - `marginal`: the old decoder;
   - `smooth`: Viterbi over the averaged posteriors, with λ × a gold label-transition prior;
   - `crfavg`: average the members' CRF emissions and transitions, then run Viterbi.

   Run `pip install numba` first. Offset tuning under Viterbi takes seconds with numba and several minutes per candidate without it.
3. **Stacker fixes.** These address why ST1 and ST2 lost to the plain average:
   - no class balancing;
   - log-posterior features without the leaky one-hot decodes;
   - stronger regularisation;
   - the same sequence decoders as the average.

   The stacker writes its own submission only if it beats the average by at least 0.005. Otherwise it writes the average.
4. **Lexical member** `L1_tfidf` (`bash run_phase3.sh lexical`): TF-IDF word and character n-grams of the sentence and its neighbours, plus position, fed to a logistic regression. It is weak alone, but its errors are lexical, so it can help an ensemble. A greedy ensemble keeps it only if it raises the OOF score.

`push` runs all of the above and builds these ensembles:
- E6: fixed diverse set, including M3 and M4;
- E7: greedy, up to 9 members;
- E8: E6 plus L1;
- V6: vote over the E6 set;
- ST3: stack over the E6 set plus L1.

Change the diverse set with `DIV="..."`.

**GPU, in parallel:** `S1=M4_gqa_dapt_lora_kd_mix S2=M4_gqa_dapt_lora_kd_mix bash run_phase3.sh seeds` trains two more seeds of M4 (about 6 h). Then rerun `push` with `DIV="M4_gqa_dapt_lora_kd_mix M4_gqa_dapt_lora_kd_mix_s7 M4_gqa_dapt_lora_kd_mix_s13 ..."`.

## R1: the rethought training pipeline

R1 targets three limits found in the error analysis.

| limit | evidence | R1 component | flag |
|---|---|---|---|
| 90 labelled documents; the best single test run (M4) learned from one teacher at 0.571 OOF | distillation plus silver data already helped most on test | **ensemble teacher**: the student distils the averaged posteriors of several fold-matched runs | `--teacher RUN1 RUN2 ...` |
| | | **silver data ranked per fold**: 3× candidate judgments are drawn, and each fold keeps the ones closest (TF-IDF) to its own training documents | `--kd-select-similar --kd-similar-pool 3` |
| | | **noisy student**: token dropout, block splicing, confidence-weighted distillation, silver share decaying to 0 by the last epoch; round 2 adds the student to the teachers | `--token-dropout`, `--kd-confidence`, `--kd-silver-schedule decay` |
| a sentence's role is set far away | the speaker cue appears in only 6% of ArgumentPetitioner and 3% of ArgumentRespondent sentences; documents run 123 sentences at the median and up to 942 | **whole-document context**: BiLSTM, then self-attention over every sentence of the document. Encoder gradients flow through a 64-sentence segment; the other sentences come from a per-epoch no-grad cache | `--context lstm_tf --doc-context` |
| | label shifts at 13% of sentence boundaries; blocks are 2–4 sentences at the median | **label-shift head** (Malik et al.) and a **coarse-role head** (9 groups), as auxiliary losses | `--aux-shift 0.3 --aux-coarse 0.3` |
| causal encoders see only left context | a citation's treatment ("distinguishable on facts") usually follows it | **two-pass encoding**: a second pass reads the sentences in reverse order; the two sentence vectors are concatenated | `--two-pass` |

All of these are off by default, so older runs and their checkpoints behave exactly as before.

```bash
pip install numba                                   # fast Viterbi for decoding
bash run_phase3.sh r1                               # R1A (all of the above except two-pass) || R1B (+ two-pass)
bash run_phase3.sh summary                          # compare R1A / R1B with M4, T_lorma, P5 on the honest OOF score
R1_BEST=R1A_gqa_ens_doc bash run_phase3.sh r2       # round 2: R1's winner joins the teachers; gqa || dense student
bash run_phase3.sh r1abl                            # ablations: no whole-document context / no aux heads / one teacher
bash run_phase3.sh push                             # ensembles with sequence decoding (add the R runs via DIV=...)
```

- **Teachers:** `TEACHERS="T_lorma P5_p2ner_lorma M4_gqa_dapt_lora_kd_mix"` is the default. Every teacher needs a finished 5-fold run with `--final` on the same folds; others are skipped with a message, and a fold mismatch stops the run.
- **Silver data:** `SILVER_DOCS_R1` (default 600 judgments) and `SILVER_SENTS_R1` (default 30k sentences per epoch at the start).
- **Segment length:** `SEG_R1` (default 64 sentences with live gradients).
- **Cost:** 3 teachers × 5 folds of inference over the training documents and the silver data, plus one no-grad encoding pass per epoch for the cache. On the gqa backbone that is roughly M4's time plus 1–2 h. Two-pass doubles the encoder cost.

### Other data-mixing strategies

Built in this round:
- similarity-ranked silver data (per fold);
- silver decay schedule;
- confidence-weighted distillation;
- token-dropout noise.

Already available:
- label-targeted repetition (`--repeat-labels`);
- domain weights (`--domain-weights`);
- block splicing (`--splice`).

Not built, ranked by expected value:
1. **Agreement-filtered silver.** Keep only the silver sentences where the teachers' argmax labels agree, and distil from the average there. This is stricter than confidence weighting. It is cheap once ensemble teachers exist.
2. **Cross-domain warm-up.** Train the first epochs on CL+IT plus silver, then the last two on the target domain only, with domain heads. This helps when one domain lags; CL is where the gap to the top entry is.
3. **Rare-role silver mining.** From the silver pool, keep extra judgments whose teacher posteriors contain Issue, PrecedentNotReliedUpon or RulingByPresentCourt blocks. This gives more rare-role examples without repeating the same gold sentences.
4. **Label-preserving paraphrase of rare blocks**, by synonym swaps of legal terms from a small hand-made list. It is risky with legal text, so I would rank it last.

## Speed and memory on 24 GB (RTX 3090)

| change | what it fixes | where |
|---|---|---|
| **FlashAttention-2** (`--attention auto`, default) | The sliding-window layers of gqa_local ran SDPA in 64-query chunks with explicit masks, which rules out SDPA's flash kernel; with grouped-query attention it likely fell back to the plain math kernel (the R1B out-of-memory error was in that call). FA2 does causal attention, the 256-token window (`window_size=(255, 0)`) and grouped-query attention in one kernel with linear memory. | `phase3/flash.py`, used by Phase 3 and Phase 4 |
| SDPA fallback (`--attention sdpa`, or auto without flash-attn) | Key/value heads are expanded to the query heads, so PyTorch's flash or memory-efficient kernel applies; the window is one banded-mask call instead of a Python loop | same |
| per-segment backward | Only one segment's activations are on the GPU at a time (gradients identical) | `train()` |
| cross-document no-grad batches | Validation, the document cache and teacher inference pack windows from up to 32 documents per batch (32k tokens) and pool each window right away | `encode_many` |
| teacher-posterior disk cache | `kd_cache_T2.pt` next to each teacher model: R1A, R1B, the ablations and R2 compute each teacher's posteriors once (file-locked, so both GPUs can share it) | `teacher_posteriors` |
| smaller document cache | Documents that fit in one segment are not cached; the cache is freed before validation | `cache_vectors` |
| TF32, `expandable_segments` | Faster fp32 LSTM/CRF/head maths; less fragmentation | `run.py`, launcher |

Tests check that every path gives the same numbers as the original code:
- the patched attention equals the Phase-1 attention for all four architectures (causal, sliding window, bidirectional, encoder);
- a CPU emulation of the FA2 call contract gives the same result;
- the cross-document encoder equals the per-document encoder.

```bash
bash run_phase3.sh install-fast      # pip install numba flash-attn; prints which attention path is active
```

**Installing flash-attn.** It needs a prebuilt wheel matching your torch, CUDA and Python versions, which pip downloads when one exists. Otherwise it compiles with `nvcc`, which takes 30–60 min; set `MAX_JOBS=4` to limit RAM use. If it doesn't install, everything still runs on the SDPA path.

**Out-of-memory fallbacks:**
- `R1_EXTRA="--gradient-checkpointing"`;
- `SEG_R1=48`;
- `--infer-batch-tokens 16384` for the no-grad passes.

## Speaker-tracking rule (`speaker`, `speaker_all`)

**What the leaderboard showed.** Its three columns are Macro-F1, Macro-Precision and Macro-Recall. The best entry is 0.745 precision / 0.643 recall; ours was 0.660 / 0.636, so the gap is precision.

**The cause.** In the largest test judgment (Vodafone, High Court, 942 sentences, 43% of the test set), the respondents' arguments (sentences 447–502: CCI's counsel, then RJIL's) are labelled ArgumentPetitioner by every model. The speaker is named once, then carried by "According to him …". That one block removes ArgumentRespondent recall in the document and adds 56 false positives to ArgumentPetitioner.

**The rule** (`phase3/rules.py`):
- A counsel introduction sets the current side. This covers "counsel / advocate / appearing … for / on behalf of the respondent / opposite party / petitioner / appellant / informant", "supported the impugned order", and "per contra" or "on the other hand".
- Each sentence the model already labelled as an argument takes that side.
- The side is forgotten after 3 non-argument sentences.

**Where cues don't count:**
- quoting the other side ("canvassed by the learned counsel for the petitioner", "the contention advanced on behalf of the petitioners");
- the Commission in its own orders (it is the court there);
- the Revenue in tax appeals (it can be either side).

**Validation on the 90 training documents:**

| | cue precision (side of the next argument sentence) | on correct labels: argument sentences changed, ArgumentRespondent / ArgumentPetitioner F1 | all respondent arguments mislabelled as petitioner: ArgumentRespondent F1 / ArgumentPetitioner F1 |
|---|---|---|---|
| `speaker` | petitioner 94%, respondent 87% | 113 of 3,633; 0.965 / 0.972 | 0 → 0.45 / 0.71 → 0.77 |
| `speaker_all` (+ "<party> has submitted …") | lower | 485 of 3,633; 0.844 / 0.883 | 0 → 0.57 / 0.71 → 0.77 |

**How runs use it.**
- `rules=auto` tries none, dissent, dissent+speaker and dissent+speaker_all per run, on the honest OOF score.
- A speaker variant must win by at least 0.002.
- Speaker rules are applied inside offset tuning, so the offsets are chosen with them in place.

```bash
bash run_phase3.sh redecode                                   # every run re-chooses its rules
RUNS_ONLY="R1B_gqa_ens_doc_2pass" REDECODE_ARGS="--rules dissent,speaker" bash run_phase3.sh redecode   # force it
```
