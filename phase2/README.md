# CS781 Phase 2 — Legal NER on the Phase-1 models

> **Run from the repository root** as `bash phase2/run_phase2.sh <command>`, or from inside `phase2/`. The launcher puts the repository root on `PYTHONPATH`.
>
> **Best submission:** `V24_all`, test strict macro-F1 **0.6626**. It is a CV-selected span vote (k = 2 of 5) over two seeds of X24_dense_coarse, X25_gqa_clm_coarse and two seeds of X22_gqa_clm_header_aug. Out-of-fold score 0.6615.
>
> **Milestones on test:**
> - X1_denseorig_lorma: 0.5962 (best plain single model)
> - X24_dense_coarse_s13: 0.6557 (merged-role training)
> - V14 vote: 0.6441
> - V23_greedy vote: 0.6616
> - V24_all vote: 0.6626
>
> **Other test results:**
> - DAPT: A2 encoder MLM 0.549 (vs 0.392 without); A3 gqa 0.574 (vs 0.538)
> - KD: E4f 37M student 0.572 (vs X9 without KD 0.464); X8 54M student 0.577
> - Failures: X20 O-penalty 0.310; X26 silver 0.611


`phase2/` sits next to `phase1/` in `$PROJECT` (default `$HOME/cs781_phase1`) and loads Phase-1
checkpoints directly (run-1 `best.pt` files and the run-2 stage checkpoints use the same
`{"model", "resolved"}` format). Only `torch`, `tokenizers` and `numpy` are needed — no
pretrained weights, no peft/transformers.

```
phase2/
  data.py   loading, end-offset convention detection, stats, BIOES alignment
  peft.py   LoRA, LoRMA+, adapters, prompt tuning, norm-only (BitFit-style), partial, full
  model.py  Encoder wrapper (causal / bidirectional / encoder), BiLSTM/linear heads, CRF, biaffine span head
  post.py   strict metrics, error breakdown, boundary rules, label vote, propagation, regex/gazetteer rules
  run.py    CLI: stats | cv | predict | ensemble
  fetch.py  gated-dataset download with explicit token + access diagnosis
  adapt.py  continued pretraining (CLM / encoder MLM) on official Phase-1 text
  vote.py   span-voting ensemble with members and threshold chosen on out-of-fold CV
  overlap.py  finds NER documents inside the Phase-1 corpus (word 12-gram fingerprints)
tests/      synthetic fixture + 38 unit tests (CPU, no network)
run_phase2.sh   launcher with your checkpoint paths (bash run_phase2.sh with no argument lists the commands)
```

## Install

```bash
cd $PROJECT
unzip cs781_phase2.zip            # adds phase2/, tests/, run_phase2.sh, README_phase2.md
python -m pytest -q tests/                    # 38 tests, CPU, ~1 min (builds a fixture in /tmp/fx_test)
```

## The six backbones (all share one tokenizer, sha `cd4848…`)

| Name in script | Checkpoint | Notes |
|---|---|---|
| `DENSE_LM` | `improvements/dense_v2/runs/dense_lm_dense_v2_s42/best.pt` | run-2 continued LM on cleaned SC/HC — **dense backbone** |
| `DENSE_SUFFIX` | `improvements/dense_v2/runs/dense_suffix_dense_v2_s42/latest.pt` | run-2 Phase-1 winner; loss only on continuations |
| `DENSE_ORIG` | `runs/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt` | run-1 dense |
| `GQA` | `runs/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt` | 3 of 4 layers use a 256-token window |
| `ENCDEC` | `runs/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt` | encoder stack is bidirectional — NER teacher |
| `HYBRID` | `runs/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt` | 10 layers, LSTM in even layers |

Because the tokenizer is shared, any model can teach any other (KD) and any token-head runs
can be ensembled. `run_phase2.sh check` re-verifies the hashes on the server.

## Order of work

```bash
bash run_phase2.sh access    # which HF account the token is, and whether it can read the gated dataset
bash run_phase2.sh fetch     # Phase_2_LNER -> $CACHE/raw/Phase_2_LNER
bash run_phase2.sh check     # checkpoints, tokenizer identity, label table, alignment ceiling
bash run_phase2.sh round1    # E2 enc-dec full FT  ||  E1 dense_lm_v2 + LoRMA+ (causal)
bash run_phase2.sh round2    # E3 gqa + LoRA       ||  E4 hybrid KD student
bash run_phase2.sh extras    # X1-X9 ablations (one variable each)
bash run_phase2.sh ensemble  # X11: X1 + E1 + X2 + E3 averaged token log-probs
bash run_phase2.sh overlap   # report statistic: NER documents found inside the Phase-1 corpus
bash run_phase2.sh round3    # E4f (KD without leak, teacher X1) || X9 (no-KD control), then X10 || X8
bash run_phase2.sh round4    # X20 O penalty || X17 header, then X18 augmentation || X19 header+augmentation
bash run_phase2.sh round5    # X21 dense: all + self-KD || X22 adapted gqa: header+augmentation
bash run_phase2.sh round6    # X23: X1 recipe + entry bonus, seeds 42/13/21/7
bash run_phase2.sh repost X1_denseorig_lorma   # re-choose post-processing with the current rules (CPU)
bash run_phase2.sh selfkd    # X14 X1 -> X1 self-distillation || X15 X1 -> adapted gqa (judge on test)
bash run_phase2.sh seeds     # X1 with seeds 7/13/21, A3 with seed 7 (any name + _s<N> works with `one`)
bash run_phase2.sh vote      # CV-chosen span-voting ensemble over every finished run (CPU, minutes)
bash run_phase2.sh summary   # CV table across all runs
```

GPUs default to 0 and 1 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; override with `GPU_A=2 GPU_B=3`. A finished run (one with `submission/submission.zip`) is skipped on
rerun. Each submission is `runsP2/<run>/submission/submission.zip`.

Hugging Face token for `access`/`fetch` (never printed): `$HF_TOKEN` if set, else
`cs781_cache/huggingface/token`, else a hidden prompt that can save it there (mode 600).
The machine-wide `~/.cache/huggingface/token` is ignored on purpose — on a shared login it
may belong to another account. `bash run_phase2.sh fetch --reset-token` forgets a saved token.
A 403 "not in the authorized list" prints the account name and what to check.

What to read in `check`:

* **end offsets**: output is written in the same convention as the gold files.
* **alignment ceiling** should be 1.0000; anything lower is strict F1 lost to tokenization —
  the printed `misaligned` examples show why.
* **mention cov** near 1.0 means every mention is annotated, so `propagate` should help.
* **document tokens**: whole documents are covered by 512-token windows, stride 256.

## Required experiments and ablations

| Run | Backbone | Method | Role | Test F1 |
|---|---|---|---|---|
| E1 | dense_lm_v2, causal | **LoRMA+** r=16, BiLSTM-CRF | compulsory method | 0.573 |
| E2 | enc-dec encoder | full fine-tuning | full-FT baseline | 0.392 |
| E3 | gqa_local, causal | LoRA r=16 | PEFT comparison | 0.538 |
| E4 | hybrid, 2 of 10 layers | full FT + **KD** from E2 (+ teacher-labelled SC text) | KD (CV leaked, see below) | 0.448 |
| E4f | hybrid, 2 of 10 layers | full FT + **KD** from X1, NER documents removed from KD text | compulsory KD, fixed | — |
| X1 | dense run-1, causal | LoRMA+ | did run-2 continued training help NER? | **0.596** |
| X2 | dense_suffix_v2 | LoRMA+ | does the suffix objective hurt representations? | 0.554 |
| X3 | dense_lm_v2, **bidirectional** | LoRMA+ | causal vs bidirectional (vs E1) | 0.467 |
| X4 | enc-dec | full FT, **biaffine span head** | span vs BIOES-CRF (vs E2) | 0.242 |
| X6 | enc-dec | full FT, `--segment bpe` | segmentation (vs E2) | 0.411 |
| X7 | enc-dec | LoRMA+ | PEFT vs full FT on the same backbone (vs E2) | 0.480 |
| X8 | hybrid, 4 layers | KD | student size | — |
| X9 | hybrid, 2 layers | full FT, **no KD** | what KD adds (vs E4f) | — |
| X10 | dense run-1 | LoRMA+, `--segment bpe` | segmentation on the best model (vs X1) | — |
| X11 | ensemble X1+E1+X2+E3 | — | extra submission | — |

**KD leak (E4).** The official Phase-1 Supreme Court corpus contains the NER judgments. E4's
teacher had been trained on both folds, so teacher-labelled "unlabeled" text included the
validation fold's own documents: CV 0.75 vs test 0.448. The KD text and continued-pretraining
data now skip every corpus document sharing ≥~160 running words with any NER document (train or
test); `bash run_phase2.sh overlap` gives the numbers for the report. E4f is the honest KD run.

The KD teacher defaults to X1 (best test score); override with `TEACHER=runsP2/<run>/model.pt`.

## How a run works

`cv` trains fold1→fold2 and fold2→fold1 (true out-of-fold), early-stops on strict macro-F1,
then greedily picks post-processing on the pooled out-of-fold predictions. With `--final`
it retrains on both folds for ≈1.15 × the CV-best epoch count, predicts fold 3 and writes a
validated `submission.zip` (root-level `prediction.json`, every test ID, offsets in range,
known labels only).

Per run: `cv_summary.json` (best epochs, per-label P/R/F1, error breakdown into boundary /
label-confusion / missed / spurious, parameter counts, seconds and peak GB per epoch,
post-processing trace), `model.pt`, `submission/token_logprobs.pt` for ensembling.

## Starting hyper-parameters

| Method | LR (encoder) | LR (head) | Epochs |
|---|---|---|---|
| full / partial / normfit | 5e-5 | 2e-3 | ≤20, patience 6 |
| LoRA / LoRMA / adapter / prompt | 3e-4 (1e-3 diverged on E1) | 2e-3 | ≤20, patience 6 |

Window 512 / stride 256. The encoder–decoder's encoder only saw 569–692-token inputs in
Phase 1, so do not raise its window much past ~640. Batch 8 fits a ~100M model on a 3090 in
bf16 without gradient checkpointing (`--gradient-checkpointing` if it does not).

## What the strict metric punishes and how the code handles it

1. Byte-level BPE offsets include the leading space — trimmed in `encode_document`.
2. Whitespace tokens inside an entity (double spaces, newlines) are tagged `I-`, so gold
   tag sequences are always valid BIOES.
3. Training windows cut through entities — the CRF loss skips start/end constraints on
   windows; document-level Viterbi applies them.
4. End-offset convention is inferred from `entity_text` and mirrored on output.
5. Window edges have one-sided context — overlapping windows are averaged with edge
   down-weighting, then decoded once per document.

## Continued pretraining before NER (more tokens per parameter)

The NER labels are fixed, but the backbones are under-trained: tokens seen per parameter in
Phase 1 were dense 3.8, enc-dec 3.6, hybrid 0.77, gqa 0.55 (compute-optimal is ~20). This
stage starts from each Phase-1 checkpoint and trains it further on official text only:

| Backbone | Objective | Why |
|---|---|---|
| `encdec_mlm` | masked-token prediction on the **encoder** (15%: 80% `<unk>`-as-mask / 10% random / 10% kept) | NER uses only the encoder; every token gets a both-sided training signal. The decoder is left untouched (keep the original checkpoint for generation phases). |
| `dense_clm`, `gqa_clm`, `hybrid_clm` | causal LM, as in Phase 1 | more tokens; gqa and hybrid gain most |

Data (`adapt-data`, CPU only): ~1B tokens from the official `train_*.json` files present
(SC 0.5, HC 0.3, Acts 0.1, Tribunals 0.1, renormalised), plus the NER **training** folds'
text sampled for 2% of sequences (task-adaptive pretraining). Corpus documents that contain an
NER document are dropped (`excluded_ner_overlap_documents` in `meta.json`), so the NER test
text is never used.
Tokenized with the same punctuation segmentation as the NER runs, 512-token blocks.
Schedule: AdamW, peak LR 1e-4, 200 warmup steps, constant, linear decay over the last 20%,
sized to `ADAPT_HOURS` from the measured speed after 30 steps.

```bash
bash run_phase2.sh adapt-data                 # CPU; can run while GPUs train
bash run_phase2.sh adapt encdec_mlm 0 &       # ~10 h each (ADAPT_HOURS)
bash run_phase2.sh adapt gqa_clm 1 &  wait
bash run_phase2.sh adapt dense_clm 0 & bash run_phase2.sh adapt hybrid_clm 1 & wait
bash run_phase2.sh adapted                    # A1-A4: same NER recipes on adapted weights
```

Each adapted backbone is a normal checkpoint in `runsP2/adapt/<name>/best.pt`; compare A*
with E* (same recipe, different starting weights) to measure what the extra tokens bought.
For the report: present this as domain/task-adaptive pretraining (Gururangan et al., 2020)
that starts from the Phase-1 model, and log start/final validation loss from `adapt_summary.json`.

## Choosing an ensemble without spending submissions (`vote`)

Each run's `oof_predictions.pt` holds out-of-fold spans for all training documents, so any set
of runs can be scored by span voting: a span is kept when at least k members predict exactly
it (overlaps go to more votes, then the longer span). Members are added greedily from the best
single run while pooled out-of-fold macro-F1 rises; post-processing is re-chosen on the voted
spans. Runs with different tokenizers or segmentations can vote together (E2/A2 with the dense
models, X10 with X1). KD runs are always refused: the teacher's final model was trained on
both folds, so a student's out-of-fold score is leaked even with no NER document in the extra
text (E4f: CV 0.80, test 0.57). Judge KD runs on test only. `VOTE_RUNS="..."` limits the candidates, `VOTE_ARGS=--fixed` keeps
all of them, `VOTE_NAME` names the output. The selection is made on the same out-of-fold data it is
scored on, so expect the test score to fall a little short of `oof_macro_f1_post`.

## Round 4/5: context, augmentation, recall (flags of `phase2.run cv`)

* `--header-tokens N` (launcher `HEADER`, default 160): every window after the first is fed as the
  document's first N tokens + `window - N` document tokens; outputs for the header are dropped.
  Party roles (APP / RESP, and WIT by elimination) are stated in the header, which a mid-document
  window otherwise never sees. Total length stays at `--window`, so dense's learned position
  embeddings stay in range.
* `--augment K` (launcher `AUGMENT`, default 2): K mention-replaced copies of every training
  document (Dai & Adel, 2020). Each mention is swapped for another training mention of the same
  label, consistently within a document, case-matched; overlapping mentions are kept. Built from
  the training fold only, so CV stays clean.
* `--o-bias auto`: a bonus for ENTERING an entity (CRF transitions O->B/S and document start),
  tuned on pooled out-of-fold scores: first one value for APP/RESP/WIT (the under-recalled party
  roles), then one for all other labels, from {0, 0.5, 1, 2, 3}. Stored as {label: bonus} in
  `model.pt` / `token_logprobs.pt` and applied by `predict`, `ensemble`, `vote`, `repost`.
  The first version (X17-X20, X22) instead lowered the "O" score at every token; that makes
  continuing an entity as cheap as starting one, so Viterbi merged neighbouring entities into
  long spans (X20: test 0.31). It is withdrawn: those runs' stored float is ignored, and `vote`
  skips them. Rerun as X23.

X17-X20 change one thing each relative to X1 for the report; X21/X22 combine them.

## Post-processing rules (all chosen on out-of-fold CV, per label where it applies)

boundary, label vote, exact propagation, regex/gazetteer rules (original), plus:
`counsel` (A./R.COUNSEL from the nearest "for the appellant/petitioner/respondent" cue),
`propagate_ci` (case-insensitive, whitespace-tolerant propagation), `parts` (first name /
surname of a predicted multi-word name, capitalised, owned by one label in the document),
`prec` extend|trim (a citation such as "(2003) 5 SCC 120" / "AIR 1990 SC 123" after or at the
end of a PREC span). `repost` applies the current rule set to any finished run from its saved
out-of-fold spans and test scores, without retraining; `--keep-config` reproduces the original
submission exactly.

## Second-stage reranker (inside `vote`) and error dumps

`vote` now trains a gradient-boosted span classifier (scikit-learn) on the out-of-fold candidates
after voting + rules: keep, relabel or drop each span from model support, form, position, cue
words around it and around every other mention of the same string. It is scored by document-
fold swap and used only when it beats plain voting by 0.3 points (`VOTE_ARGS=--no-rerank` turns
it off); `vote_summary.json` records both scores. Motivation: V5 predicted 554 APP / 127 RESP on
test where a split of this size has about 204 / 190 - names are found, roles are not.

`bash run_phase2.sh errors RUN APP RESP WIT` prints real out-of-fold misses, false hits and label
confusions with 70 characters of context each (training text only), to write targeted rules.

## Party and witness rules (from the real out-of-fold errors)

The error dump showed that APP / RESP are document-level roles: the parties named in the header
(people, companies, "Union of India", a Directorate ...) and every later mention of them, often a
first name alone; WIT is a person who testified (PW / LW / CW markers) and every mention of them.
`parties` parses the header ("X ..... Appellant(s) VERSUS Y .....Respondent(s)", "PETITIONER: X
RESPONDENT: Y", "X ..... Petitioner Through:") and labels every mention with the header's side,
overriding the model's role (never inside PREC / STAT / CASENO / DATE / COURT / JUDGE spans);
`parties:first` adds first names of people; `parties:only` drops APP/RESP spans that match no
party. `witness` does the same for names next to PW/LW/CW markers. All are chosen on CV, by
`cv`, `vote` and `repost`.

## Which runs `vote` uses

* members (CV selection, k, rules, reranker): every run with honest out-of-fold predictions,
  including X18 (its legacy 'O' penalty is reproduced at test time so CV and test match) and the
  span head X4 (its post-processed test predictions are used).
* extra voters (test-time only): KD runs trained with the NER-overlap filter (E4f, X8, X14, X15,
  X21). Their out-of-fold scores are leaked, so they never steer selection; at test time a span the
  members would drop is still kept when members + extras give it the same vote share CV chose
  (k of n -> ceil(k/n x (n+m)) of n+m). `--no-extras` turns this off.
* skipped unless `--allow-unfiltered-kd`: KD runs from before the filter (E4, A4), whose
  teacher-labelled text contained test documents.
* every member's test spans are decoded with its own learned CRF transitions (as its CV spans were).

## Per-label voting (`phase2/labelvote.py`, inside `vote`)

Macro-F1 averages per-label F1, so every label gets its own member set (greedy on that label's
out-of-fold F1) and its own threshold k. It is chosen on fold-1 documents and scored on fold-2
(and back); it replaces the global vote only if that swap score beats the global vote's swap score
by 0.3 points (`VOTE_ARGS=--no-per-label` turns it off). Overlap resolution is fully deterministic
(identical output for any PYTHONHASHSEED), so a vote is reproducible.

## Judge, counsel-side and case-number rules (from the second error dump)

`judges`: names before ", J." / ", JJ." / "(J)", after "Justice", in "(NAME) JUDGE" sign-offs and in
Indian-Kanoon "BENCH: SURNAME, GIVEN ..." lists; every mention becomes JUDGE. `counsel` now
prefers the last header-style cue before a name ("For the Appellant(s) :", "Counsel for
Petitioner :-", "Respondent Through:", "For Appellant(s) Mr. ...") and otherwise the inline cue after
it ("learned counsel for the first respondent", "for the <header party>"). `caseno:split` splits a
CASENO span holding two numbers after each "of YYYY". All are chosen on CV.

## Merged-role training (`--coarse-roles`, round 7: X24 dense, X25 adapted gqa)

The person-role labels fail because the side (appellant vs respondent, whose counsel) is decided by
the header, thousands of tokens from most mentions, while "this is a party / counsel mention" is
local. With `--coarse-roles` the tagger learns PARTY (APP+RESP) and COUNSEL (A./R.COUNSEL) from
twice the examples per class; `resolve_roles` then assigns sides from the header parties (exact,
contained, or shared distinctive word), adjacent cue words ("the appellant X", "X, the
respondent"), public bodies (-> RESP), wider cues, else APP; counsel via `counsel_side`.
Everything (early-stopping aside) is scored on the 12 original labels; test raw spans are saved
as `submission/raw_spans.json`, which `vote` uses for these runs.

Header parser (used by `parties`, `counsel` and `--coarse-roles`): besides "X ..... Appellant(s)
VERSUS Y .....Respondent(s)" and "PETITIONER: X RESPONDENT: Y", it reads role-word-after layouts
without dots ("X APPELLANT(S) VERSUS Y RESPONDENT(S)", "X …. APPELLANT"), role-first layouts
("APPLICANT : X, S/o", "Petitioner :- X Respondent :- Y", "BETWEEN: 1. X ... AND: Y") and short
titles ("<case no.> X Vs. Y For the petitioner"). On the 20 training headers it previously missed,
19 now yield both sides (the 20th has no header).
