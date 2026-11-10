# CS781 Capstone: Legal NLP from Scratch-Pretrained Models

**Author:** Aneek Roy (Roll No. 261330602)
**Course:** CS781, IIT Kanpur
**Benchmark:** Codabench, Indian legal NLP (IL-TUR tasks)

This repository holds the code for all seven phases of the capstone:
- four ~100M-parameter language models trained **from random initialisation** on official Indian legal text (Phase 1);
- their adaptation to six downstream legal tasks (Phases 2–7): named-entity recognition, rhetorical-role labelling, judgment prediction, statute identification, summarisation and prior-case retrieval.

Every neural model in Phases 2–7 starts from the Phase-1 checkpoints. The best Phase-5 and Phase-7 submissions are lexical baselines (TF-IDF, BM25) that scored above the neural runs. No outside pretrained model, LLM or external dataset is used for training or inference.

The PEFT methods are implemented from scratch: LoRA, rsLoRA, QLoRA (NF4), AdaLoRA, LoRMA, IA³, prefix, prompt and adapters. So are knowledge distillation, CRF tagging, beam search, BM25, the ensembles and the decoders.

---

## Results at a glance (Codabench test set)

| Phase | Task | Primary metric | Best submission | Test score | Rank |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt → 128-word continuation) | BERTScore-F1 | `dense_gpt` | **0.8312** | 71 |
| 2 | Legal NER (12 entity types, character spans) | strict macro-F1 | `V24_all` (CV-selected span vote of 5 taggers) | **0.6626** | 5 |
| 3 | Rhetorical-role prediction (13 roles, CL + IT domains) | pooled macro-F1 | `E6_div_mk2` (diverse tagger ensemble, sequence decoding, Dissent + speaker rules) | **0.6877** | **1** |
| 4 | Court judgment prediction (ILDC, binary) | macro-F1 | `F3_densedapt_lora` | **0.7507** | 47 |
| 5 | Legal statute identification (100 IPC statutes, multi-label) | macro-F1 | `S4_ensemble` (TF-IDF baselines averaged; neural runs scored lower) | **0.3968** | 41 |
| 6 | Abstractive summarisation (IN-Abs) | BERTScore-F1 | `C9_dense_full_c512_e5` | **0.8533** | 11 |
| 7 | Prior-case retrieval (IL-PCR) | micro-F1@7 | `B1_lexical_ctx`; Phase-1 retriever fused: `R1_ctx2` 0.4371 | **0.4400** | 40 |

Ranks were recorded on 9 November 2026. **[SUBMISSIONS.md](SUBMISSIONS.md) lists every individual Codabench submission** (214 valid submissions over the seven phases) with what it was, its test score and the command that rebuilds it. Each phase README holds the full experiment tables, including the runs that failed.

---

## Repository layout

```
.
├── README.md                ← this file: architecture, methodology, implementation and results per phase
├── requirements.txt
├── phase1/  fetch_data.sh  Phase-1 LMs: data fetch, models, training, generation, selection, submission checks
│   ├── configs/             the four architecture configs
│   ├── evaluators/          the official Phase-1 scorer (unchanged)
│   └── notebooks/           the Jupyter notebooks that produced the Phase-1 runs
├── phase2/  run_phase2.sh   NER: Phase-1 encoder + PEFT + CRF, DAPT, KD, rules, CV span voting
├── phase3/  run_phase3.sh   rhetorical roles: hierarchical sentence tagger + PEFT library + FlashAttention patch
├── phase4/  run_phase4.sh   judgment prediction: window classifier → BiGRU-attention aggregator
├── phase5/  run_phase5.sh   statute identification: windows + label-wise attention + ASL + per-statute thresholds
├── phase6/  run_phase6.sh   summarisation: sentence-aligned chunks, masks for batched decoding, beam/MBR, KD
├── phase7/  run_phase7.sh   case retrieval: n-gram BM25, Phase-1 passage retriever, re-ranker, fusion
├── tools/plot_curves.py    learning curves for every phase, read from the run logs (vector PDFs + LaTeX snippet)
├── tools/data_stats.py     corpus statistics for the report's data section (figures + LaTeX snippet)
└── tests/                   98 unit and end-to-end tests on tiny random models (CPU)
```

**Shared code.** Later phases reuse earlier packages:
- `phase2.model` (load a Phase-1 checkpoint as an encoder) and `phase2.peft` (LoRA / LoRMA / adapters);
- `phase3.peft` (all other techniques) and `phase3.flash` (FlashAttention-2 / SDPA attention patch);
- `phase6.model` (attention / LSTM masks).

That's why the packages live side by side at the repository root.

## Setup

```bash
conda create -n cs781 python=3.11 && conda activate cs781
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# optional: FlashAttention-2 wheel matching torch 2.5 / cu12 / cxx11abi=False (falls back to SDPA otherwise)
```

Every launcher reads the same variables, with defaults for the original server:

| variable | meaning | default |
|---|---|---|
| `PROJECT` | Phase-1 project: `runs/` holds the Phase-1 checkpoints; `runsP2/` … `runsP7/` receive the outputs | `$HOME/cs781_phase1` |
| `CACHE` | data cache; official data is fetched to `$CACHE/raw/Phase_N_*` | `$HOME/cs781_cache` |
| `GPU_A` | the one GPU used for paired runs (each run uses one 24 GB GPU; we used RTX 3090) | `0` |

Data is fetched from the gated official Hugging Face dataset with **your own token**, stored at `$CACHE/huggingface/token` and never printed. For example: `bash phase1/fetch_data.sh` (Phase-1 corpus) or `bash phase3/run_phase3.sh fetch`.

Run a launcher from the repository root (`bash phase6/run_phase6.sh <command>`) or from inside its folder. Each launcher prints its commands when called without arguments.

Tests: `python -m pytest -q tests/`. They build a tiny fixture of all four architectures, so no GPU or data is needed.

---

## Ground rules that shaped the design

1. **From-scratch backbones only.** Every downstream model is initialised from a Phase-1 checkpoint, or from a Phase-2 checkpoint that itself came from Phase 1.
2. **Honest model selection.** Selection never touches the test set:
   - Small labelled sets get document-level cross-validation: 2 folds in Phase 2, 5 folds in Phase 3.
   - Every tuned decision (thresholds, decoding offsets, ensemble members, post-processing rules) must also win a **half-swap check**: tune on one half, score the other, swap. Otherwise the untuned default is kept.
3. **Leakage control.** Several official splits contain copies of test documents:
   - Phase 4: every test judgment also appears in dev (1,517 of the 2,511 unique dev documents).
   - Phase 5: 833 train and 407 dev documents copy test cases.
   - Phase 2: 26 of 34 test documents are inside the Phase-1 corpus.

   Every loader drops train/dev documents that share ≥50% of sampled 8-gram shingles with a test document. KD unlabelled text is filtered the same way.
4. **Format safety.** Every submission passes a re-implementation of the official checks before it is zipped: IDs, counts, label vocabulary and types. Where an official scorer was provided, our scorer reproduces it exactly, and the tests check this.

---

## Phase 1: legal language models from scratch

**Problem.** Continue a 512-word judgment excerpt with 128 words. The score is BERTScore-F1.

**Architectures** (`phase1/models.py`). There are four genuinely different designs. All have width 768 and share one 24k BPE tokenizer.

| key | design | params |
|---|---|---|
| `dense_gpt` | 12-layer GPT, learned positions, full attention | 105.0M |
| `gqa_local` | 14 layers, RoPE, grouped-query attention (12 Q / 2 KV heads), sliding-window attention (256) with a global layer every 4th | 103.8M |
| `recurrent_hybrid` | 10 layers alternating LSTM and RoPE-attention blocks | 101.1M |
| `encoder_decoder` | 6-layer bidirectional encoder + 6-layer decoder with cross-attention | 119.1M |

**Methodology.**
- Train a tokenizer on training documents only, and build disk-backed token corpora from the official SC/HC judgments.
- Profile throughput per architecture, then train two models at a time on 2×24 GB. Settings: AdamW, cosine schedule, bf16, gradient checkpointing, sequence length 2048.
- Select the checkpoint and decoding settings with the **unchanged official scorer** on held-out continuations.

**Implementation:**
- `data.py`: streaming, tokenizer, corpora
- `train.py`: resumable training
- `generate.py`: cached, length-bucketed generation with retries
- `select.py`, `official_eval.py`: checkpoint and decoder selection
- `submission.py`: strict checks
- `notebooks/`: the full run

The first version of this code was generated with ChatGPT, then reviewed and fixed. See [phase1/README.md](phase1/README.md).

**Results.** dense 0.8312, hybrid 0.8280, gqa 0.8279, encoder–decoder 0.8216 (leader 0.8689). All four models land within 1 point of each other, so continuation quality is limited by training data, not by architecture. Training tokens consumed: dense 395M (about 3.8 per parameter), encoder–decoder 429M (but only the last quarter of each block is predicted), hybrid 78M and gqa 58M. The corpus was 256M SC and 428M HC tokens.

---

## Phase 2: legal named-entity recognition (L-NER)

**Problem.** Find character spans of 12 entity types in full judgments. The score is strict span macro-F1.
- **Labels:** APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, COURT, AUTH, STAT, PREC, CASENO, DATE.
- **Data:** 70 training documents (2 folds) and 35 test documents. The median document is 5.6k tokens.
- **Rare roles:** RESP has 83 training mentions, WIT 116.

**Architecture.** A Phase-1 model is used as a token encoder:
- decoder-only models run causally (bidirectional attention without adaptation collapsed to F1 0);
- the encoder–decoder uses its encoder.

LoRMA/LoRA adapters (rank 16) sit on all projections, followed by a linear head and a CRF over BIO tags. Documents are tagged in 512-token windows with stride 256, and window scores are averaged.

**Methodology.**
1. **Segmentation.** Punctuation is split off before BPE, so span boundaries align with tokens. The alignment ceiling is 0.993.
2. **DAPT.** Continued pretraining (`adapt.py`) on 643M official tokens: causal LM for the decoders, masked LM for the encoder. Test score effect:
   - enc-dec: +15.7 points (0.392 → 0.549)
   - gqa: +3.6 (0.538 → 0.574)
   - dense: inconclusive
3. **Merged-role training** (`--coarse-roles`). APP/RESP and A./R.COUNSEL look identical locally; only the case header says who is appellant. The tagger therefore learns PARTY and COUNSEL, and a header parser plus cue rules assign the side. This is the best single model: X24, test 0.6557.
4. **Post-processing rules** (`post.py`): counsel side, parties from the header, witnesses, and label propagation of a recognised string. Each rule is kept only if it raises out-of-fold F1; together they add about 6.5 points.
5. **KD.** A 37M hybrid student learned from the dense teacher on teacher-labelled SC text (test 0.572 vs 0.464 without KD). Its CV score is unusable: the teacher had seen both folds, so KD runs are judged on test only.
6. **Ensembling** (`vote.py`). Span voting over out-of-fold predictions. Members and the vote threshold are chosen greedily on CV, and post-processing is re-chosen on the voted spans.

**Best submission: `V24_all`, test 0.6626.**
- Members: X24 ×2 seeds, X25 (adapted gqa, merged roles), X22 ×2 seeds (gqa, header context + entity-swap augmentation). Keep spans with at least 2 of 5 votes.
- The CV score was 0.6615, matching the test score within 0.001.

```bash
bash phase2/run_phase2.sh adapt-data && bash phase2/run_phase2.sh adapt gqa_clm 0
bash phase2/run_phase2.sh round7                                     # X24_dense_coarse || X25_gqa_clm_coarse
bash phase2/run_phase2.sh one X24_dense_coarse_s13 0 & bash phase2/run_phase2.sh one X22_gqa_clm_header_aug_s13 1; wait
bash phase2/run_phase2.sh one X22_gqa_clm_header_aug_s21 0
VOTE_NAME=V24_all VOTE_ARGS="--no-extras --no-rerank --no-per-label" bash phase2/run_phase2.sh vote
```

**What failed:**
- A tuned "O"-tag penalty merged neighbouring entities (0.310).
- Silver data from the merged-role teacher scored 0.611.
- A learned span re-ranker lost to plain voting on CV.
- Single-seed differences under about 3 points are noise on 35 test documents: the same recipe scored 0.627 and 0.656 with different seeds.

---

## Phase 3: rhetorical-role labelling

**Problem.** Assign one of 13 roles to every sentence of a judgment, in two domains: competition law (CL) and income tax (IT). The score is pooled macro-F1.
- **Data:** 90 labelled documents (train + dev).
- **Missing labels in dev:** Dissent and PrecedentOverruled have no dev sentences.

**Architecture** (`phase3/model.py`). A hierarchical tagger:
1. **Sentence encoder.** A Phase-1 model with PEFT. Consecutive sentences are packed into 512-token windows, so each sentence sees its left context; each sentence is mean+last pooled.
2. **Document context.** A BiLSTM over the sentence vectors, plus relative-position and domain embeddings.
3. **Output.** A linear-chain CRF.

**Methodology:**
- **Cross-validation.** 5-fold document CV over all 90 documents, scored with the pooled OOF macro-F1 (the same computation as the leaderboard).
- **Backbones and techniques.** A backbone screen over 12 checkpoints, then all 9 techniques on the best backbone: LoRA, rsLoRA, QLoRA-NF4, AdaLoRA, LoRMA, IA³, prefix, prompt, full fine-tuning.
- **KD.** Fold-matched: the student's fold *k* learns only from the teacher's fold-*k* model, so CV stays honest.
- **Data mixing.** Rare-role repetition, same-label block splicing between documents, and in-domain silver documents.
- **Dissent rule.** A dissenting note always forms the end of a judgment, so a rule marks it (Dissent F1 0.99).
- **Sequence decoding for ensembles.** A Viterbi pass with a gold transition prior, or averaged CRFs, instead of per-sentence argmax.
- **TF-IDF member.** A logistic-regression member over the sentence and its neighbours, to diversify the ensemble.

**Findings (OOF macro-F1):**
- Document context is essential: sentence-only scored 0.436, against 0.568 with context.
- LoRMA was the best technique (0.576), ahead of rsLoRA 0.566, full fine-tuning 0.564, LoRA 0.561, AdaLoRA 0.561, IA³ 0.556, QLoRA 0.554, prefix 0.539 and prompt 0.533.
- Initialising from the Phase-2 NER model helped (0.571).
- On test, single runs vary by ±4 points. Ensembles are steadier.

**Best submission: `E6_div_mk2`, test 0.6877 (P 0.689, R 0.692), first on the leaderboard.** It is a fixed diverse set of taggers combined with posterior averaging and sequence decoding, followed by the Dissent rule and the speaker-tracking rule. The speaker rule fixes a 56-sentence block of respondent arguments that every model labelled as petitioner arguments, inside one test judgment that holds 43% of the test sentences; it added 4–5 points (E8 0.640 → E9 0.685, R1B 0.627 → 0.670).

```bash
bash phase3/run_phase3.sh screen; BEST=<backbone key from the screen report> bash phase3/run_phase3.sh techniques
bash phase3/run_phase3.sh round1 && bash phase3/run_phase3.sh round2 && bash phase3/run_phase3.sh kd
bash phase3/run_phase3.sh mix && bash phase3/run_phase3.sh mix2
bash phase3/run_phase3.sh push          # redecode + lexical member + E6/E7/E8 ensembles, vote, stack
bash phase3/run_phase3.sh redecode && bash phase3/run_phase3.sh push   # with the speaker rule in rules=auto: E6 -> E6_div_mk2 (0.6877)
```

---

## Phase 4: court judgment prediction (CJPE / ILDC)

**Problem.** Predict whether an appeal was accepted, given the judgment without its final decision. The score is macro-F1.
- **Data:** 32,305 training documents after removing 5,082 exact repeats; 2,511 dev documents; 1,517 test documents.
- **Leakage:** every test judgment is near-duplicated in dev. Those copies are dropped from training and selection, which leaves 994 dev documents.

**Architecture and methodology** (`phase4/`):
1. **Stage 0** (`baseline`): TF-IDF over the first 300 + last 2,000 words, with logistic regression. Dev macro-F1 0.677.
2. **Stage 1** (`neural.py`): a Phase-1 backbone with PEFT, fine-tuned on the **last four 512-token windows plus the first window**. Each window carries the document label plus a learned from-the-end position embedding. Fine-tuned recipes:
   - dense DAPT + LoRA (F3): dev 0.746
   - AdaLoRA (F5): 0.737
   - rsLoRA r16 (F4) collapsed to 0.428; it was retuned to r64 / α16.
   - LoRMA on all four architectures (F1, F2, F7, F8)
   - KD into a gqa and a 4-layer hybrid student (K1, K2)
3. **Stage 2/3** (`aggregate.py`): cached window embeddings of the last 8 windows, then a BiGRU + attention aggregator averaged over 5 seeds. This is the hierarchical design that was strongest in the ILDC paper.
4. **Stage 4** (`ensemble.py`): a probability average or majority vote, with members and the decision threshold chosen on dev under the half-swap check.

**Best submission: `F3_densedapt_lora`, test 0.7507.** Other test scores: dev-selected ensemble E1 0.744, AdaLoRA 0.732, TF-IDF 0.669, gqa KD student 0.666, Phase-2 init LoRMA 0.620, dense LoRMA 0.608, BiGRU aggregators 0.620–0.621, gqa/enc-dec LoRMA 0.523/0.518, rsLoRA (r16) 0.464. Details in [phase4/README.md](phase4/README.md).

---

## Phase 5: legal statute identification (ILSI)

**Problem.** From the masked facts of a criminal case, predict every applicable IPC statute (100 labels, ~3.8 per case). The score is macro-F1.
- **Data:** 41,917 training and 9,774 dev documents after test-copy removal (833 and 407 copies dropped); 13,019 test documents.

**Architecture** (`phase5/neural.py`). All 512-token windows of the facts are encoded by a Phase-1 backbone with PEFT. **Label-wise attention** (LAAT) follows: each statute attends to its own windows. Training uses the **asymmetric loss**. Thresholds are per statute, shrunk toward a global value, with a "min-one" rule, and are chosen with the honest split-half check.

**Methodology:**
- TF-IDF one-vs-rest baselines; the best is an ensemble with dev macro-F1 0.404.
- LoRMA on all four architectures.
- The technique comparison.
- KD (dense teacher → gqa / 4-layer hybrid students).
- Pooling, loss, section and rare-statute ablations.
- A greedy probability-averaging ensemble.

**Best submission: `S4_ensemble` (average of the TF-IDF baselines), test 0.3968.** The neural label-attention models scored lower within the phase deadline: dense 0.358, gqa 0.352, enc-dec 0.298. Details in [phase5/README.md](phase5/README.md).

---

## Phase 6: abstractive summarisation (IN-Abs)

**Problem.** Write the headnote of a Supreme Court judgment. The score is BERTScore-F1.
- **Data:** 6,818 training pairs, a 200-document hash-selected dev split, and 100 test judgments.
- **Lengths:** the median document has 3,139 words, the median summary 640 words.

**Architecture** (`phase6/`):
- **Sentence-aligned chunking.** Each summary sentence is aligned (TF-IDF) to the 512-token document chunk it comes from, so every chunk gets its own target.
- **Decoder-only models.** Each generates `[chunk] "Summary:" target`.
- **Encoder–decoder.** Also tried as Fusion-in-Decoder.
- **Masks.** Key-padding masks are patched into the Phase-1 attention and LSTM blocks. Batched left-padded generation is token-identical to one-by-one decoding; the tests check this.
- **Length.** Length budgets are learned from document length and alignment density.
- **Decoding.** Beam search with n-gram blocking, plus an MBR option.

**Methodology:**
- 4 architectures, 9 techniques (Houlsby/Pfeiffer adapters, prefix, prompt, LoRA, QLoRA, AdaLoRA, LoRMA, full).
- KD: word-level (logits) and sequence-level (teacher summaries).
- Chunk-size, DAPT, decoding and length ablations.
- Extractive and lead baselines.

**Findings:**
- **Fine-tuning:** full fine-tuning and LoRMA on 512-token chunks were best (0.852–0.853 test). Prompt and prefix tuning trailed (0.818–0.823). DAPT mattered most for the encoder–decoder (+3.1 points).
- **Distillation:** a 6-layer dense student (62.5M) with sequence-level KD scored 0.8527 test, matching its 105M teacher (C3, 0.8520) at 60% of the size.
- **No gains:** MBR, ensembles and FiD did not help.
- **Scorer:** BERTScore truncates to 512 sub-words, so the first ~400 words of a summary decide the score.

**Best submission: `C9_dense_full_c512_e5`, test 0.8533.** It is dense, fully fine-tuned on 512-token chunks for 5 epochs, with dev R-1 0.608.

```bash
bash phase6/run_phase6.sh longer        # C8_dense_lorma_c512_e5 || C9_dense_full_c512_e5
```

---

## Phase 7: prior-case retrieval (IL-PCR)

**Problem.** For each query judgment, return the 7 cited cases from its candidate pool. The score is micro-F1@7.
- **Data:** 827 / 118 / 237 queries; pools of 4,320 / 1,023 / 1,727 candidates.
- **Ceiling:** about a third of queries cite more than 7 cases, so the best achievable F1@7 is about 0.62–0.64.

**Architecture** (`phase7/`):
- **Lexical:**
  - BM25 over hashed word n-grams (n = 1–10), with parameters tuned per n.
  - **Citation-context queries:** sentences around each `<CITATION>` mark.
  - Pseudo-relevance feedback, and best-passage BM25.
- **Neural:** a Phase-1 bi-encoder over 200-word passages, scored by max-sim aggregation. Training details:
  - **Positives:** passage pairs mined lexically.
  - **Negatives:** BM25 hard negatives, then ANCE-style second-round negatives.
  - **KD:** passage-level and list-level, from the fused teacher.
- **Re-ranking:** a passage-pair cross-encoder.
- **Fusion** (`combine`). Z-score or RRF fusion of retrievers and BM25, plus three corrections:
  - a citation prior;
  - a later-year penalty;
  - hubness correction and neighbour-citation voting.

  All weights are chosen on dev.

**Findings:**
- Long shared n-grams are the strongest evidence, because judgments quote the cases they cite.
- Citation-context queries were the best lexical change (0.440 test).
- Phase-1 retrievers alone reach 0.31–0.36. Fused with BM25 they reach 0.437, close to but not above lexical-only.

**Best submission: `B1_lexical_ctx`, test 0.4400.** The best Phase-1 model is R1_ctx2 (fused), at 0.4371. The pool-level fusion runs (prior, kNN, hubness), B4 and the re-ranker were built but not submitted before the deadline.

```bash
bash phase7/run_phase7.sh lexical-context                         # B1
LEXRUN=B1_lexical_ctx bash phase7/run_phase7.sh context            # R1_ctx1 / R1_ctx2
bash phase7/run_phase7.sh combine C_final R1_ctx2 L1_dense_lorma B1_lexical_ctx
```

---

## Corpus statistics

```bash
python tools/data_stats.py --cache $CACHE --out data      # needs matplotlib and ijson; Phase 1 is sampled
```

Writes `data/data_stats.json`, figures and `data/appendix_data.tex` (length histograms, Phase-2 span lengths and nested/adjacent spans, labelled-vs-unlabelled strings, Phase-3 role distribution by domain, Phase-5 label histogram and long tail, Phase-6 extractiveness, Phase-7 lengths and citation ages). Copy `data/` next to the report's `.tex`.

## Learning curves

Every run already logs its curves (Phase 1 `metrics.jsonl`, DAPT `adapt_metrics.jsonl`, task runs `cv_summary.json` / `summary.json`). To draw them:

```bash
python tools/plot_curves.py --project $PROJECT --out curves      # needs matplotlib
```

This writes `curves/phase*_curves.pdf`, `curves/dapt_curves.pdf` and `curves/appendix_curves.tex`; copy the `curves/` folder next to the report's `.tex` and the appendix includes them. Use `--p2 RUN ...` (and `--p1`, `--dapt`, `--p3` … `--p7`) to choose other runs.

## Testing

`tests/` checks the following on tiny random models of all four architectures:
- **Correctness of every architecture-dependent component:** masked batching equals unbatched results, PEFT merges are exact, and KD fold matching holds.
- **Scorer agreement:** our scorers match the official ones.
- **End-to-end runs:** every phase runs all the way through.

`test_fetch.py` needs the network and a recent `huggingface_hub`.

## Use of AI assistants

- The Phase-1 base pipeline was first created by me over the harness starter code in code package.
- Code for Phases 2–7 were deliberated and added by me while the debugging, the experiment design and the documentation were developed with Claude (Anthropic) as a coding assistant.
- All experiments were run, and all results produced, on our own server.
- The report's LLM declaration gives the details.

## License

MIT (see `LICENSE`).
