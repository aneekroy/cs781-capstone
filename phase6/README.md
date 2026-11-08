# CS781 Phase 6: Abstractive Legal Summarization

> **Run from the repository root** as `bash phase6/run_phase6.sh <command>`, or from inside `phase6/`.
>
> **Best submission:** `C9_dense_full_c512_e5`, test BERTScore-F1 **0.8533**. It is the dense model, fully fine-tuned on 512-token sentence-aligned chunks for 5 epochs.
>
> **Other test results:**
> - D1 hybrid DAPT 0.8532
> - K6 6-layer dense with sequence KD 0.8527
> - C1 dense LoRMA 0.8521
> - C3 0.8520
> - B0 extractive 0.8508
> - C5 encoder–decoder 0.8210
> - L5 FiD 0.8079


**The task.** Given an Indian Supreme Court judgment, write its summary (headnote).
- **Data:** IN-Abs (Shukla et al. 2022, via IL-TUR), in `Phase_6_Summ/Train/train.json` (`{"id", "document", "summary"}`) and `Test/test.json` (`{"id", "document"}`).
- **Lengths:** documents average ~4.4k words, summaries ~1k words, a compression of about 0.24.
- **No dev split** is published. A fixed held-out set of 200 training documents (chosen by id hash) is used for selection and the dev scores.
- **Submission:** `prediction.json` = `[{"id": ..., "summary": "..."}]` in `submission.zip`.
- **Scores:** BERTScore-F1 (primary), ROUGE-1 F1, ROUGE-L F1.
- **Rules carried over:**
  - start from the Phase-1 models; all four architectures;
  - at least 4 substantially different experiments, at least 3 adaptation techniques;
  - LoRMA and knowledge distillation compulsory;
  - no outside models or LLMs.

## Design

| part | choice | why |
|---|---|---|
| long documents | **sentence-aligned chunks** (default 1024 tokens, shrunk automatically to fit the Phase-1 context with the target). Every summary sentence is assigned to the chunk holding its most similar document sentence (TF-IDF cosine), so each chunk has its own target. At inference every chunk is summarized and the pieces are joined in document order. | a ~1k-word summary of a ~6k-token judgment does not fit one Phase-1 context. Chunk-wise targets (Gidiotis & Tsoumakas 2020) were the best fine-tuned setting on IN-Abs in Shukla et al. 2022 |
| encoder-decoder | `chunk` like the others, **plus FiD** (`--strategy fid`, run L5): up to 8 chunks encoded separately, their states concatenated, one decoder pass writes the whole summary | a hierarchical encoder-decoder: the decoder sees the whole judgment, so it can order and connect content across chunks and avoid repeating itself |
| decoder-only | `[soft prompt] chunk "\n\nSummary:" target <eos>`, loss on target tokens only (`--src-lm w` adds the language-model loss on the source as in-task domain adaptation) | the Phase-1 decoders are language models; the separator turns them into conditional generators |
| length | summary length predicted from document length (log-log fit on train; `stats` prints R² and the dev error). Split over the chunks by the **alignment density learned on train** (`--budget learned`; `uniform` is the ablation). Decoding forbids `<eos>` before 0.8× and stops at 1.3× each chunk's budget | the test set gives no target length; ROUGE depends heavily on length |
| decoding | beam search (4 beams, length penalty 1) with 5-gram blocking; optional **MBR** (`--mbr N`: of the top N beam hypotheses keep the one with the highest mean unigram+bigram F1 against the others). Greedy, top-p sampling and repetition penalty are available. KV-cached batched decoding for all four architectures | the `tune` command compares them on held-out documents |
| post-processing | chunk summaries joined; sentences whose trigrams are ≥60% already present are dropped (`--dedup`); unfinished last sentence removed; trimmed to 1.05× the budget at a sentence boundary | chunked summaries repeat facts restated in several chunks |
| padding | the Phase-1 attention has no masks. Key-padding masks are patched in (`phase6/model.py`, like `phase3/flash.py`): the encoder and cross-attention ignore padding; batched decoder-only generation uses left padding with per-row positions and per-row LSTM prefill | tests check that padded batches give **exactly** the tokens of one-at-a-time decoding on all four architectures, greedy and beam |
| baselines | `lead` (first sentences up to the predicted length) and a supervised **extractive** model: TF-IDF + position features, trained on oracle labels where each summary sentence marks its 3 most similar document sentences (the "Avr" conversion of Shukla et al.) | reference points. The extractive scorer also drives `--strategy extract` (importance-based context selection) |
| KD | **word-level** (`--kd logit`): KL to the teacher's distribution at T=2 on every target token, mixed 50/50 with gold (all Phase-1 runs share one tokenizer). **Sequence-level** (`--kd seq`, Kim & Rush 2016): the teacher summarizes every training chunk once (cached), and each epoch an example uses the teacher's text with probability 0.5 | the required distillation run, with both kinds of signal the spec names (distributions and teacher-generated summaries) |
| data hygiene | training documents that copy a test document (exact or ≥50% shared 8-grams) are dropped; `stats` reports them | Phase 4 and 5 had such copies |

All the section 12.2.1 techniques are implemented (`--method`):
- `houlsby`: two adapters per layer, after the attention output and after the feed-forward down projection.
- `pfeiffer`: one adapter per layer, after the feed-forward sublayer.
- `prefix`: key/value prefixes in every decoder self-attention layer.
- `prompt`: soft tokens before the input, initialised from vocabulary embeddings.
- `lora`, `qlora` (NF4 base + LoRA), `adalora`, `lorma`.
- `full` and `partial` as references, plus KD.

**About BERTScore.** `bert_score` truncates each text to the scorer's 512 sub-words. With ~1k-word summaries, the score is therefore decided by roughly the first 350–400 words of each summary. The learned budget puts more words where the reference's content comes from, and the chunks are joined in document order, which matches how headnotes are written. Length mainly moves ROUGE.

## Runs (`run_phase6.sh`)

| run | backbone | technique / strategy | answers |
|---|---|---|---|
| B0_extractive (+ lead) | – | extractive / lead | non-neural reference, `extractor.pkl` |
| L1_dense_lorma, L2_gqa_lorma, L3_hybrid_lorma, L4_encdec_lorma | the four Phase-1 models (Phase-2 domain-adapted dense/gqa if present) | LoRMA, chunks | "all four architectures"; the 4 submissions |
| L5_encdec_fid_lorma | enc-dec | LoRMA, FiD | hierarchical encoder-decoder vs chunk-and-join |
| T_full, T_lora, T_qlora, T_adalora, T_houlsby, T_pfeiffer, T_prefix, T_prompt | `$BEST` (default gqa) | technique comparison | can PEFT match full fine-tuning; parameters / memory / minutes |
| K1_gqa_kd_logit, K2_gqa_kd_seq | gqa ← `$TEACHER` (default L1 dense; both ~105M) | word- vs sequence-level KD | does a stronger teacher help a same-size student? (control: L2) |
| K4_dense6s_kd, K5_dense6s_nokd, K6_dense6s_kdseq | 6 of 12 dense layers (evenly spaced, ~62M) ← L1 (~105M) | word KD / none / sequence KD | the compression run: KD at ~60% of the size |
| K3_dense6_kd, K0_dense6_nokd | 6-layer dense ← L1 | KD vs none | same-architecture compression |
| A_head | `$BEST` | first chunk only → whole summary | what truncation loses |
| A_extract | `$BEST` | top-salient 50% of the sentences, then chunks | importance-based selection |
| A_overlap | `$BEST` | 128 tokens of the previous chunk as context | overlapping windows |
| A_align_pos | `$BEST` | summary split over chunks by position, not similarity | does alignment matter |
| A_chunk512 | `$BEST` | 512-token chunks | chunk size |
| A_srclm | `$BEST` | + source LM loss 0.2 | in-task domain adaptation |
| A_nodapt | `$BEST` | original Phase-1 instead of Phase-2 adapted | domain adaptation |
| A_scratch | `$BEST` | random init, full training | are Phase-1 representations useful |
| C1–C5 (`combine`) | one per architecture | 512-token chunks + LoRMA (dense) or full fine-tuning (gqa, dense, hybrid, enc-dec) | the leaderboard's best factors combined |
| C6, C7 (`chunks`), C8 (`longer`) | gqa / dense | 384- and 256-token chunks; 5 epochs | chunk size and schedule |
| C9 (`longer`) | dense | full fine-tuning, 512-token chunks, 5 epochs | the best recipe (C3), trained longer |
| D1, D2 (`dapt`) | hybrid / enc-dec on their Phase-2 adapted checkpoints (`hybrid_clm`, `encdec_mlm`) | full, 512-token chunks | domain adaptation for the two weak architectures |
| E_dense, E_gqa (`ensembles`, `ens NAME R1 R2 ...`) | several finished runs of one architecture | mixture of next-token distributions at decode time | checkpoint ensembling |
| `tune NAME` | any run | length ×0.8 / ×1.2, greedy, sampling, no / 3-gram blocking, uniform / learned budget, no dedup | decoding strategy, generation length vs BERTScore / ROUGE, redundancy |

Every `summary.json` records:
- **Dev quality:** BERTScore P/R/F1, ROUGE-1/2/L, words generated vs reference, length ratio, repeated-3-gram rate, the share of chunk outputs that hit their budget.
- **Faithfulness proxies:** `number_precision` (share of the summary's numbers, i.e. sections, years and amounts, found in the judgment) and `novel_3gram` (abstractiveness; `ref_novel_3gram` for the reference).
- **Cost:** trainable/total parameters, peak GPU memory, training minutes, dev/test generation seconds.
- **Setup:** the plan (chunk, target cap, length model, learned budget profile).

`dev_pred.json` keeps the dev summaries for qualitative analysis.

## Order of work

```bash
pip install rouge-score bert-score           # scoring only (BERTScore downloads its scorer model once)
bash run_phase6.sh fetch && bash run_phase6.sh stats
bash run_phase6.sh baseline                  # CPU, ~15 min
bash run_phase6.sh arch                      # L1 || L2, L4 || L3, L5 ; check "max_context" in the first log line
bash run_phase6.sh summary                   # pick BEST (technique backbone) and TEACHER (a chunk run: L4 or L1)
BEST=gqa bash run_phase6.sh techniques
TEACHER=$PWD/../cs781_phase1/runsP6/L4_encdec_lorma bash run_phase6.sh kd
bash run_phase6.sh kd-small
BEST=gqa bash run_phase6.sh ablations
for r in L1_dense_lorma L2_gqa_lorma L3_hybrid_lorma L4_encdec_lorma L5_encdec_fid_lorma; do bash run_phase6.sh tune $r; done
bash run_phase6.sh summary && bash run_phase6.sh collect   # collect takes tuned/submission when tuning found better
```

**Quick check first:** `MAX_TRAIN=800 EPOCHS=1 EVAL_DOCS=20 OUT=$PWD/../cs781_phase1/runsP6_quick bash run_phase6.sh one L2_gqa_lorma 0`.

**Knobs** (environment variables):
- **Training:** `EPOCHS` (3); `CHUNK` (1024); `MAX_TRAIN`.
- **Decoding and scoring:** `BEAMS` (4); `EVAL_DOCS` (100 dev documents scored); `DEV_SIZE` (200); `GEN_BATCH` (8 prompts × beams per batch; lower it if generation runs out of memory).
- **Runs:** `BEST`, `TEACHER`.
- **Scorer:** `NO_BERTSCORE=1` to skip BERTScore; `BERTSCORE_MODEL` to match the course scorer. The default is the `model_type` found in `$PROJECT/evaluators/evaluate_phase6.py` or `evaluate_phase1.py`, else roberta-large.

**Memory:**
- Micro-batches are capped by tokens: 4096 for dense/enc-dec, 8192 for gqa, 6144 for hybrid.
- Full fine-tuning and FiD use gradient checkpointing.
- If a run runs out of memory, add `--micro-tokens 2048` to its line in `spec()`.

**Rough cost on one 3090** (3 epochs, ~45k chunk examples):
- gqa: ~2 h.
- hybrid: ~4 h.
- dense and enc-dec: ~6–9 h.
- Decoding 100 + 100 documents with 4 beams: 10–30 min.

`EPOCHS=2` for the 400M models halves their time.
