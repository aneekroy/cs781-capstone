# CS781 Phase 7: Prior Case Retrieval (IL-PCR)

> **Run from the repository root** as `bash phase7/run_phase7.sh <command>`, or from inside `phase7/`.
>
> **Best submission:** `B1_lexical_ctx` (n-gram BM25 with citation-context queries), test micro-F1@7 **0.4400**.
>
> **Phase-1 retrievers on test:**
> - fused with BM25: R1_ctx2 0.4371; A_chunk400 / A_bidir 0.4312; L1 0.4253
> - alone: 0.31–0.36 (best A_neg_random 0.3615)
>
> **Also on test:** zero-shot Phase-1 encoders fused with BM25 0.419–0.421 (neural only 0.309–0.320); technique runs T_full / T_partial / T_qlora fused 0.431 / 0.425 / 0.423. The pool-level fusion runs, B4 and the re-ranker were not submitted before the deadline.


**The task.** For each query judgment, rank the cases of the same split's candidate pool and submit the top 7: `prediction.json` = `{query id: [7 unique ids]}`, in `submission.zip`.
- **Metric:** Micro-F1@7, scored against the complete gold sets (no padding or truncation).
- **Data:** train 827 queries / 4,320 candidates; dev 118 / 1,023; test 237 / 1,727. Pools are never mixed across splits.
- **Rules carried over:** start from the Phase-1 models; at least 4 substantially different experiments, at least 3 adaptation techniques; LoRMA and knowledge distillation compulsory; no outside models.

## What the literature says, and the design that follows

IL-TUR's IL-PCR table (Joshi et al. 2023) gives the starting point:
- **Word n-gram BM25 is strong.** BM25 over bigrams reached F1 28.6 (K=7), and 4-gram BM25 on event-filtered text 39.2.
- **Document-level transformers were far weaker.** BERT scored 9.2 F1 and InLegalBERT 7.6, even fine-tuned.

The reason is that a judgment citing a case usually **quotes or closely paraphrases it**. Long shared word sequences are the evidence, and a single vector for a 8,000-word judgment blurs them. So:

| part | choice |
|---|---|
| lexical reference (B0) | BM25 over hashed word n-grams, n = 1..5 (numpy rolling hash, 2^24 buckets; no string vocabulary). IDF from the split's own pool; anonymisation tags (`<NAME>`, `<ORG>`, `<CITATION>`) removed; n-grams in more than `max_df` of the pool dropped as boilerplate. Per-n k1/b/max_df and the fusion weights over n (z-scores per query) are tuned on the **train** queries and reported on dev |
| case representation | 200-word chunks (up to 64, evenly spaced when longer), each encoded by the Phase-1 model and pooled (mean / last / max) into a unit vector. Decoder-only models run causally by default (`--mode bidirectional` is an ablation); the encoder-decoder uses its encoder. Padded batches are masked exactly, which tests verify for all four architectures |
| query–candidate score | aggregations of chunk-pair cosines: `maxsim` (best candidate chunk per query chunk, averaged), `maxsim_top` (the same over the 8 best query chunks: a citation is discussed in a few passages), `max`, `mean` (one vector per case). The dev set picks one per run |
| training signal | citation links are document-level, so the passages that carry them are found lexically. For every (query, cited case), the query/candidate chunk pairs with the highest word 2-3-gram TF-IDF cosine become positives (3 per link). InfoNCE (τ = 0.05) with in-batch negatives plus **hard negatives**: the most similar chunks of the top-20 BM25 candidates that are not cited. Cases cited by another query in the batch are masked, so there are no false negatives |
| final ranking | the Phase-1 retriever alone (`submission_neural/`) and fused with the lexical scores (`submission/`, z-score weight chosen on dev). `combine` fuses several retrievers + lexical |
| KD | teacher = a full-model retriever (default T_full); student = the dense model cut to 6 (or 4) **evenly spaced** layers. The student matches the teacher's similarity distribution over each batch's candidates (KL, T = 2, mixed 50/50 with InfoNCE) and its chunk vectors (MSE, `--kd-embed 0.5`) |

**On fusion.** The required experiments are Phase-1 retrievers, and every run reports and submits its neural-only ranking. The fused ranking adds lexical evidence that the IL-PCR results show is hard to beat. It is reported separately, so you can decide which to submit as the "experiment" and which as the best leaderboard entry. The fusion weight is chosen on dev, so fused dev scores are slightly optimistic.

## Runs (`run_phase7.sh`)

| run | what | answers |
|---|---|---|
| B0_lexical | n-gram BM25, tuned on train | the reference; F1 per n shows how much long quotes matter |
| Z1–Z4 | the four Phase-1 models, untrained | how Phase-1 representations transfer as they are |
| L1–L4 | LoRMA retrievers on all four architectures | the compulsory technique; which architecture retrieves best |
| T_full, T_partial, T_qlora, T_ia3, T_prefix, T_prompt | techniques on `$BEST` (default dense) | adaptation technique vs retrieval quality, parameters, memory |
| K1 (6 layers, KD), K0 (6 layers, no KD), K2 (4 layers, KD) | dense students of `$TEACHER` | does KD keep quality at 1/2 and 1/3 depth |
| A_bidir, A_pool_last | causal vs bidirectional; mean vs last-token pooling | representation choices |
| A_neg_random, A_neg_none | random or no hard negatives instead of BM25 ones | negative sampling |
| A_chunk100, A_chunk400 | 100- and 400-word chunks | granularity for long cases |
| A_proj256 | trainable projection to 256 dimensions | index size vs quality |
| A_nodapt | original Phase-1 checkpoint instead of the Phase-2 adapted one | domain adaptation |
| `combine NAME R1 R2 ...` | several retrievers + lexical, weights on dev | the best leaderboard entry |

**Per-run outputs.** Every `summary.json` records:
- **Dev F1/P/R:** neural, fused and lexical, plus F1 for every aggregation.
- **Fusion curve:** F1 over lexical weight 0…1.
- **Recall@7/20/50/100:** the candidate-generation ceiling.
- **Cost:** parameters (total / trainable), peak GPU memory, training minutes, encode seconds and index size.

`scores_{dev,test}.npz` keep the score matrices for error analysis, e.g. queries with no hit, or cited cases with no shared phrasing.

## Further experiments (accuracy push)

**What the data stats showed:**
- Every query is in its own pool; the code never ranks a query against itself.
- About a third of queries cite more than 7 cases, so the best possible F1@7 is ~0.62–0.64.
- Pools overlap across splits (740 of 1,727 test candidates are in the train pool).
- ~95% of each pool is cited by some query in that split.

| run / command | what | why |
|---|---|---|
| B1 `lexical-context` | query = sentences within ±0/1/2/5 of each `<CITATION>` mark, alone or fused with the full query (window and weights tuned on train) | the passages that discuss cited cases; the rest of a 4,500-word judgment is mostly noise for this task |
| B2 `lexical-prf` | + pseudo-relevance feedback (RM3-style, 12 settings), kept only if it helps on train | query expansion |
| B3 `lexical-dev` | tuned on dev | does the train-tuned configuration transfer |
| R1_ctx1/2 `context` | retriever with citation-context queries | the same idea for the Phase-1 model |
| R2_round2 `round2` | continue `$ROUND1` with the hard negatives the trained retriever itself confuses (lexical + neural top-20, passages by embedding) | second-round (ANCE-style) negatives |
| R3–R9 `knobs` | pairs per citation 6, min pair similarity 0.05, τ 0.02 / 0.1, 8 epochs, top passages 4 / 16 | tuning (not separate experiments) |
| K3 / K4 `kd-list` | 6-layer student taught by `$LIST_TEACHER`'s **fused** ranking (list-level KL over its top 16 candidates), alone / with passage KD | KD from the stronger teacher, aimed at the final ranking |
| X1–X4 `rerank` | passage-pair re-ranker (LoRMA / full / over lexical / depth 50) over `$FIRST` | judges each candidate by its best passage pairs read together; depth 30 = 30×4 pair passes per query |
| B4 `lexical-passage` | **best-passage BM25** (150/300-token half-overlapping passages; a candidate scores its best passage) fused with document BM25; n-grams up to 10; context windows up to 8 | a quoted holding is local; whole-document BM25 dilutes it in a 4,000-word case |
| `combine --knn` | **neighbour citations**: the cases cited by the k most similar labelled queries (train for dev; train + dev for test; k chosen on dev), softmax-weighted by query similarity | similar judgments cite the same precedents and the pools overlap across splits |
| `combine` (default flags) | per-run weights + **citation prior** (log of how often labelled queries cited the case: train labels for dev, train+dev for test) + **later-year penalty** (candidate mentions a year after every year in the query) + **hubness correction** (mix in scores standardised across queries), all chosen on dev; `FUSION=rrf` for reciprocal-rank fusion | pool structure the per-query models cannot see |

## Order of work

```bash
bash run_phase7.sh fetch && bash run_phase7.sh stats
bash run_phase7.sh lexical              # CPU, ~20-40 min (5 n-gram sizes x parameter grid on 827 x 4,320)
bash run_phase7.sh zeroshot             # encode only, ~10 min per model
bash run_phase7.sh arch                 # L1 || L2, L3 || L4  (~1 h each)
bash run_phase7.sh summary              # choose BEST (techniques/ablations) and the TEACHER
BEST=dense bash run_phase7.sh techniques
bash run_phase7.sh kd && bash run_phase7.sh kd4
bash run_phase7.sh ablations
bash run_phase7.sh combine C_all L1_dense_lorma L2_gqa_lorma L3_hybrid_lorma L4_encdec_lorma T_full
bash run_phase7.sh lexical-all                       # B1-B3; then LEXRUN=<best B run> for the runs below
bash run_phase7.sh context && bash run_phase7.sh round2 && bash run_phase7.sh knobs
bash run_phase7.sh kd-list
FIRST=$OUT/C_all bash run_phase7.sh rerank
bash run_phase7.sh combine C_final <best retrievers> <best re-ranker> <best B run>
bash run_phase7.sh summary && bash run_phase7.sh collect
```

**Knobs:**
- `EPOCHS` (4), `BATCH` (32 chunk pairs; each step encodes 3×BATCH chunks), `BEST`, `TEACHER`.
- Backbones: the Phase-2 adapted checkpoints (`runsP2/adapt/{dense_clm,gqa_clm,hybrid_clm,encdec_mlm}`) when present, else Phase-1.
- If a run runs out of memory, add `--gradient-checkpointing` or lower `BATCH`.
