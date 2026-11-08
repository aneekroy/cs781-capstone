# CS781 Phase 1: Legal Language-Model Pretraining from Scratch

**The task.** Given a 512-word prompt taken from an Indian legal judgment, generate the next 128 words.
- **Score:** BERTScore-F1 against the true continuation (primary), ROUGE-L and BLEU (secondary).
- **Submission:** `prediction.json` = `{id: continuation}` inside `submission.zip`.
- **Rules:** every model is randomly initialised and trained only on the official Phase-1 legal corpus: Supreme Court (SC) and High Court (HC) judgments. No pretrained weights, no outside data. Each model must have 50M–300M parameters.

Every later phase starts from the checkpoints trained here, so this phase also fixes the backbones for the whole capstone.

## Provenance of this code

The first version of the Phase-1 pipeline was generated with ChatGPT. It was then reviewed, debugged and extended with Claude:
- precision handling (bf16 on Ampere, fp16 elsewhere);
- exact-state resume;
- KV/recurrent caches for generation;
- the strict submission checks;
- the dense continued-LM / suffix variants used later.

See the LLM declaration in the report.

## Architectures (`models.py`)

All four models share one 24k byte-level BPE tokenizer, trained on training documents only. They also share width 768, 12 heads, an FFN of width 2048 and a maximum context of 2048 tokens.

| key | design | layers | trainable parameters |
|---|---|---|---|
| `dense_gpt` | GPT-style decoder: learned absolute positions, full causal multi-head attention | 12 | 105.0M |
| `gqa_local` | decoder with RoPE and grouped-query attention (12 query / 2 KV heads); sliding-window attention (256 tokens, chunked queries, no quadratic mask) in 3 of every 4 layers, global attention every 4th | 14 | 103.8M |
| `recurrent_hybrid` | decoder alternating LSTM blocks (even layers) with RoPE attention blocks (odd layers); the LSTM state replaces attention in half the layers | 10 | 101.1M |
| `encoder_decoder` | 6-layer bidirectional encoder + 6-layer decoder with cross-attention, learned positions; the prompt is encoded, the continuation decoded | 6 + 6 | 119.1M (encoder 62.5M) |

All four expose `hidden` / `encode`, so later phases can attach task heads. Generation uses attention KV caches and the LSTM state for the hybrid.

## Pipeline (`notebooks/CS781_Phase1_Jupyter_2x24GB_run.ipynb`)

The notebook writes every source file with `%%writefile`, so it is self-contained. The same files are in this folder, together with `configs/`, `evaluators/` and `requirements.txt`.

1. **Data** (`data.py`): the official JSON is streamed with `ijson`. A training-only tokenizer is trained on round-robin samples from the sources. Disk-backed token corpora are built, and held-out continuation examples are drawn per source.
2. **Profiling** (`profile.py`, `scheduler.py`, `runtime.py`): 50 updates per architecture, one isolated process per GPU. Token budgets come from the measured throughput, with a safety factor.
3. **Training** (`train.py`, `experiments.py`): sequence length 2048, micro-batch 2 × 8 accumulation, AdamW (lr 3e-4, cosine to 0.1×, 3% warm-up, weight decay 0.1, clip 1.0), mixed precision and gradient checkpointing. Two models train concurrently, one per GPU. Run IDs encode the measured token budget; dense saw the most tokens (~395M, about 3.8 tokens per parameter) and gqa the fewest (~0.55 per parameter).
4. **Selection** (`select.py`, `official_eval.py`): best-loss and latest checkpoints are compared with several decoding settings using the **unchanged official scorer** (`evaluators/evaluate_phase1.py`) on held-out continuations.
5. **Generation and packaging** (`generate.py`, `submission.py`): length-bucketed cached generation of exactly 128 words, with up to two seeded retries for short or stalled outputs. ID, word-count and ZIP checks run before writing.

## Results (Codabench test, BERTScore-F1)

| submission | BERTScore-F1 |
|---|---|
| **dense_gpt** | **0.8312** |
| recurrent_hybrid | 0.8280 |
| gqa_local | 0.8279 |
| encoder_decoder | 0.8216 |
| first-round submissions (four models, shorter training) | 0.8168 – 0.8275 |
| leaderboard leader (for reference) | 0.8592 |

## Reproduce

Open `notebooks/CS781_Phase1_Jupyter_2x24GB_run.ipynb` on a machine with two ~24 GB GPUs. Set your own Hugging Face token (the notebook reads it from the environment or a Kaggle secret), then run the cells top to bottom. To use the package directly:

```bash
pip install -r phase1/requirements.txt
python -m phase1.experiments matrix --data <prepared corpus> --output runs \
    --configs phase1/configs/dense_gpt.json phase1/configs/gqa_local.json \
              phase1/configs/recurrent_hybrid.json phase1/configs/encoder_decoder.json --execute
```

Phases 2–7 expect the trained runs under `$PROJECT/runs/<run_id>/best.pt`, with `tokenizer.json` beside them.
