# CS781 capstone: model weights

Google Drive link : https[ ]

Checkpoints of the submitted systems, to be used with the code in this repository. Load a downstream checkpoint together with the base it **needs**; place each file at its original path under `$PROJECT` (rename `__` back to `/`) and the launchers find it.

| asset | original path | size | needs | sha256 |
|---|---|---|---|---|
| `runs__dense_gpt_full_395280384_s42_jupyter24gb_v1__best.pt` | `runs/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt` | 420.0 MB | – | 4e7f0d0ca02fee3b |
| `runsP2__X24_dense_coarse__model.pt` | `runsP2/X24_dense_coarse/model.pt` | 439.1 MB | `runs__dense_gpt_full_395280384_s42_jupyter24gb_v1__best.pt` | 0147a94e555dcda5 |
| `runsP2__X24_dense_coarse_s13__model.pt` | `runsP2/X24_dense_coarse_s13/model.pt` | 439.1 MB | `runs__dense_gpt_full_395280384_s42_jupyter24gb_v1__best.pt` | 0353b23081d34955 |
| `runsP2__X22_gqa_clm_header_aug_s13__model.pt` | `runsP2/X22_gqa_clm_header_aug_s13/model.pt` | 435.7 MB | `runsP2__adapt__gqa_clm__best.pt` | 369ddf4165b97411 |
| `runsP2__X25_gqa_clm_coarse__model.pt` | `runsP2/X25_gqa_clm_coarse/model.pt` | 435.7 MB | `runsP2__adapt__gqa_clm__best.pt` | 74d6caf15a967766 |
| `runsP2__X22_gqa_clm_header_aug_s21__model.pt` | `runsP2/X22_gqa_clm_header_aug_s21/model.pt` | 435.7 MB | `runsP2__adapt__gqa_clm__best.pt` | 27cdffa1582437e3 |
| `runsP3__M4_gqa_dapt_lora_kd_mix__final__model.pt` | `runsP3/M4_gqa_dapt_lora_kd_mix/final/model.pt` | 18.2 MB | `runsP2__adapt__gqa_clm__best.pt` | 7d4fc7fc6258bf80 |
| `runsP3__M3_dense_lorma_rare_mix__final__model.pt` | `runsP3/M3_dense_lorma_rare_mix/final/model.pt` | 16.9 MB | `runs__dense_gpt_full_395280384_s42_jupyter24gb_v1__best.pt` | 45b80ffdb389e244 |
| `runsP3__K2_gqa_lora_kd__final__model.pt` | `runsP3/K2_gqa_lora_kd/final/model.pt` | 18.2 MB | `runs__gqa_local_full_57507840_s42_jupyter24gb_v1__best.pt` | a41fefd80fe1a553 |
| `runsP3__S_p2_x1__final__model.pt` | `runsP3/S_p2_x1/final/model.pt` | 16.9 MB | – | 40a7fa9b602c83cf |
| `runsP3__T_ia3__final__model.pt` | `runsP3/T_ia3/final/model.pt` | 6.4 MB | `runsP2__adapt__denselm_clm__best.pt` | 0270617868d51985 |
| `runsP3__S_hybrid_dapt__final__model.pt` | `runsP3/S_hybrid_dapt/final/model.pt` | 14.4 MB | `runsP2__adapt__hybrid_clm__best.pt` | 6c098b124ad46534 |
| `runsP3__P5_p2ner_lorma__final__model.pt` | `runsP3/P5_p2ner_lorma/final/model.pt` | 16.9 MB | – | 066e1583784463de |
| `runsP3__T_lorma__final__model.pt` | `runsP3/T_lorma/final/model.pt` | 16.9 MB | `runsP2__adapt__denselm_clm__best.pt` | d06b62f7a4bdf5cb |
| `runsP3__P7_dense_lorma_wce__final__model.pt` | `runsP3/P7_dense_lorma_wce/final/model.pt` | 16.9 MB | `runs__dense_gpt_full_395280384_s42_jupyter24gb_v1__best.pt` | 63b742d9cad677d6 |
| `runsP4__F3_densedapt_lora__model.pt` | `runsP4/F3_densedapt_lora/model.pt` | 10.7 MB | `runsP2__adapt__denselm_clm__best.pt` | c6deae4e28fda266 |
| `runsP6__C9_dense_full_c512_e5__model.pt` | `runsP6/C9_dense_full_c512_e5/model.pt` | 419.9 MB | `runsP2__adapt__denselm_clm__best.pt` | ce2f6916682b4ee0 |
| `runsP2__adapt__gqa_clm__best.pt` | `runsP2/adapt/gqa_clm/best.pt` | 415.2 MB | – | f1d0a6cccdff1b5d |
| `runs__gqa_local_full_57507840_s42_jupyter24gb_v1__best.pt` | `runs/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt` | 415.2 MB | – | 0307a6fb21c474f3 |
| `runsP2__adapt__denselm_clm__best.pt` | `runsP2/adapt/denselm_clm/best.pt` | 420.0 MB | – | cc39fdb8eb287f6a |
| `runsP2__adapt__hybrid_clm__best.pt` | `runsP2/adapt/hybrid_clm/best.pt` | 404.2 MB | – | f461c48d33c9b1f9 |

Total 4.83 GB in 21 files, plus `tokenizer.json` (shared by all models).

Restore all files to their original paths:
```bash
for f in *.pt; do d=$PROJECT/${f//__//}; mkdir -p "$(dirname "$d")"; mv "$f" "$d"; done
```
