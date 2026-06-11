# Benchmarks — 2026-06-11 PLUTO

All numbers measured on pluto (RTX 3060 12GB + GTX 1080 8GB, Ryzen 5 5600X, 128GB).
Method: `POST /v1/chat/completions` with a ~2.4K-token prompt, reading llama.cpp's
`timings.prompt_per_second` (prefill) and `timings.predicted_per_second` (generation).
Raw data in [`benchmarks.csv`](benchmarks.csv).

## Headline: before vs after

| Metric | Before — 1 GPU (3060), 30 experts on CPU | After — 2 GPU, tuned |
|---|---:|---:|
| Prefill (2.4K-token prompt) | 700 tok/s | **1015 tok/s** |
| Generation (with real context) | 4.5 tok/s | **45.6 tok/s** |
| Tool call (grammar-constrained) | hung ~15 min (421% CPU) | **3.1 s**, structured |
| Real context window | 40960 | 40960 |

The generation collapse and the tool-call grind had the **same root cause**: experts on the
CPU. Moving them to the 1080's VRAM fixed both. Generation on a *short* prompt was ~31 tok/s
before, but collapsed to 4.5 as context grew (CPU attention over the KV) — that growth-collapse
is what made agent turns unusable.

## Dual-GPU tuning sweep

| # | ctx | n-cpu-moe | tensor-split (3060/1080) | Result | Prefill | Gen |
|---|---|---|---|---|---:|---:|
| 1 | 40960 | 8 | 0.55 / 0.45 | ❌ OOM on 1080 | — | — |
| 2 | 32768 | 16 | 0.62 / 0.38 | ✅ fit (1080 tight, 3060 idle) | 857 | 40.0 |
| 3 | 32768 | 8 | 0.78 / 0.22 | ❌ OOM on 3060 | — | — |
| 4 | 32768 | 12 | 0.72 / 0.28 | ✅ balanced | 1006 | 45.3 |
| yarn64k | 65536 (YaRN) | 16 | 0.68 / 0.32 | ⚠ fit but clamped to 40960 | 906 | 40.5 |
| **FINAL** | **40960** | **12** | **0.72 / 0.28** | ✅ **live** | **1015** | **45.6** |

**Lessons:**
- The 1080 is the *smaller* card — its layer share must account for KV, not just weights
  (attempt 1 gave it 45% and it OOM'd on the KV alloc).
- Change one knob at a time (attempt 3 moved both split and n-cpu-moe and OOM'd the 3060).
- `0.72/0.28` + `n-cpu-moe 12` is the sweet spot: 3060 ~10.5 GB, 1080 ~5.5 GB, both with margin.
- **YaRN-64K does not help this model**: it was trained to 40960, so llama.cpp clamps the
  usable window there regardless (`/props` reported `n_ctx=40960`). It only cost ~10% speed.

## Final config VRAM placement

| Card | Used | Free |
|---|---:|---:|
| RTX 3060 12 GB (CUDA0) | 10505 MiB | 1403 MiB |
| GTX 1080 8 GB (CUDA1) | 5459 MiB | 2649 MiB |

## Caveat

The 1080 (Pascal) is the slow card; in layer-split mode a token passes through both cards in
sequence, so the 1080 caps peak throughput — but it's far ahead of CPU, which is the point.
