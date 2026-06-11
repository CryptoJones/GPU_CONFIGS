# GPU_CONFIGS

[![snapshot: 2026-06-11 PLUTO](https://img.shields.io/badge/snapshot-2026--06--11_PLUTO-1f6feb?logo=nvidia&logoColor=white)](./2026-06-11%20-%20PLUTO/)

> **The config that worked, captured the day it worked — drop in the files, hand the runbook to an agent, rebuild it anywhere.**

Versioned snapshots of working local-LLM / GPU serving configurations across the fleet.

Each dated folder is a **self-contained, drop-in snapshot** of one machine's setup: the actual
config files, the benchmarks and charts that justify it, captured system logs, and an
`IMPLEMENTATION.md` runbook a Claude Code agent can execute to rebuild it from scratch.

## Snapshots

| Date | Host | Summary |
|---|---|---|
| [2026-06-11 - PLUTO](./2026-06-11%20-%20PLUTO/) | pluto (Rocky 9.8) | Uncensored Qwen3-30B-A3B served on dual GPUs (RTX 3060 + GTX 1080) via llama.cpp, wired to Hermes. ~10× faster than single-GPU; tool calls 15 min → 3 s. |

## How to use a snapshot to revert/reproduce
1. Open the folder's `README.md` for the what/why/results.
2. Hand `IMPLEMENTATION.md` + the `configs/` files to a Claude Code agent: *"implement this
   configuration on <host>, verifying paths/UUIDs against the live machine."*
3. `benchmarks.md` + `charts/` are the acceptance criteria — reproduce them to confirm success.
