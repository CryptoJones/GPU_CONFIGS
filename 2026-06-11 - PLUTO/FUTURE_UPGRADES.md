# Future upgrades — bigger models once the V100 lands

Context: the live config runs **Qwen3-30B-A3B-abliterated** (Q4_K_M, 18.5 GB) split across the
RTX 3060 (12 GB) + GTX 1080 (8 GB) = **20 GB**, with ~12 expert layers spilled to CPU.

Swapping the 1080 for a **Tesla V100 16 GB** (see [`IMPLEMENTATION.md`](IMPLEMENTATION.md) →
"V100 variant") raises total VRAM to **12 + 16 = 28 GB**, enough to run the whole model on GPU
*and* step up to a larger model. After ~3-4 GB for KV/context/buffers, budget **~24 GB for
weights**. The practical ceiling is ~40B params at Q4; the 70B class stays out of reach.

**Keep the A3B (≈3B-active) MoE design** — that's what makes a *bigger* model still *fast*. A
dense 32B fits but runs ~10× the active params per token, so it's much slower. Anything ≥48B
(e.g. the Qwen3-48B-A4B merges) is too tight at 28 GB.

## Shortlist (sizes verified from HF, 2026-06-11)

### ★ Recommended — Huihui-Qwen3.6-35B-A3B-abliterated  (the clean upgrade)
Newer generation (Qwen3.6), **+5B total params, still ~3B active** (same speed class),
abliterated, **same Qwen family whose structured tool-calling we already validated** with Hermes.
- Q4_K_M **19.7 GB** → ~8 GB KV room on 28 GB (generous, 40K+ context) — **daily driver**
- Q4_K_S 18.5 GB · Q5_K_M 23.0 GB (quality, ~16-32K context)
- Repo: `mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF`
- **Staged on disk** at `/home/akclark/models/Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf`
  (downloaded 2026-06-11, ahead of the V100).

### Go-bigger (more params, tighter, validate tool-calling first)
**GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B** — 42B-A3B MoE, 202K context.
- Q4_K_M **25.8 GB** → fits but tight (~2 GB KV → q8 cache + 8-16K context).
- Caveats: different family (GLM, not Qwen); it's a "Brainstorm 20x" *creative* merge, so its
  agentic **tool-calling must be smoke-tested** before trusting it in Hermes.
- Repo: `DavidAU/GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B-GGUF`

### Dense (more capability per param, but slower)
**Huihui-Qwen3-VL-32B-Thinking-abliterated** — dense 32B (+vision), Qwen tool-calling.
- Q4_K_M **19.8 GB** → fits easily, but dense = all 32B active = noticeably slower than the A3B MoEs.
- Repo: `mradermacher/Huihui-Qwen3-VL-32B-Thinking-abliterated-GGUF`

### Already on disk (no download; not real upgrades)
- `qwen3.5-coder:35b` (~22 GB) — coder-tuned; coder variants flagged for unreliable tool calls. Skip for agent use.
- `Nemotron-3-Nano-Omni-30B-A3B` (22.8 GB) — 30B-A3B multimodal; a sideways move.

## How to switch the serving model
1. Put the new GGUF at `/home/akclark/models/<file>.gguf`.
2. In `configs/serve-qwen.sh` set `MODEL=` to the new path and `ALIAS=` to a new id.
3. With the V100's 28 GB, set `N_CPU_MOE=0` (full model on GPU) and retune `TENSOR_SPLIT`
   (see IMPLEMENTATION Step 5). Watch VRAM; leave a few hundred MiB free per card.
4. Update `~/.hermes/config.yaml` `model.default` / `providers.custom.default_model` to the new alias.
5. **Smoke-test tool calling** before trusting it (the make-or-break for Hermes):
   `hermes -z "list files in /home/akclark"` must return a real listing via the tool.
6. Re-run `benchmarks.md`'s method and save a new dated snapshot folder.

## Verify the model still does structured tool_calls (any new model)
```bash
curl -s http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"<alias>","messages":[{"role":"user","content":"/no_think List files in /home/akclark using the tool."}],
  "tools":[{"type":"function","function":{"name":"list_dir","description":"List files",
    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
  "tool_choice":"auto","max_tokens":256}' | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0].get("finish_reason"), d["choices"][0]["message"].get("tool_calls"))'
# want: finish_reason "tool_calls" + a structured list_dir call (NOT JSON-in-content)
```

Sources: [Qwen3.6-35B-A3B-abliterated](https://huggingface.co/mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF) ·
[GLM-4.7-Flash-Grande-42B-A3B](https://huggingface.co/DavidAU/GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B-GGUF) ·
[Qwen3-VL-32B-Thinking-abliterated](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-32B-Thinking-abliterated-GGUF)
