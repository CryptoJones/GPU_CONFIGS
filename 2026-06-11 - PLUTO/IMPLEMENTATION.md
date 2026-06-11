# IMPLEMENTATION — rebuild the 2026-06-11 PLUTO config

**Audience:** a Claude Code agent (or a human) reproducing this setup on __HOSTNAME__-Linux or a
similar host. Read this top-to-bottom; the files in `configs/` are the drop-in artifacts.
Verify every path/UUID against the live machine before trusting it — hardware UUIDs and IPs
are host-specific (see `logs/gpu-state.txt`, `logs/host.txt`).

## Target end state
Uncensored Qwen3-30B-A3B served on `0.0.0.0:8081` (OpenAI `/v1`) across **both** GPUs via
llama.cpp, as a boot-persistent **user** systemd service, with Hermes Agent pointed at it.
Expected: prefill ~1000 tok/s, gen ~45 tok/s, tool calls ~3 s.

## Step 0 — populate placeholders
This public snapshot ships with `__LAN_IP__` and `__HOSTNAME__` placeholders instead of the
original host's network coordinates. Fill them in for your machine first:
```bash
./repopulate.sh                 # auto-detect this host's LAN IP + short hostname
./repopulate.sh --dry-run       # preview without editing
./repopulate.sh --ip 10.0.0.5 --host mybox   # or set them explicitly
```
(These placeholders are only in the docs — the actual `configs/` bind `0.0.0.0` / `localhost`,
so they don't block installation; populating just makes the endpoint URLs correct for your LAN.)
Don't commit the populated files back to the public repo.

## Preconditions to check first
- GPUs present: `nvidia-smi -L` → expect an Ampere card (sm_86) + the Pascal GTX 1080 (sm_61).
  **Get the real UUIDs** here; the ones in `configs/serve-qwen.sh` are __HOSTNAME__-specific.
- Driver supports CUDA 13 (`nvidia-smi` top-right). Driver stays untouched throughout.
- `git`, `gcc/g++`, `make`, `cmake` present (`dnf install cmake` if missing).
- SELinux state: `getenforce`. If `Enforcing`, you MUST use a **user** systemd service
  (system services can't exec files under `/home`). This is why the unit lives in
  `~/.config/systemd/user/`, not `/etc/systemd/system/`.
- The model GGUF on fast local storage. If absent, pull
  `hf.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF:Q4_K_M` and place at
  `/home/akclark/models/qwen3-abliterated-30b-Q4_K_M.gguf` (verify magic bytes = `GGUF`).

## Step 1 — CUDA toolkits (minimal dev sets, NOT the driver)
The 1080 is Pascal/sm_61, which **CUDA 13 dropped**. So the multi-GPU build needs **CUDA 12.x**.
Install only the dev packages llama.cpp links (skips ~2 GB of cuFFT/NPP/Nsight):
```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
# CUDA 12.9 = last 12.x with Pascal sm_61 support (for the 1080 / dual-GPU build)
sudo dnf -y install cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcublas-devel-12-9 \
                    cuda-nvrtc-devel-12-9 cuda-crt-12-9
# (optional) CUDA 13.0 for a single-GPU 3060-only fallback build
sudo dnf -y install cuda-nvcc-13-0 cuda-cudart-devel-13-0 libcublas-devel-13-0 \
                    cuda-nvrtc-devel-13-0 cuda-crt-13-0
```

## Step 2 — build llama.cpp (dual-arch, the live build)
```bash
cd /home/akclark && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git || true
cd /home/akclark/llama.cpp
export PATH=/usr/local/cuda-12.9/bin:$PATH CUDACXX=/usr/local/cuda-12.9/bin/nvcc
cmake -B build-multigpu -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;86" \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-multigpu --config Release -j "$(nproc)" --target llama-server
ldd build-multigpu/bin/llama-server | grep -E 'cudart|cublas'   # expect libcudart.so.12
```
Optional single-GPU fallback (3060 only): same but `-B build`, archs `"86"`, CUDA 13.0 nvcc.

## Step 3 — drop in the launch wrapper
Copy `configs/serve-qwen.sh` to `/home/akclark/models/serve-qwen.sh`, `chmod +x`.
**Edit the two GPU UUIDs** (`CUDA_VISIBLE_DEVICES=`) to match `nvidia-smi -L` on this host,
ordered **fast card first** (the 3060) so `--tensor-split 0.72,0.28` maps 3060:1080 correctly.
Key tunables (env-overridable): `CTX=40960`, `N_CPU_MOE=12`, `TENSOR_SPLIT=0.72,0.28`,
`KV_TYPE=q8_0`, `LLAMA_BIN=.../build-multigpu/bin/llama-server`.

## Step 4 — install as a boot-persistent USER service
```bash
sudo loginctl enable-linger akclark                       # start at boot w/o login
mkdir -p ~/.config/systemd/user
cp configs/llama-qwen.service ~/.config/systemd/user/llama-qwen.service
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user daemon-reload
systemctl --user enable --now llama-qwen.service
# verify it binds and exposes the real window:
curl -s http://localhost:8081/props | python3 -c 'import sys,json;print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'  # -> 40960
```
Open the firewall: `sudo firewall-cmd --permanent --add-port=8081/tcp && sudo firewall-cmd --reload`.

## Step 5 — tune the split (only if GPUs/VRAM differ)
If it OOMs or VRAM is unbalanced, sweep manually (stop the service first) — see `benchmarks.md`.
Rules: the **smaller** card's share must leave room for its KV; change **one** knob at a time;
raise `N_CPU_MOE` to free VRAM, shift `TENSOR_SPLIT` toward the card with headroom. Watch with
`nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv`. Target both cards with
a few hundred MiB free.

## Step 6 — Hermes
```bash
cd /home/akclark && git clone https://github.com/NousResearch/hermes-agent.git || true
cd hermes-agent && uv venv venv --python 3.11 && \
  VIRTUAL_ENV=$PWD/venv uv pip install -e ".[all]"
ln -sf /home/akclark/hermes-agent/venv/bin/hermes ~/.local/bin/hermes   # put on PATH
cp /path/to/configs/hermes-config.yaml ~/.hermes/config.yaml            # provider->localhost:8081
touch ~/.hermes/.env
hermes -z "/no_think say hi"                                            # smoke test
```
**Critical:** `model.context_length: 65536` in the Hermes config is mandatory — Hermes refuses
models reporting <64K. The server's real window is 40960; Hermes compresses at ~32K so it never
overflows. Tool-call test (the thing that was broken): `hermes -z "list files in /home/akclark"`
should return a real listing (uses the filesystem tool).

## Reverting / variants
- **Remove the 1080** (e.g. V100 swap): set `LLAMA_BIN` to `build/bin/llama-server` (CUDA13,
  sm_86), drop `--tensor-split`, raise `N_CPU_MOE` until it fits one card; `systemctl --user restart`.
- **OOM after any change:** raise `N_CPU_MOE` (+2 at a time) or lower `CTX`.
- **Manage:** `XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user {status,restart} llama-qwen`,
  logs via `journalctl --user -u llama-qwen -f`.
