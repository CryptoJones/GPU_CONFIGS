#!/usr/bin/env python3
"""Generate the benchmark charts for the 2026-06-11 PLUTO GPU config.
Run with the repo's .charts-venv:  ../../.charts-venv/bin/python make_charts.py
All numbers are measured values from the tuning session (see benchmarks.csv)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- Chart 1: Before vs After (the headline result) -------------------------
labels = ["Prefill\n(tok/s)", "Generation\n(tok/s)", "Tool call\n(seconds)"]
before = [700, 4.5, 900]      # tool-call hung ~15min; capped to 900s for the bar
after  = [1015, 45.6, 3.1]
note   = ["", "", "(was ~15 min)"]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
fig.suptitle("PLUTO Qwen3-30B-abliterated: single-GPU (before) vs dual-GPU (after)",
             fontsize=13, fontweight="bold")
for ax, lab, b, a, n in zip(axes, labels, before, after, note):
    is_sec = "second" in lab
    bars = ax.bar(["before", "after"], [b, a], color=["#c0504d", "#4f81bd"])
    ax.set_title(lab, fontsize=11)
    # value labels: for the tool-call chart the "before" bar is a capped placeholder
    # (it actually hung ~15 min), so label it descriptively instead of with the cap.
    labels_txt = ["~15 min\n(hung)" if is_sec else f"{b:g}", f"{a:g}"]
    for bar, txt in zip(bars, labels_txt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                txt, ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#c0504d" if (is_sec and bar is bars[0]) else "black")
    if is_sec:
        ax.set_ylim(0, 1050)
        ax.set_ylabel("seconds (lower = better)")
    else:
        ax.set_ylabel("tokens/sec (higher = better)")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "before_after.png"), dpi=130)
plt.close(fig)

# ---- Chart 2: tuning sweep (fit + throughput across attempts) ---------------
attempts = ["1\n0.55/.45\nmoe8", "2\n0.62/.38\nmoe16", "3\n0.78/.22\nmoe8",
            "4\n0.72/.28\nmoe12", "yarn64k\n0.68/.32\nmoe16", "FINAL\n40960\nmoe12"]
gen     = [0, 40.0, 0, 45.3, 40.5, 45.6]   # 0 = OOM (did not load)
prefill = [0, 857, 0, 1006, 906, 1015]
status  = ["OOM-1080", "fit", "OOM-3060", "fit", "fit(clamped)", "FINAL"]

fig, ax = plt.subplots(figsize=(11, 4.6))
x = range(len(attempts))
ax.bar([i-0.2 for i in x], prefill, width=0.4, label="prefill tok/s", color="#9bbb59")
ax.bar([i+0.2 for i in x], [g*20 for g in gen], width=0.4,
       label="generation tok/s (x20 scale)", color="#4f81bd")
ax.set_xticks(list(x)); ax.set_xticklabels(attempts, fontsize=8)
ax.set_title("Dual-GPU tuning sweep — tensor-split / n-cpu-moe (0 = OOM, did not load)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("prefill tok/s  /  gen tok/s x20")
for i, s in enumerate(status):
    ax.text(i, 30, s, ha="center", fontsize=7, rotation=90,
            color="#c0504d" if "OOM" in s else "#1f5c1f")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tuning_sweep.png"), dpi=130)
plt.close(fig)

# ---- Chart 3: VRAM placement of the final config ----------------------------
fig, ax = plt.subplots(figsize=(8, 3.2))
cards = ["RTX 3060 (12GB) CUDA0", "GTX 1080 (8GB) CUDA1"]
used  = [10505, 5459]
free  = [12288-10505, 8192-5459]
ax.barh(cards, used, color="#4f81bd", label="used by model+KV (MiB)")
ax.barh(cards, free, left=used, color="#d9d9d9", label="free (MiB)")
for i,(u,f) in enumerate(zip(used,free)):
    ax.text(u/2, i, f"{u} MiB", va="center", ha="center", color="white", fontweight="bold")
    ax.text(u+f/2, i, f"{f} free", va="center", ha="center", fontsize=8)
ax.set_title("Final config VRAM placement (tensor-split 0.72/0.28, n-cpu-moe 12, ctx 40960)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "vram_placement.png"), dpi=130)
plt.close(fig)
print("wrote before_after.png, tuning_sweep.png, vram_placement.png")
