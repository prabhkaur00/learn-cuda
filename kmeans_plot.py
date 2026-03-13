"""
kmeans_plot.py
==============
Reads kmeans_profile_results.csv and generates:
  1. Per-step timing breakdown for n=8000 (bar chart + line per iter)
  2. Step % composition across all cluster sizes (stacked bar)
  3. Total execution time vs cluster size (line + scatter)
  4. Memory traffic breakdown per step across cluster sizes
  5. Arithmetic intensity heatmap
  6. Convergence curve (centroid distance) for n=8000

Usage:
    python kmeans_plot.py                          # reads ./kmeans_profile_results.csv
    python kmeans_plot.py path/to/results.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    "#0d1117",
    "figure.facecolor":  "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "axes.grid":         True,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.25,
    "legend.edgecolor":  "#30363d",
    "figure.dpi":        130,
})

# Palette — CUDA-green accent on dark github theme
COLORS = {
    "assign":  "#58a6ff",   # blue
    "accum":   "#3fb950",   # green
    "update":  "#d29922",   # yellow
    "d2h":     "#f85149",   # red
    "seed":    "#bc8cff",   # purple
    "total":   "#79c0ff",
}

STEP_ORDER   = ["seed", "assign", "accum", "update", "d2h"]
STEP_LABELS  = {"seed": "Seed", "assign": "Assign (E-step)",
                "accum": "Accumulate (M-step)", "update": "Update centroids", "d2h": "D→H copy"}

# ─────────────────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "kmeans_profile_results.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}\nRun kmeans_profile first.")

df = pd.read_csv(csv_path)
df["n"] = df["n"].astype(int)
print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Cluster sizes: {sorted(df['n'].unique())}")
print(f"Iterations per size: {df.groupby('n')['iter'].max().to_dict()}\n")

sizes    = sorted(df["n"].unique())
df_8k    = df[df["n"] == 8000].copy()

os.makedirs("plots", exist_ok=True)

# =============================================================================
# FIG 1: Per-step timing for n=8000, one bar per iteration
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("GPU K-Means Split — Per-Step Timing  (n=8000, dim=128)",
             fontsize=13, color="#e6edf3", fontweight="bold", y=0.97)

ax = axes[0]
iters = df_8k["iter"].values
x     = np.arange(len(iters))
w     = 0.65

step_cols = {
    "assign": "assign_ms",
    "accum":  "accum_ms",
    "update": "update_ms",
    "d2h":    "d2h_ms",
}
bottoms = np.zeros(len(iters))
for step, col in step_cols.items():
    vals = df_8k[col].values
    bars = ax.bar(x, vals, w, bottom=bottoms, color=COLORS[step],
                  label=STEP_LABELS[step], alpha=0.9, zorder=3)
    # Label bars that are tall enough
    for b, v, bot in zip(bars, vals, bottoms):
        if v > 0.002:
            ax.text(b.get_x() + b.get_width()/2, bot + v/2,
                    f"{v:.3f}", ha="center", va="center",
                    fontsize=6.5, color="#0d1117", fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([f"iter {i}" for i in iters], rotation=35, ha="right")
ax.set_ylabel("Time (ms)")
ax.set_title("Stacked step time per iteration")
ax.legend(loc="upper right", ncol=2)
ax.set_xlim(-0.5, len(iters) - 0.5)

# Percentage donut inset for last-iteration split
ax2 = axes[1]
totals_per_iter = {step: df_8k[col].mean() for step, col in step_cols.items()}
labels_pie  = [STEP_LABELS[s] for s in totals_per_iter]
sizes_pie   = list(totals_per_iter.values())
colors_pie  = [COLORS[s] for s in totals_per_iter]
wedges, texts, autotexts = ax2.pie(sizes_pie, labels=None, colors=colors_pie,
                                   autopct="%1.1f%%", startangle=90,
                                   wedgeprops={"linewidth": 0.5, "edgecolor": "#0d1117"})
for at in autotexts:
    at.set_fontsize(8)
    at.set_color("#0d1117")
ax2.set_title("Average step composition (all iterations)", pad=4)
ax2.legend(wedges, labels_pie, loc="center left",
           bbox_to_anchor=(1.05, 0.5), fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/fig1_step_timing_8k.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig1_step_timing_8k.png")


# =============================================================================
# FIG 2: Stacked step % across all cluster sizes
# =============================================================================
# Build mean step times per cluster size
summary = df.groupby("n").agg(
    assign_ms  = ("assign_ms",  "mean"),
    accum_ms   = ("accum_ms",   "mean"),
    update_ms  = ("update_ms",  "mean"),
    d2h_ms     = ("d2h_ms",     "mean"),
    total_ms   = ("total_iter_ms", "mean"),
    total_run  = ("total_run_ms",  "first"),
    iters_run  = ("iters_run",     "first"),
).reset_index()

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Step Time Composition Across Cluster Sizes (mean per iteration)",
             fontsize=13, color="#e6edf3", fontweight="bold")

x   = np.arange(len(summary))
w   = 0.55
bot = np.zeros(len(summary))
for step, col in [("assign", "assign_ms"), ("accum", "accum_ms"),
                  ("update", "update_ms"), ("d2h",   "d2h_ms")]:
    vals  = summary[col].values
    total = summary["total_ms"].values
    pct   = np.where(total > 0, vals / total * 100, 0)
    bars  = ax.bar(x, vals, w, bottom=bot, color=COLORS[step],
                   label=STEP_LABELS[step], alpha=0.9, zorder=3)
    for b, v, p, bo in zip(bars, vals, pct, bot):
        if p > 5:
            ax.text(b.get_x() + b.get_width()/2, bo + v/2,
                    f"{p:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="#0d1117", fontweight="bold")
    bot += vals

ax.set_xticks(x)
ax.set_xticklabels([f"n={n}" for n in summary["n"]], rotation=15)
ax.set_ylabel("Mean iteration time (ms)")
ax.set_xlabel("Cluster size (n)")
ax.legend(loc="upper left", ncol=2)
ax.set_title("Absolute + relative step breakdown")

# Twin axis: total run time
ax3 = ax.twinx()
ax3.plot(x, summary["total_run"].values, "o--",
         color=COLORS["total"], linewidth=2, markersize=7, label="Total run (ms)")
ax3.set_ylabel("Total run time (ms)", color=COLORS["total"])
ax3.tick_params(axis="y", labelcolor=COLORS["total"])
ax3.legend(loc="upper right")

plt.tight_layout()
plt.savefig("plots/fig2_step_composition_all_sizes.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig2_step_composition_all_sizes.png")


# =============================================================================
# FIG 3: Total execution time scaling (log-log)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Total Execution Time vs Cluster Size", fontsize=13,
             color="#e6edf3", fontweight="bold")

ns   = summary["n"].values
trun = summary["total_run"].values
tper = summary["total_ms"].values  # mean per-iter time

for ax, y, title, ylabel in [
    (axes[0], trun, "Total run time", "ms (total)"),
    (axes[1], tper, "Mean per-iteration time", "ms (per iter)"),
]:
    ax.plot(ns, y, "o-", color=COLORS["total"], lw=2.5, ms=9,
            markerfacecolor="#0d1117", markeredgewidth=2.5, zorder=4)
    for xi, yi in zip(ns, y):
        ax.annotate(f"{yi:.2f}ms", (xi, yi),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="#c9d1d9")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (cluster size)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

# Fit power law on total run
mask = trun > 0
log_n = np.log(ns[mask])
log_t = np.log(trun[mask])
slope, intercept = np.polyfit(log_n, log_t, 1)
axes[0].set_title(f"Total run time  (scaling exponent ≈ {slope:.2f})")

plt.tight_layout()
plt.savefig("plots/fig3_scaling.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig3_scaling.png")


# =============================================================================
# FIG 4: Memory traffic breakdown across cluster sizes
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Estimated Memory Traffic per Iteration vs Cluster Size",
             fontsize=13, color="#e6edf3", fontweight="bold")

traffic_specs = [
    ("assign",  "assign_global_read_MB",  "assign_global_write_MB",  "assign_smem_MB",  "assign_ai"),
    ("accum",   "accum_global_read_MB",   "accum_global_write_MB",   "accum_smem_MB",   "accum_ai"),
    ("update",  None,                     None,                      None,              "update_ai"),
]

df_mean = df.groupby("n").mean(numeric_only=True).reset_index()

for ax_idx, (step, read_col, write_col, smem_col, ai_col) in enumerate(traffic_specs):
    ax = axes[ax_idx]
    ax.set_title(f"{STEP_LABELS.get(step, step)}")
    ax.set_xlabel("n (cluster size)")

    ns_m = df_mean["n"].values

    if read_col and read_col in df_mean.columns:
        ax.plot(ns_m, df_mean[read_col].values, "o-",
                color="#58a6ff", lw=2, ms=6, label="Global read (MB)")
        ax.plot(ns_m, df_mean[write_col].values, "s--",
                color="#3fb950", lw=2, ms=6, label="Global write (MB)")
        ax.plot(ns_m, df_mean[smem_col].values, "^:",
                color="#d29922", lw=1.5, ms=5, label="Smem (MB)")
        ax.set_ylabel("MB per iteration")
    else:
        # update: read/write in KB
        read_k  = f"update_global_read_KB"
        write_k = f"update_global_write_KB"
        if read_k in df_mean.columns:
            ax.plot(ns_m, df_mean[read_k].values,  "o-",
                    color="#58a6ff", lw=2, ms=6, label="Global read (KB)")
            ax.plot(ns_m, df_mean[write_k].values, "s--",
                    color="#3fb950", lw=2, ms=6, label="Global write (KB)")
        ax.set_ylabel("KB per iteration")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns_m)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="upper left", fontsize=7)

    # AI on twin axis
    if ai_col in df_mean.columns:
        ax2 = ax.twinx()
        ax2.plot(ns_m, df_mean[ai_col].values, "D-.",
                 color="#f85149", lw=1.5, ms=5, alpha=0.8, label="AI (FLOP/B)")
        ax2.set_ylabel("Arithmetic Intensity", color="#f85149", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#f85149", labelsize=7)
        ax2.legend(loc="lower right", fontsize=7)

plt.tight_layout()
plt.savefig("plots/fig4_memory_traffic.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig4_memory_traffic.png")


# =============================================================================
# FIG 5: Arithmetic intensity heatmap
# =============================================================================
ai_steps = ["assign_ai", "accum_ai", "update_ai"]
ai_labels = ["Assign\n(E-step)", "Accumulate\n(M-step)", "Update\ncentroids"]

ai_matrix = np.zeros((len(ai_steps), len(sizes)))
for j, n in enumerate(sizes):
    row = df_mean[df_mean["n"] == n]
    if len(row) == 0:
        continue
    for i, col in enumerate(ai_steps):
        if col in row.columns:
            ai_matrix[i, j] = row[col].values[0]

cmap = LinearSegmentedColormap.from_list(
    "ai_cmap", ["#161b22", "#1f4068", "#1b6ca8", "#58a6ff", "#79c0ff", "#b3d9ff"])

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("Arithmetic Intensity (FLOPs/byte) — All Steps × All Cluster Sizes",
             fontsize=12, color="#e6edf3", fontweight="bold")

im = ax.imshow(ai_matrix, aspect="auto", cmap=cmap, interpolation="nearest")
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"n={n}" for n in sizes])
ax.set_yticks(range(len(ai_labels)))
ax.set_yticklabels(ai_labels)
ax.set_xlabel("Cluster size")
ax.set_title("Lower = more memory-bound   |   Roofline crossover ~5 FLOP/byte on A100")

for i in range(len(ai_steps)):
    for j in range(len(sizes)):
        v = ai_matrix[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < ai_matrix.max() * 0.6 else "#0d1117",
                fontsize=9, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("FLOP/byte", color="#c9d1d9")
cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

plt.tight_layout()
plt.savefig("plots/fig5_arithmetic_intensity.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig5_arithmetic_intensity.png")


# =============================================================================
# FIG 6: Convergence curve for n=8000
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Convergence Metrics — n=8000, dim=128",
             fontsize=13, color="#e6edf3", fontweight="bold")

iters_8k = df_8k["iter"].values
cdist     = df_8k["centroid_dist"].values
cntA      = df_8k["cntA"].values
cntB      = df_8k["cntB"].values

ax = axes[0]
ax.plot(iters_8k, cdist, "o-", color="#58a6ff", lw=2.5, ms=7,
        markerfacecolor="#0d1117", markeredgewidth=2)
ax.fill_between(iters_8k, cdist, alpha=0.15, color="#58a6ff")
ax.set_xlabel("Iteration")
ax.set_ylabel("L2 distance between centroids")
ax.set_title("Centroid separation (should stabilise)")
ax.set_yscale("log" if cdist.min() > 0 else "linear")

# Mark convergence point
conv_idx = np.where(df_8k["converged"].values)[0]
if len(conv_idx) > 0:
    ci = conv_idx[0]
    ax.axvline(iters_8k[ci], color="#3fb950", ls="--", lw=1.5, alpha=0.8)
    ax.annotate("converged", (iters_8k[ci], cdist[ci]),
                xytext=(10, 10), textcoords="offset points",
                color="#3fb950", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#3fb950"})

ax = axes[1]
ax.bar(iters_8k - 0.2, cntA, 0.38, color=COLORS["assign"],
       label="Cluster A", alpha=0.85, zorder=3)
ax.bar(iters_8k + 0.2, cntB, 0.38, color=COLORS["accum"],
       label="Cluster B", alpha=0.85, zorder=3)
ax.axhline(4000, ls=":", color="#8b949e", lw=1, label="n/2 (ideal balance)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Cluster size")
ax.set_title("Cluster sizes per iteration (balance check)")
ax.legend()

plt.tight_layout()
plt.savefig("plots/fig6_convergence_8k.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: plots/fig6_convergence_8k.png")


# =============================================================================
# FIG 7: Combined overview — all 6 panels in one poster figure
# =============================================================================
fig = plt.figure(figsize=(22, 15), facecolor="#0d1117")
fig.suptitle("GPU K-Means Split — Complete Profiling Overview",
             fontsize=16, color="#e6edf3", fontweight="bold", y=0.98)

imgs = [
    "plots/fig1_step_timing_8k.png",
    "plots/fig2_step_composition_all_sizes.png",
    "plots/fig3_scaling.png",
    "plots/fig4_memory_traffic.png",
    "plots/fig5_arithmetic_intensity.png",
    "plots/fig6_convergence_8k.png",
]
titles = [
    "Per-step timing (n=8000)",
    "Step composition (all sizes)",
    "Total time scaling",
    "Memory traffic breakdown",
    "Arithmetic intensity heatmap",
    "Convergence (n=8000)",
]

for idx, (img_path, title) in enumerate(zip(imgs, titles)):
    ax = fig.add_subplot(2, 3, idx + 1)
    if os.path.exists(img_path):
        from PIL import Image
        img = Image.open(img_path)
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10, color="#58a6ff", pad=6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/fig7_overview_poster.png", bbox_inches="tight",
            facecolor=fig.get_facecolor(), dpi=100)
plt.close()
print("Saved: plots/fig7_overview_poster.png")

print("\n✓ All plots saved to ./plots/")
print("  fig1  — Per-step timing (n=8000)")
print("  fig2  — Step composition stacked bar (all sizes)")
print("  fig3  — Total execution time scaling (log-log)")
print("  fig4  — Memory traffic: global read/write/smem + AI per step")
print("  fig5  — Arithmetic intensity heatmap")
print("  fig6  — Convergence: centroid distance + cluster sizes")
print("  fig7  — Combined overview poster")
