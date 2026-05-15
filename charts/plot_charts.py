#!/usr/bin/env python3
"""Generate charts for REPORT.md from the 2026-05-14 bench results.

Run with: .venv/bin/python plot_charts.py
Outputs to ./*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


@dataclass
class Row:
    cfg_id: str
    workload: str  # decode | prefill | balanced | longctx | sharegpt
    x: float  # concurrency or request rate
    total_tps: float  # total token throughput
    out_tps: float  # output-only throughput
    ttft_mean: float
    ttft_median: float
    ttft_p99: float
    tpot_mean: float
    tpot_median: float
    tpot_p99: float
    itl_mean: float
    itl_median: float
    itl_p99: float


# (cfg_id, workload, x, total_tps, out_tps,
#  ttft_mean, ttft_median, ttft_p99,
#  tpot_mean, tpot_median, tpot_p99,
#  itl_mean, itl_median, itl_p99)
# fmt: off
SGLANG = [
    Row("01-decode-c1",      "decode",   1,   187.75, 186.48,   15.75,   15.75,    15.98,    5.35,   5.35,   5.35,    5.42,   5.35,   5.58),
    Row("02-decode-c16",     "decode",   16,  2594.66, 2577.04, 92.15,   48.18,   179.25,    6.12,   6.10,   6.26,    6.20,   6.06,  11.99),
    Row("03-decode-c64",     "decode",   64,  6538.36, 6493.97, 141.28, 118.22,   337.16,    9.72,   9.71,   9.86,    9.86,   8.34,  19.85),
    Row("04-decode-c256",    "decode",   256, 8548.58, 8490.54, 451.34, 446.44,   875.71,   29.72,  29.70,  30.10,   30.10,  25.16,  58.27),
    Row("05-prefill-c1",     "prefill",  1,   20207.64, 39.40, 166.34, 166.55,   169.75,    5.22,   5.23,   5.28,    4.07,   5.50,   5.73),
    Row("06-prefill-c16",    "prefill",  16,  26124.85, 50.94, 1308.96, 1246.04, 2449.28,  170.26, 160.09, 333.26,  132.76,   8.88, 2126.02),
    Row("07-prefill-c64",    "prefill",  64,  26145.08, 50.98, 5073.16, 5032.48, 9791.83,  696.06, 699.34, 1368.49, 543.09,  22.23, 8638.88),
    Row("08-balanced-c1",    "balanced", 1,   368.85, 184.60,   32.55,   32.78,    33.38,    5.36,   5.36,   5.37,    5.39,   5.36,   5.51),
    Row("09-balanced-c16",   "balanced", 16,  4624.45, 2314.49, 195.95, 190.57,   388.45,    6.54,   6.54,   7.12,    6.57,   6.14,   7.87),
    Row("10-balanced-c64",   "balanced", 64,  9624.08, 4816.74, 689.50, 683.34,  1358.47,   11.95,  11.96,  13.22,   12.00,   8.82,  20.03),
    Row("11-balanced-c256",  "balanced", 256, 10963.17, 5486.94, 2773.29, 2764.29, 5389.09, 41.28,  41.31,  45.93,   41.42,  37.72,  59.81),
    Row("12-longctx-c1",     "longctx",  1,   6335.24, 97.47,   1808.07, 1921.85, 1928.89,   6.74,   6.74,   6.75,    7.14,   6.78,  13.61),
    Row("13-longctx-c8",     "longctx",  8,   9711.95, 149.42,  6691.45, 6560.37, 12962.33, 40.47,  40.80,  51.50,   42.83,  26.57,  77.07),
    Row("14-longctx-c32",    "longctx",  32,  9355.08, 143.93,  68411.92, 75351.15, 103389.48, 47.19, 48.24, 59.10,  51.17,  30.41,  89.86),
    Row("15-sharegpt-r1",    "sharegpt", 1,   416.21, 204.34,   23.78,   18.11,   133.96,    5.48,   5.41,   6.65,    5.43,   5.38,   5.66),
    Row("16-sharegpt-r2",    "sharegpt", 2,   824.10, 404.60,   24.90,   19.09,   131.08,    5.62,   5.57,   7.06,    5.58,   5.43,  11.53),
    Row("17-sharegpt-r4",    "sharegpt", 4,   1613.52, 791.95,  25.26,   18.69,   133.99,    5.95,   5.84,   7.43,    5.89,   5.60,  21.95),
    Row("18-sharegpt-r8",    "sharegpt", 8,   3023.43, 1489.57, 26.49,   21.55,   127.36,    6.73,   6.59,  10.53,    6.57,   5.91,  24.04),
    Row("19-sharegpt-r16",   "sharegpt", 16,  5285.28, 2585.14, 31.08,   25.12,   149.65,    8.20,   7.98,  12.43,    7.82,   6.26,  39.11),
]

VLLM = [
    Row("01-decode-c1",      "decode",   1,   131.41, 130.52,   13.89,   13.85,    14.38,    7.66,   7.64,   7.77,    7.65,   7.64,   7.89),
    Row("02-decode-c16",     "decode",   16,  1921.27, 1908.23, 49.28,   54.32,    65.02,    8.34,   8.34,   8.36,    8.33,   8.33,   8.61),
    Row("03-decode-c64",     "decode",   64,  5662.16, 5623.72, 116.17, 122.95,   202.01,   11.26,  11.26,  11.32,   11.25,  11.18,  12.13),
    Row("04-decode-c256",    "decode",   256, 10394.15, 10323.58, 379.25, 362.72, 752.99,   24.43,  24.45,  24.58,   24.41,  23.97,  27.30),
    Row("05-prefill-c1",     "prefill",  1,   19091.27, 37.22, 160.53, 160.70,   164.75,    7.75,   7.75,   7.79,    6.78,   7.74,   7.86),
    Row("06-prefill-c16",    "prefill",  16,  28207.14, 55.00, 1813.12, 1821.79, 1913.79,   71.86,  71.90,  74.10,   62.88,  71.98,  76.60),
    Row("07-prefill-c64",    "prefill",  64,  28219.70, 55.02, 8544.31, 8792.76, 8905.25,   71.75,  71.82,  74.42,   62.81,  71.89,  76.61),
    Row("08-balanced-c1",    "balanced", 1,   261.81, 131.03,   40.20,   40.60,    41.29,    7.57,   7.56,   7.62,    7.55,   7.56,   7.69),
    Row("09-balanced-c16",   "balanced", 16,  3553.07, 1778.27, 168.72, 168.16,   294.59,    8.67,   8.64,   8.94,    8.66,   8.44,   8.81),
    Row("10-balanced-c64",   "balanced", 64,  9322.32, 4665.72, 332.23, 208.02,  1255.85,   13.05,  13.15,  13.64,   13.03,  11.55,  69.25),
    Row("11-balanced-c256",  "balanced", 256, 15131.98, 7573.38, 918.63, 245.95, 5403.87,   31.79,  32.95,  33.22,   31.73,  25.06,  81.07),
    Row("12-longctx-c1",     "longctx",  1,   5172.98, 79.59,   1728.71, 1837.79, 1844.33,   9.21,   9.20,   9.28,    9.19,   9.20,   9.35),
    Row("13-longctx-c8",     "longctx",  8,   11638.65, 179.06, 4609.05, 2761.05, 12896.97, 35.04,  39.77,  41.42,   34.97,  19.52, 155.54),
    Row("14-longctx-c32",    "longctx",  32,  11673.19, 179.59, 51854.31, 62263.54, 84626.69, 49.11, 52.05, 52.17,   49.02,  22.56, 164.79),
    Row("15-sharegpt-r1",    "sharegpt", 1,   414.15, 203.15,   24.60,   21.02,    47.97,    7.76,   7.72,   8.77,    7.71,   7.67,   8.28),
    Row("16-sharegpt-r2",    "sharegpt", 2,   817.00, 400.86,   25.40,   20.90,    47.50,    7.88,   7.84,   8.91,    7.83,   7.73,   9.25),
    Row("17-sharegpt-r4",    "sharegpt", 4,   1577.71, 777.13,  26.44,   22.04,    50.55,    8.21,   8.15,   9.38,    8.16,   7.92,  19.29),
    Row("18-sharegpt-r8",    "sharegpt", 8,   2876.84, 1414.11, 28.67,   23.36,    68.71,    8.81,   8.76,  10.43,    8.75,   8.26,  35.09),
    Row("19-sharegpt-r16",   "sharegpt", 16,  4875.66, 2398.49, 34.85,   29.64,    77.10,   10.71,  10.83,  15.83,   10.46,   9.62,  36.63),
]
# fmt: on


WORKLOADS = ["decode", "prefill", "balanced", "longctx", "sharegpt"]
WORKLOAD_TITLES = {
    "decode": "Decode-heavy (8 in / 1024 out)",
    "prefill": "Prefill-heavy (4096 in / 8 out)",
    "balanced": "Balanced (512 in / 512 out)",
    "longctx": "Long context (32768 in / 512 out)",
    "sharegpt": "ShareGPT open-loop (mixed in/out)",
}
WORKLOAD_XLABEL = {
    "decode": "Max concurrency",
    "prefill": "Max concurrency",
    "balanced": "Max concurrency",
    "longctx": "Max concurrency",
    "sharegpt": "Request rate (req/s)",
}

COLOR_S = "#1f77b4"  # sglang-mini
COLOR_V = "#d62728"  # vllm-tpu
LBL_S = "tpu-sglang-mini"
LBL_V = "vllm-tpu"


def rows_for(rows: list[Row], workload: str) -> list[Row]:
    return [r for r in rows if r.workload == workload]


def setup_axes_grid():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    # We use 5 of 6 slots; hide the last.
    axes_flat[-1].axis("off")
    return fig, axes_flat


def annotate_xticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])


def plot_throughput_overview(out_dir: Path):
    fig, axes = setup_axes_grid()
    for ax, wl in zip(axes, WORKLOADS, strict=False):
        s_rows = rows_for(SGLANG, wl)
        v_rows = rows_for(VLLM, wl)
        xs = [r.x for r in s_rows]
        s_vals = [r.total_tps for r in s_rows]
        v_vals = [r.total_tps for r in v_rows]
        idx = np.arange(len(xs))
        width = 0.38
        ax.bar(idx - width / 2, s_vals, width, color=COLOR_S, label=LBL_S)
        ax.bar(idx + width / 2, v_vals, width, color=COLOR_V, label=LBL_V)
        ax.set_xticks(idx)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel(WORKLOAD_XLABEL[wl])
        ax.set_ylabel("Total throughput (tok/s)")
        ax.set_title(WORKLOAD_TITLES[wl])
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate ratio above each pair.
        for i, (sv, vv) in enumerate(zip(s_vals, v_vals, strict=True)):
            top = max(sv, vv)
            ratio = sv / vv
            ax.text(idx[i], top * 1.02, f"{ratio:.2f}×", ha="center", fontsize=8, color="#333")
        ax.set_ylim(0, max(max(s_vals), max(v_vals)) * 1.18)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Total token throughput by workload — sglang-mini vs vllm-tpu (ratio = sglang/vllm)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "01_throughput_overview.png", dpi=140)
    plt.close(fig)


def _plot_latency_metric(
    out_dir: Path, fname: str, attr_prefix: str, ylabel: str, title: str, log: bool = True
):
    """attr_prefix in {'ttft','tpot','itl'} — pulls median/mean/p99 fields."""
    fig, axes = setup_axes_grid()
    for ax, wl in zip(axes, WORKLOADS, strict=False):
        s_rows = rows_for(SGLANG, wl)
        v_rows = rows_for(VLLM, wl)
        xs = np.array([r.x for r in s_rows], dtype=float)

        def get(rows, suffix):
            return np.array([getattr(r, f"{attr_prefix}_{suffix}") for r in rows], dtype=float)

        s_med = get(s_rows, "median")
        s_mean = get(s_rows, "mean")
        s_p99 = get(s_rows, "p99")
        v_med = get(v_rows, "median")
        v_mean = get(v_rows, "mean")
        v_p99 = get(v_rows, "p99")

        # fmt: off
        # sglang
        ax.plot(xs, s_med,  color=COLOR_S, linestyle="-",  marker="o", label=f"{LBL_S} — median")
        ax.plot(xs, s_mean, color=COLOR_S, linestyle="--", marker="s", label=f"{LBL_S} — mean", alpha=0.85)
        ax.plot(xs, s_p99,  color=COLOR_S, linestyle=":",  marker="^", label=f"{LBL_S} — p99", alpha=0.7)
        # vllm
        ax.plot(xs, v_med,  color=COLOR_V, linestyle="-",  marker="o", label=f"{LBL_V} — median")
        ax.plot(xs, v_mean, color=COLOR_V, linestyle="--", marker="s", label=f"{LBL_V} — mean", alpha=0.85)
        ax.plot(xs, v_p99,  color=COLOR_V, linestyle=":",  marker="^", label=f"{LBL_V} — p99", alpha=0.7)
        # fmt: on

        ax.set_xscale("log", base=2)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel(WORKLOAD_XLABEL[wl])
        ax.set_ylabel(ylabel)
        ax.set_title(WORKLOAD_TITLES[wl])
        ax.grid(True, which="both", alpha=0.3)

    # Shared legend in the empty slot.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / fname, dpi=140)
    plt.close(fig)


def plot_ttft(out_dir: Path):
    _plot_latency_metric(
        out_dir,
        "02_ttft.png",
        "ttft",
        "TTFT (ms, log scale)",
        "Time to First Token — median / mean / p99 (log y; lower is better)",
    )


def plot_tpot(out_dir: Path):
    _plot_latency_metric(
        out_dir,
        "03_tpot.png",
        "tpot",
        "TPOT (ms, log scale)",
        "Time per Output Token — median / mean / p99 (log y; lower is better)",
    )


def plot_itl(out_dir: Path):
    _plot_latency_metric(
        out_dir,
        "04_itl.png",
        "itl",
        "Inter-token latency (ms, log scale)",
        "Inter-Token Latency — median / mean / p99 (log y; lower is better)",
    )


def plot_speedup_ratio(out_dir: Path):
    """One bar chart: sglang/vllm throughput ratio for every config, grouped & color-coded by workload."""
    all_s = SGLANG
    all_v = VLLM
    ratios = []
    labels = []
    colors = []
    palette = {
        "decode": "#1f77b4",
        "prefill": "#ff7f0e",
        "balanced": "#2ca02c",
        "longctx": "#9467bd",
        "sharegpt": "#8c564b",
    }
    for s, v in zip(all_s, all_v, strict=True):
        assert s.cfg_id == v.cfg_id
        ratios.append(s.total_tps / v.total_tps)
        labels.append(s.cfg_id.split("-", 1)[1])
        colors.append(palette[s.workload])

    fig, ax = plt.subplots(figsize=(14, 6))
    idx = np.arange(len(ratios))
    ax.bar(idx, ratios, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Throughput ratio  (sglang-mini ÷ vllm-tpu)")
    ax.set_title(
        "Per-config throughput ratio — bars above 1.0 favor sglang-mini, below favor vllm-tpu"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    # Legend by workload color.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, list(palette.keys()), title="workload", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "05_speedup_ratio.png", dpi=140)
    plt.close(fig)


# fmt: off
def plot_scheduling_callout(out_dir: Path):
    """Two-panel figure highlighting the priority trade-off:
       LEFT  — prefill workloads: TTFT (sglang wins) vs P99 ITL (vllm wins).
       RIGHT — decode C=256: per-token latency & throughput crossover.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # LEFT: prefill — TTFT bars and P99 ITL bars side by side.
    s_pre = rows_for(SGLANG, "prefill")
    v_pre = rows_for(VLLM, "prefill")
    cs = [int(r.x) for r in s_pre]
    idx = np.arange(len(cs))
    width = 0.2

    axL.bar(idx - 1.5 * width, [r.ttft_mean for r in s_pre], width, color=COLOR_S, label=f"{LBL_S} mean TTFT")
    axL.bar(idx - 0.5 * width, [r.ttft_mean for r in v_pre], width, color=COLOR_V, label=f"{LBL_V} mean TTFT")
    axL.bar(idx + 0.5 * width, [r.itl_p99 for r in s_pre], width, color=COLOR_S, alpha=0.45, hatch="///",
            label=f"{LBL_S} P99 ITL")
    axL.bar(idx + 1.5 * width, [r.itl_p99 for r in v_pre], width, color=COLOR_V, alpha=0.45, hatch="///",
            label=f"{LBL_V} P99 ITL")
    axL.set_yscale("log")
    axL.set_xticks(idx)
    axL.set_xticklabels([f"C={c}" for c in cs])
    axL.set_ylabel("Latency (ms, log scale)")
    axL.set_title("Prefill 4096/8 — sglang clears prefill faster (lower TTFT)\nbut blocks decoders (higher P99 ITL)")
    axL.grid(True, axis="y", alpha=0.3)
    axL.legend(fontsize=8, loc="upper left")

    # RIGHT: decode — total throughput crossover & TPOT.
    s_dec = rows_for(SGLANG, "decode")
    v_dec = rows_for(VLLM, "decode")
    xs = [int(r.x) for r in s_dec]
    idx = np.arange(len(xs))
    width = 0.38
    axR.bar(idx - width / 2, [r.total_tps for r in s_dec], width, color=COLOR_S, label=f"{LBL_S} throughput")
    axR.bar(idx + width / 2, [r.total_tps for r in v_dec], width, color=COLOR_V, label=f"{LBL_V} throughput")
    axR.set_xticks(idx)
    axR.set_xticklabels([f"C={x}" for x in xs])
    axR.set_ylabel("Total throughput (tok/s)")
    axR.set_title("Decode 8/1024 — sglang leads at C≤64,\nvllm pulls ahead at C=256 (priority cliff)")
    axR.grid(True, axis="y", alpha=0.3)

    # Overlay mean TPOT on a secondary axis as lines.
    axR2 = axR.twinx()
    axR2.plot(idx, [r.tpot_mean for r in s_dec], color=COLOR_S, marker="o", linestyle="--", label=f"{LBL_S} mean TPOT")
    axR2.plot(idx, [r.tpot_mean for r in v_dec], color=COLOR_V, marker="o", linestyle="--", label=f"{LBL_V} mean TPOT")
    axR2.set_ylabel("Mean TPOT (ms)")

    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axR2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.suptitle("Scheduling priority trade-off: prefill-first (sglang) vs decode-first (vllm)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "06_scheduling_callout.png", dpi=140)
    plt.close(fig)


def plot_sharegpt_detail(out_dir: Path):
    """ShareGPT is the most realistic mixed workload — give it its own figure."""
    s_rows = rows_for(SGLANG, "sharegpt")
    v_rows = rows_for(VLLM, "sharegpt")
    xs = [r.x for r in s_rows]
    idx = np.arange(len(xs))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    width = 0.38
    ax = axes[0]
    ax.bar(idx - width / 2, [r.total_tps for r in s_rows], width, color=COLOR_S, label=LBL_S)
    ax.bar(idx + width / 2, [r.total_tps for r in v_rows], width, color=COLOR_V, label=LBL_V)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"r={int(x)}" for x in xs])
    ax.set_ylabel("Total throughput (tok/s)")
    ax.set_title("Total throughput")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # TTFT median/mean/p99
    ax = axes[1]
    for rows, label, color in [(s_rows, LBL_S, COLOR_S), (v_rows, LBL_V, COLOR_V)]:
        ax.plot(xs, [r.ttft_median for r in rows], color=color, marker="o", linestyle="-",  label=f"{label} median")
        ax.plot(xs, [r.ttft_mean for r in rows],   color=color, marker="s", linestyle="--", label=f"{label} mean", alpha=0.85)
        ax.plot(xs, [r.ttft_p99 for r in rows],    color=color, marker="^", linestyle=":",  label=f"{label} p99",  alpha=0.7)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TTFT (ms, log)")
    ax.set_title("TTFT — median / mean / p99")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # TPOT
    ax = axes[2]
    for rows, label, color in [(s_rows, LBL_S, COLOR_S), (v_rows, LBL_V, COLOR_V)]:
        ax.plot(xs, [r.tpot_median for r in rows], color=color, marker="o", linestyle="-",  label=f"{label} median")
        ax.plot(xs, [r.tpot_mean for r in rows],   color=color, marker="s", linestyle="--", label=f"{label} mean", alpha=0.85)
        ax.plot(xs, [r.tpot_p99 for r in rows],    color=color, marker="^", linestyle=":",  label=f"{label} p99",  alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("TPOT — median / mean / p99")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("ShareGPT open-loop — the most realistic mixed workload", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "07_sharegpt_detail.png", dpi=140)
    plt.close(fig)
# fmt: on


def main():
    here = Path(__file__).resolve().parent
    out = here
    out.mkdir(exist_ok=True)
    plot_throughput_overview(out)
    plot_ttft(out)
    plot_tpot(out)
    plot_itl(out)
    plot_speedup_ratio(out)
    plot_scheduling_callout(out)
    plot_sharegpt_detail(out)
    print("Wrote charts to", out)
    for p in sorted(out.glob("*.png")):
        print(" -", p.name, f"{p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
