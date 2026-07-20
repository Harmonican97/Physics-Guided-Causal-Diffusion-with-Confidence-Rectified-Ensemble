"""Publication exports for corrected-protocol robustness and sensitivity data."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
FIGURE_DIR = ROOT / "results_corrected_ablation" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#0F4D92"
VIOLET = "#7C6CCF"
TEAL = "#42949E"
GREY = "#767676"
BLACK = "#272727"


def style():
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def band(ax, x, mean, std, color, label, marker, linestyle="-"):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ax.fill_between(
        x,
        np.clip(mean - std, 0, None),
        mean + std,
        color=color,
        alpha=0.14,
        linewidth=0,
    )
    ax.plot(
        x,
        mean,
        color=color,
        marker=marker,
        markersize=3.2,
        markeredgewidth=0.6,
        markerfacecolor="white",
        linestyle=linestyle,
        label=label,
    )


def panel_label(ax, label):
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def save_bundle(fig, stem):
    base = FIGURE_DIR / stem
    for suffix in ("svg", "pdf"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    # Manuscript-compatible raster replacement.
    fig.savefig(WORKSPACE / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_noise_robustness():
    rows = read_csv(
        ROOT
        / "results_corrected_ablation"
        / "hust_diagnostic"
        / "noise_robustness.csv"
    )
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    for values in by_method.values():
        values.sort(key=lambda row: float(row["snr_db"]))

    fig, axes = plt.subplots(2, 1, figsize=(3.50, 4.15), sharex=True)
    specifications = [
        ("unseen_accuracy", "Unseen accuracy (%)"),
        ("h_score", "H-score (%)"),
    ]
    methods = [
        ("K-Means + CRE", VIOLET, "D", "--"),
        ("HDBSCAN-CRE", BLUE, "o", "-"),
    ]
    for panel_index, (metric, ylabel) in enumerate(specifications):
        ax = axes[panel_index]
        for method, color, marker, linestyle in methods:
            values = by_method[method]
            x = np.asarray([float(row["snr_db"]) for row in values])
            mean = 100 * np.asarray([float(row[f"{metric}_mean"]) for row in values])
            std = 100 * np.asarray(
                [float(row[f"{metric}_sample_std"]) for row in values]
            )
            band(ax, x, mean, std, color, method, marker, linestyle)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
        panel_label(ax, "a" if panel_index == 0 else "b")
    axes[0].legend(loc="lower right", ncol=1)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_xticks(np.arange(-10, 11, 2))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.12, hspace=0.16)
    save_bundle(fig, "ablation_clustering_robustness")


def plot_guidance_sensitivity():
    rows = read_csv(
        ROOT
        / "results_corrected_ablation"
        / "xjtu_generation"
        / "sensitivity"
        / "summary.csv"
    )
    rows.sort(key=lambda row: float(row["guidance_scale"]))
    scale = np.asarray([float(row["guidance_scale"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(3.50, 4.15), sharex=True)
    metric_lines = [
        ("pointwise_h_score", "Shared CNN", VIOLET, "D", "--"),
        ("transductive_h_score", "Full CRE", BLUE, "o", "-"),
    ]
    for prefix, label, color, marker, linestyle in metric_lines:
        mean = 100 * np.asarray([float(row[f"{prefix}_mean"]) for row in rows])
        std = 100 * np.asarray(
            [float(row[f"{prefix}_sample_std"]) for row in rows]
        )
        band(axes[0], scale, mean, std, color, label, marker, linestyle)
    axes[0].axvline(10, color=GREY, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("H-score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axes[0].legend(loc="center right")
    panel_label(axes[0], "a")

    mmd_mean = np.asarray([float(row["mmd2_mean"]) for row in rows])
    mmd_std = np.asarray([float(row["mmd2_sample_std"]) for row in rows])
    band(axes[1], scale, mmd_mean, mmd_std, TEAL, r"MMD$^2$", "o", "-")
    axes[1].axvline(10, color=GREY, linewidth=0.8, linestyle=":")
    axes[1].set_ylabel(r"Synthetic--real MMD$^2$")
    axes[1].set_xlabel(r"Guidance scale $s$")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    axes[1].grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axes[1].set_xticks(scale)
    axes[1].set_xticklabels([f"{value:g}" for value in scale])
    panel_label(axes[1], "b")

    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.12, hspace=0.16)
    save_bundle(fig, "sensitivity_analysis_normalized")


def main():
    style()
    plot_noise_robustness()
    plot_guidance_sensitivity()


if __name__ == "__main__":
    main()
