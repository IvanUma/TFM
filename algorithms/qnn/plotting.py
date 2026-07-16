from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"


def plot_evolution_progress(
    generations,
    train_soft,
    val_acc,
    val_champion,
    depth_per_gen,
    depth_champion,
    val_champion_best,
    depth_champion_best,
    train_best,
    val_best,
    approach,
    output_stem,
    output_dir,
):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Accuracy")

    sns.lineplot(
        x=generations,
        y=val_acc,
        label="Per-Gen Best",
        color="#2c7bb6",
        linestyle=":",
        ax=ax1,
        legend=False,
    )
    sns.lineplot(
        x=generations,
        y=val_champion,
        label="Best-ever Champion",
        color="#d7191c",
        ax=ax1,
        legend=False,
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Profundidad (Depth)")
    sns.lineplot(
        x=generations,
        y=depth_per_gen,
        label="Per-Gen Depth",
        color="#fdae61",
        linestyle=":",
        ax=ax2,
        legend=False,
    )
    sns.lineplot(
        x=generations,
        y=depth_champion,
        label="Champion Depth",
        color="#fdae61",
        ax=ax2,
        legend=False,
    )

    ax1.legend(
        handles=ax1.get_lines() + ax2.get_lines(),
        labels=[l.get_label() for l in ax1.get_lines() + ax2.get_lines()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=1,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(output_dir / f"{output_stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig2, ax_left = plt.subplots(figsize=(8, 5))
    ax_left.set_xlabel("Generación")
    ax_left.set_ylabel("Training Soft Score")
    sns.lineplot(
        x=generations,
        y=train_soft,
        label="Train Soft Score",
        color="#2c7bb6",
        ax=ax_left,
        legend=False,
    )

    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Validation Accuracy")
    sns.lineplot(
        x=generations,
        y=val_acc,
        label="Val Acc",
        color="#d7191c",
        linestyle="--",
        ax=ax_right,
        legend=False,
    )

    ax_left.legend(
        handles=ax_left.get_lines() + ax_right.get_lines(),
        labels=[l.get_label() for l in ax_left.get_lines() + ax_right.get_lines()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=1,
    )

    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.25)
    fig2.savefig(
        output_dir / f"{output_stem}_train_vs_val.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig2)
