"""Generate publication tables and figures from the completed experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiment_results.json"
FIGURES = ROOT / "figures"

REGIMES = {
    "label_budget_25pct": "25% (179 images)",
    "label_budget_50pct": "50% (358 images)",
    "label_budget_100pct": "100% (716 images)",
}
METHODS = {
    "compact_unet_bce_softdice_single_view": "Compact U-Net",
    "compact_unet_mean_probability_four_view_tta": "Mean-probability TTA",
    "uniform_distance_boundary_unet_single_view": "Uniform boundary loss",
    "uniform_boundary_mean_probability_four_view_factorial_arm": "Boundary loss + mean TTA",
    "uncertainty_gated_distance_boundary_unet_single_view": "Gated boundary loss",
    "compact_unet_robust_logit_consensus_four_view_tta": "Robust-logit consensus",
    "latency_matched_wider_unet_single_view": "Wider U-Net",
}
BASELINE = "compact_unet_bce_softdice_single_view"


def load_rows() -> tuple[dict, pd.DataFrame]:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for regime, methods in data["conditions"].items():
        for method, seeds in methods.items():
            for seed, result in seeds.items():
                if result.get("status") != "success":
                    continue
                row = {"regime": regime, "method": method, "seed": int(seed)}
                row.update({k: v for k, v in result.items() if k not in {"status", "dice_score_per_image"}})
                rows.append(row)
    return data, pd.DataFrame(rows)


def main() -> None:
    data, frame = load_rows()
    FIGURES.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "results" / "per_seed_metrics.csv", index=False)

    metrics = [
        "dice_score", "intersection_over_union", "boundary_f_score",
        "normalized_surface_dice", "hd95", "foreground_balanced_brier_score",
        "negative_log_likelihood", "expected_calibration_error",
    ]
    summary = frame.groupby(["regime", "method"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(ROOT / "results" / "summary_metrics.csv", index=False)

    paired = []
    for regime in REGIMES:
        base = frame[(frame.regime == regime) & (frame.method == BASELINE)].set_index("seed")
        for method in METHODS:
            if method == BASELINE:
                continue
            other = frame[(frame.regime == regime) & (frame.method == method)].set_index("seed")
            common = sorted(set(base.index) & set(other.index))
            diff = other.loc[common, "dice_score"].to_numpy() - base.loc[common, "dice_score"].to_numpy()
            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            rng = np.random.default_rng(20260926)
            boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(20000)])
            paired.append({
                "regime": regime, "method": method, "mean_dice_difference": diff.mean(),
                "ci95_low": np.quantile(boot, 0.025), "ci95_high": np.quantile(boot, 0.975),
                "wilcoxon_statistic": stat, "wilcoxon_p": p,
            })
    pd.DataFrame(paired).to_csv(ROOT / "results" / "paired_dice_comparisons.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(3)
    for color, (method, label) in zip(colors, METHODS.items()):
        means, errors = [], []
        for regime in REGIMES:
            values = frame[(frame.regime == regime) & (frame.method == method)].dice_score
            means.append(values.mean()); errors.append(values.std(ddof=1))
        ax.errorbar(x, means, yerr=errors, marker="o", capsize=3, linewidth=2, label=label, color=color)
    ax.set_xticks(x, list(REGIMES.values())); ax.set_ylabel("Mean per-image Dice")
    ax.set_title("Segmentation accuracy across label budgets (mean ± sample SD, n=5 seeds)")
    ax.set_ylim(0.82, 0.91); ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "dice_by_label_budget.png", dpi=220); plt.close(fig)

    effect = pd.DataFrame(paired)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    nonbase = [m for m in METHODS if m != BASELINE]
    for panel_index, (ax, regime) in enumerate(zip(axes, REGIMES)):
        sub = effect[effect.regime == regime].set_index("method").loc[nonbase]
        y = np.arange(len(nonbase))
        ax.errorbar(sub.mean_dice_difference, y,
                    xerr=[sub.mean_dice_difference-sub.ci95_low, sub.ci95_high-sub.mean_dice_difference],
                    fmt="o", capsize=3, color="#1f5a94")
        ax.axvline(0, color="black", linewidth=1); ax.set_title(REGIMES[regime]); ax.set_xlabel("Paired Dice difference")
        ax.set_yticks(y)
        if panel_index == 0:
            ax.set_yticklabels([METHODS[m] for m in nonbase])
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Effect relative to compact U-Net (paired seed bootstrap 95% CI)")
    fig.tight_layout(); fig.savefig(FIGURES / "paired_dice_effects.png", dpi=220); plt.close(fig)

    chosen = ["dice_score", "boundary_f_score", "normalized_surface_dice", "foreground_balanced_brier_score"]
    labels = ["Dice ↑", "Boundary F-score ↑", "Normalized surface Dice ↑", "Balanced Brier ↓"]
    full = frame[frame.regime == "label_budget_100pct"].groupby("method")[chosen].mean().loc[list(METHODS)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(axes.flat, chosen, labels):
        ax.barh(list(METHODS.values()), full[metric], color=colors)
        ax.set_title(label); ax.tick_params(axis="y", labelsize=8)
        if metric == "dice_score": ax.set_xlim(0.88, 0.90)
    fig.suptitle("Full-label test performance (mean across five seeds)")
    fig.tight_layout(); fig.savefig(FIGURES / "full_label_metrics.png", dpi=220); plt.close(fig)

    run_summary = {
        "successful_runs": int(len(frame)), "expected_runs": 3 * 7 * 5,
        "elapsed_seconds": data["harness"]["elapsed_seconds"],
        "device": data["metadata"]["device"], "torch_version": data["metadata"]["torch_version"],
        "time_guard_triggered": data["metadata"]["time_guard_triggered"],
    }
    (ROOT / "results" / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
