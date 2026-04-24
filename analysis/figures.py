"""Generate all paper figures from benchmark results.

Usage:
    python analysis/figures.py --results results/uplift_benchmark_v1/results.csv
    python analysis/figures.py --results results/uplift_benchmark_v1/results.csv --out paper/figures/

Figures produced:
    fig1_pehe_heatmap.pdf         -- method x alpha heatmap of PEHE
    fig2_degradation_curves.pdf   -- PEHE vs alpha per method (one panel per DGP)
    fig3_wasted_spend.pdf         -- wasted spend fraction vs alpha per method
    fig4_qini_bars.pdf            -- Qini coefficient across DGP types
    fig5_iroas_efficiency.pdf     -- iROAS efficiency vs oracle across methods
    fig6_calibration.pdf          -- calibration curves (illustrative single run)
    fig7_uplift_curves.pdf        -- uplift curves for one DGP configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

# Consistent color palette and model display names
MODEL_ORDER = [
    "slearner", "tlearner", "xlearner", "rlearner", "uplift_rf", "causal_forest"
]
MODEL_LABELS = {
    "slearner": "S-Learner",
    "tlearner": "T-Learner",
    "xlearner": "X-Learner",
    "rlearner": "R-Learner",
    "uplift_rf": "Uplift RF",
    "causal_forest": "Causal Forest",
}
LEARNER_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGB"}
DGP_LABELS = {
    "linear": "Linear CATE",
    "nonlinear": "Nonlinear CATE",
    "heterogeneous": "Heterogeneous CATE",
}
DGP_ORDER = ["linear", "nonlinear", "heterogeneous"]

PALETTE = sns.color_palette("tab10", n_colors=10)
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_ORDER)}

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
}

sns.set_theme(style="whitegrid", rc=STYLE)


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    # Composite model label for grouping
    df["model_label"] = df.apply(
        lambda r: f"{MODEL_LABELS.get(r['model'], r['model'])}"
                  f"[{LEARNER_LABELS.get(r['learner'], r['learner'])}]",
        axis=1,
    )
    df["model_base"] = df["model"].map(MODEL_LABELS)
    return df


def _savefig(fig: plt.Figure, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 0: Propensity overlap diagnostic — validates confounding is real
# ---------------------------------------------------------------------------

def fig_propensity_overlap(out_dir: Path):
    """Show propensity score distributions by treatment arm across alpha levels.

    Validates that:
    1. At alpha=0, treated/control propensity distributions overlap perfectly
    2. As alpha grows, the distributions separate (real confounding)
    3. At alpha=2.0, there is meaningful but not catastrophic overlap violation

    This is a methodological validation figure that reviewers appreciate.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dgp import HeterogeneousDGP

    alphas = [0.0, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)

    for ax, alpha in zip(axes, alphas):
        dgp = HeterogeneousDGP(alpha=alpha, rho=0.05, seed=42)
        sample = dgp.sample(10000)

        treated = sample.propensity[sample.T == 1]
        control = sample.propensity[sample.T == 0]

        ax.hist(control, bins=40, alpha=0.6, density=True,
                color="steelblue", label=f"Control (n={len(control):,})",
                edgecolor="none")
        ax.hist(treated, bins=40, alpha=0.6, density=True,
                color="coral", label=f"Treated (n={len(treated):,})",
                edgecolor="none")

        # Overlap statistic: fraction of treated in [0.1, 0.9] propensity range
        overlap = ((treated > 0.1) & (treated < 0.9)).mean()
        ax.set_title(f"α = {alpha}  |  Overlap = {overlap:.0%}")
        ax.set_xlabel("Propensity score P(T=1|X)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle(
        "Propensity Score Overlap by Confounding Level\n"
        "(Heterogeneous DGP — validates that α controls selection bias as intended)",
        fontsize=11, y=1.02,
    )
    _savefig(fig, out_dir, "fig0_propensity_overlap.pdf")


# ---------------------------------------------------------------------------
# Figure 1: PEHE heatmap — method x confounding level (best learner per model)
# ---------------------------------------------------------------------------

def fig_pehe_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap: x=alpha (confounding), y=model, color=PEHE. One panel per DGP."""
    # Take best learner per (model, alpha, dgp)
    best = (
        df.groupby(["dgp", "model", "learner", "alpha"])["pehe"]
        .mean()
        .reset_index()
        .loc[lambda d: d.groupby(["dgp", "model", "alpha"])["pehe"].transform("min") == d["pehe"]]
        .drop_duplicates(["dgp", "model", "alpha"])
    )

    dgps = [d for d in DGP_ORDER if d in best["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 3.5), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        pivot = (
            best[best["dgp"] == dgp]
            .pivot(index="model", columns="alpha", values="pehe")
            .reindex([m for m in MODEL_ORDER if m in best["model"].unique()])
        )
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="YlOrRd", linewidths=0.5, cbar=(ax == axes[-1]),
        )
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Confounding strength (α)")
        if ax == axes[0]:
            ax.set_ylabel("Model")
        else:
            ax.set_ylabel("")

    fig.suptitle("PEHE by Model and Confounding Strength", fontsize=12, y=1.01)
    _savefig(fig, out_dir, "fig1_pehe_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Degradation curves — PEHE vs alpha, one panel per DGP
# ---------------------------------------------------------------------------

def fig_degradation_curves(df: pd.DataFrame, out_dir: Path):
    """Line plot: PEHE vs confounding strength per method. Best learner per method."""
    best = (
        df.groupby(["dgp", "model", "learner", "alpha", "seed"])["pehe"]
        .mean()
        .groupby(["dgp", "model", "learner", "alpha"])
        .mean()
        .reset_index()
    )
    # For each (dgp, model, alpha) take the learner with lowest PEHE
    best = (
        best.loc[best.groupby(["dgp", "model", "alpha"])["pehe"].idxmin()]
        .drop_duplicates(["dgp", "model", "alpha"])
    )

    dgps = [d for d in DGP_ORDER if d in best["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 3.5), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = best[best["dgp"] == dgp]
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].sort_values("alpha")
            if len(msub) == 0:
                continue
            ax.plot(
                msub["alpha"], msub["pehe"],
                marker="o", label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS[model],
            )
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Confounding strength (α)")
        if ax == axes[0]:
            ax.set_ylabel("PEHE (lower is better)")
        ax.legend(fontsize=8)

    fig.suptitle("PEHE Degradation Under Increasing Confounding", fontsize=12, y=1.01)
    _savefig(fig, out_dir, "fig2_degradation_curves.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Wasted spend vs confounding — the "sure thing" story
# ---------------------------------------------------------------------------

def fig_wasted_spend(df: pd.DataFrame, out_dir: Path):
    """Line plot: wasted spend fraction vs alpha. Key figure for the paper narrative."""
    if "wasted_spend" not in df.columns:
        print("  Skipping wasted spend figure (metric not available)")
        return

    best = (
        df.groupby(["dgp", "model", "learner", "alpha", "seed"])["wasted_spend"]
        .mean()
        .groupby(["dgp", "model", "learner", "alpha"])
        .mean()
        .reset_index()
        .loc[lambda d: d.groupby(["dgp", "model", "alpha"])["wasted_spend"]
             .transform("min") == d["wasted_spend"]]
        .drop_duplicates(["dgp", "model", "alpha"])
    )

    dgps = [d for d in DGP_ORDER if d in best["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 3.5), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = best[best["dgp"] == dgp]
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].sort_values("alpha")
            if len(msub) == 0:
                continue
            ax.plot(
                msub["alpha"], msub["wasted_spend"] * 100,
                marker="o", label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS[model],
            )
        # Random baseline: treating top 30% at random → wasted_spend ≈ fraction sure-things
        ax.axhline(y=30, color="gray", linestyle="--", linewidth=1, label="Random policy")
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Confounding strength (α)")
        if ax == axes[0]:
            ax.set_ylabel("Wasted spend on sure-things (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Fraction of Ad Budget Wasted on Non-Incrementals\n(lower is better; top-30% treatment policy)",
        fontsize=11, y=1.02,
    )
    _savefig(fig, out_dir, "fig3_wasted_spend.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Qini coefficient bar chart — across DGP types
# ---------------------------------------------------------------------------

def fig_qini_bars(df: pd.DataFrame, out_dir: Path):
    """Bar chart: Qini coefficient grouped by DGP type."""
    if "qini" not in df.columns:
        return

    agg = (
        df.groupby(["dgp", "model", "learner"])["qini"]
        .mean()
        .reset_index()
    )
    agg["model_label"] = agg.apply(
        lambda r: f"{MODEL_LABELS.get(r['model'], r['model'])}\n"
                  f"[{LEARNER_LABELS.get(r['learner'], r['learner'])}]",
        axis=1,
    )

    dgps = [d for d in DGP_ORDER if d in agg["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 3.5), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = agg[agg["dgp"] == dgp].sort_values("qini", ascending=False)
        colors = [MODEL_COLORS.get(m, PALETTE[0]) for m in sub["model"]]
        bars = ax.barh(sub["model_label"], sub["qini"], color=colors)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("Qini coefficient")
        ax.set_xlim(-0.2, None)

    fig.suptitle("Qini Coefficient by DGP Type\n(higher is better; averaged over α and seeds)",
                 fontsize=11, y=1.02)
    _savefig(fig, out_dir, "fig4_qini_bars.pdf")


# ---------------------------------------------------------------------------
# Figure 5: iROAS efficiency — incremental revenue per ad dollar
# ---------------------------------------------------------------------------

def fig_iroas_efficiency(df: pd.DataFrame, out_dir: Path):
    """Bar chart: iROAS efficiency (model iROAS / oracle iROAS)."""
    if "iroas_efficiency" not in df.columns:
        return

    agg = (
        df.groupby(["dgp", "model", "learner"])["iroas_efficiency"]
        .mean()
        .reset_index()
    )

    dgps = [d for d in DGP_ORDER if d in agg["dgp"].unique()]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 3.5), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgps):
        sub = agg[agg["dgp"] == dgp].sort_values("iroas_efficiency", ascending=False)
        colors = [MODEL_COLORS.get(m, PALETTE[0]) for m in sub["model"]]
        ax.barh(
            sub.apply(lambda r: f"{MODEL_LABELS.get(r['model'], r['model'])}"
                                f"[{LEARNER_LABELS.get(r['learner'], r['learner'])}]", axis=1),
            sub["iroas_efficiency"],
            color=colors,
        )
        ax.axvline(1.0, color="green", linewidth=1, linestyle="--", label="Oracle")
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_xlabel("iROAS / Oracle iROAS")
        ax.set_xlim(0, None)
        ax.legend(fontsize=8)

    fig.suptitle("Incremental ROAS Efficiency Relative to Oracle\n(top-30% treatment policy)",
                 fontsize=11, y=1.02)
    _savefig(fig, out_dir, "fig5_iroas_efficiency.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Calibration plot (illustrative single run)
# ---------------------------------------------------------------------------

def fig_calibration_illustrative(out_dir: Path):
    """Generate an illustrative calibration figure using a single fresh run."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dgp import HeterogeneousDGP
    from models import SLearner, RLearner
    from metrics import calibration_data

    dgp = HeterogeneousDGP(alpha=1.5, rho=0.05, seed=42)
    sample = dgp.sample(20000)
    train, test = sample.train_test_split(test_size=0.3, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    models_to_show = [
        ("S-Learner [RF]", SLearner(learner_type="rf", seed=42)),
        ("R-Learner [RF]", RLearner(learner_type="rf", seed=42)),
    ]

    for ax, (label, model) in zip(axes, models_to_show):
        model.fit(train.X, train.T, train.Y)
        tau_pred = model.predict_cate(test.X)
        cal = calibration_data(tau_pred, test.tau, n_bins=10)

        ax.scatter(cal["pred_mean"], cal["true_mean"], s=cal["count"] / 20,
                   alpha=0.8, color="steelblue", edgecolors="k", linewidths=0.5)
        lims = [
            min(cal["pred_mean"].min(), cal["true_mean"].min()),
            max(cal["pred_mean"].max(), cal["true_mean"].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Predicted CATE")
        ax.set_ylabel("True CATE (bin mean)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("CATE Calibration: Predicted vs. True Uplift\n(Heterogeneous DGP, α=1.5)",
                 fontsize=11, y=1.02)
    _savefig(fig, out_dir, "fig6_calibration.pdf")


# ---------------------------------------------------------------------------
# Figure 7: Uplift curves — model ranking comparison
# ---------------------------------------------------------------------------

def fig_uplift_curves(out_dir: Path):
    """Uplift curves for one illustrative DGP/alpha configuration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dgp import HeterogeneousDGP
    from models import SLearner, TLearner, XLearner, RLearner, UpliftRF
    from metrics import uplift_curve

    dgp = HeterogeneousDGP(alpha=1.0, rho=0.05, seed=0)
    sample = dgp.sample(20000)
    train, test = sample.train_test_split(test_size=0.3, seed=0)

    models_to_plot = [
        ("S-Learner [RF]", SLearner(learner_type="rf", seed=0)),
        ("T-Learner [RF]", TLearner(learner_type="rf", seed=0)),
        ("X-Learner [RF]", XLearner(learner_type="rf", seed=0)),
        ("R-Learner [RF]", RLearner(learner_type="rf", seed=0)),
        ("Uplift RF", UpliftRF(seed=0)),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for (label, model), color in zip(models_to_plot, PALETTE):
        model.fit(train.X, train.T, train.Y)
        tau_pred = model.predict_cate(test.X)
        phi, lift = uplift_curve(tau_pred, test.Y, test.T)
        ax.plot(phi * 100, lift, label=label, linewidth=1.5, color=color)

    # Oracle curve (sort by true tau)
    phi_oracle, lift_oracle = uplift_curve(test.tau, test.Y, test.T)
    ax.plot(phi_oracle * 100, lift_oracle, "k--", linewidth=2, label="Oracle (true CATE)")

    # Random baseline
    ax.plot([0, 100], [0, lift_oracle[-1]], color="gray",
            linewidth=1, linestyle=":", label="Random")

    ax.set_xlabel("Population treated (%)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Uplift Curves — Heterogeneous DGP (α=1.0)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 100)

    _savefig(fig, out_dir, "fig7_uplift_curves.pdf")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_main_table(df: pd.DataFrame, out_dir: Path):
    """Generate main results table as LaTeX."""
    if "pehe" not in df.columns:
        return

    agg = (
        df.groupby(["dgp", "model", "learner"])
        .agg(
            pehe_mean=("pehe", "mean"),
            pehe_std=("pehe", "std"),
            qini_mean=("qini", "mean"),
            wasted_spend_mean=("wasted_spend", "mean"),
        )
        .reset_index()
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Benchmark results across DGP types (mean $\pm$ std over 5 seeds, $\alpha=1.0$, $\rho=0.05$)}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"DGP & Model & Learner & PEHE $\downarrow$ & Qini $\uparrow$ & Wasted Spend $\downarrow$ \\",
        r"\midrule",
    ]

    alpha1 = df[np.isclose(df["alpha"], 1.0)] if "alpha" in df.columns else df

    agg1 = (
        alpha1.groupby(["dgp", "model", "learner"])
        .agg(
            pehe_mean=("pehe", "mean"),
            pehe_std=("pehe", "std"),
            qini_mean=("qini", "mean"),
            ws_mean=("wasted_spend", "mean"),
        )
        .reset_index()
    )

    for dgp in DGP_ORDER:
        sub = agg1[agg1["dgp"] == dgp]
        if len(sub) == 0:
            continue
        lines.append(r"\multicolumn{6}{l}{\textit{" + DGP_LABELS.get(dgp, dgp) + r"}} \\")
        for _, row in sub.iterrows():
            pehe_str = f"{row['pehe_mean']:.3f} $\\pm$ {row['pehe_std']:.3f}"
            lines.append(
                f"  & {MODEL_LABELS.get(row['model'], row['model'])} "
                f"& {LEARNER_LABELS.get(row['learner'], row['learner'])} "
                f"& {pehe_str} "
                f"& {row['qini_mean']:.3f} "
                f"& {row['ws_mean']:.3f} \\\\"
            )
        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table_path = out_dir / "tab_main_results.tex"
    table_path.write_text("\n".join(lines))
    print(f"  Saved: {table_path}")


# ---------------------------------------------------------------------------
# Figure 8: Criteo real-data results — external validity check
# ---------------------------------------------------------------------------

def fig_criteo_qini(criteo_path: Path, out_dir: Path):
    """Bar chart of Qini coefficients on real Criteo data."""
    if not criteo_path.exists():
        print(f"  Skipping Criteo figure (no results at {criteo_path})")
        return

    df = pd.read_csv(criteo_path)
    df = df[df["status"] == "ok"].copy()

    agg = (
        df.groupby(["model", "learner"])["qini"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    agg["model_label"] = agg.apply(
        lambda r: f"{MODEL_LABELS.get(r['model'], r['model'])}"
                  f"[{LEARNER_LABELS.get(r['learner'], r['learner'])}]",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [MODEL_COLORS.get(m, PALETTE[0]) for m in agg["model"]]
    ax.barh(agg["model_label"], agg["mean"], xerr=agg["std"],
            color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Qini coefficient (higher is better)")
    ax.set_title("Real-Data Validation: Qini on Criteo Uplift Dataset\n"
                 "(500k users, random A/B treatment, 3 seeds)")

    _savefig(fig, out_dir, "fig8_criteo_qini.pdf")


def generate_criteo_table(criteo_path: Path, out_dir: Path):
    """Generate LaTeX table of Criteo results."""
    if not criteo_path.exists():
        return

    df = pd.read_csv(criteo_path)
    df = df[df["status"] == "ok"]

    agg = (
        df.groupby(["model", "learner"])
        .agg(
            qini_mean=("qini", "mean"),
            qini_std=("qini", "std"),
            auuc_mean=("auuc", "mean"),
            pv_mean=("policy_value_30pct", "mean"),
        )
        .reset_index()
        .sort_values("qini_mean", ascending=False)
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-data results on Criteo Uplift Dataset (500k users, mean $\pm$ std over 3 seeds). "
        r"No ground-truth CATE available; ranking metrics only.}",
        r"\label{tab:criteo_results}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Learner & Qini $\uparrow$ & AUUC $\uparrow$ & Policy Value $\uparrow$ \\",
        r"\midrule",
    ]

    for _, row in agg.iterrows():
        lines.append(
            f"{MODEL_LABELS.get(row['model'], row['model'])} "
            f"& {LEARNER_LABELS.get(row['learner'], row['learner'])} "
            f"& {row['qini_mean']:.4f} $\\pm$ {row['qini_std']:.4f} "
            f"& {row['auuc_mean']:.4f} "
            f"& {row['pv_mean']:.5f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = out_dir / "tab_criteo_results.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument("--results", required=False,
                        default="results/uplift_benchmark_v1/results.csv",
                        help="Path to benchmark results CSV")
    parser.add_argument("--out", default="paper/figures/",
                        help="Output directory for figures")
    parser.add_argument("--illustrative-only", action="store_true",
                        help="Only generate illustrative figures (no results CSV needed)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")

    # Always generate illustrative + diagnostic figures (no results CSV needed)
    print("\nGenerating illustrative and diagnostic figures...")
    fig_propensity_overlap(out_dir)
    fig_calibration_illustrative(out_dir)
    fig_uplift_curves(out_dir)

    if args.illustrative_only:
        print("Done (illustrative only).")
        return

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"\nResults file not found: {results_path}")
        print("Run: python experiments/run_benchmark.py first, then re-run this script.")
        print("Generated illustrative figures only.")
        return

    print(f"\nLoading results from {results_path}...")
    df = load_results(str(results_path))
    print(f"  Loaded {len(df)} rows")

    print("\nGenerating figures from results...")
    fig_pehe_heatmap(df, out_dir)
    fig_degradation_curves(df, out_dir)
    fig_wasted_spend(df, out_dir)
    fig_qini_bars(df, out_dir)
    fig_iroas_efficiency(df, out_dir)
    generate_main_table(df, out_dir)

    # Criteo figures (generated if criteo results exist)
    criteo_path = Path("results/criteo/criteo_results.csv")
    fig_criteo_qini(criteo_path, out_dir)
    generate_criteo_table(criteo_path, out_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
