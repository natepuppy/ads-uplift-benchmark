"""Run all uplift models on the Criteo Uplift Dataset v2.

The Criteo dataset is a large-scale real A/B test with random treatment
assignment, so there is no confounding. We cannot compute PEHE (no ground
truth CATE), but we can compute:
    - Qini coefficient
    - AUUC
    - Policy value at top-30%

These real-data results serve as external validity for the synthetic benchmark:
if a method ranks well on synthetic data, does it also rank well on Criteo?

Usage:
    python experiments/run_criteo.py
    python experiments/run_criteo.py --sample 500000  # use 500k rows
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.criteo.preprocess import load_criteo, preprocess, RAW_PATH, OUT_PATH
from models import MODEL_REGISTRY
from metrics import evaluate

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results/criteo/")

MODEL_CONFIGS = [
    ("slearner",      "lr"),
    ("slearner",      "rf"),
    ("slearner",      "xgb"),
    ("tlearner",      "lr"),
    ("tlearner",      "rf"),
    ("tlearner",      "xgb"),
    ("xlearner",      "lr"),
    ("xlearner",      "rf"),
    ("xlearner",      "xgb"),
    ("rlearner",      "lr"),
    ("rlearner",      "rf"),
    ("rlearner",      "xgb"),
    ("uplift_rf",     "rf"),
    ("causal_forest", "rf"),
]


def run_single_criteo(
    X_train, T_train, Y_train,
    X_test, T_test, Y_test,
    model_name: str,
    learner_type: str,
    seed: int,
    top_k: float = 0.30,
) -> dict:
    try:
        ModelClass = MODEL_REGISTRY[model_name]
        kwargs = {"seed": seed}
        if model_name not in ("uplift_rf", "causal_forest"):
            kwargs["learner_type"] = learner_type

        t0 = time.perf_counter()
        model = ModelClass(**kwargs)
        model.fit(X_train, T_train, Y_train)
        fit_time = time.perf_counter() - t0

        tau_pred = model.predict_cate(X_test)

        # No tau_true on real data — only ranking metrics
        metrics = evaluate(
            tau_pred=tau_pred,
            Y=Y_test,
            T=T_test,
            tau_true=None,
            top_k=top_k,
        )

        return {
            "model": model_name,
            "learner": learner_type,
            "seed": seed,
            "n_train": len(Y_train),
            "n_test": len(Y_test),
            "fit_time_s": round(fit_time, 3),
            **{k: round(v, 5) for k, v in metrics.items()},
            "status": "ok",
        }
    except Exception as e:
        return {
            "model": model_name,
            "learner": learner_type,
            "seed": seed,
            "status": f"error: {e}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500_000,
                        help="Number of Criteo rows to use (default 500k for speed)")
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--top_k", type=float, default=0.30)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    # Preprocess if needed
    if not OUT_PATH.exists():
        print("Preprocessing Criteo dataset...")
        preprocess(RAW_PATH, OUT_PATH, sample=args.sample)
    else:
        print(f"Using preprocessed Criteo data: {OUT_PATH}")

    print("Loading Criteo data...")
    X, T, Y = load_criteo(OUT_PATH)
    print(f"  {len(X):,} rows, {X.shape[1]} features  "
          f"| Treatment rate: {T.mean():.3f}  "
          f"| CVR: {Y.mean():.4f}")

    # Subsample if needed
    if args.sample and args.sample < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=args.sample, replace=False)
        X, T, Y = X[idx], T[idx], Y[idx]
        print(f"  Subsampled to {len(X):,} rows")

    # Train/test split
    n = len(X)
    cut = int(n * (1 - args.test_size))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Build job list
    jobs = [
        (model_name, learner_type, seed)
        for model_name, learner_type in MODEL_CONFIGS
        for seed in range(args.n_seeds)
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "criteo_results.csv"

    print(f"\nRunning {len(jobs)} model configurations on Criteo...")

    rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(run_single_criteo)(
            X_train, T_train, Y_train,
            X_test, T_test, Y_test,
            model_name, learner_type, seed, args.top_k,
        )
        for model_name, learner_type, seed in tqdm(jobs, desc="Criteo")
    )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    ok = df[df["status"] == "ok"]
    errors = df[df["status"] != "ok"]
    print(f"\nCompleted: {len(ok)}/{len(df)} successful")
    if len(errors):
        print(errors[["model", "learner", "seed", "status"]].to_string())

    if len(ok):
        print("\n--- Criteo Results: Qini Coefficient (mean over seeds) ---")
        summary = (
            ok.groupby(["model", "learner"])["qini"]
            .agg(["mean", "std"])
            .round(4)
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        print(summary.to_string(index=False))

        print("\n--- Criteo Results: Policy Value @ 30% ---")
        pv = (
            ok.groupby(["model", "learner"])["policy_value_30pct"]
            .mean()
            .round(5)
            .sort_values(ascending=False)
        )
        print(pv.to_string())

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
