"""Main experiment runner for the uplift benchmark.

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --config experiments/configs/benchmark.yaml
    python experiments/run_benchmark.py --fast   # small grid for smoke test
    python experiments/run_benchmark.py --dgp linear --alpha 1.0  # single config

Results are saved to results/<experiment_name>/results.csv
A summary table is printed to stdout on completion.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dgp import DGP_REGISTRY
from models import MODEL_REGISTRY
from metrics import evaluate

warnings.filterwarnings("ignore")


def run_single(
    dgp_name: str,
    dgp_kwargs: dict,
    alpha: float,
    rho: float,
    model_name: str,
    learner_type: str,
    seed: int,
    test_size: float,
    top_k: float,
    economics: dict,
) -> dict:
    """Run one (DGP, alpha, rho, model, learner, seed) combination.

    Returns a flat dict of results to be aggregated into the results DataFrame.
    """
    try:
        # Build DGP
        DGPClass = DGP_REGISTRY[dgp_name]
        dgp = DGPClass(
            alpha=alpha,
            rho=rho,
            seed=seed,
            **{k: v for k, v in dgp_kwargs.items()
               if k not in ("name", "n_samples")},
        )
        n_samples = dgp_kwargs.get("n_samples", 20000)
        sample = dgp.sample(n_samples, seed_offset=seed * 1000)
        train, test = sample.train_test_split(test_size=test_size, seed=seed)

        # Build model
        ModelClass = MODEL_REGISTRY[model_name]
        model_kwargs: dict[str, Any] = {"seed": seed}
        if model_name != "uplift_rf":
            model_kwargs["learner_type"] = learner_type

        t0 = time.perf_counter()
        model = ModelClass(**model_kwargs)
        model.fit(train.X, train.T, train.Y)
        fit_time = time.perf_counter() - t0

        # Predict CATE on test set
        tau_pred = model.predict_cate(test.X)

        # Evaluate
        metrics = evaluate(
            tau_pred=tau_pred,
            Y=test.Y,
            T=test.T,
            tau_true=test.tau,
            top_k=top_k,
        )

        return {
            "dgp": dgp_name,
            "alpha": alpha,
            "rho": rho,
            "model": model_name,
            "learner": learner_type,
            "seed": seed,
            "n_train": len(train.Y),
            "n_test": len(test.Y),
            "fit_time_s": round(fit_time, 3),
            "true_ate": round(float(test.tau.mean()), 5),
            **{k: round(v, 5) for k, v in metrics.items()},
            "status": "ok",
        }

    except Exception as e:
        return {
            "dgp": dgp_name,
            "alpha": alpha,
            "rho": rho,
            "model": model_name,
            "learner": learner_type,
            "seed": seed,
            "status": f"error: {e}",
        }


def build_job_list(cfg: dict, fast: bool = False) -> list[dict]:
    """Expand config into list of individual job kwargs."""
    alpha_grid = [0.0, 1.0] if fast else cfg["experiment"].get("alpha_grid", [0.0, 1.0, 2.0])
    rho_grid = [0.05] if fast else cfg["experiment"].get("rho_grid", [0.05, 0.10])
    n_seeds = 2 if fast else cfg["experiment"].get("n_seeds", 5)
    dgps = cfg["dgps"][:1] if fast else cfg["dgps"]
    models = cfg["models"][:2] if fast else cfg["models"]

    jobs = []
    for dgp_cfg, alpha, rho, model_cfg, seed in itertools.product(
        dgps, alpha_grid, rho_grid, models, range(n_seeds)
    ):
        for learner_type in model_cfg.get("learner_types", ["rf"]):
            jobs.append({
                "dgp_name": dgp_cfg["name"],
                "dgp_kwargs": dgp_cfg,
                "alpha": alpha,
                "rho": rho,
                "model_name": model_cfg["name"],
                "learner_type": learner_type,
                "seed": seed,
                "test_size": cfg["experiment"].get("test_size", 0.30),
                "top_k": cfg.get("top_k", 0.30),
                "economics": cfg.get("economics", {}),
            })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run uplift benchmark experiments.")
    parser.add_argument("--config", default="experiments/configs/benchmark.yaml")
    parser.add_argument("--fast", action="store_true",
                        help="Run a small smoke-test grid (2 alphas, 1 DGP, 2 seeds)")
    parser.add_argument("--dgp", default=None, help="Override: run only this DGP")
    parser.add_argument("--model", default=None, help="Override: run only this model")
    parser.add_argument("--alpha", type=float, default=None, help="Override: single alpha")
    parser.add_argument("--n_jobs", type=int, default=None, help="Override parallelism")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_jobs is not None:
        cfg["experiment"]["n_jobs"] = args.n_jobs

    # Apply CLI overrides
    if args.dgp:
        cfg["dgps"] = [d for d in cfg["dgps"] if d["name"] == args.dgp]
    if args.model:
        cfg["models"] = [m for m in cfg["models"] if m["name"] == args.model]
    if args.alpha is not None:
        cfg["experiment"]["alpha_grid"] = [args.alpha]

    jobs = build_job_list(cfg, fast=args.fast)
    n_jobs_parallel = cfg["experiment"].get("n_jobs", -1)

    results_dir = Path(cfg["experiment"]["results_dir"]) / cfg["experiment"]["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "results.csv"

    print(f"Running {len(jobs)} experiment configurations "
          f"({'smoke test' if args.fast else 'full grid'})")
    print(f"Results will be saved to: {out_path}\n")

    t_start = time.perf_counter()

    rows = Parallel(n_jobs=n_jobs_parallel, verbose=0)(
        delayed(run_single)(**job) for job in tqdm(jobs, desc="Experiments")
    )

    elapsed = time.perf_counter() - t_start
    df = pd.DataFrame(rows)

    # Merge with existing results if present (for targeted reruns)
    if out_path.exists() and (args.model or args.dgp or args.alpha is not None):
        existing = pd.read_csv(out_path)
        # Drop rows that match the current run's scope (will be replaced)
        mask = pd.Series([True] * len(existing))
        if args.model:
            mask &= existing["model"] == args.model
        if args.dgp:
            mask &= existing["dgp"] == args.dgp
        if args.alpha is not None:
            mask &= existing["alpha"].apply(lambda a: abs(a - args.alpha) < 1e-9)
        existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(out_path, index=False)

    # Print summary
    ok = df[df["status"] == "ok"]
    errors = df[df["status"] != "ok"]
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Successful: {len(ok)} / {len(df)}")
    if len(errors) > 0:
        print(f"  Errors: {len(errors)}")
        print(errors[["dgp", "model", "learner", "alpha", "seed", "status"]].to_string())

    if len(ok) > 0:
        print("\n--- PEHE Summary (mean ± std across seeds) ---")
        summary = (
            ok.groupby(["dgp", "alpha", "model", "learner"])["pehe"]
            .agg(["mean", "std"])
            .round(4)
            .reset_index()
        )
        print(summary.to_string(index=False))

        print("\n--- Wasted Spend @ 30% Summary ---")
        ws_summary = (
            ok.groupby(["dgp", "alpha", "model", "learner"])["wasted_spend"]
            .agg(["mean", "std"])
            .round(4)
            .reset_index()
        )
        print(ws_summary.to_string(index=False))

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
