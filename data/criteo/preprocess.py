"""Preprocess the Criteo Uplift Dataset for use in the benchmark.

The Criteo dataset is a large-scale A/B test with random treatment assignment.
It has no ground-truth CATE (since it's real observational + RCT data), so we
cannot compute PEHE. However, we can compute:
    - Qini coefficient
    - AUUC
    - Policy value

These serve as external validity checks: if our synthetic benchmark rankings
correlate with Criteo rankings, the synthetic benchmark generalizes.

Dataset columns:
    f0..f11 : anonymized user features (continuous)
    treatment: 1 = saw ad, 0 = did not
    conversion: 1 = converted (purchased), 0 = did not
    visit: 1 = visited site

We use `conversion` as the outcome Y.

Usage:
    python data/criteo/preprocess.py
    python data/criteo/preprocess.py --sample 500000  # use a 500k subsample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path(__file__).parent / "criteo_uplift_v2.csv"
OUT_PATH = Path(__file__).parent / "criteo_processed.parquet"

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


def preprocess(raw_path: Path, out_path: Path, sample: int | None = None) -> pd.DataFrame:
    print(f"Loading Criteo data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Keep relevant columns
    keep = FEATURE_COLS + [TREATMENT_COL, OUTCOME_COL]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()

    # Drop rows with nulls
    before = len(df)
    df = df.dropna()
    print(f"  Dropped {before - len(df):,} rows with nulls. Remaining: {len(df):,}")

    # Treatment and outcome as int
    df[TREATMENT_COL] = df[TREATMENT_COL].astype(int)
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)

    # Summary statistics
    n_treated = (df[TREATMENT_COL] == 1).sum()
    n_control = (df[TREATMENT_COL] == 0).sum()
    cvr_treated = df[df[TREATMENT_COL] == 1][OUTCOME_COL].mean()
    cvr_control = df[df[TREATMENT_COL] == 0][OUTCOME_COL].mean()
    print(f"  Treated: {n_treated:,} ({100*n_treated/len(df):.1f}%),  "
          f"CVR: {100*cvr_treated:.3f}%")
    print(f"  Control: {n_control:,} ({100*n_control/len(df):.1f}%),  "
          f"CVR: {100*cvr_control:.3f}%")
    print(f"  ATE (observed): {100*(cvr_treated - cvr_control):.4f}%")

    if sample is not None and sample < len(df):
        # Stratified sample to preserve treatment ratio
        rng = np.random.default_rng(42)
        t1_idx = df.index[df[TREATMENT_COL] == 1]
        t0_idx = df.index[df[TREATMENT_COL] == 0]
        n1 = int(sample * len(t1_idx) / len(df))
        n0 = sample - n1
        chosen = np.concatenate([
            rng.choice(t1_idx, size=min(n1, len(t1_idx)), replace=False),
            rng.choice(t0_idx, size=min(n0, len(t0_idx)), replace=False),
        ])
        df = df.loc[chosen].copy()
        print(f"  Subsampled to {len(df):,} rows")

    # Standardize features
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df[feat_cols] = (df[feat_cols] - df[feat_cols].mean()) / (df[feat_cols].std() + 1e-8)

    df.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")
    return df


def load_criteo(path: Path = OUT_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed Criteo dataset.

    Returns:
        X: (n, 12) feature array
        T: (n,) treatment array
        Y: (n,) outcome (conversion) array
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Criteo preprocessed data not found at {path}.\n"
            f"Run: bash data/criteo/download.sh && python data/criteo/preprocess.py"
        )
    df = pd.read_parquet(path)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].values.astype(np.float64)
    T = df[TREATMENT_COL].values.astype(np.float64)
    Y = df[OUTCOME_COL].values.astype(np.float64)
    return X, T, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample to N rows (stratified by treatment)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Raw Criteo CSV not found: {args.input}")
        print("Run: bash data/criteo/download.sh")
        raise SystemExit(1)

    preprocess(Path(args.input), Path(args.output), sample=args.sample)
