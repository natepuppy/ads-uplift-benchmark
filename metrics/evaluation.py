"""Evaluation metrics for uplift / CATE models.

Metrics are grouped by whether they require synthetic ground truth:

Ground-truth metrics (synthetic only):
    - PEHE: Precision in Estimation of Heterogeneous Effects
    - ATE error: bias in the average treatment effect estimate

Ranking metrics (work on real data):
    - Qini coefficient
    - AUUC: Area Under the Uplift Curve
    - Policy value at top-k%

Cost/ROI metrics (synthetic + real):
    - Wasted spend fraction: fraction of treated budget spent on sure-things
    - Incremental ROAS: revenue lift per dollar of ad spend under a policy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# NumPy 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ---------------------------------------------------------------------------
# Ground-truth metrics (require true CATE)
# ---------------------------------------------------------------------------

def pehe(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effects.

    sqrt(E[(tau_pred(X) - tau_true(X))^2])

    Lower is better. This is only computable when true CATE is known (synthetic).
    """
    return float(np.sqrt(np.mean((tau_pred - tau_true) ** 2)))


def ate_error(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """Absolute error in average treatment effect estimate."""
    return float(abs(tau_pred.mean() - tau_true.mean()))


def r_squared_cate(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """R² of CATE prediction (how much variance in true CATE is explained)."""
    ss_res = np.sum((tau_true - tau_pred) ** 2)
    ss_tot = np.sum((tau_true - tau_true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Ranking / uplift curve metrics (work on real data)
# ---------------------------------------------------------------------------

def uplift_curve(
    tau_pred: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the uplift curve.

    Sort users by predicted CATE descending. At each fraction phi of the
    population treated, compute the realized uplift vs. random policy.

    Returns:
        phi: array of population fractions [0, 1]
        uplift: cumulative incremental conversions at each phi

    The uplift curve is the primary visual tool in the paper.
    """
    n = len(Y)
    order = np.argsort(-tau_pred)  # descending
    Y_sorted = Y[order]
    T_sorted = T[order]

    # Running treated/control counts and outcomes
    treated_mask = T_sorted == 1
    control_mask = T_sorted == 0

    n_treated_cum = np.cumsum(treated_mask).astype(float)
    n_control_cum = np.cumsum(control_mask).astype(float)
    y_treated_cum = np.cumsum(Y_sorted * treated_mask)
    y_control_cum = np.cumsum(Y_sorted * control_mask)

    # Avoid division by zero
    safe_nt = np.where(n_treated_cum > 0, n_treated_cum, 1)
    safe_nc = np.where(n_control_cum > 0, n_control_cum, 1)

    # Incremental lift vs. random at each rank
    lift = (y_treated_cum / safe_nt - y_control_cum / safe_nc) * (
        n_treated_cum + n_control_cum
    )

    phi = np.arange(1, n + 1) / n
    return phi, lift


def auuc(
    tau_pred: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    normalize: bool = True,
) -> float:
    """Area Under the Uplift Curve.

    Measures how much incremental lift is captured by targeting users
    in order of predicted CATE, relative to random targeting.

    Args:
        normalize: if True, divide by AUUC of perfect model (oracle ranking).
    """
    phi, lift = uplift_curve(tau_pred, Y, T)
    area = float(_trapz(lift, phi))

    if normalize:
        random_area = float(_trapz(np.linspace(0, lift[-1], len(lift)), phi))
        denom = abs(area - random_area)
        if denom < 1e-10:
            return 0.0
        return (area - random_area) / abs(random_area) if random_area != 0 else area
    return area


def qini_coefficient(
    tau_pred: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> float:
    """Qini coefficient (Radcliffe 2007).

    The Qini coefficient is the area between the model's uplift curve
    and the random-targeting baseline, normalized by the total observed uplift.

    A Qini of 1.0 means perfect ranking. A Qini of 0 means no better than random.
    Negative values mean the model is harmful (worse than random).
    """
    phi, lift = uplift_curve(tau_pred, Y, T)
    n = len(Y)
    n_t = T.sum()
    n_c = n - n_t

    # Random baseline: linear from 0 to total observed lift
    total_treated_outcome = Y[T == 1].sum()
    total_control_outcome = Y[T == 0].sum()
    if n_t > 0 and n_c > 0:
        total_lift = (total_treated_outcome / n_t - total_control_outcome / n_c) * n
    else:
        total_lift = 0.0

    random_lift = np.linspace(0, total_lift, len(phi))

    model_area = float(_trapz(lift, phi))
    random_area = float(_trapz(random_lift, phi))

    denom = abs(random_area)
    if denom < 1e-10:
        return 0.0
    return float((model_area - random_area) / denom)


def policy_value(
    tau_pred: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    top_k: float = 0.3,
) -> float:
    """Expected conversion rate improvement under a top-k% treatment policy.

    Given a fixed budget to treat top_k fraction of users:
        1. Rank by predicted CATE descending
        2. Treat top_k * n users
        3. Measure realized lift vs. random-treat policy

    Returns the realized lift (positive = better than random).
    """
    n = len(Y)
    k = int(top_k * n)
    order = np.argsort(-tau_pred)
    top_idx = order[:k]
    rest_idx = order[k:]

    def arm_rate(idx: np.ndarray) -> float:
        t = T[idx]
        y = Y[idx]
        nt = t.sum()
        nc = len(t) - nt
        if nt == 0 or nc == 0:
            return 0.0
        return float(y[t == 1].mean() - y[t == 0].mean())

    top_rate = arm_rate(top_idx)
    rest_rate = arm_rate(rest_idx) if len(rest_idx) > 0 else 0.0
    return float(top_rate - rest_rate)


# ---------------------------------------------------------------------------
# Cost / ROI metrics — the "sure thing" wasted spend story
# ---------------------------------------------------------------------------

def wasted_spend_fraction(
    tau_pred: np.ndarray,
    tau_true: np.ndarray,
    top_k: float = 0.3,
    sure_thing_threshold: float = 0.01,
) -> float:
    """Fraction of ad budget spent on "sure things" under a top-k% policy.

    Sure things: users with true CATE < sure_thing_threshold (they would
    convert regardless of the ad). Targeting them wastes spend.

    Args:
        tau_pred: predicted CATE scores (used for ranking)
        tau_true: ground-truth CATE (only available with synthetic data)
        top_k: fraction of population to target
        sure_thing_threshold: CATE below this = sure thing (or lost cause)

    Returns:
        Fraction of targeted users who are sure-things (wasted spend rate).
    """
    n = len(tau_pred)
    k = int(top_k * n)
    order = np.argsort(-tau_pred)
    targeted_idx = order[:k]

    true_cate_targeted = tau_true[targeted_idx]
    n_sure_things = (true_cate_targeted < sure_thing_threshold).sum()
    return float(n_sure_things / k)


def incremental_roas(
    tau_pred: np.ndarray,
    tau_true: np.ndarray,
    top_k: float = 0.3,
    revenue_per_conversion: float = 50.0,
    cost_per_impression: float = 1.0,
) -> float:
    """Incremental Return on Ad Spend under top-k% targeting policy.

    iROAS = (incremental conversions * revenue) / (ad spend)

    Higher is better. A random policy has iROAS ~ (ATE * revenue / cost).
    A perfect policy concentrates spend on high-tau users.

    Args:
        tau_pred: predicted CATE for ranking
        tau_true: ground-truth CATE (synthetic only)
        top_k: fraction of population to target
        revenue_per_conversion: assumed revenue per conversion (dollars)
        cost_per_impression: cost per user shown the ad (dollars)
    """
    n = len(tau_pred)
    k = int(top_k * n)
    order = np.argsort(-tau_pred)
    targeted_idx = order[:k]

    avg_true_cate = tau_true[targeted_idx].mean()
    incremental_conversions = avg_true_cate * k
    incremental_revenue = incremental_conversions * revenue_per_conversion
    ad_spend = k * cost_per_impression

    return float(incremental_revenue / ad_spend) if ad_spend > 0 else 0.0


def oracle_incremental_roas(
    tau_true: np.ndarray,
    top_k: float = 0.3,
    revenue_per_conversion: float = 50.0,
    cost_per_impression: float = 1.0,
) -> float:
    """iROAS achievable by a perfect model (oracle upper bound)."""
    return incremental_roas(tau_true, tau_true, top_k,
                            revenue_per_conversion, cost_per_impression)


def calibration_data(
    tau_pred: np.ndarray,
    tau_true: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration of predicted CATE vs. true CATE.

    Bins users by predicted uplift quantile; checks whether mean predicted
    uplift matches mean true uplift within each bin.

    Returns a DataFrame with columns: bin_center, pred_mean, true_mean, count.
    """
    bins = np.quantile(tau_pred, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9  # ensure last point included

    records = []
    for i in range(n_bins):
        mask = (tau_pred >= bins[i]) & (tau_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        records.append({
            "bin_center": float((bins[i] + bins[i + 1]) / 2),
            "pred_mean": float(tau_pred[mask].mean()),
            "true_mean": float(tau_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Composite evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    tau_pred: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    tau_true: Optional[np.ndarray] = None,
    top_k: float = 0.30,
) -> dict:
    """Run all applicable metrics and return as a dict.

    Args:
        tau_pred: predicted CATE from the model
        Y: observed binary outcome
        T: binary treatment indicator
        tau_true: ground-truth CATE (None for real datasets)
        top_k: fraction of population for policy-based metrics

    Returns:
        dict of metric_name -> float
    """
    results = {}

    # Ranking metrics (always available)
    results["qini"] = qini_coefficient(tau_pred, Y, T)
    results["auuc"] = auuc(tau_pred, Y, T)
    results["policy_value_30pct"] = policy_value(tau_pred, Y, T, top_k=top_k)

    # Ground-truth metrics (synthetic only)
    if tau_true is not None:
        results["pehe"] = pehe(tau_true, tau_pred)
        results["ate_error"] = ate_error(tau_true, tau_pred)
        results["r2_cate"] = r_squared_cate(tau_true, tau_pred)
        results["wasted_spend"] = wasted_spend_fraction(tau_pred, tau_true, top_k)
        results["iroas"] = incremental_roas(tau_pred, tau_true, top_k)
        results["oracle_iroas"] = oracle_incremental_roas(tau_true, top_k)
        results["iroas_efficiency"] = (
            results["iroas"] / results["oracle_iroas"]
            if results["oracle_iroas"] > 1e-6 else 0.0
        )

    return results
