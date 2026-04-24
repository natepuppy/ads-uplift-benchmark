"""R-Learner (Robinson decomposition) uplift estimator.

Based on the partial linear model decomposition. Uses cross-fitting to
de-bias the CATE estimate by partialling out the main effects of X and
the propensity score.

Algorithm (simplified, single-fold version):
    1. Fit m(X) = E[Y | X]  (outcome nuisance)
    2. Fit e(X) = E[T | X]  (propensity nuisance)
    3. Compute residuals:
        Y_res = Y - m(X)
        T_res = T - e(X)
    4. Fit tau(X) by regressing Y_res on T_res * X (Robinson 1988):
        tau(X) = argmin_tau sum_i (Y_res_i - tau(X_i) * T_res_i)^2

The R-Learner is doubly robust in the sense that misspecification of either
the outcome or propensity model (but not both) leads to a consistent estimator.

We implement cross-fitting (k=5) to avoid overfitting in the nuisance models,
which is critical for valid inference.

Reference: Nie & Wager (2021) "Quasi-oracle estimation of heterogeneous treatment
effects." Biometrika.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

from .base import BaseUpliftModel


def _make_regressor(learner_type: str, seed: int = 0):
    if learner_type == "lr":
        return Ridge(alpha=1.0)
    elif learner_type == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=seed,
                                     n_jobs=-1, min_samples_leaf=20)
    else:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=seed,
                            verbosity=0, n_jobs=-1)


def _make_classifier(learner_type: str, seed: int = 0):
    if learner_type == "lr":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    elif learner_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1, min_samples_leaf=20)
    else:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, random_state=seed,
                             eval_metric="logloss", verbosity=0, n_jobs=-1)


class RLearner(BaseUpliftModel):
    """R-Learner with 5-fold cross-fitting for nuisance models."""

    def __init__(self, learner_type: str = "rf", seed: int = 0, n_folds: int = 5):
        super().__init__(learner_type=learner_type, seed=seed)
        self.n_folds = n_folds

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "RLearner":
        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        m_hat = np.zeros(n)   # E[Y|X] predictions
        e_hat = np.zeros(n)   # E[T|X] predictions

        # Cross-fitting: fit nuisance on fold complement, predict on fold
        for train_idx, val_idx in kf.split(X):
            m_fold = _make_regressor(self.learner_type, self.seed)
            m_fold.fit(X[train_idx], Y[train_idx])
            m_hat[val_idx] = m_fold.predict(X[val_idx])

            e_fold = _make_classifier(self.learner_type, self.seed)
            e_fold.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = e_fold.predict_proba(X[val_idx])[:, 1]

        # Clip propensity to avoid division instability
        e_hat = np.clip(e_hat, 0.01, 0.99)

        Y_res = Y - m_hat
        T_res = T - e_hat

        # Final stage: regress Y_res ~ tau(X) * T_res
        # Equivalent to weighted regression: use T_res as sample weights
        # and (Y_res / T_res) as pseudo-outcome (numerically unstable),
        # or equivalently fit on (X, Y_res/T_res) with weights T_res^2.
        weights = T_res ** 2
        pseudo_outcome = np.where(np.abs(T_res) > 1e-6, Y_res / T_res, 0.0)

        self._final_model = _make_regressor(self.learner_type, self.seed)
        self._final_model.fit(X, pseudo_outcome, sample_weight=weights)

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self._final_model.predict(X)
