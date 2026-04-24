"""X-Learner uplift estimator.

Three-stage meta-learner designed to handle imbalanced treatment/control splits
and to "cross-impute" treatment effects, which reduces variance under imbalance.

Algorithm:
    Stage 1: Fit mu0, mu1 (same as T-Learner)
    Stage 2: Compute imputed effects:
        D1 = Y1 - mu0(X1)   (treated units: actual - predicted counterfactual)
        D0 = mu1(X0) - Y0   (control units: predicted counterfactual - actual)
        Fit tau1(X) on (X1, D1)
        Fit tau0(X) on (X0, D0)
    Stage 3: Combine with propensity-weighted average:
        tau(X) = e(X) * tau0(X) + (1 - e(X)) * tau1(X)

The propensity-weighted combination puts more weight on whichever group
is smaller (fewer treated -> trust tau0 more, since it uses all control data).

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment
effects using machine learning." PNAS.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseUpliftModel


class XLearner(BaseUpliftModel):
    """X-Learner with propensity-weighted CATE combination."""

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        T_bool = T.astype(bool)
        X0, X1 = X[~T_bool], X[T_bool]
        Y0, Y1 = Y[~T_bool], Y[T_bool]

        # Stage 1: base models
        self._mu0 = self._make_learner()
        self._mu1 = self._make_learner()
        self._mu0.fit(X0, Y0)
        self._mu1.fit(X1, Y1)

        # Stage 2: imputed treatment effects
        D1 = Y1 - self._mu0.predict_proba(X1)[:, 1]
        D0 = self._mu1.predict_proba(X0)[:, 1] - Y0

        self._tau1 = self._make_learner()
        self._tau0 = self._make_learner()

        # Imputed effects may be continuous — binarize for classifiers via
        # binning trick, or switch to regressor. We use a regression proxy:
        # fit on shifted binary labels using clipped imputed effects.
        # For simplicity (and to keep classifier API consistent) we clip to [0,1]
        # and treat as soft labels via a regression fallback.
        self._tau1 = self._fit_effect_model(X1, D1)
        self._tau0 = self._fit_effect_model(X0, D0)

        # Stage 3: propensity model
        self._propensity = LogisticRegression(max_iter=1000, C=1.0)
        self._propensity.fit(X, T)

        return self

    def _fit_effect_model(self, X: np.ndarray, D: np.ndarray):
        """Fit a regression model for imputed effects using sklearn-compatible API."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from xgboost import XGBRegressor

        if self.learner_type == "lr":
            m = Ridge(alpha=1.0)
        elif self.learner_type == "rf":
            m = RandomForestRegressor(n_estimators=100, random_state=self.seed,
                                      n_jobs=-1, min_samples_leaf=20)
        else:  # xgb
            m = XGBRegressor(n_estimators=100, random_state=self.seed,
                             verbosity=0, n_jobs=-1)
        m.fit(X, D)
        return m

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        e = self._propensity.predict_proba(X)[:, 1]
        tau1_hat = self._tau1.predict(X)
        tau0_hat = self._tau0.predict(X)
        return e * tau0_hat + (1 - e) * tau1_hat
