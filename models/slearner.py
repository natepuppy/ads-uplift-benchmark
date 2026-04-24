"""S-Learner (Single model) uplift estimator.

Fits a single model mu(X, T) on all data with T as a feature.
CATE is estimated as:
    tau(X) = mu(X, 1) - mu(X, 0)

Weakness: the model may learn to ignore T as a feature (regularization),
especially when treatment is a small signal relative to X. Under strong
confounding, the model conflates selection bias with the treatment effect.

Reference: Hill (2011) "Bayesian Nonparametric Modeling for Causal Inference."
"""

from __future__ import annotations

import numpy as np

from .base import BaseUpliftModel


class SLearner(BaseUpliftModel):
    """Single-model uplift estimator. T is concatenated to X as a feature."""

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        XT = np.column_stack([X, T])
        self._model = self._make_learner()
        self._model.fit(XT, Y)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        X1 = np.column_stack([X, np.ones(n)])
        X0 = np.column_stack([X, np.zeros(n)])
        p1 = self._model.predict_proba(X1)[:, 1]
        p0 = self._model.predict_proba(X0)[:, 1]
        return p1 - p0
