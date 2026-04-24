"""T-Learner (Two model) uplift estimator.

Fits separate models on treated and control populations:
    mu0(X) = E[Y | X, T=0]
    mu1(X) = E[Y | X, T=1]

CATE estimate:
    tau(X) = mu1(X) - mu0(X)

Weakness: each model is fit on a subset of data, so sample efficiency is lower.
Under strong confounding (covariate distributions of treated/control differ greatly),
each model extrapolates into regions of low support, causing bias.

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment
effects using machine learning." PNAS.
"""

from __future__ import annotations

import numpy as np

from .base import BaseUpliftModel


class TLearner(BaseUpliftModel):
    """Two-model uplift estimator with separate models per treatment arm."""

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        T = T.astype(bool)
        self._model0 = self._make_learner()
        self._model1 = self._make_learner()
        self._model0.fit(X[~T], Y[~T])
        self._model1.fit(X[T], Y[T])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        p1 = self._model1.predict_proba(X)[:, 1]
        p0 = self._model0.predict_proba(X)[:, 1]
        return p1 - p0
