"""Nonlinear CATE DGP.

Treatment effect involves nonlinear interactions and polynomial terms.
This stresses linear meta-learners and rewards flexible base estimators.

CATE(X) = effect_scale * sin(X[:,0] * X[:,1]) + 0.5 * (X[:,2] > 0) * X[:,3]

The interaction terms mean S-Learner (with linear base) and T-Learner (linear)
will systematically misestimate treatment effects.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDGP


class NonlinearDGP(BaseDGP):
    """DGP where CATE contains interaction and polynomial terms.

    Designed to expose failures of linear base learners and reward
    flexible models (tree-based, neural) in the meta-learner.
    """

    def __init__(
        self,
        n_features: int = 10,
        alpha: float = 1.0,
        rho: float = 0.05,
        effect_scale: float = 0.4,
        seed: int = 42,
    ):
        super().__init__(n_features=n_features, alpha=alpha, rho=rho, seed=seed)
        self.effect_scale = effect_scale
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        d = self.n_features

        self._beta0 = rng.standard_normal(d) / np.sqrt(d)
        self._offset0 = np.log(self.rho / (1 - self.rho))

        # Confounding uses a nonlinear propensity too
        self._conf_coef = rng.standard_normal(d) / np.sqrt(d)
        # Which feature indices to use for interaction terms
        self._interaction_idx = list(range(min(4, d)))

    def _sample_covariates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.n_features))

    def _compute_propensity(self, X: np.ndarray) -> np.ndarray:
        # Nonlinear propensity: sigmoid of linear + squared term
        idx = self._interaction_idx
        logit = self.alpha * (X @ self._conf_coef)
        if len(idx) >= 2:
            # Add a nonlinear component to confounding
            logit += 0.5 * self.alpha * X[:, idx[0]] * X[:, idx[1]]
        return np.clip(1.0 / (1.0 + np.exp(-logit)), 0.05, 0.95)

    def _compute_potential_outcomes(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        mu0 = X @ self._beta0 + self._offset0

        # Nonlinear CATE construction
        idx = self._interaction_idx
        tau = np.zeros(len(X))

        if len(idx) >= 2:
            # Interaction term
            tau += self.effect_scale * np.sin(X[:, idx[0]] * X[:, idx[1]])
        if len(idx) >= 4:
            # Threshold interaction
            tau += 0.5 * self.effect_scale * (X[:, idx[2]] > 0) * X[:, idx[3]]
        if len(idx) >= 1:
            # Squared term
            tau += 0.2 * self.effect_scale * X[:, idx[0]] ** 2

        mu1 = mu0 + tau
        Y0 = self._binary_outcome(mu0, rng)
        Y1 = self._binary_outcome(mu1, rng)
        return Y0, Y1

    @property
    def name(self) -> str:
        return "NonlinearDGP"
