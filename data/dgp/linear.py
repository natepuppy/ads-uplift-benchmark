"""Linear CATE DGP.

Treatment effect is a linear function of covariates:
    tau(X) = X @ gamma

This is the easiest setting for meta-learners that assume linearity.
Under strong confounding, even linear methods can fail if propensity
is not properly accounted for.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .base import BaseDGP


class LinearDGP(BaseDGP):
    """DGP where CATE is a linear function of covariates.

    Y0 = sigmoid(X @ beta0 + offset0) + noise
    Y1 = Y0 + X @ gamma          (CATE = X @ gamma)

    Treatment assignment: T ~ Bernoulli(sigmoid(alpha * X @ conf_coef))
    """

    def __init__(
        self,
        n_features: int = 10,
        alpha: float = 1.0,
        rho: float = 0.05,
        effect_scale: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            effect_scale: Scale of treatment effects relative to baseline.
                          Higher = larger average CATE.
        """
        super().__init__(n_features=n_features, alpha=alpha, rho=rho, seed=seed)
        self.effect_scale = effect_scale
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        d = self.n_features

        # Baseline outcome coefficients
        self._beta0 = rng.standard_normal(d) / np.sqrt(d)
        # Outcome intercept tuned so E[Y0] ≈ rho
        self._offset0 = np.log(self.rho / (1 - self.rho))

        # CATE coefficients — only first half of features have effects
        self._gamma = np.zeros(d)
        active = rng.choice(d, size=max(1, d // 2), replace=False)
        self._gamma[active] = rng.standard_normal(len(active)) * self.effect_scale

        # Confounding coefficients — independent of gamma to allow control
        self._conf_coef = rng.standard_normal(d) / np.sqrt(d)

    def _sample_covariates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.n_features))

    def _compute_propensity(self, X: np.ndarray) -> np.ndarray:
        return self._confounded_propensity(X, self._conf_coef)

    def _compute_potential_outcomes(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        mu0 = X @ self._beta0 + self._offset0
        tau = X @ self._gamma  # linear CATE
        mu1 = mu0 + tau

        Y0 = self._binary_outcome(mu0, rng)
        Y1 = self._binary_outcome(mu1, rng)
        return Y0, Y1

    @property
    def name(self) -> str:
        return "LinearDGP"
