"""Heterogeneous (subgroup) CATE DGP.

The population is a mixture of subgroups with qualitatively different treatment
responses: some users are helped by ads, some are hurt (persuadables vs.
sleeping dogs), and some are unaffected. This mirrors real ad targeting scenarios.

Subgroup structure:
    - Persuadables  (40%): tau > 0, benefit from being shown ad
    - Sure things   (20%): high Y0 regardless, tau ≈ 0
    - Lost causes   (20%): low Y0 regardless, tau ≈ 0
    - Sleeping dogs (20%): tau < 0, hurt by ad (e.g., competitor awareness)

The presence of sleeping dogs is particularly important: a naive uplift model
that doesn't capture subgroup heterogeneity may inadvertently target them,
reducing overall conversion.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDGP


class HeterogeneousDGP(BaseDGP):
    """DGP with discrete subgroup structure and sign-varying CATE.

    Subgroup membership is determined by a latent partition of X-space.
    This creates a setting where global linear estimators are systematically
    wrong for subgroups, even with unlimited data.
    """

    # Subgroup labels for interpretability in plots
    SUBGROUP_NAMES = ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]
    SUBGROUP_WEIGHTS = [0.40, 0.20, 0.20, 0.20]

    def __init__(
        self,
        n_features: int = 10,
        alpha: float = 1.0,
        rho: float = 0.05,
        effect_scale: float = 0.5,
        seed: int = 42,
    ):
        super().__init__(n_features=n_features, alpha=alpha, rho=rho, seed=seed)
        self.effect_scale = effect_scale
        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        d = self.n_features

        # Each subgroup defined by a centroid in X-space
        self._centroids = rng.standard_normal((4, d))

        # Per-subgroup CATE magnitudes
        # Persuadable: positive effect
        # Sure thing: near-zero effect
        # Lost cause: near-zero effect
        # Sleeping dog: negative effect
        self._tau_means = np.array([
            self.effect_scale,       # persuadable
            0.02,                    # sure thing
            0.02,                    # lost cause
            -self.effect_scale * 0.6 # sleeping dog
        ])

        # Per-subgroup baseline conversion rates
        self._rho_per_group = np.array([
            self.rho,
            self.rho * 4,   # sure things convert anyway
            self.rho * 0.2, # lost causes rarely convert
            self.rho,
        ])

        self._beta0 = rng.standard_normal(d) / np.sqrt(d)
        self._conf_coef = rng.standard_normal(d) / np.sqrt(d)

    def _assign_subgroups(self, X: np.ndarray) -> np.ndarray:
        """Assign each user to nearest centroid subgroup."""
        dists = np.array([
            np.sum((X - c) ** 2, axis=1)
            for c in self._centroids
        ])  # (4, n)
        return np.argmin(dists, axis=0)  # (n,)

    def _sample_covariates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.n_features))

    def _compute_propensity(self, X: np.ndarray) -> np.ndarray:
        # Confounding: treatment is more likely for "persuadables" who also look
        # like high-value users to the ad system
        groups = self._assign_subgroups(X)
        base_logit = self.alpha * (X @ self._conf_coef)
        # Sleeping dogs (group 3) are sometimes OVER-targeted (realistic mistake)
        group_bias = np.where(groups == 0, 0.3 * self.alpha,
                     np.where(groups == 3, 0.4 * self.alpha, 0.0))
        return np.clip(1.0 / (1.0 + np.exp(-(base_logit + group_bias))), 0.05, 0.95)

    def _compute_potential_outcomes(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        groups = self._assign_subgroups(X)
        n = len(X)

        # Per-sample baseline
        offset = np.log(self._rho_per_group[groups] /
                        (1 - self._rho_per_group[groups] + 1e-8))
        mu0 = X @ self._beta0 * 0.3 + offset

        # Per-sample CATE (add small within-group noise)
        tau = (self._tau_means[groups] +
               rng.standard_normal(n) * 0.05 * self.effect_scale)

        mu1 = mu0 + tau
        Y0 = self._binary_outcome(mu0, rng)
        Y1 = self._binary_outcome(mu1, rng)
        return Y0, Y1

    @property
    def name(self) -> str:
        return "HeterogeneousDGP"
