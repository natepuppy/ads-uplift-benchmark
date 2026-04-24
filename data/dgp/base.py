"""Abstract base class for all Data Generating Processes (DGPs).

Each DGP samples users with covariates X, a binary treatment T (potentially
confounded by X), observed outcome Y, and both potential outcomes Y0/Y1
(which are never jointly observed in practice but available here for evaluation).

The fundamental quantity of interest is the Conditional Average Treatment Effect:
    CATE(x) = E[Y(1) - Y(0) | X = x]

Having access to true CATE is the key advantage of synthetic benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DGPSample:
    """Container for one draw from a DGP."""
    X: np.ndarray          # (n, d) covariate matrix
    T: np.ndarray          # (n,) binary treatment indicator
    Y: np.ndarray          # (n,) observed outcome  Y = T*Y1 + (1-T)*Y0
    Y0: np.ndarray         # (n,) potential outcome under control
    Y1: np.ndarray         # (n,) potential outcome under treatment
    tau: np.ndarray        # (n,) true CATE = Y1 - Y0
    propensity: np.ndarray # (n,) P(T=1|X) — true propensity score
    feature_names: list[str] = field(default_factory=list)

    @property
    def ate(self) -> float:
        """Average Treatment Effect over this sample."""
        return float(self.tau.mean())

    def to_dataframe(self) -> pd.DataFrame:
        cols = {f"x{i}": self.X[:, i] for i in range(self.X.shape[1])}
        cols.update({"T": self.T, "Y": self.Y, "Y0": self.Y0,
                     "Y1": self.Y1, "tau": self.tau,
                     "propensity": self.propensity})
        return pd.DataFrame(cols)

    def train_test_split(
        self, test_size: float = 0.3, seed: int = 0
    ) -> tuple["DGPSample", "DGPSample"]:
        rng = np.random.default_rng(seed)
        n = len(self.Y)
        idx = rng.permutation(n)
        cut = int(n * (1 - test_size))
        train_idx, test_idx = idx[:cut], idx[cut:]
        return self._subset(train_idx), self._subset(test_idx)

    def _subset(self, idx: np.ndarray) -> "DGPSample":
        return DGPSample(
            X=self.X[idx],
            T=self.T[idx],
            Y=self.Y[idx],
            Y0=self.Y0[idx],
            Y1=self.Y1[idx],
            tau=self.tau[idx],
            propensity=self.propensity[idx],
            feature_names=self.feature_names,
        )


class BaseDGP(ABC):
    """Abstract base for all DGPs.

    Subclasses must implement:
        - _sample_covariates(n, rng) -> np.ndarray
        - _compute_propensity(X) -> np.ndarray  (values in (0,1))
        - _compute_potential_outcomes(X, rng) -> tuple[Y0, Y1]
    """

    def __init__(
        self,
        n_features: int = 10,
        alpha: float = 1.0,
        rho: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            n_features: Dimensionality of covariate space.
            alpha: Confounding strength. 0 = random treatment assignment,
                   higher values = stronger dependence of T on X.
            rho: Base conversion rate under control (rough target).
            seed: Master random seed.
        """
        self.n_features = n_features
        self.alpha = alpha
        self.rho = rho
        self.seed = seed

    def sample(self, n: int, seed_offset: int = 0) -> DGPSample:
        """Draw n samples from this DGP."""
        rng = np.random.default_rng(self.seed + seed_offset)

        X = self._sample_covariates(n, rng)
        propensity = self._compute_propensity(X)
        T = rng.binomial(1, propensity).astype(np.float64)
        Y0, Y1 = self._compute_potential_outcomes(X, rng)
        tau = Y1 - Y0
        Y = T * Y1 + (1 - T) * Y0

        feature_names = [f"x{i}" for i in range(self.n_features)]
        return DGPSample(X=X, T=T, Y=Y, Y0=Y0, Y1=Y1,
                         tau=tau, propensity=propensity,
                         feature_names=feature_names)

    @abstractmethod
    def _sample_covariates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, n_features) covariate matrix."""

    @abstractmethod
    def _compute_propensity(self, X: np.ndarray) -> np.ndarray:
        """Return (n,) propensity scores P(T=1|X) in (0,1)."""

    @abstractmethod
    def _compute_potential_outcomes(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (Y0, Y1) as (n,) binary outcome arrays."""

    def _confounded_propensity(
        self, X: np.ndarray, coef: np.ndarray
    ) -> np.ndarray:
        """Sigmoid propensity with confounding controlled by alpha.

        At alpha=0, propensity = 0.5 everywhere (no confounding).
        At alpha>0, propensity depends on X via coef.
        Clipped to [0.05, 0.95] to avoid extreme overlap violations.
        """
        logit = self.alpha * (X @ coef)
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(p, 0.05, 0.95)

    @staticmethod
    def _binary_outcome(mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample binary outcomes from Bernoulli(sigmoid(mu))."""
        p = np.clip(1.0 / (1.0 + np.exp(-mu)), 1e-6, 1 - 1e-6)
        return rng.binomial(1, p).astype(np.float64)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (f"{self.name}(n_features={self.n_features}, "
                f"alpha={self.alpha}, rho={self.rho})")
