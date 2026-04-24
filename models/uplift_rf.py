"""Uplift Random Forest.

A tree-based method that directly optimizes for treatment effect heterogeneity
by using a modified splitting criterion: instead of maximizing outcome purity
in each node, it maximizes the *difference* in treatment effects between
child nodes (KL divergence between treatment/control distributions).

This avoids the two-step estimation problem of meta-learners and can
capture sharp nonlinear treatment effect boundaries.

We implement two variants:
    1. DDPSplit (Rzepakowski & Jaroszewicz 2012): KL-divergence criterion
    2. Transformed Outcome approach: convert the problem to a weighted
       regression problem solvable by a standard RF.

The transformed outcome (TO) approach is simpler and more numerically stable.
It transforms each observation into a pseudo-outcome:
    Z_i = Y_i * (T_i - e(X_i)) / (e(X_i) * (1 - e(X_i)))

where e(X) is the propensity score. E[Z | X] = CATE(X) under correct
propensity specification. A standard RF fit on (X, Z) gives uplift scores.

References:
    Rzepakowski & Jaroszewicz (2012). "Decision trees for uplift modeling
    with single and multiple treatments." DMKD.

    Athey & Imbens (2016). "Recursive partitioning for heterogeneous causal
    effects." PNAS. (Causal Forest generalization)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from .base import BaseUpliftModel


class UpliftRF(BaseUpliftModel):
    """Uplift Random Forest via Transformed Outcome.

    Estimates propensity internally unless provided at predict time.
    Uses a RF regressor on the transformed outcome Z as the CATE estimator.
    """

    def __init__(
        self,
        learner_type: str = "rf",
        seed: int = 0,
        n_estimators: int = 100,
        min_samples_leaf: int = 20,
    ):
        super().__init__(learner_type=learner_type, seed=seed)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "UpliftRF":
        # Step 1: estimate propensity
        self._propensity_model = LogisticRegression(max_iter=1000, C=1.0,
                                                     random_state=self.seed)
        self._propensity_model.fit(X, T)
        e = self._propensity_model.predict_proba(X)[:, 1]
        e = np.clip(e, 0.01, 0.99)

        # Step 2: compute transformed outcome
        # Z_i = Y_i * (T_i - e_i) / (e_i * (1 - e_i))
        Z = Y * (T - e) / (e * (1 - e))

        # Step 3: fit RF on (X, Z)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
            min_samples_leaf=self.min_samples_leaf,
        )
        self._model.fit(X, Z)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def name(self) -> str:
        return "UpliftRF[TO]"
