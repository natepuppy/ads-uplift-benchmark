"""Causal Forest DML — the current gold standard in CATE estimation.

CausalForestDML combines:
  1. Double Machine Learning (DML) debiasing — cross-fitted residualization
     of Y and T on X, removing the confounding signal before CATE estimation
  2. Generalized Random Forest (GRF) — forest that optimizes splitting on
     treatment effect heterogeneity, not outcome prediction

This is the most robust method available for CATE estimation under
confounding, and is the primary benchmark any new method must beat.

Reference:
    Wager & Athey (2018). "Estimation and inference of heterogeneous
    treatment effects using random forests." JASA.

    Athey, Tibshirani & Wager (2019). "Generalized random forests."
    Annals of Statistics.

    Chernozhukov et al. (2018). "Double/debiased machine learning."
    Econometrics Journal.
"""

from __future__ import annotations

import numpy as np

from .base import BaseUpliftModel


class CausalForest(BaseUpliftModel):
    """Causal Forest via econml's CausalForestDML.

    Uses 5-fold cross-fitting for nuisance models (outcome and propensity),
    then fits a causal forest on the residualized data.

    The DML step makes this doubly robust: misspecification of either
    the outcome or propensity model (but not both) does not bias estimates.
    """

    def __init__(
        self,
        learner_type: str = "rf",   # used for nuisance models
        seed: int = 0,
        n_estimators: int = 100,
        n_folds: int = 5,
    ):
        super().__init__(learner_type=learner_type, seed=seed)
        self.n_estimators = n_estimators
        self.n_folds = n_folds

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "CausalForest":
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import Ridge, LogisticRegression

        discrete_outcome = len(np.unique(Y)) == 2

        # model_y must be a classifier when discrete_outcome=True,
        # a regressor otherwise. model_t is always a classifier for binary T.
        if self.learner_type == "lr":
            model_y = (LogisticRegression(max_iter=1000, C=1.0,
                                          random_state=self.seed)
                       if discrete_outcome else Ridge(alpha=1.0))
            model_t = LogisticRegression(max_iter=1000, C=1.0,
                                         random_state=self.seed)
        elif self.learner_type == "rf":
            model_y = (RandomForestClassifier(n_estimators=100,
                                              random_state=self.seed,
                                              n_jobs=-1, min_samples_leaf=20)
                       if discrete_outcome
                       else RandomForestRegressor(n_estimators=100,
                                                  random_state=self.seed,
                                                  n_jobs=-1, min_samples_leaf=20))
            model_t = RandomForestClassifier(n_estimators=100,
                                             random_state=self.seed,
                                             n_jobs=-1, min_samples_leaf=20)
        else:  # xgb
            from xgboost import XGBRegressor, XGBClassifier
            model_y = (XGBClassifier(n_estimators=100, random_state=self.seed,
                                     eval_metric="logloss", verbosity=0, n_jobs=-1)
                       if discrete_outcome
                       else XGBRegressor(n_estimators=100, random_state=self.seed,
                                         verbosity=0, n_jobs=-1))
            model_t = XGBClassifier(n_estimators=100, random_state=self.seed,
                                    eval_metric="logloss", verbosity=0, n_jobs=-1)

        discrete_treatment = len(np.unique(T)) == 2

        self._model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            cv=self.n_folds,
            n_jobs=-1,
            discrete_outcome=discrete_outcome,
            discrete_treatment=discrete_treatment,
        )
        self._model.fit(Y, T, X=X)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self._model.effect(X).flatten()

    @property
    def name(self) -> str:
        return f"CausalForest[{self.learner_type}]"
