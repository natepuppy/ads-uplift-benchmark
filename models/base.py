"""Base class for all uplift meta-learners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def make_base_learner(learner_type: str, seed: int = 0):
    """Factory for base learners used inside meta-learners."""
    if learner_type == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    elif learner_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1, min_samples_leaf=20)
    elif learner_type == "xgb":
        return XGBClassifier(n_estimators=100, random_state=seed,
                             eval_metric="logloss", verbosity=0,
                             use_label_encoder=False, n_jobs=-1)
    else:
        raise ValueError(f"Unknown learner_type: {learner_type!r}. "
                         f"Choose from 'lr', 'rf', 'xgb'.")


class BaseUpliftModel(ABC):
    """Abstract interface for all uplift/CATE estimators.

    All models expose:
        fit(X, T, Y) -> self
        predict_cate(X) -> np.ndarray of shape (n,)
    """

    def __init__(self, learner_type: str = "rf", seed: int = 0):
        self.learner_type = learner_type
        self.seed = seed

    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "BaseUpliftModel":
        """Fit the model on training data."""

    @abstractmethod
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Return predicted CATE for each row of X."""

    def _make_learner(self):
        return make_base_learner(self.learner_type, self.seed)

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}[{self.learner_type}]"

    def __repr__(self) -> str:
        return self.name
