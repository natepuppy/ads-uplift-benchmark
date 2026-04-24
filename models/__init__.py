from .slearner import SLearner
from .tlearner import TLearner
from .xlearner import XLearner
from .rlearner import RLearner
from .uplift_rf import UpliftRF
from .causal_forest import CausalForest
from .base import BaseUpliftModel, make_base_learner

MODEL_REGISTRY = {
    "slearner": SLearner,
    "tlearner": TLearner,
    "xlearner": XLearner,
    "rlearner": RLearner,
    "uplift_rf": UpliftRF,
    "causal_forest": CausalForest,
}

LEARNER_TYPES = ["lr", "rf", "xgb"]

__all__ = [
    "SLearner", "TLearner", "XLearner", "RLearner", "UpliftRF", "CausalForest",
    "BaseUpliftModel", "make_base_learner",
    "MODEL_REGISTRY", "LEARNER_TYPES",
]
