from .base import BaseDGP, DGPSample
from .linear import LinearDGP
from .nonlinear import NonlinearDGP
from .heterogeneous import HeterogeneousDGP

DGP_REGISTRY = {
    "linear": LinearDGP,
    "nonlinear": NonlinearDGP,
    "heterogeneous": HeterogeneousDGP,
}

__all__ = [
    "BaseDGP",
    "DGPSample",
    "LinearDGP",
    "NonlinearDGP",
    "HeterogeneousDGP",
    "DGP_REGISTRY",
]
