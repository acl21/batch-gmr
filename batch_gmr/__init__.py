"""
batch-gmr
===

Batch Gaussian Mixture Regression in Python with PyTorch.
"""

__version__ = "0.1.0"

from . import batch_gmm, batch_mvn

__all__ = ["batch_gmm", "batch_mvn"]

from .batch_mvn import BatchMVN
from .batch_gmm import BatchGMM

__all__.extend(
    [
        "BatchMVN",
        "BatchGMM",
    ]
)
