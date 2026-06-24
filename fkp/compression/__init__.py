from .ridge import ridge_primal, ridge_dual, ridge_auto
from .svd import spectral_compress, select_rank
from .pcls import compute_pcls

__all__ = [
    "ridge_primal",
    "ridge_dual",
    "ridge_auto",
    "spectral_compress",
    "select_rank",
    "compute_pcls",
]
