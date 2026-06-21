"""
Feature Knowledge Projection (FKP)
===================================
A tuning-free framework for label-free, certifiable edge deployment
of frozen foundation models via split-feature inference.

Reference:
    Gaurav, S. (2026). Feature Knowledge Projection: Label-Free, Certifiable
    Edge Deployment of Frozen Foundation Models. ICLR 2027.
"""

from fkp.compression.ridge import ridge_primal, ridge_dual, ridge_auto
from fkp.compression.svd import spectral_compress, select_rank
from fkp.compression.pcls import compute_pcls
from fkp.conditioning.zca import ZCAWhitener
from fkp.conditioning.centering import center_logits
from fkp.theory.bounds import stability_bound, feature_norm_bound, projection_loss_bound
from fkp.theory.certificate import FKPCertificate

__version__ = "0.1.0"
__all__ = [
    "ridge_primal",
    "ridge_dual",
    "ridge_auto",
    "spectral_compress",
    "select_rank",
    "compute_pcls",
    "ZCAWhitener",
    "center_logits",
    "stability_bound",
    "feature_norm_bound",
    "projection_loss_bound",
    "FKPCertificate",
]
