"""
Layer-wise ZCA Whitening with PCA-ZCA hybrid for the D > m case.

ZCA whitening transforms raw multi-layer embeddings so that each layer's
features have zero mean and identity covariance.  This is critical for
FKP because it ensures the Frobenius norm of the weight matrix directly
corresponds to the functional logit error (Theorem 1 in the paper).

When the total feature dimension D exceeds the calibration set size m
(common for multi-layer ViT features), the empirical covariance is
rank-deficient.  The PCA-ZCA hybrid first reduces each layer's features
via PCA to retain 99% of explained variance, then applies ZCA whitening
in the reduced space.

Reference:
    §3.1 "Preliminaries: Feature Conditioning and Teacher as Linear
    Functional" in the FKP paper.
"""

from __future__ import annotations

import torch
import torch.linalg as LA
from dataclasses import dataclass, field


@dataclass
class LayerWhiteningState:
    """Stores all statistics needed to whiten a single layer at inference time.

    Attributes
    ----------
    mean : torch.Tensor
        Layer-wise mean vector of shape (d_k,).
    W_zca : torch.Tensor
        ZCA whitening matrix of shape (d_reduced, d_k) where d_reduced ≤ d_k.
        At inference: tilde_phi = W_zca @ (phi - mean).
    pca_components : torch.Tensor or None
        PCA projection matrix of shape (d_reduced, d_k), present only when
        the PCA-ZCA hybrid is used (D > m for the layer).  None otherwise.
    d_original : int
        Original layer feature dimension d_k.
    d_reduced : int
        Dimension after PCA reduction (equals d_k if no PCA was applied).
    used_pca : bool
        True if the PCA-ZCA hybrid was applied to this layer.
    """
    mean: torch.Tensor
    W_zca: torch.Tensor
    pca_components: torch.Tensor | None
    d_original: int
    d_reduced: int
    used_pca: bool


class ZCAWhitener:
    """Layer-wise ZCA whitening with optional PCA-ZCA hybrid.

    Fits whitening statistics on a calibration set and applies them at
    inference time.  Each layer is whitened independently, then the
    whitened features are concatenated into the conditioned embedding.

    Parameters
    ----------
    lambda_zca : float
        Regularization constant added to the diagonal of each layer's
        covariance matrix to ensure positive-definiteness.  Default: 1e-4.
    pca_variance_threshold : float
        Fraction of explained variance retained by PCA when the PCA-ZCA
        hybrid is triggered (D > m for a given layer).  Default: 0.99.
    device : str or torch.device
        Device for tensor computations.  Default: 'cpu'.

    Example
    -------
    >>> whitener = ZCAWhitener(lambda_zca=1e-4)
    >>> layer_features = [phi_1, phi_2, phi_3]  # list of (m, d_k) tensors
    >>> whitener.fit(layer_features, calibration_size=m)
    >>> E_tilde = whitener.transform(layer_features)  # (m, D)
    """

    def __init__(
        self,
        lambda_zca: float = 1e-4,
        pca_variance_threshold: float = 0.99,
        device: str | torch.device = "cpu",
    ) -> None:
        if lambda_zca <= 0:
            raise ValueError(f"lambda_zca must be positive, got {lambda_zca}")
        if not (0 < pca_variance_threshold <= 1.0):
            raise ValueError(
                f"pca_variance_threshold must be in (0, 1], got {pca_variance_threshold}"
            )
        self.lambda_zca = lambda_zca
        self.pca_variance_threshold = pca_variance_threshold
        self.device = torch.device(device)
        self._layer_states: list[LayerWhiteningState] = []
        self._fitted = False

    def fit(self, layer_features: list[torch.Tensor], calibration_size: int) -> "ZCAWhitener":
        """Compute and store ZCA whitening statistics for each layer.

        Parameters
        ----------
        layer_features : list[torch.Tensor]
            List of L tensors, each of shape (m, d_k), representing the
            pooled intermediate representations extracted from the teacher
            at each of the L layers.
        calibration_size : int
            Number of calibration samples m.  Used to detect the D > m case.

        Returns
        -------
        self : ZCAWhitener
            Fitted whitener (for method chaining).
        """
        self._layer_states = []
        for k, phi_k in enumerate(layer_features):
            phi_k = phi_k.to(dtype=torch.float64, device=self.device)
            m, d_k = phi_k.shape

            # Layer-wise mean (Eq. 2 in the paper)
            mean_k = phi_k.mean(dim=0)            # (d_k,)
            phi_c = phi_k - mean_k.unsqueeze(0)   # (m, d_k), centred

            state = self._fit_single_layer(phi_c, mean_k, d_k, calibration_size, layer_idx=k)
            self._layer_states.append(state)

        self._fitted = True
        return self

    def _fit_single_layer(
        self,
        phi_c: torch.Tensor,
        mean_k: torch.Tensor,
        d_k: int,
        calibration_size: int,
        layer_idx: int,
    ) -> LayerWhiteningState:
        """Fit ZCA (or PCA-ZCA) for one layer.

        Parameters
        ----------
        phi_c : torch.Tensor
            Zero-mean centred features of shape (m, d_k).
        mean_k : torch.Tensor
            Layer mean of shape (d_k,).
        d_k : int
            Original feature dimension.
        calibration_size : int
            Calibration set size m.
        layer_idx : int
            Layer index (used only for informative errors).

        Returns
        -------
        LayerWhiteningState
            Fitted state for this layer.
        """
        m = phi_c.shape[0]
        used_pca = False
        pca_components = None
        phi_reduced = phi_c  # will be overwritten if PCA is applied

        if d_k > calibration_size:
            # PCA-ZCA hybrid (Remark 1 in the paper):
            # reduce to top-r components that retain >= pca_variance_threshold variance.
            used_pca = True
            pca_components, phi_reduced = self._apply_pca(phi_c, layer_idx)

        d_reduced = phi_reduced.shape[1]

        # Regularised empirical covariance (Eq. 1 in the paper)
        C_k = (phi_reduced.T @ phi_reduced) / m + self.lambda_zca * torch.eye(
            d_reduced, dtype=torch.float64, device=self.device
        )

        # ZCA whitening matrix: C_k^{-1/2} via eigendecomposition
        # C_k = V Lambda V^T  =>  C_k^{-1/2} = V Lambda^{-1/2} V^T
        eigenvalues, eigenvectors = LA.eigh(C_k)  # ascending order
        # Clamp to avoid numerical negatives from floating point
        eigenvalues = eigenvalues.clamp(min=1e-12)
        inv_sqrt_eigenvalues = eigenvalues.pow(-0.5)            # (d_reduced,)
        W_zca = eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0)  # (d_reduced, d_reduced)
        W_zca = W_zca @ eigenvectors.T                            # (d_reduced, d_reduced)

        return LayerWhiteningState(
            mean=mean_k.to(torch.float32),
            W_zca=W_zca.to(torch.float32),
            pca_components=pca_components.to(torch.float32) if pca_components is not None else None,
            d_original=d_k,
            d_reduced=d_reduced,
            used_pca=used_pca,
        )

    def _apply_pca(self, phi_c: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """PCA dimensionality reduction retaining ``pca_variance_threshold`` variance.

        Uses the economy SVD of phi_c: phi_c = U S V^T, so the principal
        components are the rows of V^T.

        Parameters
        ----------
        phi_c : torch.Tensor
            Zero-mean centred features of shape (m, d_k).
        layer_idx : int
            Layer index (for error messages).

        Returns
        -------
        V_r : torch.Tensor
            PCA projection matrix of shape (r, d_k).  phi_reduced = phi_c @ V_r.T
        phi_reduced : torch.Tensor
            PCA-projected features of shape (m, r).
        """
        # Economy SVD: U (m,k) @ diag(S) @ V^T (k, d_k), k = min(m, d_k)
        U, S, Vh = LA.svd(phi_c, full_matrices=False)
        # Explained variance proportions
        variance = S.pow(2)
        total_variance = variance.sum()
        cumulative_variance_ratio = variance.cumsum(dim=0) / total_variance
        # Number of components to retain
        r = int((cumulative_variance_ratio >= self.pca_variance_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        r = max(r, 1)
        V_r = Vh[:r, :]          # (r, d_k)
        phi_reduced = phi_c @ V_r.T  # (m, r)
        return V_r, phi_reduced

    def transform(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        """Apply ZCA whitening and concatenate all layers.

        Parameters
        ----------
        layer_features : list[torch.Tensor]
            List of L tensors, each of shape (n, d_k), where n is the
            number of samples (can differ from calibration size m).

        Returns
        -------
        E_tilde : torch.Tensor
            Conditioned multi-layer embedding of shape (n, D_reduced),
            where D_reduced = sum of d_reduced_k across all layers.
            By construction, the calibration-set version of this matrix
            has approximately zero mean and identity covariance.
        """
        if not self._fitted:
            raise RuntimeError("ZCAWhitener must be fitted before calling transform().")
        if len(layer_features) != len(self._layer_states):
            raise ValueError(
                f"Expected {len(self._layer_states)} layers, got {len(layer_features)}."
            )

        whitened_layers: list[torch.Tensor] = []
        for phi_k, state in zip(layer_features, self._layer_states):
            phi_k = phi_k.to(dtype=torch.float32, device=self.device)
            # Subtract layer mean
            phi_c = phi_k - state.mean.unsqueeze(0)   # (n, d_k)
            # Apply PCA reduction if needed
            if state.used_pca and state.pca_components is not None:
                phi_c = phi_c @ state.pca_components.T  # (n, d_reduced)
            # Apply ZCA whitening: tilde_phi = phi_c @ W_zca^T
            phi_white = phi_c @ state.W_zca.T          # (n, d_reduced)
            whitened_layers.append(phi_white)

        return torch.cat(whitened_layers, dim=1)  # (n, D_reduced)

    def fit_transform(
        self, layer_features: list[torch.Tensor], calibration_size: int
    ) -> torch.Tensor:
        """Convenience method: fit and transform in one call.

        Parameters
        ----------
        layer_features : list[torch.Tensor]
            List of L tensors, each of shape (m, d_k).
        calibration_size : int
            Number of calibration samples m.

        Returns
        -------
        E_tilde : torch.Tensor
            Conditioned embedding of shape (m, D_reduced).
        """
        self.fit(layer_features, calibration_size)
        return self.transform(layer_features)

    @property
    def output_dim(self) -> int:
        """Total dimension D_reduced of the conditioned embedding."""
        if not self._fitted:
            raise RuntimeError("ZCAWhitener is not fitted yet.")
        return sum(s.d_reduced for s in self._layer_states)

    @property
    def layer_states(self) -> list[LayerWhiteningState]:
        """Read-only access to per-layer whitening states."""
        return list(self._layer_states)

    def summary(self) -> str:
        """Human-readable summary of per-layer dimensionality changes."""
        if not self._fitted:
            return "ZCAWhitener (not fitted)"
        lines = ["ZCAWhitener summary:"]
        for k, s in enumerate(self._layer_states):
            tag = " [PCA-ZCA]" if s.used_pca else ""
            lines.append(
                f"  Layer {k}: d_original={s.d_original} -> d_reduced={s.d_reduced}{tag}"
            )
        lines.append(f"  Total D_reduced = {self.output_dim}")
        return "\n".join(lines)
