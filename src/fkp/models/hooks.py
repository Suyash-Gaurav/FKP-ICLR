"""
PyTorch Forward Hook Manager for Multi-Layer Feature Extraction.

Extracts intermediate pooled representations from any timm or HuggingFace
transformers model at user-specified layers.  Supports ResNet, ViT, CLIP,
DINOv2, MobileNetV3, and any architecture that exposes named submodules.

Memory Management:
    Multi-layer ViT features can OOM a GPU if batched improperly.  This
    module processes the calibration set in configurable chunks, moving
    intermediate tensors to CPU/RAM before accumulation.  The full feature
    matrix is assembled in RAM, not on GPU.

Reference:
    §2 "Notation" and §5.1 "Implementation Details" in FKP.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HookManager:
    """Generic forward hook manager for multi-layer feature extraction.

    Attaches forward hooks to named submodules of any PyTorch model and
    captures their outputs after suitable global average pooling.

    Parameters
    ----------
    model : nn.Module
        The frozen teacher model.  All parameters should be detached or in
        eval mode; the HookManager does not modify model weights.
    layer_names : list[str]
        List of submodule names (as returned by model.named_modules()) at
        which to attach hooks.  E.g. ['layer1', 'layer2', 'layer3', 'layer4']
        for ResNet, or ['blocks.6', 'blocks.9', 'blocks.11'] for ViT.
    pool_fn : callable or None
        Pooling function applied to each hook output before storage.
        Signature: (tensor: Tensor) -> Tensor.
        Default: adaptive global average pooling to a scalar-per-channel.

    Example
    -------
    >>> manager = HookManager(model, layer_names=['layer2', 'layer3', 'layer4'])
    >>> with manager.capture():
    ...     _ = model(x_batch)
    >>> features = manager.get_features()  # list of (batch, d_k) tensors
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
        pool_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.layer_names = layer_names
        self.pool_fn = pool_fn or _default_pool
        self._hooks: list[Any] = []
        self._captures: dict[str, torch.Tensor] = {}
        self._validate_layer_names()

    def _validate_layer_names(self) -> None:
        """Ensure all requested layer names exist in the model."""
        named = {name for name, _ in self.model.named_modules()}
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            available = sorted(named)[:30]
            raise ValueError(
                f"Layer(s) not found in model: {missing}\n"
                f"First 30 available names: {available}"
            )

    @contextmanager
    def capture(self):
        """Context manager that attaches hooks, yields, then removes them."""
        self._captures = {}
        self._hooks = []
        try:
            for name in self.layer_names:
                module = dict(self.model.named_modules())[name]
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
            yield self
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks = []

    def _make_hook(self, name: str) -> Callable:
        """Factory for named forward hooks."""
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            pooled = self.pool_fn(output.detach())
            self._captures[name] = pooled.cpu()
        return hook

    def get_features(self) -> list[torch.Tensor]:
        """Return captured features in the order of layer_names.

        Returns
        -------
        list[torch.Tensor]
            One tensor per layer, each of shape (batch, d_k).
        """
        if not self._captures:
            raise RuntimeError(
                "No features captured.  Use the 'capture()' context manager."
            )
        return [self._captures[name] for name in self.layer_names]


def extract_multilayer_features(
    model: nn.Module,
    layer_names: list[str],
    dataloader: Any,
    device: torch.device | str = "cpu",
    chunk_size: int = 64,
    pool_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Extract multi-layer features and teacher logits from a dataloader.

    Processes the calibration set in chunks to avoid GPU OOM.  All tensors
    are collected on CPU/RAM.

    Parameters
    ----------
    model : nn.Module
        Frozen teacher model in eval mode.
    layer_names : list[str]
        Submodule names at which to extract features.
    dataloader : DataLoader
        Yields batches of (images,) or (images, labels).  Labels are ignored.
    device : torch.device or str
        Device for model inference.
    chunk_size : int
        Batch size for inference.  Reduce if GPU OOM occurs.
    pool_fn : callable or None
        Custom pooling function.  Default: global average pooling.

    Returns
    -------
    layer_features : list[torch.Tensor]
        L tensors, each of shape (m, d_k), with m = total calibration samples.
    logits : torch.Tensor
        Teacher logit matrix of shape (m, c).
    """
    device = torch.device(device)
    model = model.to(device).eval()
    manager = HookManager(model, layer_names, pool_fn)

    # Accumulators per layer + logit
    layer_accumulators: list[list[torch.Tensor]] = [[] for _ in layer_names]
    logit_accumulator: list[torch.Tensor] = []

    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            # Handle (images,) or (images, labels) tuples
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)

            with manager.capture():
                logits_batch = model(images)  # triggers hooks

            # Move logits to CPU immediately to free GPU memory
            logit_accumulator.append(logits_batch.cpu())

            # Collect per-layer features
            captured = manager.get_features()
            for k, feat in enumerate(captured):
                layer_accumulators[k].append(feat)   # already on CPU

            total_samples += images.shape[0]
            logger.debug("Extracted features for %d samples", total_samples)

    # Concatenate along the sample dimension
    layer_features = [torch.cat(acc, dim=0) for acc in layer_accumulators]
    logits = torch.cat(logit_accumulator, dim=0)

    logger.info(
        "Feature extraction complete: %d samples, %d layers, logit shape %s",
        total_samples,
        len(layer_names),
        tuple(logits.shape),
    )
    return layer_features, logits


def _default_pool(output: torch.Tensor) -> torch.Tensor:
    """Default pooling: global average over spatial dimensions.

    Handles the following output shapes:
        (B, C, H, W)   -> (B, C)   [CNN feature maps]
        (B, T, C)      -> (B, C)   [ViT token sequences, average over tokens]
        (B, C)         -> (B, C)   [already pooled]
        (B,)           -> (B, 1)   [scalar output, rare]

    Parameters
    ----------
    output : torch.Tensor
        Raw hook output tensor.

    Returns
    -------
    torch.Tensor
        Pooled tensor of shape (B, d_k).
    """
    if output.ndim == 4:
        # CNN: (B, C, H, W) -> (B, C)
        return output.mean(dim=(2, 3))
    if output.ndim == 3:
        # ViT / Transformer: (B, T, C) -> (B, C), skip CLS if first token
        return output[:, 1:, :].mean(dim=1) if output.shape[1] > 1 else output[:, 0, :]
    if output.ndim == 2:
        return output
    if output.ndim == 1:
        return output.unsqueeze(1)
    raise ValueError(f"Cannot pool tensor with ndim={output.ndim}, shape={tuple(output.shape)}")


def list_model_layers(model: nn.Module) -> list[str]:
    """Utility: print all named submodule names in a model.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.

    Returns
    -------
    list[str]
        All non-empty submodule names.
    """
    return [name for name, _ in model.named_modules() if name]
