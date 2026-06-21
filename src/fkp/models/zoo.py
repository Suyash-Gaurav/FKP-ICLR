"""
Teacher Model Loader (Zoo).

Loads pretrained teacher models from timm and HuggingFace transformers.
All models are loaded in eval mode with frozen parameters — FKP never
modifies teacher weights.

Supported architectures:
    - ResNet-50 (timm)
    - ViT-B/16 (timm)
    - CLIP-ViT-B/32 (openai/clip-vit-base-patch32 via transformers)
    - DINOv2-Small (facebook/dinov2-small via transformers)
    - MobileNetV3-Large (timm)

Default layer configurations per architecture are provided to ensure the
multi-layer embedding covers early, mid, and final representations.

Reference:
    §5.1 "Teacher Models" in FKP.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --- Default layer hook configurations per architecture ---
DEFAULT_LAYERS: dict[str, list[str]] = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "vit_base_patch16_224": [
        "blocks.2", "blocks.5", "blocks.8", "blocks.11"
    ],
    "vit_base_patch32_224_clip_laion2b": [
        "blocks.2", "blocks.5", "blocks.8", "blocks.11"
    ],
    "mobilenetv3_large_100": [
        "blocks.1", "blocks.2", "blocks.3", "blocks.4", "conv_head"
    ],
}


def load_teacher(
    model_name: str,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> tuple[nn.Module, list[str]]:
    """Load a pretrained teacher model and return its default layer names.

    Parameters
    ----------
    model_name : str
        Model identifier.  Supported values:
          - 'resnet50'
          - 'vit_b16' or 'vit_base_patch16_224'
          - 'clip' or 'clip_vit_b32'
          - 'dinov2_small' or 'dinov2_vits14'
          - 'mobilenetv3' or 'mobilenetv3_large_100'
          - Any timm model name (default layers not provided; use layer_names arg).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.  Default: True.
    device : str or torch.device
        Device to load the model onto.
    **kwargs :
        Additional keyword arguments forwarded to the loader.

    Returns
    -------
    model : nn.Module
        Frozen teacher model in eval mode.
    layer_names : list[str]
        Default layer names for multi-layer feature extraction.

    Raises
    ------
    ImportError
        If the required backend (timm or transformers) is not installed.
    ValueError
        If the model name is unrecognised and no fallback exists.
    """
    device = torch.device(device)
    model_name_lower = model_name.lower().replace("-", "_").replace("/", "_")

    model: nn.Module
    layer_names: list[str]

    if model_name_lower in ("clip", "clip_vit_b32", "clip_vit_base_patch32"):
        model, layer_names = _load_clip(device, **kwargs)
    elif model_name_lower in ("dinov2_small", "dinov2_vits14", "dinov2_s"):
        model, layer_names = _load_dinov2(model_name_lower, device, **kwargs)
    else:
        model, layer_names = _load_timm(model_name_lower, pretrained, device, **kwargs)

    # Freeze all parameters — FKP never backpropagates through the teacher
    for param in model.parameters():
        param.requires_grad_(False)

    model = model.to(device).eval()
    logger.info("Loaded teacher '%s' on %s (%s frozen parameters)", model_name, device,
                sum(p.numel() for p in model.parameters()))
    return model, layer_names


def _load_timm(
    model_name: str,
    pretrained: bool,
    device: torch.device,
    **kwargs: Any,
) -> tuple[nn.Module, list[str]]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for ResNet / ViT / MobileNetV3 teachers. "
            "Install with: pip install timm"
        ) from exc

    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    # Match to default layer config (exact or prefix match)
    layer_names = _resolve_layer_names(model, model_name)
    return model, layer_names


def _load_clip(device: torch.device, **kwargs: Any) -> tuple[nn.Module, list[str]]:
    try:
        from transformers import CLIPModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for CLIP.  Install with: pip install transformers"
        ) from exc

    model_id = kwargs.pop("hf_model_id", "openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained(model_id, **kwargs)
    # Wrap vision encoder for hook-based extraction
    vision_model = _CLIPVisionWrapper(clip)
    layer_names = [f"encoder.layers.{i}" for i in [2, 5, 8, 11]]
    return vision_model, layer_names


def _load_dinov2(name: str, device: torch.device, **kwargs: Any) -> tuple[nn.Module, list[str]]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for DINOv2.  Install with: pip install transformers"
        ) from exc

    hub_name_map = {
        "dinov2_small": "facebook/dinov2-small",
        "dinov2_s": "facebook/dinov2-small",
        "dinov2_vits14": "facebook/dinov2-small",
    }
    hf_id = kwargs.pop("hf_model_id", hub_name_map.get(name, "facebook/dinov2-small"))
    model = AutoModel.from_pretrained(hf_id, **kwargs)
    layer_names = [f"encoder.layer.{i}" for i in [2, 5, 8, 11]]
    return model, layer_names


def _resolve_layer_names(model: nn.Module, model_name: str) -> list[str]:
    """Find the best-matching default layer config for a timm model."""
    for key, layers in DEFAULT_LAYERS.items():
        if model_name.startswith(key) or key.startswith(model_name):
            # Validate that these names actually exist
            named = {n for n, _ in model.named_modules()}
            valid = [l for l in layers if l in named]
            if valid:
                return valid
    # Fallback: use last 4 non-trivial submodule names
    all_names = [n for n, m in model.named_modules() if n and len(list(m.children())) == 0]
    logger.warning(
        "No default layer config for '%s'. Using last 4 leaf modules: %s",
        model_name, all_names[-4:],
    )
    return all_names[-4:]


class _CLIPVisionWrapper(nn.Module):
    """Thin wrapper that exposes CLIP's vision encoder as a standalone model."""

    def __init__(self, clip_model: Any) -> None:
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output    # (B, hidden_dim)
        return self.visual_projection(pooled)  # (B, embed_dim)
