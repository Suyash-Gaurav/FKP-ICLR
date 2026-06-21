# `models/` — Teacher Wrappers and Multi-Layer Feature Extraction

This module wraps frozen foundation models for multi-layer feature extraction
using PyTorch forward hooks. The teacher model is **never fine-tuned** — weights
are frozen and used purely for feature extraction.

| File | Purpose |
|---|---|
| `hooks.py` | `HookManager`: registers forward hooks, stacks per-layer features |
| `zoo.py` | `load_teacher()`: one-liner loader for timm / transformers / CLIP / DINOv2 |

---

## `hooks.py` — HookManager

Registers `forward_hook` callbacks on named submodules to extract activations
without modifying the model graph. Features are spatial-pooled (global average)
to produce one vector per sample per layer.

```python
from fkp.models.hooks import HookManager

hook_mgr = HookManager(
    model=teacher,
    layer_names=["layer1", "layer2", "layer3", "layer4"],  # ResNet-50
    pool="avg",   # "avg" | "max" | "flatten" | "cls_token"
)

with torch.no_grad():
    _ = teacher(images)

# layer_features[k]: Tensor of shape (batch, d_k)
layer_features = hook_mgr.get_features()
hook_mgr.clear()
hook_mgr.remove()   # always remove hooks when done
```

### Pooling strategies

| Strategy | Use for | Output |
|---|---|---|
| `"avg"` | CNNs (ResNet, MobileNet) | Global average pool of spatial maps |
| `"cls_token"` | ViT / CLIP / DINOv2 | First token of sequence output |
| `"flatten"` | Small feature maps (CIFAR) | Flattened spatial map |

---

## `zoo.py` — Teacher Model Zoo

```python
from fkp.models.zoo import load_teacher

teacher, layer_names = load_teacher("resnet50", device="cpu")
teacher, layer_names = load_teacher("vit_b16",  device="cuda")
teacher, layer_names = load_teacher("clip",     device="cpu")
teacher, layer_names = load_teacher("dinov2_small", device="cpu")
teacher, layer_names = load_teacher("mobilenetv3",  device="cpu")
```

All returned models are in **eval mode** with **all parameters frozen**
(`requires_grad=False`).

### Supported backends

| `name` argument | Backend | Pretrained weights |
|---|---|---|
| `resnet50`, `resnet101`, `mobilenetv3_large_100` | timm | ImageNet-1k |
| `vit_b16`, `vit_s16` | timm | ImageNet-21k |
| `dinov2_small`, `dinov2_base` | transformers | DINOv2 |
| `clip` | transformers | CLIP (WIT-400M) |

### Adding a new teacher

1. Add an entry to the `_TEACHER_REGISTRY` dict in `zoo.py`.
2. Specify `backend`, `model_id`, `hook_layers`, and `pool_strategy`.
3. Add the corresponding YAML file to `configs/teachers/<name>.yaml`.
4. Run `make test-math` to confirm ZCA covariance tests still pass with the new feature dim.
