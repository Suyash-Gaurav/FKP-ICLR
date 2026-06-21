#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FKP Dataset Download Script
#
# Downloads all datasets used in the FKP paper experiments.
# Run from the repository root: bash data/download_datasets.sh
#
# Datasets:
#   - PlantVillage  (via HuggingFace datasets)
#   - CIFAR-10      (via torchvision, auto-download)
#   - CIFAR-100     (via torchvision, auto-download)
#   - Cassava       (via HuggingFace datasets)
#   - DomainNet     (manual download — see instructions below)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "=== FKP Dataset Downloader ==="
echo "Target directory: $DATA_DIR"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.10+."
    exit 1
fi

# ── CIFAR-10 and CIFAR-100 ────────────────────────────────────────────────────
echo "[1/4] Downloading CIFAR-10 and CIFAR-100 via torchvision..."
python - <<'EOF'
import torchvision
import os
data_dir = os.environ.get("DATA_DIR", "data")
print("  Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root=f"{data_dir}/cifar10", train=True, download=True)
torchvision.datasets.CIFAR10(root=f"{data_dir}/cifar10", train=False, download=True)
print("  Downloading CIFAR-100...")
torchvision.datasets.CIFAR100(root=f"{data_dir}/cifar100", train=True, download=True)
torchvision.datasets.CIFAR100(root=f"{data_dir}/cifar100", train=False, download=True)
print("  CIFAR-10 and CIFAR-100 ready.")
EOF
export DATA_DIR="$DATA_DIR"

echo ""

# ── PlantVillage ──────────────────────────────────────────────────────────────
echo "[2/4] Downloading PlantVillage via HuggingFace datasets..."
python - <<'EOF'
import os
try:
    from datasets import load_dataset
    data_dir = os.environ.get("DATA_DIR", "data")
    ds = load_dataset("jkang37/plant-village", cache_dir=f"{data_dir}/plantvillage")
    print(f"  PlantVillage: {len(ds['train'])} train samples, {len(ds['test'])} test samples.")
except ImportError:
    print("  [WARN] 'datasets' package not installed. Run: pip install datasets")
    print("  Alternatively, download from https://www.kaggle.com/emmarex/plantdisease")
except Exception as e:
    print(f"  [WARN] PlantVillage download failed: {e}")
    print("  Download manually from https://huggingface.co/datasets/jkang37/plant-village")
EOF

echo ""

# ── Cassava ───────────────────────────────────────────────────────────────────
echo "[3/4] Downloading Cassava Leaf Disease dataset..."
python - <<'EOF'
import os
try:
    from datasets import load_dataset
    data_dir = os.environ.get("DATA_DIR", "data")
    ds = load_dataset("cassava", cache_dir=f"{data_dir}/cassava")
    print(f"  Cassava: {len(ds['train'])} train samples.")
except ImportError:
    print("  [WARN] 'datasets' package not installed.")
except Exception as e:
    print(f"  [WARN] Cassava download failed: {e}")
    print("  Download from https://huggingface.co/datasets/cassava")
EOF

echo ""

# ── DomainNet ─────────────────────────────────────────────────────────────────
echo "[4/4] DomainNet — manual download required."
echo ""
echo "  DomainNet is not redistributable via script."
echo "  Download from: http://ai.bu.edu/M3SDA/"
echo "  Steps:"
echo "    1. Visit http://ai.bu.edu/M3SDA/ and agree to the license."
echo "    2. Download 'Real' and 'Sketch' domain zip files."
echo "    3. Extract to $DATA_DIR/domainnet/real/ and $DATA_DIR/domainnet/sketch/"
echo "    4. Download split files from the same page."
echo ""

echo "=== Download complete (DomainNet requires manual steps above) ==="
echo "Data directory: $DATA_DIR"
ls -lh "$DATA_DIR" 2>/dev/null || true
