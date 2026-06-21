# `edge_deployment/` — Bare-Metal C for ESP32 / STM32 / RP2040

This directory contains everything needed to flash FKP's edge inference onto a
microcontroller. The entire edge computation is a single matrix-vector multiply:

```c
// logits[c] = W_edge[p × c]^T @ z[p]  +  bias[c]
for (int i = 0; i < FKP_RANK_P; ++i)
    for (int j = 0; j < FKP_N_CLASSES; ++j)
        logits[j] += fkp_W_edge[i * FKP_N_CLASSES + j] * z[i];
```

**Complexity**: `p × c` MACs. For p=64, c=38 (PlantVillage): **2,432 MACs**, ~500 µs on Cortex-M4.

---

## Files

| File | Purpose |
|---|---|
| `export_to_c.py` | Python script: converts `.npy` weights → `.h` file |
| `include/fkp_inference.h` | Auto-generated C header with weight arrays |
| `src/main.c` | Bare-metal inference loop + gateway projection reference |
| `CMakeLists.txt` | CMake build for local sim + ARM cross-compile target |

---

## Workflow

### Step 1: Calibrate the pipeline (Python, on gateway)

```bash
python scripts/run_calibration.py \
    --teacher resnet50 \
    --dataset plantvillage \
    --output_dir outputs/plantvillage_resnet50
```

Produces `outputs/plantvillage_resnet50/{U_p.npy, W_edge.npy, bias.npy}`.

### Step 2: Export to C header

```bash
python edge_deployment/export_to_c.py \
    --weights_dir outputs/plantvillage_resnet50 \
    --output edge_deployment/include/fkp_inference.h \
    --model_name FKP_PlantVillage_ResNet50
```

Or simply:

```bash
make export-c WEIGHTS_DIR=outputs/plantvillage_resnet50
```

### Step 3: Build local C simulation (Linux/macOS)

```bash
make compile-c-sim
./build_c/fkp_sim
```

Expected output:

```
=== FKP Edge Inference Simulator ===
    Rank p = 64, Classes c = 38
    Edge model: 9.75 KB
    Payload  : 256 bytes (p x float32)

[FKP] Input: z = [1.0, 1.0, ..., 1.0] (p=64)
[FKP] Predicted class: 12
[FKP] Top-5 logits:
  class  12 : 4.2315
  class   7 : 3.1184
  ...
[FKP] MACs = p * c = 64 * 38 = 2432
```

### Step 4: Flash to MCU

For Arduino / PlatformIO (ESP32):

```bash
cp edge_deployment/include/fkp_inference.h <your_arduino_project>/src/
# Include in your sketch: #include "fkp_inference.h"
# Call fkp_infer(z, logits) in loop()
```

For STM32CubeIDE: include `fkp_inference.h` and `main.c` in your project sources.

---

## Memory footprint (p=64, c=38, float32)

| Component | On device? | Size |
|---|---|---|
| `fkp_W_edge` (p × c) | ✅ Edge | 9.5 KB |
| `fkp_bias` (c) | ✅ Edge | 0.15 KB |
| `fkp_U_p` (D × p) | ❌ Gateway | 192 KB |
| BLE/LoRa payload `z` | Transmitted | 256 bytes |

The edge device only stores 9.65 KB of weights. `U_p` stays on the gateway.

---

## CMSIS-DSP acceleration (optional)

Uncomment the CMSIS-DSP block in `CMakeLists.txt` and replace the inner loop
with `arm_mat_vec_mult_f32()` for hardware-accelerated MACC on Cortex-M4/M33.
Expected speedup: 2–4× over the reference loop implementation.
