# `docker/` — Reproducibility Container

The Dockerfile provides an exact environment for reproducing all paper results.
It pins PyTorch, CUDA, and all Python dependencies to the versions used during
paper submission.

## Build

```bash
docker build -t fkp-iclr2027:latest -f docker/Dockerfile .
```

Or:

```bash
make docker-build
```

Build time: ~10 minutes (downloads PyTorch 2.2.2 base image).

## Run

```bash
# Interactive shell
make docker-run

# Mount your data and outputs
docker run --rm -it \
    -v $(PWD)/data:/workspace/FKP/data \
    -v $(PWD)/outputs:/workspace/FKP/outputs \
    fkp-iclr2027:latest bash
```

## GPU support

```bash
docker run --rm -it --gpus all \
    -v $(PWD)/data:/workspace/FKP/data \
    -v $(PWD)/outputs:/workspace/FKP/outputs \
    fkp-iclr2027:latest bash
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## What the build does

1. Starts from `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`
2. Installs system dependencies (`gcc`, `cmake`, `wget`, etc.)
3. Copies the FKP source tree
4. Installs pinned Python dependencies from `requirements.txt`
5. Installs FKP in editable mode (`pip install -e ".[models]"`)
6. **Automatically runs** `make test-math` to verify the three math unit tests pass
7. Fails the build if any test fails

## Reproduce Table 1 inside Docker

```bash
docker run --rm -it \
    -v $(PWD)/data:/workspace/FKP/data \
    -v $(PWD)/outputs:/workspace/FKP/outputs \
    fkp-iclr2027:latest \
    make reproduce-main-table
```

## Image size

~8 GB (dominated by the PyTorch CUDA base image).
CPU-only variant would be ~3 GB — replace the base image with
`pytorch/pytorch:2.2.2-cpu` if GPU is not needed.
