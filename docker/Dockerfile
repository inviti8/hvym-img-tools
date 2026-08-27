# hvym-img-tools GPU image.
#
# The *light* TripoSR path (AGENTS.md §6): no tiny-cuda-nn, no NeuS, no Wonder3D,
# no pytorch3d. That collapses the prototype's environment recipe to
# torch + transformers + TripoSR + onnxruntime, and keeps the licensing clean
# (MIT end to end -- Wonder3D's CC-BY-NC weights are deliberately absent).
#
# Multi-stage: the CUDA toolchain is only needed to compile torchmcubes, so the
# builder is thrown away and the runtime image never carries nvcc.
#
# Weights are baked in (isnet ~176 MB, TripoSR ~1.7 GB) so a cold worker does not
# download 1.9 GB before serving its first job.

# ---------------------------------------------------------------------------
# Stage 1 -- build torchmcubes (the only thing needing nvcc)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# torchmcubes is compiled here, so this list is exactly the hardware the image
# can run on. A card outside it fails at RUNTIME with "no kernel image is
# available for execution on the device" -- the build says nothing.
#
# 8.6 (A40/A6000/A5000/3090) + 8.9 (4090/L4/L40S) alone left us on two GPU
# pools in EU-RO-1 and RunPod throttled us ~9 min waiting for one. Widened to
# the pools that actually hold stock there:
#   8.0  = A100 40/80GB                  (AMPERE_80)
#   9.0  = H100                          (HOPPER_141)
#   12.0 = RTX PRO 4500/6000 Blackwell -- the only High-stock card in EU-RO-1
# +PTX embeds forward-compatible PTX so a newer arch can JIT instead of fail.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip wheel \
    && python3.10 -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX} \
    && python3.10 -m pip install --no-cache-dir scikit-build-core cmake ninja pybind11

# --no-build-isolation so the build sees torch; without it the extension cannot
# compile (and without scikit-build-core the backend itself is missing).
RUN python3.10 -m pip wheel --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/tatsy/torchmcubes.git" -w /wheels \
    && ls -la /wheels

# ---------------------------------------------------------------------------
# Stage 2 -- runtime
# ---------------------------------------------------------------------------
# 3.10 to match the builder's ABI (see the note in stage 1).
FROM python:3.10-slim-bookworm AS runtime

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# torch's pip wheels bundle CUDA (cudart/cudnn/cublas) via the nvidia-* packages,
# so we need only the driver from the host -- no CUDA base image required.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/hf \
    ISNET_PATH=/opt/models/isnet_dis.onnx \
    TRIPOSR_PATH=/opt/TripoSR \
    HVYM_CACHE_DIR=/cache \
    HVYM_WORKSPACE_DIR=/tmp/hvym \
    HVYM_MODELS_DIR=/opt/models \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# --- python deps -----------------------------------------------------------
# Version pins here are load-bearing; see docs/BENCHMARK.md §5.
RUN pip install "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX}

RUN pip install \
        # transformers 5.x renamed ViT weights -- the TripoSR checkpoint will NOT load
        "transformers==4.49.0" \
        "huggingface_hub" \
        omegaconf einops trimesh xatlas pillow numpy imageio "imageio[ffmpeg]" \
        # TripoSR imports rembg at module scope even though we matte with isnet
        rembg \
        fastapi "uvicorn[standard]" python-multipart pydantic httpx \
        runpod

# rembg pulls in CPU onnxruntime, which SHADOWS onnxruntime-gpu and silently
# sends the matte back to the CPU: 6.4s instead of 0.03s, with no error.
# Install it last so it wins, and assert the GPU provider is actually present.
RUN pip uninstall -y onnxruntime || true \
    && pip install "onnxruntime-gpu==1.22.0" \
    && python -c "import onnxruntime as ort; \
assert 'CUDAExecutionProvider' in ort.get_available_providers(), \
  'onnxruntime-gpu missing CUDA provider: %s' % ort.get_available_providers(); \
print('ok: onnxruntime', ort.__version__, ort.get_available_providers())"

COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels \
    && python -c "import torchmcubes; print('ok: torchmcubes')"

# --- TripoSR source --------------------------------------------------------
ARG TRIPOSR_REF=main
RUN git clone --depth 1 --branch ${TRIPOSR_REF} \
        https://github.com/VAST-AI-Research/TripoSR.git ${TRIPOSR_PATH} \
    && rm -rf ${TRIPOSR_PATH}/.git

# --- weights, baked in for fast cold start ---------------------------------
RUN mkdir -p /opt/models \
    && curl -fsSL -o ${ISNET_PATH} \
        https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx \
    && python -c "size=__import__('os').path.getsize('${ISNET_PATH}'); \
assert size > 100_000_000, 'isnet download looks truncated: %d bytes' % size; \
print('ok: isnet', size, 'bytes')"

RUN python -c "\
from huggingface_hub import hf_hub_download; \
[hf_hub_download('stabilityai/TripoSR', f) for f in ('config.yaml','model.ckpt')]; \
print('ok: TripoSR weights cached')"

# --- app -------------------------------------------------------------------
WORKDIR /app
COPY hvym_img_tools /app/hvym_img_tools
COPY pyproject.toml README.md /app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /cache /tmp/hvym

# Sanity check at BUILD time: the tool must import and register on a CPU-only
# builder. Catches a broken package before it ever reaches a GPU worker.
RUN python -c "\
from hvym_img_tools.core import registry; \
registry.discover(); \
assert 'reangle' in registry.names(), registry.names(); \
print('ok: tools registered ->', registry.names())"

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serverless"]

# Placed last so it never invalidates the expensive layers above. The source
# label links the GHCR package to the repo, which is also what lets the package
# inherit/settle its visibility.
LABEL org.opencontainers.image.source="https://github.com/inviti8/hvym-img-tools"       org.opencontainers.image.description="Self-hosted AI image tools (reangle: style-preserving camera reangle) - GPU worker"       org.opencontainers.image.licenses="MIT"
