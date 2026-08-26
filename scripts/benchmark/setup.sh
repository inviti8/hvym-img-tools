#!/usr/bin/env bash
# TripoSR benchmark env setup (light path: no tiny-cuda-nn / NeuS / Wonder3D)
set -euo pipefail
cd /workspace 2>/dev/null || cd /root

echo "=== SYSTEM ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import sys,torch;print("python",sys.version.split()[0],"| torch",torch.__version__,"| cuda",torch.version.cuda,"| dev",torch.cuda.get_device_name(0))'
echo "nvcc: $(which nvcc || echo NONE)"
df -h . | tail -1

echo "=== UV ==="
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== CLONE TripoSR ==="
[ -d TripoSR ] || git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
echo "--- requirements.txt ---"; cat requirements.txt

echo "=== DEPS (into system env; torch already present) ==="
uv pip install --system omegaconf einops transformers trimesh rembg onnxruntime \
    "huggingface_hub" imageio "imageio[ffmpeg]" xatlas moderngl pillow numpy opencv-python-headless

echo "=== torchmcubes (CUDA marching cubes) ==="
if uv pip install --system "git+https://github.com/tatsy/torchmcubes.git" 2>&1 | tail -5; then
  python -c "import torchmcubes; print('torchmcubes OK')" || echo "TORCHMCUBES_IMPORT_FAILED"
else
  echo "TORCHMCUBES_BUILD_FAILED -- will need PyMCubes CPU fallback"
  uv pip install --system PyMCubes
fi

echo "=== isnet matting model ==="
mkdir -p /workspace/models
[ -f /workspace/models/isnet_dis.onnx ] || \
  wget -q --show-progress https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx \
     -O /workspace/models/isnet_dis.onnx
ls -la /workspace/models/isnet_dis.onnx

echo "=== SETUP DONE ==="
