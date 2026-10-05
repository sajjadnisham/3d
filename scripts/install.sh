#!/usr/bin/env bash
# Installs Hunyuan3D-2 and the create3d tool into the current Python environment.
#   ./scripts/install.sh            # shape generation only
#   ./scripts/install.sh --texture  # also build the CUDA texture extensions
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HY_DIR="$ROOT/third_party/Hunyuan3D-2"
HY_COMMIT=f8db63096c8282cb27354314d896feba5ba6ff8a   # keep in sync with requirements.txt
WITH_TEXTURE=0
[[ "${1:-}" == "--texture" ]] && WITH_TEXTURE=1

if ! python -c "import torch" 2>/dev/null; then
  echo ">> PyTorch not found. Install it first for your platform: https://pytorch.org/get-started/locally/"
  echo "   e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
  exit 1
fi

if [[ ! -d "$HY_DIR" ]]; then
  echo ">> Cloning Hunyuan3D-2 ..."
  git init -q "$HY_DIR"
  git -C "$HY_DIR" remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  git -C "$HY_DIR" fetch -q --depth 1 origin "$HY_COMMIT"
  git -C "$HY_DIR" checkout -q FETCH_HEAD
fi

echo ">> Installing Hunyuan3D-2 requirements ..."
pip install -r "$HY_DIR/requirements.txt"
pip install -e "$HY_DIR"
pip install scipy scikit-image pyyaml   # imported by Hunyuan3D-2 but not in its requirements

if [[ $WITH_TEXTURE == 1 ]]; then
  echo ">> Building texture extensions (needs CUDA toolkit + matching compiler) ..."
  # --no-build-isolation so the build can see the installed torch/pybind11
  pip install --no-build-isolation "$HY_DIR/hy3dgen/texgen/custom_rasterizer"
  pip install --no-build-isolation "$HY_DIR/hy3dgen/texgen/differentiable_renderer"
fi

echo ">> Installing create3d ..."
pip install -e "$ROOT"

echo ">> Done. Try:  create3d image $HY_DIR/assets/demo.png -o demo.glb"
