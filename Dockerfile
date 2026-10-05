# create3d: Hunyuan3D-2 image/text -> 3D model, with GPU support.
#
# Build:   docker build -t create3d .
#          docker build -t create3d --build-arg WITH_TEXTURE=0 .   # smaller, shape only
# Web UI:  docker run --gpus all -p 7860:7860 -v create3d-cache:/cache create3d
# CLI:     docker run --gpus all -v create3d-cache:/cache -v "$PWD":/work create3d \
#              image /work/chair.png -o /work/chair.fbx --texture
#
# Needs an NVIDIA GPU + driver >= 550 and the NVIDIA Container Toolkit on the host.

# "devel" (not "runtime") because the texture extensions are compiled with nvcc.
# cuDNN comes with the PyTorch wheel, so the plain CUDA image is enough.
# Override with --build-arg BASE_IMAGE=... to use a registry mirror.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG WITH_TEXTURE=1
# Keep in sync with requirements.txt / scripts/install.sh
ARG HY_COMMIT=f8db63096c8282cb27354314d896feba5ba6ff8a
# No GPU is visible during `docker build`, so list the GPU generations to compile for:
# Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx), Ada (L4, RTX 40xx), Hopper (H100).
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    # Model weights and caches go to /cache: mount a volume there to keep them between runs.
    HY3DGEN_MODELS=/cache/hy3dgen \
    HF_HOME=/cache/huggingface \
    U2NET_HOME=/cache/u2net \
    GRADIO_SERVER_NAME=0.0.0.0

# Python 3.10, build tools, OpenGL/GLib for OpenCV, assimp for FBX export.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python-is-python3 \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
        assimp-utils \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04's pip 22.0 ignores the [project] table in pyproject.toml (builds "UNKNOWN-0.0.0"),
# so upgrade the packaging tools first.
RUN python -m pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

# PyTorch first: PyPI's Linux build of 2.5.1 targets CUDA 12.4, matching the base image.
RUN pip install torch==2.5.1 torchvision==0.20.1

# Pinned dependencies (everything except create3d itself), cached separately from the source.
COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^-e ' /tmp/requirements.txt > /tmp/requirements-deps.txt \
    && pip install -r /tmp/requirements-deps.txt

# Texture extensions (custom rasterizer + mesh processor), compiled from the Hunyuan3D-2 source.
RUN if [ "$WITH_TEXTURE" = "1" ]; then \
        git init -q /opt/Hunyuan3D-2 \
        && git -C /opt/Hunyuan3D-2 remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2 \
        && git -C /opt/Hunyuan3D-2 fetch -q --depth 1 origin "$HY_COMMIT" \
        && git -C /opt/Hunyuan3D-2 checkout -q FETCH_HEAD \
        && pip install --no-build-isolation /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer \
        && pip install --no-build-isolation /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer \
        && rm -rf /opt/Hunyuan3D-2; \
    fi

# create3d itself
COPY pyproject.toml README.md /app/
COPY create3d /app/create3d
RUN pip install --no-deps /app

VOLUME /cache
WORKDIR /work
EXPOSE 7860

ENTRYPOINT ["create3d"]
CMD ["ui", "--host", "0.0.0.0", "--port", "7860"]
