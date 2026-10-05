# create3d

A simple tool for making 3D models with [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).
It wraps Hunyuan3D-2 in a command-line tool and a browser UI:

- **Image → 3D**: one photo or render of an object becomes a mesh
- **Text → 3D**: a prompt goes through HunyuanDiT to make an image, then a mesh
- **Multi-view → 3D**: front/left/back/right images of one object
- **Batch**: turn a whole folder of images into models
- Optional **texture painting**, automatic background removal, mesh cleanup and face reduction
- Exports `.glb`, `.obj`, `.ply`, `.stl`, `.fbx`

## Run it in Google Colab (no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sajjadnisham/3d/blob/main/create3d_colab.ipynb)

Open the notebook, pick *Runtime → Change runtime type → T4 GPU*, and run the cells top to bottom:
install → load model → upload an image (or type a prompt) → generate → preview → download (GLB/OBJ/FBX/STL/PLY).

## Requirements

| Task | Hardware |
|---|---|
| Shape only (`mini-turbo`, default) | NVIDIA GPU with ~6 GB VRAM (Apple MPS / CPU work, but slowly) |
| Shape + texture | NVIDIA GPU with ~16 GB VRAM (use `--low-vram` on smaller cards) |
| Text → 3D | NVIDIA GPU (adds HunyuanDiT) |

Python 3.10–3.12, and PyTorch installed for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)).

For **FBX** export you also need one of:
- [Blender](https://www.blender.org/) on your `PATH` (or `BLENDER=/path/to/blender`) — preferred, embeds textures
- the assimp command-line tool: `apt install assimp-utils` / `brew install assimp`

## Install

Install PyTorch for your GPU first (pick your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/)),
then, from the repo root, either:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Option A: pip only (shape generation)
pip install -r requirements.txt

# Option B: install script
./scripts/install.sh            # shape generation
./scripts/install.sh --texture  # also build the texture CUDA extensions
```

All packages are pinned to exact versions in `requirements.txt` (both options use it). PyTorch is the
exception: install the build that matches your GPU/CUDA/OS yourself (2.5.1 is the recommended version), and the
requirements keep it instead of replacing it.

Texture painting needs two CUDA extensions compiled from the Hunyuan3D-2 source, so for textures
run `./scripts/install.sh --texture` (it works after Option A too). A virtual environment
(`python -m venv .venv && source .venv/bin/activate`) is recommended.

Model weights download from Hugging Face the first time you use them (into `~/.cache/hy3dgen`,
or wherever `HY3DGEN_MODELS` points).

## Docker

Needs an NVIDIA GPU, driver 550 or newer, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image includes PyTorch, all pinned packages, the compiled texture extensions and FBX export.

```bash
docker compose up --build            # web UI at http://localhost:7860
```

Or with plain Docker:

```bash
docker build -t create3d .                              # add --build-arg WITH_TEXTURE=0 for a smaller, shape-only image
docker run --gpus all -p 7860:7860 -v create3d-cache:/cache create3d          # web UI
docker run --gpus all -v create3d-cache:/cache -v "$PWD":/work create3d \
    image /work/chair.png -o /work/chair.fbx --texture                         # CLI
```

**Prebuilt images** are published by GitHub Actions (`.github/workflows/docker.yml`) on every push to `main`:

```bash
docker pull ghcr.io/sajjadnisham/3d:latest   # shape + texture
docker pull ghcr.io/sajjadnisham/3d:shape    # shape only, smaller
docker run --gpus all -p 7860:7860 -v create3d-cache:/cache ghcr.io/sajjadnisham/3d:latest
```

The `create3d-cache` volume keeps the downloaded model weights (several GB) between runs.
Files you want to convert go in the folder mounted at `/work`.

## Command line

```bash
create3d image chair.png                          # writes chair.glb
create3d image chair.png -o out/chair.glb --texture
create3d text "a cute cartoon robot" --texture -o robot.glb
create3d views --front f.png --left l.png --back b.png -o toy.glb
create3d batch ./photos -o ./models --format .obj
create3d image chair.png --texture -o chair.fbx   # FBX for Unity/Unreal/Maya
```

Useful options (shared by `image`, `text`, `views` and `batch`):

| Option | Meaning |
|---|---|
| `--model` | `mini-turbo` (default, fastest), `mini`, `turbo`, `full`, `mv`, `mv-turbo` |
| `--texture` | paint a texture onto the mesh (CUDA only) |
| `--texture-model` | `turbo` (default) or `full` |
| `--octree 384` | finer geometry (more VRAM/time); default 256 |
| `--steps N` | diffusion steps; default 5 for turbo models, 50 otherwise |
| `--seed N` | change the seed for a different result |
| `--max-faces N` | simplify to N faces (default 40000, `0` = off) |
| `--keep-background` | don't run background removal |
| `--low-vram` | offload the texture model to CPU when idle (for GPUs under 16 GB) |
| `--save-image` | also save the background-removed / generated input image |

Run `create3d --help` or `create3d image --help` for everything.

## Browser UI

```bash
create3d ui                      # http://127.0.0.1:7860
create3d ui --model mv-turbo     # enables the multi-view tab
create3d ui --host 0.0.0.0 --share
```

Upload an image (or type a prompt), press **Generate 3D model**, rotate the result in the viewer, and download it.

## Python

```python
from create3d.engine import Generator3D, GenerationSettings, export_mesh

gen = Generator3D(shape_model="mini-turbo")
mesh, ref = gen.generate(image="chair.png", settings=GenerationSettings(texture=True))
export_mesh(mesh, "chair.glb")
```

## Tips for good results

- Use one object, centred, on a plain background, viewed from slightly above the front.
- Images with a transparent background skip background removal automatically.
- If the result has holes or blobs, try another `--seed` or a higher `--octree`.

## License

Hunyuan3D-2 and its weights are under the
[Tencent Hunyuan Non-Commercial License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE);
it does not apply in the EU, UK or South Korea. Check it before commercial use.
