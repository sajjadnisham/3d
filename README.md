# create3d

A simple tool for making 3D models with [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).
It wraps Hunyuan3D-2 in a command-line tool and a browser UI:

- **Image → 3D**: one photo or render of an object becomes a mesh
- **Text → 3D**: a prompt goes through HunyuanDiT to make an image, then a mesh
- **Multi-view → 3D**: front/left/back/right images of one object
- **Batch**: turn a whole folder of images into models
- Optional **texture painting**, automatic background removal, mesh cleanup and face reduction
- Exports `.glb`, `.obj`, `.ply`, `.stl`

## Requirements

| Task | Hardware |
|---|---|
| Shape only (`mini-turbo`, default) | NVIDIA GPU with ~6 GB VRAM (Apple MPS / CPU work, but slowly) |
| Shape + texture | NVIDIA GPU with ~16 GB VRAM (use `--low-vram` on smaller cards) |
| Text → 3D | NVIDIA GPU (adds HunyuanDiT) |

Python 3.9+, and PyTorch installed for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)).

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA version
./scripts/install.sh            # shape generation
./scripts/install.sh --texture  # also build the texture CUDA extensions
```

This clones Hunyuan3D-2 into `third_party/` and installs it along with `create3d`.
Model weights download from Hugging Face the first time you use them (into `~/.cache/hy3dgen`,
or wherever `HY3DGEN_MODELS` points).

## Command line

```bash
create3d image chair.png                          # writes chair.glb
create3d image chair.png -o out/chair.glb --texture
create3d text "a cute cartoon robot" --texture -o robot.glb
create3d views --front f.png --left l.png --back b.png -o toy.glb
create3d batch ./photos -o ./models --format .obj
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
| `--low-vram` | offload models to CPU between steps |
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
