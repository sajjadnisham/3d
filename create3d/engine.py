"""Thin wrapper around the Hunyuan3D-2 pipelines.

All heavy imports (torch, hy3dgen, ...) happen lazily so that `--help` and
argument validation work without a GPU or the models installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from PIL import Image

# Shape-generation model presets: name -> (hf repo, subfolder, default steps, uses flashvdm)
SHAPE_MODELS: Dict[str, tuple] = {
    "full":       ("tencent/Hunyuan3D-2",     "hunyuan3d-dit-v2-0",          50, False),
    "turbo":      ("tencent/Hunyuan3D-2",     "hunyuan3d-dit-v2-0-turbo",     5, True),
    "mini":       ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini",       50, False),
    "mini-turbo": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo",  5, True),
    "mv":         ("tencent/Hunyuan3D-2mv",   "hunyuan3d-dit-v2-mv",         50, False),
    "mv-turbo":   ("tencent/Hunyuan3D-2mv",   "hunyuan3d-dit-v2-mv-turbo",    5, True),
}

# Texture-generation presets: name -> (hf repo, subfolder)
TEXTURE_MODELS: Dict[str, tuple] = {
    "full":  ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"),
    "turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"),
}

MULTIVIEW_KEYS = ("front", "left", "back", "right")
EXPORT_FORMATS = (".glb", ".obj", ".ply", ".stl", ".fbx")


@dataclass
class GenerationSettings:
    steps: Optional[int] = None          # None -> preset default
    guidance_scale: float = 5.0
    octree_resolution: int = 256
    num_chunks: int = 8000
    seed: int = 1234
    remove_floaters: bool = True
    max_faces: Optional[int] = 40000     # None/0 disables face reduction
    texture: bool = False


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log(msg: str) -> None:
    print(f"[create3d] {msg}", flush=True)


class Generator3D:
    """Loads Hunyuan3D-2 pipelines on demand and turns images/text into meshes."""

    def __init__(
        self,
        shape_model: str = "mini-turbo",
        texture_model: str = "turbo",
        device: str = "auto",
        low_vram: bool = False,
    ):
        if shape_model not in SHAPE_MODELS:
            raise ValueError(f"Unknown shape model '{shape_model}'. Choose from {list(SHAPE_MODELS)}")
        if texture_model not in TEXTURE_MODELS:
            raise ValueError(f"Unknown texture model '{texture_model}'. Choose from {list(TEXTURE_MODELS)}")
        self.shape_model = shape_model
        self.texture_model = texture_model
        self.device = pick_device(device)
        self.low_vram = low_vram
        self._shape = None
        self._paint = None
        self._rembg = None
        self._t2i = None

    @property
    def is_multiview(self) -> bool:
        return self.shape_model.startswith("mv")

    # ------------------------------------------------------------------ loaders
    def _shape_pipeline(self):
        if self._shape is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            repo, subfolder, _, flashvdm = SHAPE_MODELS[self.shape_model]
            _log(f"Loading shape model {repo}/{subfolder} on {self.device} ...")
            self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                repo, subfolder=subfolder, use_safetensors=True, device=self.device
            )
            if flashvdm:
                # "dmc" would need the extra `diso` package; "mc" works everywhere.
                self._shape.enable_flashvdm(mc_algo="mc")
        return self._shape

    def _paint_pipeline(self):
        if self._paint is None:
            if self.device != "cuda":
                raise RuntimeError("Texture generation requires an NVIDIA GPU (CUDA).")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            repo, subfolder = TEXTURE_MODELS[self.texture_model]
            _log(f"Loading texture model {repo}/{subfolder} ...")
            self._paint = Hunyuan3DPaintPipeline.from_pretrained(repo, subfolder=subfolder)
            if self.low_vram:
                self._paint.enable_model_cpu_offload()
        return self._paint

    def _background_remover(self):
        if self._rembg is None:
            from hy3dgen.rembg import BackgroundRemover

            self._rembg = BackgroundRemover()
        return self._rembg

    def _text_to_image(self):
        if self._t2i is None:
            if self.device != "cuda":
                raise RuntimeError("Text-to-3D requires an NVIDIA GPU (CUDA).")
            from hy3dgen.text2image import HunyuanDiTPipeline

            _log("Loading text-to-image model (HunyuanDiT) ...")
            self._t2i = HunyuanDiTPipeline(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", device=self.device
            )
        return self._t2i

    # ------------------------------------------------------------------ helpers
    def prepare_image(self, image, remove_background: bool = True):
        """Load an image (path or PIL) and strip its background if it has none."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255
        if remove_background and not has_alpha:
            _log("Removing background ...")
            image = self._background_remover()(image.convert("RGB"))
        return image.convert("RGBA")

    def text_to_image(self, prompt: str, seed: int = 1234):
        _log(f"Generating reference image for prompt: {prompt!r}")
        return self._text_to_image()(prompt, seed=seed)

    # ------------------------------------------------------------------ main API
    def generate(
        self,
        image=None,
        views: Optional[Dict[str, Union[str, Path, "Image.Image"]]] = None,
        prompt: Optional[str] = None,
        settings: Optional[GenerationSettings] = None,
        remove_background: bool = True,
    ):
        """Generate a trimesh.Trimesh from one image, multiple views, or a text prompt.

        Returns (mesh, reference_image).
        """
        import torch

        s = settings or GenerationSettings()
        sources = sum(x is not None for x in (image, views, prompt))
        if sources != 1:
            raise ValueError("Provide exactly one of: image, views, prompt")

        if s.texture and self.device != "cuda":
            raise RuntimeError("Texture generation requires an NVIDIA GPU (CUDA).")

        if prompt is not None:
            image = self.text_to_image(prompt, seed=s.seed)

        if views is not None:
            if not self.is_multiview:
                raise ValueError("Multi-view input requires the 'mv' or 'mv-turbo' shape model.")
            bad = set(views) - set(MULTIVIEW_KEYS)
            if bad:
                raise ValueError(f"Unknown view(s) {sorted(bad)}; use {MULTIVIEW_KEYS}")
            cond = {k: self.prepare_image(v, remove_background) for k, v in views.items()}
            reference = cond["front"] if "front" in cond else next(iter(cond.values()))
        else:
            reference = self.prepare_image(image, remove_background)
            cond = {"front": reference} if self.is_multiview else reference

        steps = s.steps or SHAPE_MODELS[self.shape_model][2]
        _log(f"Generating shape ({self.shape_model}, {steps} steps, octree {s.octree_resolution}) ...")
        start = time.time()
        mesh = self._shape_pipeline()(
            image=cond,
            num_inference_steps=steps,
            guidance_scale=s.guidance_scale,
            octree_resolution=s.octree_resolution,
            num_chunks=s.num_chunks,
            generator=torch.manual_seed(s.seed),
            output_type="trimesh",
        )[0]
        _log(f"Shape done in {time.time() - start:.1f}s ({len(mesh.faces)} faces)")

        mesh = self.postprocess(mesh, s)

        if s.texture:
            _log("Painting texture ...")
            start = time.time()
            mesh = self._paint_pipeline()(mesh, image=reference)
            _log(f"Texture done in {time.time() - start:.1f}s")

        if self.low_vram and self.device == "cuda":
            torch.cuda.empty_cache()
        return mesh, reference

    @staticmethod
    def postprocess(mesh, s: GenerationSettings):
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        if s.remove_floaters:
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
        if s.max_faces and len(mesh.faces) > s.max_faces:
            mesh = FaceReducer()(mesh, max_facenum=s.max_faces)
            _log(f"Reduced to {len(mesh.faces)} faces")
        return mesh


# Blender script: import a GLB and write an FBX with textures embedded.
_BLENDER_FBX_SCRIPT = """
import sys, bpy
src, dst = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
bpy.ops.export_scene.fbx(filepath=dst, path_mode="COPY", embed_textures=True)
"""


def _find_blender() -> Optional[str]:
    return os.environ.get("BLENDER") or shutil.which("blender")


def _export_fbx(mesh, output: Path) -> None:
    """trimesh can't write FBX, so write a GLB and convert it with Blender or assimp."""
    blender = _find_blender()
    assimp = shutil.which("assimp")
    if not blender and not assimp:
        raise RuntimeError(
            "FBX export needs Blender (on PATH or set BLENDER=/path/to/blender) "
            "or the assimp command-line tool (e.g. `apt install assimp-utils`, `brew install assimp`)."
        )
    with tempfile.TemporaryDirectory() as tmp:
        glb = Path(tmp) / "mesh.glb"
        mesh.export(str(glb))
        if blender:
            script = Path(tmp) / "to_fbx.py"
            script.write_text(_BLENDER_FBX_SCRIPT)
            cmd = [blender, "-b", "--factory-startup", "--python", str(script), "--", str(glb), str(output)]
        else:
            cmd = [assimp, "export", str(glb), str(output), "-ffbx"]
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output.is_file():
        tail = (result.stderr or result.stdout).strip()[-800:]
        raise RuntimeError(f"FBX conversion failed ({Path(cmd[0]).name}):\n{tail}")


def export_mesh(mesh, output: Union[str, Path]) -> Path:
    output = Path(output)
    if output.suffix.lower() not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported output format '{output.suffix}'. Use one of {EXPORT_FORMATS}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".fbx":
        _export_fbx(mesh, output.resolve())
    else:
        mesh.export(str(output))
    _log(f"Saved {output}")
    return output
