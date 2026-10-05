"""Shared fixtures.

Unit tests never load the real Hunyuan3D-2 models: `fake_backend` swaps lightweight
fake `torch` and `hy3dgen` modules into sys.modules for the duration of a test and
records every call the tool makes into them.
"""

import os
import sys
import types

import pytest
import trimesh
from PIL import Image

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


class Recorder:
    def __init__(self):
        self.calls = []
        self.cuda = False

    def names(self):
        return [c[0] for c in self.calls]

    def last(self, name):
        return [c for c in self.calls if c[0] == name][-1]


def _sphere(subdivisions=3, radius=1.0, offset=(0, 0, 0)):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius).apply_translation(offset)


@pytest.fixture
def fake_backend(monkeypatch):
    rec = Recorder()

    # --- torch -----------------------------------------------------------------
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: rec.cuda, empty_cache=lambda: None)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.manual_seed = lambda seed: ("generator", seed)

    # --- hy3dgen ---------------------------------------------------------------
    class ShapePipeline:
        @classmethod
        def from_pretrained(cls, repo, **kw):
            rec.calls.append(("shape.load", repo, kw))
            return cls()

        def enable_flashvdm(self, **kw):
            rec.calls.append(("shape.flashvdm", kw))

        def __call__(self, **kw):
            rec.calls.append(("shape.generate", kw))
            # main body + a tiny floating blob far away
            return [trimesh.util.concatenate([_sphere(4), _sphere(1, 0.05, (5, 0, 0))])]

    class PaintPipeline:
        @classmethod
        def from_pretrained(cls, repo, **kw):
            rec.calls.append(("paint.load", repo, kw))
            return cls()

        def enable_model_cpu_offload(self):
            rec.calls.append(("paint.offload",))

        def __call__(self, mesh, image):
            rec.calls.append(("paint.generate", image.mode))
            return mesh

    class BackgroundRemover:
        def __call__(self, image):
            rec.calls.append(("rembg", image.mode))
            out = image.convert("RGBA")
            out.putalpha(128)
            return out

    class HunyuanDiTPipeline:
        def __init__(self, model_path, device):
            rec.calls.append(("t2i.load", model_path, device))

        def __call__(self, prompt, seed=0):
            rec.calls.append(("t2i.generate", prompt, seed))
            return Image.new("RGB", (64, 64), "white")

    class FloaterRemover:
        def __call__(self, mesh):
            rec.calls.append(("floater",))
            return _sphere(4)   # the main body, without the floating blob

    class DegenerateFaceRemover:
        def __call__(self, mesh):
            rec.calls.append(("degenerate",))
            return mesh

    class FaceReducer:
        def __call__(self, mesh, max_facenum):
            rec.calls.append(("reduce", max_facenum))
            return _sphere(2)

    hy = types.ModuleType("hy3dgen")
    shapegen = types.ModuleType("hy3dgen.shapegen")
    shapegen.Hunyuan3DDiTFlowMatchingPipeline = ShapePipeline
    shapegen.FloaterRemover = FloaterRemover
    shapegen.DegenerateFaceRemover = DegenerateFaceRemover
    shapegen.FaceReducer = FaceReducer
    texgen = types.ModuleType("hy3dgen.texgen")
    texgen.Hunyuan3DPaintPipeline = PaintPipeline
    rembg = types.ModuleType("hy3dgen.rembg")
    rembg.BackgroundRemover = BackgroundRemover
    t2i = types.ModuleType("hy3dgen.text2image")
    t2i.HunyuanDiTPipeline = HunyuanDiTPipeline

    for name, mod in {"torch": torch, "hy3dgen": hy, "hy3dgen.shapegen": shapegen,
                      "hy3dgen.texgen": texgen, "hy3dgen.rembg": rembg,
                      "hy3dgen.text2image": t2i}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return rec


@pytest.fixture
def rgb_image(tmp_path):
    path = tmp_path / "chair.png"
    Image.new("RGB", (64, 64), "red").save(path)
    return path


@pytest.fixture
def transparent_image(tmp_path):
    path = tmp_path / "cutout.png"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (16, 16, 48, 48))
    img.save(path)
    return path


@pytest.fixture
def sphere():
    return _sphere(3)
