"""Runs the real Hunyuan3D-2 code (CPU only, no model weights).

Skipped unless the full pinned requirements are installed. The CI integration job sets
CREATE3D_REQUIRE_INTEGRATION=1 so a broken install fails instead of silently skipping.
"""

import importlib
import os

import pytest
import trimesh

for _mod in ("torch", "hy3dgen.shapegen"):
    if os.environ.get("CREATE3D_REQUIRE_INTEGRATION") == "1":
        importlib.import_module(_mod)
    else:
        pytest.importorskip(_mod)

pytestmark = pytest.mark.integration


def test_hunyuan_modules_import():
    from hy3dgen.rembg import BackgroundRemover  # noqa: F401
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    from hy3dgen.text2image import HunyuanDiTPipeline  # noqa: F401
    import hy3dgen.texgen  # noqa: F401


def test_flashvdm_mc_algorithm_exists():
    # The tool always asks for "mc"; make sure the pinned hy3dgen still has it.
    from hy3dgen.shapegen.models.autoencoders.surface_extractors import SurfaceExtractors

    assert "mc" in SurfaceExtractors


def test_real_postprocessing_removes_floaters_and_reduces_faces():
    from create3d.engine import GenerationSettings, Generator3D

    body = trimesh.creation.icosphere(subdivisions=6)
    blob = trimesh.creation.icosphere(subdivisions=1, radius=0.05).apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([body, blob])

    out = Generator3D.postprocess(mesh, GenerationSettings(max_faces=5000))
    assert len(out.split(only_watertight=False)) == 1
    assert len(out.faces) <= 5000


def test_cli_help_with_real_dependencies():
    from create3d.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
