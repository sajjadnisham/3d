import shutil
import subprocess

import pytest
import trimesh

from create3d import engine
from create3d.engine import EXPORT_FORMATS, export_mesh


@pytest.mark.parametrize("ext", [e for e in EXPORT_FORMATS if e != ".fbx"])
def test_export_roundtrip(tmp_path, sphere, ext):
    out = export_mesh(sphere, tmp_path / "sub" / f"model{ext}")
    assert out.is_file()
    loaded = trimesh.load(out, force="mesh")
    assert len(loaded.faces) == len(sphere.faces)


def test_unsupported_format(tmp_path, sphere):
    with pytest.raises(ValueError, match="Unsupported output format"):
        export_mesh(sphere, tmp_path / "model.usdz")


def test_fbx_without_converter(tmp_path, sphere, monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Blender .* or the assimp"):
        export_mesh(sphere, tmp_path / "model.fbx")


def test_fbx_prefers_blender(tmp_path, sphere, monkeypatch):
    monkeypatch.setenv("BLENDER", "/opt/blender/blender")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/assimp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--") + 2]
        open(out, "wb").write(b"Kaydara FBX Binary  \x00")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = export_mesh(sphere, tmp_path / "model.fbx")
    assert seen["cmd"][:2] == ["/opt/blender/blender", "-b"]
    assert out.read_bytes().startswith(b"Kaydara FBX")


def test_fbx_conversion_failure_is_reported(tmp_path, sphere, monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name if name == "assimp" else None)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom: bad mesh"))
    with pytest.raises(RuntimeError, match=r"(?s)FBX conversion failed \(assimp\).*boom: bad mesh"):
        export_mesh(sphere, tmp_path / "model.fbx")


@pytest.mark.skipif(shutil.which("assimp") is None, reason="assimp not installed")
def test_fbx_with_real_assimp(tmp_path, sphere, monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr(engine, "_find_blender", lambda: None)
    out = export_mesh(sphere, tmp_path / "model.fbx")
    assert out.read_bytes().startswith(b"Kaydara FBX Binary")
    info = subprocess.run(["assimp", "info", str(out)], capture_output=True, text=True).stdout
    assert f"Faces:              {len(sphere.faces)}" in info
