"""Checks that keep the repo's install paths consistent with each other."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _commits(path, pattern):
    return set(re.findall(pattern, (ROOT / path).read_text()))


def test_hunyuan_commit_pinned_consistently():
    req = _commits("requirements.txt", r"Hunyuan3D-2\.git@([0-9a-f]{40})")
    install = _commits("scripts/install.sh", r"HY_COMMIT=([0-9a-f]{40})")
    docker = _commits("Dockerfile", r"HY_COMMIT=([0-9a-f]{40})")
    notebook = _commits("create3d_colab.ipynb", r"Hunyuan3D-2/([0-9a-f]{40})/")
    assert len(req) == 1
    assert req == install == docker == notebook


def test_requirements_are_pinned():
    unpinned = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-e") or "@ git+" in line:
            continue
        name = re.split(r"[<>=!~ ]", line)[0]
        if name in ("torch", "torchvision"):   # range on purpose: GPU/OS-specific builds
            continue
        if "==" not in line:
            unpinned.append(line)
    assert unpinned == []


def test_notebook_code_cells_are_valid_python():
    ipython = pytest.importorskip("IPython.core.inputtransformer2")
    nb = json.loads((ROOT / "create3d_colab.ipynb").read_text())
    tm = ipython.TransformerManager()
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert cells
    for i, cell in enumerate(cells):
        compile(tm.transform_cell("".join(cell["source"])), f"cell {i}", "exec")


def test_notebook_has_no_saved_outputs():
    nb = json.loads((ROOT / "create3d_colab.ipynb").read_text())
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
