import pytest

gr = pytest.importorskip("gradio")

from create3d.app import build_ui  # noqa: E402
from create3d.engine import Generator3D  # noqa: E402


def test_ui_builds(fake_backend):
    demo = build_ui(Generator3D())
    assert isinstance(demo, gr.Blocks)
    labels = {getattr(c, "label", None) for c in demo.blocks.values()}
    assert {"Input", "Prompt", "Preview", "Download", "Download format"} <= labels


def test_ui_offers_every_export_format(fake_backend):
    from create3d.engine import EXPORT_FORMATS

    demo = build_ui(Generator3D())
    dropdown = next(c for c in demo.blocks.values() if getattr(c, "label", None) == "Download format")
    assert [choice[0] for choice in dropdown.choices] == list(EXPORT_FORMATS)
