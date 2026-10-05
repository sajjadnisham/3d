import pytest
from PIL import Image

from create3d.cli import build_parser, main


def run(*argv):
    main([str(a) for a in argv])


# ------------------------------------------------- validation (no model loads)
@pytest.mark.parametrize("argv, message", [
    (["image", "missing.png"], "image not found"),
    (["views", "-o", "x.glb"], "give at least one of"),
    (["batch", "no_such_dir"], "not a directory"),
])
def test_bad_arguments(fake_backend, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        run(*argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert fake_backend.calls == []


def test_bad_output_format(fake_backend, rgb_image, capsys):
    with pytest.raises(SystemExit):
        run("image", rgb_image, "-o", "model.usdz")
    assert "output must end with one of" in capsys.readouterr().err


def test_views_reject_single_view_model(fake_backend, rgb_image, capsys):
    with pytest.raises(SystemExit):
        run("views", "--front", rgb_image, "--model", "mini", "-o", "x.glb")
    assert "needs --model mv or mv-turbo" in capsys.readouterr().err


def test_empty_batch_dir(fake_backend, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run("batch", tmp_path)
    assert "no images found" in capsys.readouterr().err


def test_parser_defaults():
    args = build_parser().parse_args(["image", "a.png"])
    assert args.model is None and args.octree == 256 and args.max_faces == 40000
    assert args.texture_model == "turbo" and not args.texture


# ----------------------------------------------------------- full runs (fake)
def test_image_default_output_next_to_input(fake_backend, rgb_image):
    run("image", rgb_image)
    assert rgb_image.with_suffix(".glb").is_file()
    assert fake_backend.last("shape.load")[2]["subfolder"] == "hunyuan3d-dit-v2-mini-turbo"


def test_image_options(fake_backend, rgb_image, tmp_path):
    out = tmp_path / "out" / "chair.obj"
    run("image", rgb_image, "-o", out, "--model", "full", "--steps", "20", "--octree", "384",
        "--seed", "5", "--max-faces", "0", "--keep-floaters", "--keep-background", "--save-image")
    assert out.is_file()
    assert (tmp_path / "out" / "chair_input.png").is_file()
    kw = fake_backend.last("shape.generate")[1]
    assert (kw["num_inference_steps"], kw["octree_resolution"], kw["generator"]) == (20, 384, ("generator", 5))
    assert not {"rembg", "floater", "reduce"} & set(fake_backend.names())


def test_views_default_to_mv_turbo(fake_backend, rgb_image, tmp_path):
    run("views", "--front", rgb_image, "--left", rgb_image, "-o", tmp_path / "toy.ply")
    assert (tmp_path / "toy.ply").is_file()
    assert fake_backend.last("shape.load")[2]["subfolder"] == "hunyuan3d-dit-v2-mv-turbo"
    assert set(fake_backend.last("shape.generate")[1]["image"]) == {"front", "left"}


def test_batch(fake_backend, tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    for name in ("a.png", "b.jpg"):
        Image.new("RGB", (32, 32)).save(src / name)
    (src / "notes.txt").write_text("ignored")
    run("batch", src, "-o", tmp_path / "models", "--format", ".stl")
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["a.stl", "b.stl"]
    assert fake_backend.names().count("shape.load") == 1


def test_batch_keeps_going_after_a_bad_image(fake_backend, tmp_path, capsys):
    src = tmp_path / "photos"
    src.mkdir()
    Image.new("RGB", (32, 32)).save(src / "good.png")
    (src / "broken.png").write_bytes(b"not an image")
    with pytest.raises(SystemExit) as exc:
        run("batch", src, "-o", tmp_path / "models")
    assert exc.value.code == 1
    assert (tmp_path / "models" / "good.glb").is_file()
    out = capsys.readouterr()
    assert "FAILED broken.png" in out.err and "1 ok, 1 failed" in out.out


def test_texture_without_cuda_is_a_clean_error(fake_backend, rgb_image, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run("image", rgb_image, "-o", tmp_path / "x.glb", "--texture")
    assert "requires an NVIDIA GPU" in str(exc.value.code)
    assert not (tmp_path / "x.glb").exists()


def test_text_command_on_cuda(fake_backend, tmp_path):
    fake_backend.cuda = True
    run("text", "a red teapot", "-o", tmp_path / "teapot.glb", "--texture")
    assert (tmp_path / "teapot.glb").is_file()
    assert fake_backend.names()[0] == "t2i.load"
    assert fake_backend.names()[-1] == "paint.generate"
