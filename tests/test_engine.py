import pytest
import trimesh
from PIL import Image

from create3d.engine import SHAPE_MODELS, GenerationSettings, Generator3D


def test_unknown_models_are_rejected(fake_backend):
    with pytest.raises(ValueError, match="Unknown shape model"):
        Generator3D(shape_model="huge")
    with pytest.raises(ValueError, match="Unknown texture model"):
        Generator3D(texture_model="huge")


@pytest.mark.parametrize("cuda, expected", [(True, "cuda"), (False, "cpu")])
def test_auto_device(fake_backend, cuda, expected):
    fake_backend.cuda = cuda
    assert Generator3D().device == expected
    assert Generator3D(device="mps").device == "mps"


@pytest.mark.parametrize("model", list(SHAPE_MODELS))
def test_presets_load_the_right_weights_and_steps(fake_backend, rgb_image, model):
    repo, subfolder, steps, flashvdm = SHAPE_MODELS[model]
    gen = Generator3D(shape_model=model)
    gen.generate(image=rgb_image)

    _, loaded_repo, kw = fake_backend.last("shape.load")
    assert (loaded_repo, kw["subfolder"]) == (repo, subfolder)
    assert fake_backend.last("shape.generate")[1]["num_inference_steps"] == steps
    if flashvdm:
        # "dmc" needs the uninstalled `diso` package and would crash on CUDA.
        assert fake_backend.last("shape.flashvdm")[1] == {"mc_algo": "mc"}
    else:
        assert "shape.flashvdm" not in fake_backend.names()


def test_settings_reach_the_pipeline(fake_backend, rgb_image):
    s = GenerationSettings(steps=12, guidance_scale=7.5, octree_resolution=384, num_chunks=20000, seed=99)
    Generator3D().generate(image=rgb_image, settings=s)
    kw = fake_backend.last("shape.generate")[1]
    assert kw["num_inference_steps"] == 12
    assert kw["guidance_scale"] == 7.5
    assert kw["octree_resolution"] == 384
    assert kw["num_chunks"] == 20000
    assert kw["generator"] == ("generator", 99)
    assert kw["output_type"] == "trimesh"


def test_pipeline_is_loaded_once(fake_backend, rgb_image):
    gen = Generator3D()
    gen.generate(image=rgb_image)
    gen.generate(image=rgb_image)
    assert fake_backend.names().count("shape.load") == 1
    assert fake_backend.names().count("shape.generate") == 2


def test_background_removed_for_opaque_image(fake_backend, rgb_image):
    _, ref = Generator3D().generate(image=rgb_image)
    assert ("rembg", "RGB") in fake_backend.calls
    assert ref.mode == "RGBA"


def test_background_kept_for_transparent_image(fake_backend, transparent_image):
    Generator3D().generate(image=transparent_image)
    assert "rembg" not in fake_backend.names()


def test_background_removal_can_be_disabled(fake_backend, rgb_image):
    Generator3D().generate(image=rgb_image, remove_background=False)
    assert "rembg" not in fake_backend.names()


def test_accepts_pil_image(fake_backend):
    Generator3D().generate(image=Image.new("RGB", (32, 32)))
    assert "shape.generate" in fake_backend.names()


def test_exactly_one_source_required(fake_backend, rgb_image):
    gen = Generator3D()
    with pytest.raises(ValueError, match="exactly one"):
        gen.generate()
    with pytest.raises(ValueError, match="exactly one"):
        gen.generate(image=rgb_image, prompt="a chair")


def test_postprocess_removes_floaters_and_reduces_faces(fake_backend, rgb_image):
    mesh, _ = Generator3D().generate(image=rgb_image, settings=GenerationSettings(max_faces=1000))
    assert fake_backend.names()[-3:] == ["floater", "degenerate", "reduce"]
    assert fake_backend.last("reduce") == ("reduce", 1000)
    assert len(mesh.faces) == len(trimesh.creation.icosphere(subdivisions=2).faces)  # fake reducer output


def test_postprocess_can_be_disabled(fake_backend, rgb_image):
    Generator3D().generate(image=rgb_image, settings=GenerationSettings(remove_floaters=False, max_faces=None))
    assert not {"floater", "degenerate", "reduce"} & set(fake_backend.names())


def test_no_reduction_when_already_small(fake_backend, rgb_image):
    Generator3D().generate(image=rgb_image, settings=GenerationSettings(max_faces=10**7))
    assert "reduce" not in fake_backend.names()


# ---------------------------------------------------------------- multi-view
def test_multiview_passes_dict_of_views(fake_backend, rgb_image, transparent_image):
    gen = Generator3D(shape_model="mv-turbo")
    _, ref = gen.generate(views={"front": rgb_image, "back": transparent_image})
    cond = fake_backend.last("shape.generate")[1]["image"]
    assert set(cond) == {"front", "back"}
    assert all(img.mode == "RGBA" for img in cond.values())
    assert ref is cond["front"]


def test_single_image_on_multiview_model_becomes_front_view(fake_backend, rgb_image):
    Generator3D(shape_model="mv").generate(image=rgb_image)
    assert set(fake_backend.last("shape.generate")[1]["image"]) == {"front"}


def test_views_need_multiview_model(fake_backend, rgb_image):
    with pytest.raises(ValueError, match="mv"):
        Generator3D(shape_model="mini-turbo").generate(views={"front": rgb_image})


def test_unknown_view_name(fake_backend, rgb_image):
    with pytest.raises(ValueError, match="Unknown view"):
        Generator3D(shape_model="mv-turbo").generate(views={"top": rgb_image})


# ------------------------------------------------------------- texture / text
def test_texture_requires_cuda_before_doing_any_work(fake_backend, rgb_image):
    with pytest.raises(RuntimeError, match="CUDA"):
        Generator3D().generate(image=rgb_image, settings=GenerationSettings(texture=True))
    assert "shape.generate" not in fake_backend.names()


def test_texture_on_cuda(fake_backend, rgb_image):
    fake_backend.cuda = True
    Generator3D(texture_model="full").generate(image=rgb_image, settings=GenerationSettings(texture=True))
    _, repo, kw = fake_backend.last("paint.load")
    assert (repo, kw["subfolder"]) == ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0")
    assert fake_backend.names()[-1] == "paint.generate"
    assert "paint.offload" not in fake_backend.names()


def test_low_vram_offloads_texture_model_only(fake_backend, rgb_image):
    fake_backend.cuda = True
    Generator3D(low_vram=True).generate(image=rgb_image, settings=GenerationSettings(texture=True))
    assert "paint.offload" in fake_backend.names()


def test_text_prompt_on_cuda(fake_backend):
    fake_backend.cuda = True
    Generator3D().generate(prompt="a wooden chair", settings=GenerationSettings(seed=7))
    assert ("t2i.generate", "a wooden chair", 7) in fake_backend.calls
    assert "shape.generate" in fake_backend.names()


def test_text_prompt_requires_cuda(fake_backend):
    with pytest.raises(RuntimeError, match="CUDA"):
        Generator3D().generate(prompt="a wooden chair")
