"""Gradio browser interface for create3d."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from .engine import GenerationSettings, Generator3D, export_mesh

OUTPUT_DIR = Path(tempfile.gettempdir()) / "create3d_outputs"


def build_ui(gen: Generator3D):
    import gradio as gr

    lock = threading.Lock()  # one generation at a time on the GPU

    def run(mode, image, prompt, front, left, back, right, texture, steps, guidance,
            octree, seed, max_faces, remove_bg, fmt, progress=gr.Progress()):
        settings = GenerationSettings(
            steps=int(steps) or None,
            guidance_scale=float(guidance),
            octree_resolution=int(octree),
            seed=int(seed),
            max_faces=int(max_faces) or None,
            texture=bool(texture),
        )
        kwargs = {}
        if mode == "Image":
            if image is None:
                raise gr.Error("Upload an image first.")
            kwargs["image"] = image
        elif mode == "Text":
            if not prompt or not prompt.strip():
                raise gr.Error("Enter a prompt first.")
            kwargs["prompt"] = prompt.strip()
        else:
            views = {k: v for k, v in
                     {"front": front, "left": left, "back": back, "right": right}.items() if v is not None}
            if not views:
                raise gr.Error("Upload at least one view.")
            if not gen.is_multiview:
                raise gr.Error("Multi-view needs the app started with --model mv or mv-turbo.")
            kwargs["views"] = views

        progress(0.1, desc="Generating ...")
        with lock:
            try:
                mesh, ref = gen.generate(settings=settings, remove_background=remove_bg, **kwargs)
            except (RuntimeError, ValueError) as exc:
                raise gr.Error(str(exc))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"model_{time.strftime('%Y%m%d_%H%M%S')}"
        # The viewer always gets a GLB; the download uses the chosen format.
        preview = export_mesh(mesh, OUTPUT_DIR / f"{stem}.glb")
        download = preview if fmt == ".glb" else export_mesh(mesh, OUTPUT_DIR / f"{stem}{fmt}")
        info = f"{len(mesh.vertices):,} vertices · {len(mesh.faces):,} faces"
        return str(preview), str(download), ref, info

    with gr.Blocks(title="create3d · Hunyuan3D-2") as demo:
        gr.Markdown(
            "# create3d\nTurn an image, a few views, or a text prompt into a 3D model "
            f"with Tencent Hunyuan3D-2. Shape model: **{gen.shape_model}** on **{gen.device}**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["Image", "Text", "Multi-view"], value="Image", label="Input")
                with gr.Group(visible=True) as image_box:
                    image = gr.Image(type="pil", label="Image", image_mode="RGBA")
                with gr.Group(visible=False) as text_box:
                    prompt = gr.Textbox(label="Prompt", placeholder="a wooden treasure chest, game asset")
                with gr.Group(visible=False) as mv_box:
                    with gr.Row():
                        front = gr.Image(type="pil", label="Front", image_mode="RGBA")
                        left = gr.Image(type="pil", label="Left", image_mode="RGBA")
                    with gr.Row():
                        back = gr.Image(type="pil", label="Back", image_mode="RGBA")
                        right = gr.Image(type="pil", label="Right", image_mode="RGBA")
                texture = gr.Checkbox(label="Paint texture (CUDA, slower)", value=False)
                with gr.Accordion("Advanced", open=False):
                    steps = gr.Slider(0, 100, value=0, step=1, label="Steps (0 = model default)")
                    guidance = gr.Slider(0, 15, value=5.0, step=0.5, label="Guidance scale")
                    octree = gr.Slider(64, 512, value=256, step=16, label="Octree resolution")
                    seed = gr.Number(value=1234, precision=0, label="Seed")
                    max_faces = gr.Slider(0, 200000, value=40000, step=1000, label="Max faces (0 = no reduction)")
                    remove_bg = gr.Checkbox(value=True, label="Auto-remove background")
                    fmt = gr.Dropdown([".glb", ".obj", ".ply", ".stl", ".fbx"], value=".glb", label="Download format")
                btn = gr.Button("Generate 3D model", variant="primary")
            with gr.Column(scale=2):
                viewer = gr.Model3D(label="Preview", height=520, clear_color=[0.92, 0.92, 0.92, 1.0])
                info = gr.Markdown()
                with gr.Row():
                    download = gr.File(label="Download")
                    ref_out = gr.Image(label="Reference image", height=200)

        def switch(m):
            return (gr.update(visible=m == "Image"), gr.update(visible=m == "Text"),
                    gr.update(visible=m == "Multi-view"))

        mode.change(switch, mode, [image_box, text_box, mv_box])
        btn.click(
            run,
            [mode, image, prompt, front, left, back, right, texture, steps, guidance,
             octree, seed, max_faces, remove_bg, fmt],
            [viewer, download, ref_out, info],
        )
    return demo


def launch(args) -> None:
    gen = Generator3D(
        shape_model=args.model,
        texture_model=args.texture_model,
        device=args.device,
        low_vram=args.low_vram,
    )
    demo = build_ui(gen)
    demo.queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=args.share)
