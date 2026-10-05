"""Command-line interface: `create3d` / `python -m create3d`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import (
    EXPORT_FORMATS,
    MULTIVIEW_KEYS,
    SHAPE_MODELS,
    TEXTURE_MODELS,
    GenerationSettings,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

EXAMPLES = """\
examples:
  create3d image chair.png                         # chair.glb (untextured, fast)
  create3d image chair.png -o out/chair.glb --texture
  create3d text "a cute cartoon robot" --texture   # needs CUDA
  create3d views --front f.png --left l.png --back b.png -o toy.glb
  create3d batch ./photos -o ./models --format .obj
  create3d ui --port 7860                          # browser interface
"""


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", choices=list(SHAPE_MODELS), default=None,
                   help="shape model preset (default: mini-turbo; mv-turbo for 'views')")
    g.add_argument("--texture-model", choices=list(TEXTURE_MODELS), default="turbo")
    g.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    g.add_argument("--low-vram", action="store_true", help="offload models to CPU when idle")

    g = p.add_argument_group("generation")
    g.add_argument("--texture", action="store_true", help="also paint a texture (CUDA only)")
    g.add_argument("--steps", type=int, default=None, help="diffusion steps (default: 5 turbo / 50 full)")
    g.add_argument("--guidance", type=float, default=5.0, help="guidance scale")
    g.add_argument("--octree", type=int, default=256, help="octree resolution: higher = finer detail (e.g. 384)")
    g.add_argument("--chunks", type=int, default=8000, help="decoder chunk size (raise to go faster on big GPUs)")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--max-faces", type=int, default=40000, help="reduce mesh to this face count (0 = off)")
    g.add_argument("--keep-floaters", action="store_true", help="skip floater/degenerate face cleanup")
    g.add_argument("--keep-background", action="store_true", help="don't auto-remove the image background")
    g.add_argument("--save-image", action="store_true", help="also save the (background-removed) reference image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create3d",
        description="Create 3D models from images or text with Tencent Hunyuan3D-2.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("image", help="single image -> 3D model")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path, help="output file (default: <image>.glb)")
    _add_common(p)

    p = sub.add_parser("text", help="text prompt -> 3D model (via HunyuanDiT, CUDA only)")
    p.add_argument("prompt")
    p.add_argument("-o", "--output", type=Path, default=Path("model.glb"))
    _add_common(p)

    p = sub.add_parser("views", help="multiple views of one object -> 3D model")
    for key in MULTIVIEW_KEYS:
        p.add_argument(f"--{key}", type=Path, help=f"{key} view image")
    p.add_argument("-o", "--output", type=Path, default=Path("model.glb"))
    _add_common(p)

    p = sub.add_parser("batch", help="every image in a folder -> 3D models")
    p.add_argument("input_dir", type=Path)
    p.add_argument("-o", "--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--format", choices=EXPORT_FORMATS, default=".glb")
    _add_common(p)

    p = sub.add_parser("ui", help="launch the browser interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public Gradio link")
    p.add_argument("--model", choices=list(SHAPE_MODELS), default="mini-turbo")
    p.add_argument("--texture-model", choices=list(TEXTURE_MODELS), default="turbo")
    p.add_argument("--device", default="auto")
    p.add_argument("--low-vram", action="store_true")
    return parser


def _settings(args) -> GenerationSettings:
    return GenerationSettings(
        steps=args.steps,
        guidance_scale=args.guidance,
        octree_resolution=args.octree,
        num_chunks=args.chunks,
        seed=args.seed,
        remove_floaters=not args.keep_floaters,
        max_faces=args.max_faces or None,
        texture=args.texture,
    )


def _check_output(path: Path, parser: argparse.ArgumentParser) -> None:
    if path.suffix.lower() not in EXPORT_FORMATS:
        parser.error(f"output must end with one of {', '.join(EXPORT_FORMATS)}")


def _save(mesh, reference, output: Path, save_image: bool) -> None:
    from .engine import export_mesh

    export_mesh(mesh, output)
    if save_image:
        img_path = output.with_name(output.stem + "_input.png")
        reference.save(img_path)
        print(f"[create3d] Saved {img_path}")


def main(argv=None) -> None:
    try:
        _main(argv)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"[create3d] error: {exc}")


def _main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ui":
        from .app import launch

        launch(args)
        return

    # Validate inputs before loading any model.
    views = None
    if args.command == "image":
        if not args.image.is_file():
            parser.error(f"image not found: {args.image}")
        args.output = args.output or args.image.with_suffix(".glb")
    elif args.command == "views":
        views = {k: getattr(args, k) for k in MULTIVIEW_KEYS if getattr(args, k)}
        if not views:
            parser.error("give at least one of " + ", ".join(f"--{k}" for k in MULTIVIEW_KEYS))
        for path in views.values():
            if not path.is_file():
                parser.error(f"image not found: {path}")
        args.model = args.model or "mv-turbo"
        if not args.model.startswith("mv"):
            parser.error("'views' needs --model mv or mv-turbo")
    elif args.command == "batch":
        if not args.input_dir.is_dir():
            parser.error(f"not a directory: {args.input_dir}")
        images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            parser.error(f"no images found in {args.input_dir}")

    if args.command != "batch":
        _check_output(args.output, parser)

    from .engine import Generator3D

    gen = Generator3D(
        shape_model=args.model or "mini-turbo",
        texture_model=args.texture_model,
        device=args.device,
        low_vram=args.low_vram,
    )
    settings = _settings(args)
    remove_bg = not args.keep_background

    if args.command == "image":
        mesh, ref = gen.generate(image=args.image, settings=settings, remove_background=remove_bg)
        _save(mesh, ref, args.output, args.save_image)
    elif args.command == "text":
        mesh, ref = gen.generate(prompt=args.prompt, settings=settings, remove_background=remove_bg)
        _save(mesh, ref, args.output, args.save_image)
    elif args.command == "views":
        mesh, ref = gen.generate(views=views, settings=settings, remove_background=remove_bg)
        _save(mesh, ref, args.output, args.save_image)
    elif args.command == "batch":
        failed = []
        for i, path in enumerate(images, 1):
            print(f"[create3d] ({i}/{len(images)}) {path.name}")
            try:
                mesh, ref = gen.generate(image=path, settings=settings, remove_background=remove_bg)
                _save(mesh, ref, args.output_dir / (path.stem + args.format), args.save_image)
            except Exception as exc:  # keep going on a bad image
                print(f"[create3d] FAILED {path.name}: {exc}", file=sys.stderr)
                failed.append(path.name)
        print(f"[create3d] Done: {len(images) - len(failed)} ok, {len(failed)} failed")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
