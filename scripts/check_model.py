"""Sanity-check a generated 3D model and render a preview image.

    python scripts/check_model.py model.glb --preview preview.png --min-faces 1000

Exits non-zero if the mesh is empty, degenerate or mostly fragments. Used by the
end-to-end workflow; handy locally too (preview needs matplotlib).
"""

import argparse
import json
import sys

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--preview", help="write a PNG render (needs matplotlib)")
    ap.add_argument("--min-faces", type=int, default=1000)
    args = ap.parse_args()

    mesh = trimesh.load(args.model, force="mesh")
    extents = mesh.extents
    stats = {
        "file": args.model,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents": [round(float(e), 3) for e in extents],
        "watertight": bool(mesh.is_watertight),
        "area": round(float(mesh.area), 4),
    }
    print(json.dumps(stats, indent=2))

    problems = []
    if stats["faces"] < args.min_faces:
        problems.append(f"only {stats['faces']} faces (< {args.min_faces})")
    if not np.all(np.isfinite(mesh.vertices)):
        problems.append("non-finite vertex coordinates")
    if extents.min() <= 1e-3 * max(extents.max(), 1e-9):
        problems.append(f"degenerate (flat) bounding box {stats['extents']}")

    if args.preview:
        render(mesh, args.preview)
        print(f"preview written to {args.preview}")

    if problems:
        print("FAILED: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)
    print("OK")


def render(mesh, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(mesh.faces) > 60000:  # keep the render quick
        idx = np.random.default_rng(0).choice(len(mesh.faces), 60000, replace=False)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[idx], process=False)
    # trimesh/glTF is Y-up, matplotlib is Z-up: swap Y and Z for both vertices and normals
    tris = mesh.vertices[mesh.faces][:, :, [0, 2, 1]]
    normals = mesh.face_normals[:, [0, 2, 1]]
    shade = 0.35 + 0.65 * np.abs(normals @ (np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])))
    colors = np.stack([shade * 0.55, shade * 0.7, shade * 0.95, np.ones_like(shade)], axis=1)

    fig = plt.figure(figsize=(9, 4.5), dpi=110)
    for i, (elev, azim) in enumerate([(15, -60), (15, 120)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, linewidths=0))
        lo, hi = mesh.bounds[:, [0, 2, 1]]
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
