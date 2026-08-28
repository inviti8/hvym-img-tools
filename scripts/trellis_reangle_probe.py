"""Does the artist's art project cleanly onto a TRELLIS mesh?

The one question that decides whether TRELLIS can replace TripoSR in `reangle`
(docs/tools/mesh.md §5 anticipated the swap; nothing has measured it). Every
TRELLIS number we hold comes from the `mesh` tool -- untextured geometry judged
by largest-connected-component. Reangle asks something different: its mesh is a
**depth proxy** whose value is hugging the drawing's outline, because the
artist's own pixels get front-projected onto it. The metric for that is
silhouette IoU, and TripoSR's is **0.776** (docs/BENCHMARK.md §2).

A more "complete" generative model is not automatically better here. It can add
geometry the artist never drew, which under front-planar UV means *more*
mirror-smear on the back faces, not less. This script settles it by eye and by
number, without building an image or touching a live endpoint.

It deliberately drives `run_pipeline` -- the real guts, not a reimplementation --
so a good result is directly the production path.

## Running it

Everything needed is already in the `mesh` worker image; only the isnet weights
are missing, and they are fetched on demand.

    docker run --rm --gpus all \
        -v "$PWD:/probe" \
        -e PYTHONPATH=/app:/probe \
        ghcr.io/inviti8/hvym-img-mesh:0.3.1 \
        python /probe/scripts/trellis_reangle_probe.py \
            --image /probe/docs/benchmark/paint3d/source_drawing.png \
            --baseline-glb /probe/docs/benchmark/char.glb \
            --out /probe/probe-out

isnet has no CUDA provider in that image, so the matte runs on CPU at ~6 s
instead of ~0.03 s. Irrelevant here, and not worth a rebuild to avoid.

## Reading the output

| artefact | what it answers |
|---|---|
| `IoU` vs the baseline | does the art still land on the silhouette? |
| `*_overlay.png` | *where* it misses -- red art-only, green mesh-only, yellow agreeing |
| `*_orbit.png` | how the projection holds across the +/-20 deg window |
| `*.glb` | the shippable artefact, for a look in any viewer |
| `summary.json` | the numbers, for pasting into a doc |

A verdict needs the overlay *and* the number. IoU alone cannot tell "slightly
mis-scaled" from "projected onto the wrong surface".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Same source the GPU images bake in (docker/Dockerfile).
ISNET_URL = "https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx"

#: docs/BENCHMARK.md §2 -- TripoSR on the character drawing, best of six axes.
TRIPOSR_REFERENCE_IOU = 0.776


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #

def ensure_isnet(path: Path) -> Path:
    """Fetch the matting weights if this image did not bake them in."""
    if path.exists() and path.stat().st_size > 100_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching isnet -> {path} (~176 MB, once)")
    urllib.request.urlretrieve(ISNET_URL, path)
    size = path.stat().st_size
    if size < 100_000_000:
        raise SystemExit(f"isnet download looks truncated: {size} bytes")
    return path


def build_models(device: str):
    """Wire the real `ModelCache` exactly as a worker does.

    Going through the registry rather than calling a loader directly also
    exercises the backbone -> model-key resolution `ReangleTool` depends on.
    """
    from hvym_img_tools.core import registry
    from hvym_img_tools.core.models import ModelCache

    registry.discover()
    models = ModelCache(device=device)
    for tool_cls in registry.all_tools():
        for key, loader in tool_cls().model_loaders().items():
            if key not in models.registered():
                models.register(key, loader)
    return models


# --------------------------------------------------------------------------- #
# alignment diagnostics
# --------------------------------------------------------------------------- #

def splat(px: np.ndarray, size: int, radius: int = 2) -> np.ndarray:
    """Dilated point splat, matching `uvbake._splat_mask`'s approximation."""
    ix = np.clip(px[:, 0].astype(np.int32), 0, size - 1)
    iy = np.clip(px[:, 1].astype(np.int32), 0, size - 1)
    mask = np.zeros((size, size), bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            mask[np.clip(iy + dy, 0, size - 1), np.clip(ix + dx, 0, size - 1)] = True
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def overlay(art_alpha: np.ndarray, mesh_mask: np.ndarray) -> Image.Image:
    """Red = art with no mesh under it. Green = mesh outside the art. Yellow = agreement.

    The failure this exists to catch is a projection that scores respectably
    while sitting slightly off -- which no single number distinguishes.
    """
    art = art_alpha > 12
    rgb = np.full(art.shape + (3,), 255, np.uint8)
    rgb[art & ~mesh_mask] = (220, 40, 40)
    rgb[mesh_mask & ~art] = (40, 170, 60)
    rgb[art & mesh_mask] = (250, 205, 40)
    return Image.fromarray(rgb)


def uv_to_pixels(uv: np.ndarray, size: int) -> np.ndarray:
    """glTF UV (origin bottom-left) -> image pixel coords (origin top-left)."""
    return np.column_stack([uv[:, 0] * (size - 1), (1.0 - uv[:, 1]) * (size - 1)])


def coverage_mask(uv_px: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Where the delivered mesh's UVs actually land on the art.

    Densified over the faces first, with the same helper and threshold
    `detect_front_view` uses. Skipping that is not a small inaccuracy: a
    20k-face mesh carries ~10k vertices against a `MIN_SPLAT_POINTS` of 40k, and
    splatting the bare vertices scored 0.095 against the pipeline's own 0.344 on
    the same geometry -- an apparent catastrophe that was purely the metric.
    """
    from hvym_img_tools.tools.reangle.uvbake import MIN_SPLAT_POINTS, densify

    return splat(densify(uv_px, faces, MIN_SPLAT_POINTS), size)


def orbit_strip(vertices, faces, uv_px, art, axes, sign, angles, size):
    """Re-render the textured proxy across the angle window.

    Reuses `preview_reangle`'s z-buffered rasteriser rather than a GL path, so
    what appears is exactly the artist's own pixels -- no glTF UV-convention
    ambiguity sitting between this and the viewer.
    """
    import preview_reangle as pv

    h_axis, v_axis, d_axis = axes
    V = np.asarray(vertices, float).copy()
    if sign < 0:
        V[:, h_axis] = -V[:, h_axis]

    tiles = []
    for deg in angles:
        t = np.radians(deg)
        ct, st = np.cos(t), np.sin(t)
        h, d = V[:, h_axis], V[:, d_axis]
        cx, cd = (h.min() + h.max()) / 2, (d.min() + d.max()) / 2
        hh, dd = h - cx, d - cd
        V2 = np.empty_like(V)
        V2[:, h_axis] = hh * ct - dd * st + cx
        V2[:, d_axis] = hh * st + dd * ct + cd
        V2[:, v_axis] = V[:, v_axis]
        img = pv.rasterize(pv.fit(V2[:, [h_axis, v_axis]]), V2[:, d_axis], uv_px, faces, art)
        rgb = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4] / 255.0
        tiles.append(Image.fromarray((rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)))

    strip = Image.new("RGB", (size * len(tiles), size), (255, 255, 255))
    for i, tile in enumerate(tiles):
        strip.paste(tile.resize((size, size)), (i * size, 0))
    return strip


def load_mesh(path):
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(tuple(scene.geometry.values()))
    return scene


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="TRELLIS texture-projection probe for reangle")
    ap.add_argument("--image", required=True, nargs="+", help="drawing(s) to probe")
    ap.add_argument("--backbone", default="trellis")
    ap.add_argument("--out", default="probe-out")
    ap.add_argument("--isnet", default=os.environ.get("ISNET_PATH", "/opt/models/isnet_dis.onnx"))
    ap.add_argument("--target-faces", type=int, default=20_000,
                    help="0 disables; TripoSR needs no cap, TRELLIS does")
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-resolution", type=int, default=256,
                    help="TripoSR only; TRELLIS ignores it")
    ap.add_argument("--baseline-glb", default=None,
                    help="an existing TripoSR .glb, scored with the SAME metric")
    ap.add_argument("--angles", default="-20,-12,-6,0,6,12,20")
    ap.add_argument("--no-orbit", action="store_true", help="skip the slow rasteriser")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    isnet_path = ensure_isnet(Path(args.isnet))
    os.environ["ISNET_PATH"] = str(isnet_path)

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"device={device}  backbone={args.backbone}  target_faces={args.target_faces}")

    from hvym_img_tools.backbones import get_backbone, model_key_for
    from hvym_img_tools.core.imageio import DEFAULT_SIZE
    from hvym_img_tools.tools.reangle.pipeline import ISNET_MODEL_KEY, run_pipeline
    from hvym_img_tools.tools.reangle.uvbake import detect_front_view

    models = build_models(device)
    started = time.perf_counter()
    backbone_model = models.get(model_key_for(args.backbone))
    print(f"loaded {args.backbone} in {time.perf_counter() - started:.1f}s")
    backbone = get_backbone(args.backbone, backbone_model)
    isnet = models.get(ISNET_MODEL_KEY)

    angles = [float(a) for a in args.angles.split(",")]
    results = []

    for image_path in args.image:
        name = Path(image_path).stem
        print(f"\n=== {name} ===")
        started = time.perf_counter()
        res = run_pipeline(
            Path(image_path).read_bytes(),
            isnet_session=isnet,
            backbone=backbone,
            mc_resolution=args.mc_resolution,
            texture_size=args.texture_size,
            target_faces=args.target_faces or None,
            seed=args.seed,
        )
        wall = time.perf_counter() - started

        (out / f"{name}.glb").write_bytes(res.glb)
        (out / f"{name}_matte.png").write_bytes(res.matte_png)

        # Score the DELIVERED artefact, not an intermediate: reload the .glb and
        # splat its UVs back onto the matte. If this disagrees with the pipeline's
        # own IoU, the bake is losing something between the two.
        mesh = load_mesh(out / f"{name}.glb")
        uv = np.asarray(mesh.visual.uv, float)
        matte = Image.open(out / f"{name}_matte.png").convert("RGBA")
        art_small = matte.resize((DEFAULT_SIZE, DEFAULT_SIZE), Image.LANCZOS)
        alpha = np.asarray(art_small)[:, :, 3]

        uv_px = uv_to_pixels(uv, DEFAULT_SIZE)
        mesh_mask = coverage_mask(uv_px, np.asarray(mesh.faces), DEFAULT_SIZE)
        delivered_iou = iou(mesh_mask, alpha > 12)
        overlay(alpha, mesh_mask).save(out / f"{name}_overlay.png")

        row = {
            "subject": name,
            "backbone": args.backbone,
            "wall_s": round(wall, 2),
            "timings": res.timings,
            "faces_raw": res.faces_raw,
            "faces_out": res.faces,
            "vertices": res.vertices,
            "glb_kb": round(len(res.glb) / 1024, 1),
            "pipeline_iou": res.silhouette_iou,
            "delivered_iou": round(delivered_iou, 4),
            "front_axis": res.front_axis,
            "triposr_reference_iou": TRIPOSR_REFERENCE_IOU,
        }

        if args.baseline_glb:
            base = load_mesh(Path(args.baseline_glb))
            bview = detect_front_view(
                np.asarray(base.vertices, float), alpha, np.asarray(base.faces)
            )
            row["baseline_iou"] = round(bview.silhouette_iou, 4)
            row["baseline_faces"] = len(base.faces)

        results.append(row)
        print(json.dumps(row, indent=2))

        if not args.no_orbit:
            view = detect_front_view(
                np.asarray(mesh.vertices, float), alpha, np.asarray(mesh.faces)
            )
            strip = orbit_strip(
                mesh.vertices, np.asarray(mesh.faces), uv_px, np.asarray(art_small),
                (view.h_axis, view.v_axis, view.d_axis), view.sign, angles, DEFAULT_SIZE,
            )
            strip.save(out / f"{name}_orbit.png")
            print(f"wrote {name}_orbit.png ({len(angles)} views)")

    (out / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n=== verdict ===")
    print(f"{'subject':<20}{'IoU':>8}{'vs TripoSR':>12}{'faces':>10}{'glb KB':>9}")
    for r in results:
        base = r.get("baseline_iou", r["triposr_reference_iou"])
        print(f"{r['subject']:<20}{r['delivered_iou']:>8.3f}{base:>12.3f}"
              f"{r['faces_out']:>10}{r['glb_kb']:>9.1f}")
    print(f"\nartefacts in {out.resolve()} -- look at the overlays before deciding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
