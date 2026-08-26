"""Preview the reangle: orbit the camera with the artist's ORIGINAL art
front-projected onto the TripoSR proxy (REANGLE_PIPELINE.md §7.4).

Deliberately a self-contained numpy rasteriser -- no GL, no glTF UV-convention
ambiguity. It samples the artist's pixels directly, so what you see is exactly
what the viewer should show.

IMPORTANT: TripoSR's front axis is +X (see docs/BENCHMARK.md §5), so the
front-facing image plane is (Y, Z) with X as depth -- NOT (X, Y).

    uv run python scripts/preview_reangle.py
"""
from __future__ import annotations

import argparse
import numpy as np
import trimesh
from PIL import Image

SIZE = 512
MARGIN = 0.08  # must match scripts/reangle/prep_input.py so art aligns to silhouette

# TripoSR convention, established empirically by scripts/benchmark/depthcheck.py
# (silhouette IoU 0.776 vs 0.600 for the next-best axis).
H_AXIS, V_AXIS, D_AXIS = 1, 2, 0  # image-x = Y, image-y = Z, depth = X


def fit(p2: np.ndarray) -> np.ndarray:
    """Aspect-preserving map into SIZE^2 with prep_input.py's margin, centred."""
    mn, mx = p2.min(0), p2.max(0)
    long_side = max(*(mx - mn)) * (1 + 2 * MARGIN)
    out = (p2 - (mn + mx) / 2.0) * ((SIZE - 1) / (long_side + 1e-9)) + (SIZE - 1) / 2.0
    out[:, 1] = (SIZE - 1) - out[:, 1]  # image rows grow downward
    return out


def rasterize(pix: np.ndarray, depth: np.ndarray, uv: np.ndarray,
              faces: np.ndarray, art: np.ndarray) -> np.ndarray:
    """Z-buffered rasteriser that interpolates texture coords barycentrically."""
    zbuf = np.full((SIZE, SIZE), -np.inf)
    out = np.zeros((SIZE, SIZE, 4), np.uint8)
    for tri in faces:
        a, b, c = pix[tri]
        za, zb, zc = depth[tri]
        ua, ub, uc = uv[tri]
        x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        x1 = min(int(np.ceil(max(a[0], b[0], c[0]))), SIZE - 1)
        y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        y1 = min(int(np.ceil(max(a[1], b[1], c[1]))), SIZE - 1)
        if x1 < x0 or y1 < y0:
            continue
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / det
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / det
        w2 = 1.0 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        zz = w0 * za + w1 * zb + w2 * zc
        m &= zz > zbuf[y0:y1 + 1, x0:x1 + 1]
        if not m.any():
            continue
        # barycentric texture coords -> sample the artist's own pixels
        tu = np.clip((w0 * ua[0] + w1 * ub[0] + w2 * uc[0]), 0, SIZE - 1).astype(np.int32)
        tv = np.clip((w0 * ua[1] + w1 * ub[1] + w2 * uc[1]), 0, SIZE - 1).astype(np.int32)
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        sub_o = out[y0:y1 + 1, x0:x1 + 1]
        sub_z[m] = zz[m]
        sub_o[m] = art[tv[m], tu[m]]
        zbuf[y0:y1 + 1, x0:x1 + 1] = sub_z
        out[y0:y1 + 1, x0:x1 + 1] = sub_o
    return out


def render(V, F, art, uv_pix, deg):
    """Orbit about the vertical axis by `deg`, re-project, rasterise."""
    t = np.radians(deg)
    ct, st = np.cos(t), np.sin(t)
    h, d = V[:, H_AXIS], V[:, D_AXIS]
    cx, cd = (h.min() + h.max()) / 2, (d.min() + d.max()) / 2
    hh, dd = h - cx, d - cd
    V2 = np.empty_like(V)
    V2[:, H_AXIS] = hh * ct - dd * st + cx
    V2[:, D_AXIS] = hh * st + dd * ct + cd
    V2[:, V_AXIS] = V[:, V_AXIS]
    pix = fit(V2[:, [H_AXIS, V_AXIS]])
    return rasterize(pix, V2[:, D_AXIS], uv_pix, F, art)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="docs/benchmark/char.glb")
    ap.add_argument("--art", default="docs/benchmark/matte_texture.png")
    ap.add_argument("--out", default="docs/benchmark/reangle_preview.png")
    ap.add_argument("--angles", default="-18,-12,-6,0,6,12,18")
    args = ap.parse_args()

    scene = trimesh.load(args.mesh, process=False)
    mesh = (trimesh.util.concatenate(tuple(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    art = np.asarray(Image.open(args.art).convert("RGBA"))
    print(f"mesh {len(V)} verts / {len(F)} faces   art {art.shape}")

    # front-planar UV, baked from the TRUE front view (Y,Z) -- this is the fix
    uv_pix = fit(V[:, [H_AXIS, V_AXIS]])

    angles = [float(a) for a in args.angles.split(",")]
    tiles = []
    for a in angles:
        img = render(V, F, art, uv_pix, a)
        rgb = img[:, :, :3].astype(np.float32)
        al = img[:, :, 3:4] / 255.0
        tiles.append(Image.fromarray((rgb * al + 255 * (1 - al)).astype(np.uint8)))
        print(f"  rendered {a:+.0f}deg")

    strip = Image.new("RGB", (SIZE * len(tiles), SIZE), (255, 255, 255))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * SIZE, 0))
    strip.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
