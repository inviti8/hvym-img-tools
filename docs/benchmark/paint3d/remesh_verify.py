"""Remesh, properly measured: chart coherence AND what it costs.

Two corrections to the first pass:
  * real triangle rasterisation instead of bounding-box fill, which over-merged
    neighbouring islands and made every method look better than it was
  * silhouette IoU against the original mesh -- reangle's whole value rests on
    the proxy matching the artist's outline (BENCHMARK.md measured 0.776), so a
    remesh that buys clean charts by rounding off the silhouette is not free
"""
import numpy as np, trimesh, xatlas
from scipy import ndimage

ATLAS = 1024
SIL = 512


def raster_tris(uv, idx, size):
    """Scanline-fill each triangle. Slower than a bbox, but it is the difference
    between counting islands and counting clumps."""
    occ = np.zeros((size, size), bool)
    tri = uv[idx] * (size - 1)
    for t in tri:
        x0 = max(int(np.floor(t[:, 0].min())), 0)
        x1 = min(int(np.ceil(t[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(t[:, 1].min())), 0)
        y1 = min(int(np.ceil(t[:, 1].max())) + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        ax, ay = t[0]; bx, by = t[1]; cx, cy = t[2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-9:
            occ[y0:y1, x0:x1] |= True
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        occ[y0:y1, x0:x1] |= (w0 >= -1e-6) & (w1 >= -1e-6) & (w0 + w1 <= 1 + 1e-6)
    return occ


def silhouette(m, size=SIL):
    """Orthographic front silhouette, same convention as the reangle bake."""
    v = m.vertices - m.bounds.mean(axis=0)
    v = v / np.abs(v).max()
    occ = np.zeros((size, size), bool)
    tri = v[m.faces][:, :, [1, 2]]            # H,V axes per preview_reangle
    tri = (tri * 0.45 + 0.5) * (size - 1)
    return raster_tris(tri.reshape(-1, 2) / (size - 1),
                       np.arange(len(tri) * 3).reshape(-1, 3), size)


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def analyse(m, label, ref_sil):
    try:
        vm, idx, uv = xatlas.parametrize(m.vertices, m.faces)
    except Exception as e:
        print(f"  {label:32s} xatlas FAILED {type(e).__name__}"); return None
    occ = raster_tris(uv, idx, ATLAS)
    lab, n = ndimage.label(occ)
    sizes = np.bincount(lab.ravel())[1:]
    if not len(sizes):
        return None
    s = silhouette(m)
    keep = iou(s, ref_sil)
    print(f"  {label:32s} faces={len(m.faces):6d} islands={n:5d} "
          f"median={np.median(sizes):8.0f} tiny={(sizes<100).mean()*100:4.0f}% "
          f"silIoU={keep:.3f}")
    return dict(label=label, faces=len(m.faces), islands=int(n),
                median=float(np.median(sizes)),
                tiny=float((sizes < 100).mean() * 100), sil_iou=float(keep))


def prep(m):
    m = m.copy(); m.merge_vertices()
    try: m.update_faces(m.nondegenerate_faces())
    except AttributeError: m.remove_degenerate_faces()
    m.remove_unreferenced_vertices(); return m


def dec(m, n):
    try: return m.simplify_quadric_decimation(face_count=n)
    except TypeError: return m.simplify_quadric_decimation(n)


src = trimesh.load("docs/benchmark/tool_char.glb", process=False)
src = src if isinstance(src, trimesh.Trimesh) else list(src.geometry.values())[0]
base = prep(trimesh.Trimesh(vertices=src.vertices, faces=src.faces, process=False))
ref = silhouette(base)
print(f"source {len(base.faces)} faces | watertight={base.is_watertight}\n")

rows = [analyse(base, "0. baseline", ref)]
for f in (0.25, 0.1, 0.05):
    rows.append(analyse(prep(dec(base, int(len(base.faces) * f))), f"1. decimate {int(f*100)}%", ref))
for pitch in (0.015, 0.02, 0.03):
    try:
        r = prep(base.voxelized(pitch=pitch).fill().marching_cubes)
        trimesh.smoothing.filter_taubin(r, iterations=8)
        rows.append(analyse(prep(r), f"2. voxel {pitch}", ref))
    except Exception as e:
        print(f"  voxel {pitch}: {type(e).__name__}: {str(e)[:60]}")

rows = [r for r in rows if r]
b = rows[0]
print(f"\n{'method':32s} {'islands':>8s} {'vs base':>8s} {'silhouette kept':>16s}")
print("-" * 68)
for r in sorted(rows, key=lambda r: r["islands"]):
    print(f"  {r['label']:30s} {r['islands']:8d} {b['islands']/max(1,r['islands']):7.1f}x "
          f"{r['sil_iou']:15.3f}")
