"""Do PROPS unwrap better than a character, and does remeshing cost less on them?

The hallucinate design rests on the claim that props are the easy case. Every
measurement so far used alice_char2 -- thin limbs, a ponytail, a gap between the
legs -- which is close to the worst case for both xatlas and voxel remeshing.

The chair is deliberately not a soft test: thin legs with large open gaps between
them are exactly what voxel fill bridges. The rock is the convex easy case. If
the claim holds anywhere it should hold on the rock; the chair is where it earns
or loses credibility.
"""
import sys, numpy as np, trimesh, xatlas
from scipy import ndimage
sys.path.insert(0, "C:/Users/surfa/AppData/Local/Temp/claude/D--repos-hvym-img-tools/bc2d19af-4249-447c-8f5a-49dc9877e929/scratchpad")
from remesh_verify import raster_tris, silhouette, prep, iou, ATLAS

P = "C:/Users/surfa/AppData/Local/Temp/claude/D--repos-hvym-img-tools/bc2d19af-4249-447c-8f5a-49dc9877e929/scratchpad/props"
SUBJECTS = [
    ("character", "docs/benchmark/tool_char.glb"),
    ("chair",     f"{P}/chair.glb"),
    ("rock",      f"{P}/rock.glb"),
]


def load(path):
    m = trimesh.load(path, process=False)
    m = m if isinstance(m, trimesh.Trimesh) else list(m.geometry.values())[0]
    return prep(trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False))


def dec(m, n):
    try: return m.simplify_quadric_decimation(face_count=n)
    except TypeError: return m.simplify_quadric_decimation(n)


def measure(m, ref_sil):
    vm, idx, uv = xatlas.parametrize(m.vertices, m.faces)
    occ = raster_tris(uv, idx, ATLAS)
    lab, n = ndimage.label(occ)
    sizes = np.bincount(lab.ravel())[1:]
    s = silhouette(m)
    return dict(faces=len(m.faces), islands=int(n),
                median=float(np.median(sizes)) if len(sizes) else 0,
                tiny=float((sizes < 100).mean() * 100) if len(sizes) else 0,
                iou=iou(s, ref_sil),
                lost=(ref_sil & ~s).sum() / ref_sil.sum() * 100,
                added=(~ref_sil & s).sum() / ref_sil.sum() * 100)


for name, path in SUBJECTS:
    base = load(path)
    ref = silhouette(base)
    print(f"\n=== {name}  ({len(base.faces)} faces, watertight={base.is_watertight}) ===")
    b = measure(base, ref)
    print(f"  {'baseline':26s} islands={b['islands']:5d} median={b['median']:8.0f} "
          f"tiny={b['tiny']:4.0f}%  IoU=1.000")

    d = prep(dec(base, max(200, int(len(base.faces) * 0.10))))
    r = measure(d, ref)
    print(f"  {'decimate 10%':26s} islands={r['islands']:5d} median={r['median']:8.0f} "
          f"tiny={r['tiny']:4.0f}%  IoU={r['iou']:.3f}  lost={r['lost']:4.1f}% added={r['added']:5.1f}%")

    for pitch in (0.02, 0.03):
        try:
            v = prep(base.voxelized(pitch=pitch).fill().marching_cubes)
            trimesh.smoothing.filter_taubin(v, iterations=8)
            r = measure(prep(v), ref)
            print(f"  {'voxel ' + str(pitch):26s} islands={r['islands']:5d} median={r['median']:8.0f} "
                  f"tiny={r['tiny']:4.0f}%  IoU={r['iou']:.3f}  lost={r['lost']:4.1f}% added={r['added']:5.1f}%")
        except Exception as e:
            print(f"  voxel {pitch}: {type(e).__name__}: {str(e)[:50]}")
