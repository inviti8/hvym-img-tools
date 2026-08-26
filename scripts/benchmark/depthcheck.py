"""Dense front-ortho depth from the TripoSR mesh + automatic front-axis detection.

REANGLE_PIPELINE.md Stage 2 needs a depth map pixel-aligned to texture.png. Which
axis faces the camera is a TripoSR convention detail, so detect it by silhouette IoU
against the matte alpha instead of assuming one.
"""
import sys, json, numpy as np, trimesh
from PIL import Image

SIZE = 512
mesh = trimesh.load("/workspace/out2/mesh.obj", process=False)
alpha = np.asarray(Image.open("/workspace/out2/texture.png"))[:, :, 3] > 12
V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)


MARGIN = 0.08
def fit(v2, flipy=True):
    """Aspect-preserving map into SIZE^2 with prep_input.py's 8% margin, centred."""
    mn, mx = v2.min(0), v2.max(0)
    ext = mx - mn
    long_side = max(ext[0], ext[1]) * (1 + 2*MARGIN)
    sc = (SIZE - 1) / (long_side + 1e-9)
    c = (mn + mx) / 2.0
    p = (v2 - c) * sc + (SIZE - 1) / 2.0
    if flipy: p[:, 1] = (SIZE - 1) - p[:, 1]
    return p

def raster(v2, z, flipy=True):
    """Scanline z-buffer over triangles -> (depth, mask), nearest wins."""
    p = fit(v2, flipy)
    zb = np.full((SIZE, SIZE), -np.inf, np.float64)
    for tri in F:
        a, b, c = p[tri]; za, zb_, zc = z[tri]
        x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0); x1 = min(int(np.ceil(max(a[0], b[0], c[0]))), SIZE-1)
        y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0); y1 = min(int(np.ceil(max(a[1], b[1], c[1]))), SIZE-1)
        if x1 < x0 or y1 < y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        d = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(d) < 1e-12: continue
        w0 = ((b[1]-c[1])*(xs-c[0]) + (c[0]-b[0])*(ys-c[1])) / d
        w1 = ((c[1]-a[1])*(xs-c[0]) + (a[0]-c[0])*(ys-c[1])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any(): continue
        zz = w0*za + w1*zb_ + w2*zc
        sub = zb[y0:y1+1, x0:x1+1]
        np.putmask(sub, m & (zz > sub), zz)
        zb[y0:y1+1, x0:x1+1] = sub
    return zb, np.isfinite(zb)

def iou(m): return (m & alpha).sum() / ((m | alpha).sum() + 1e-9)

# try each of 6 view directions; keep the best silhouette match
def splat_mask(v2, flipy=True):
    """Cheap silhouette approximation for axis detection (dilated point splat)."""
    p = fit(v2, flipy).astype(np.int32)
    m = np.zeros((SIZE, SIZE), bool)
    for dy in (-2,-1,0,1,2):
        for dx in (-2,-1,0,1,2):
            m[np.clip(p[:,1]+dy,0,SIZE-1), np.clip(p[:,0]+dx,0,SIZE-1)] = True
    return m

cands = []
for ax in range(3):
    for sgn in (1, -1):
        others = [i for i in range(3) if i != ax]
        v2 = V[:, others].copy()
        if sgn < 0: v2[:, 0] = -v2[:, 0]
        cands.append((iou(splat_mask(v2)), ax, sgn))
cands.sort(key=lambda c: -c[0])
score, ax, sgn = cands[0]
others = [i for i in range(3) if i != ax]
v2 = V[:, others].copy()
if sgn < 0: v2[:, 0] = -v2[:, 0]
d, m = raster(v2, V[:, ax] * sgn)
print(json.dumps({"front_axis": int(ax), "sign": int(sgn), "silhouette_iou": round(float(score), 4),
                  "coverage": round(float(m.mean()), 4),
                  "all_iou": [round(float(c[0]), 3) for c in cands]}))
out = np.zeros((SIZE, SIZE), np.float32)
out[m] = (d[m] - d[m].min()) / (np.ptp(d[m]) + 1e-9)
Image.fromarray((out*255).astype(np.uint8)).save("/workspace/out2/front_depth_dense.png")
# relief stats INSIDE the silhouette -- the number that decides "real object relief"
inside = out[m]
print(json.dumps({"relief_p05_p95_spread": round(float(np.percentile(inside,95)-np.percentile(inside,5)),4),
                  "relief_std": round(float(inside.std()),4)}))
