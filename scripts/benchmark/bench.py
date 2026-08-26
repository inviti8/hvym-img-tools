"""TripoSR benchmark for hvym-img-tools AGENTS.md §0 (cost-model gate).

Measures, cold and warm: matte (isnet) / reconstruct (the key number) /
mesh extract / UV-bake + glb export / total. Emits JSON + quality renders.
"""
import os, sys, time, json, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/TripoSR")

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="/workspace/in/alice_char2.png")
ap.add_argument("--out", default="/workspace/out")
ap.add_argument("--mc-resolution", type=int, default=256)
ap.add_argument("--chunk-size", type=int, default=8192)
ap.add_argument("--warm-runs", type=int, default=3)
ap.add_argument("--render-views", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

DEV = "cuda:0"
R = {"gpu": torch.cuda.get_device_name(0), "mc_resolution": args.mc_resolution,
     "chunk_size": args.chunk_size, "torch": torch.__version__}

def sync(): torch.cuda.synchronize()
class t:
    def __init__(s, k, into): s.k, s.into = k, into
    def __enter__(s): sync(); s.t0 = time.perf_counter(); return s
    def __exit__(s, *a):
        sync(); s.into[s.k] = round(time.perf_counter() - s.t0, 3)
        print(f"  {s.k:<22} {s.into[s.k]:>8.3f}s", flush=True)

# ---------- COLD: model loads ----------
print("=== COLD START ===", flush=True)
cold = {}
with t("isnet_load", cold):
    import onnxruntime as ort
    isnet = ort.InferenceSession("/workspace/models/isnet_dis.onnx",
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
with t("triposr_load", cold):
    from tsr.system import TSR
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml",
                                weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(DEV)
R["cold"] = cold

# ---------- stages ----------
SIZE, MARGIN, LO, HI = 512, 0.08, 0.30, 0.65

def matte(src):
    """isnet matte -> 512^2 RGBA, identical math to scripts/reangle/prep_input.py."""
    img = Image.open(src).convert("RGB"); W0, H0 = img.size
    inp = isnet.get_inputs()[0]
    IH = inp.shape[2] if isinstance(inp.shape[2], int) else 1024
    IW = inp.shape[3] if isinstance(inp.shape[3], int) else 1024
    a = np.asarray(img.resize((IW, IH), Image.BILINEAR), dtype=np.float32)
    a = np.transpose((a / 255.0 - 0.5), (2, 0, 1))[None].astype(np.float32)
    m = isnet.run(None, {inp.name: a})[0][0][0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W0, H0), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    alpha = (np.clip((m - LO) / (HI - LO), 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([np.asarray(img, dtype=np.uint8), alpha])
    ys, xs = np.where(alpha > 12)
    crop = Image.fromarray(rgba).crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
    cw, ch = crop.size
    side = int(round(max(cw, ch) * (1 + 2*MARGIN)))
    canvas = Image.new("RGBA", (side, side), (0,0,0,0))
    canvas.paste(crop, ((side-cw)//2, (side-ch)//2), crop)
    return canvas.resize((SIZE, SIZE), Image.LANCZOS)

def to_model_input(rgba):
    """TripoSR expects RGB composited on 0.5 gray (run.py convention)."""
    a = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = a[:, :, :3] * a[:, :, 3:4] + 0.5 * (1 - a[:, :, 3:4])
    return Image.fromarray((rgb * 255).astype(np.uint8))

def frontplanar_glb(mesh, art_rgba, path):
    """The shippable bake (REANGLE_PIPELINE.md §7.5 'simplest'):
    uv = normalized vertex.xy, texture = the artist's ORIGINAL art."""
    import trimesh
    v = mesh.vertices
    mn, mx = v[:, :2].min(0), v[:, :2].max(0)
    uv = (v[:, :2] - mn) / (mx - mn + 1e-9)
    uv[:, 1] = 1.0 - uv[:, 1]
    m = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=art_rgba))
    m.export(path)
    return os.path.getsize(path)

# ---------- WARM RUNS ----------
runs = []
for i in range(args.warm_runs):
    print(f"=== WARM RUN {i+1}/{args.warm_runs} ===", flush=True)
    s = {}
    with t("matte_isnet", s):
        rgba = matte(args.image)
    with t("reconstruct", s):                      # THE KEY NUMBER
        with torch.no_grad():
            scene_codes = model([to_model_input(rgba)], device=DEV)
    with t("mesh_extract", s):
        meshes = model.extract_mesh(scene_codes, True, resolution=args.mc_resolution)
    with t("uvbake_glb_export", s):
        nbytes = frontplanar_glb(meshes[0], rgba, os.path.join(args.out, "char.glb"))
    s["total"] = round(sum(s.values()), 3)
    s["glb_bytes"] = nbytes
    s["mesh_verts"] = int(len(meshes[0].vertices)); s["mesh_faces"] = int(len(meshes[0].faces))
    s["vram_peak_gb"] = round(torch.cuda.max_memory_allocated()/1e9, 2)
    print(f"  {'TOTAL':<22} {s['total']:>8.3f}s   ({s['mesh_verts']} verts, {s['mesh_faces']} faces)", flush=True)
    runs.append(s)
R["warm_runs"] = runs

# ---------- QUALITY ARTIFACTS ----------
print("=== QUALITY ARTIFACTS ===", flush=True)
rgba.save(os.path.join(args.out, "texture.png"))
meshes[0].export(os.path.join(args.out, "mesh.obj"))
try:
    with torch.no_grad():
        views = model.render(scene_codes, n_views=args.render_views, return_type="pil")
    W = views[0][0].size[0]
    strip = Image.new("RGB", (W*len(views[0]), W), (255,255,255))
    for j, im in enumerate(views[0]): strip.paste(im.convert("RGB"), (j*W, 0))
    strip.save(os.path.join(args.out, "orbit_strip.png"))
    print("  wrote orbit_strip.png", flush=True)
except Exception as e:
    print("  render failed:", e, flush=True)

# front-ortho depth from the mesh == the relief test that monocular depth failed (§4.4)
try:
    v = meshes[0].vertices
    g = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    mn, mx = v[:, :2].min(0), v[:, :2].max(0)
    px = ((v[:, 0]-mn[0])/(mx[0]-mn[0]+1e-9)*(SIZE-1)).astype(np.int32)
    py = ((1-(v[:, 1]-mn[1])/(mx[1]-mn[1]+1e-9))*(SIZE-1)).astype(np.int32)
    order = np.argsort(v[:, 2])          # far -> near, nearest wins
    for xi, yi, zi in zip(px[order], py[order], v[order][:, 2]): g[yi, xi] = zi
    valid = ~np.isnan(g)
    d = np.zeros_like(g); d[valid] = (g[valid]-g[valid].min())/(np.ptp(g[valid])+1e-9)
    Image.fromarray((d*255).astype(np.uint8)).save(os.path.join(args.out, "front_depth.png"))
    R["depth_relief_range"] = float(np.ptp(g[valid]))
    print(f"  wrote front_depth.png (z-range {R['depth_relief_range']:.3f})", flush=True)
except Exception as e:
    print("  depth failed:", e, flush=True)

with open(os.path.join(args.out, "bench.json"), "w") as f: json.dump(R, f, indent=2)
print("\n=== JSON ===\n" + json.dumps(R, indent=2), flush=True)
