"""Sweep the style-injection dials on a SEEDED atlas.

The question: how little of the model can we use and still fill the holes the
front projection cannot reach? Every run starts from the artist's own pixels
(seed_albedo.png) rather than from noise, so `denoising_strength` reads directly
as "how much of the artist's drawing am I willing to lose".

Calls the ControlNet wrappers directly instead of re-running the whole pipeline,
so the mesh render and model load happen once and each sweep point costs seconds.

Reports, per run:
  drift  mean |Δ| on texels the artist actually painted -- lower is better,
         this is style override and should ideally be ~0
  fill   fraction of the holes that received paint -- higher is better
"""
import sys, os, json, itertools, time
import numpy as np, cv2, torch
sys.path.insert(0, "/workspace/Paint3D")
from omegaconf import OmegaConf
from PIL import Image
from controlnet.diffusers_cnet_inpaint import inpaintControlNet

SEED = "/workspace/seed"
OUT = "/workspace/sweep"
os.makedirs(OUT, exist_ok=True)

seed_rgb = cv2.cvtColor(cv2.imread(f"{SEED}/seed_albedo.png"), cv2.COLOR_BGR2RGB).astype(np.float32)
mask = cv2.imread(f"{SEED}/seed_mask.png", 0)
front = np.load(f"{SEED}/front_face.npy")
holes = mask > 0
print(f"seeded {front.mean()*100:.1f}% | holes {holes.mean()*100:.1f}%")

base = OmegaConf.load("/workspace/Paint3D/controlnet/config/UV_based_inpaint_template.yaml")

# Grid. denoise is the dominant dial; ip_adapter pulls toward the drawing;
# guidance pulls toward the prompt (i.e. away from the artist).
DENOISE   = [0.30, 0.45, 0.60, 0.75, 1.00]
GUIDANCE  = [3.0, 7.0]
USE_IPA   = [True]
PROMPT = "flat 2D cartoon character, clean black linework, cel shaded, greyscale, white background"

cnet = None
results = []
for dn, gs, ipa in itertools.product(DENOISE, GUIDANCE, USE_IPA):
    cfg = base.copy()
    p = cfg.inpaint
    p.image_path = f"{SEED}/seed_albedo.png"
    p.mask_path = f"{SEED}/seed_mask.png"
    p.ip_adapter_image_path = "/workspace/alice_char2.png" if ipa else None
    p.prompt = PROMPT
    p.denoising_strength = dn
    p.guidance_scale = gs
    p.num_inference_steps = 20
    p.seed = 1234                      # fixed, so differences are the dials only
    p.controlnet_units[0].condition_image_path = f"{SEED}/UV_pos.png"
    p.controlnet_units[1].condition_image_path = f"{SEED}/seed_albedo.png"

    if cnet is None:
        cnet = inpaintControlNet(p)     # loads once; params are per-call below
    t0 = time.time()
    try:
        imgs = cnet.infernece(config=p)
    except Exception as e:
        print(f"  dn={dn} gs={gs} ipa={ipa} FAILED: {type(e).__name__}: {str(e)[:90]}")
        continue
    dt = time.time() - t0

    out = np.asarray(imgs[0].convert("RGB").resize((seed_rgb.shape[1], seed_rgb.shape[0]))).astype(np.float32)
    drift = float(np.abs(out[front] - seed_rgb[front]).mean()) if front.any() else float("nan")
    painted = out[holes].std(axis=0).mean() if holes.any() else 0.0
    fill = float((np.abs(out[holes]).sum(axis=1) > 12).mean()) if holes.any() else 0.0

    tag = f"dn{dn:.2f}_gs{gs:.0f}_ipa{int(ipa)}"
    imgs[0].save(f"{OUT}/{tag}.png")
    results.append(dict(denoise=dn, guidance=gs, ip_adapter=ipa,
                        drift=round(drift, 2), fill=round(fill, 3),
                        variety=round(float(painted), 2), secs=round(dt, 1)))
    print(f"  {tag}: drift={drift:6.2f}  fill={fill:5.3f}  {dt:4.1f}s")

json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("\n=== ranked by lowest style drift ===")
for r in sorted(results, key=lambda r: r["drift"]):
    print(f"  denoise={r['denoise']:.2f} guidance={r['guidance']:.0f} "
          f"-> drift={r['drift']:6.2f}  fill={r['fill']:.3f}")
print("SWEEP_OK")
