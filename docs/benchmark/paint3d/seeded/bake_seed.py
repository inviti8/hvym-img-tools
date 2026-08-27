"""Bake the artist's drawing into Paint3D's own xatlas atlas.

This is what makes the "seeded" experiment meaningful. Paint3D's UV models
expect a texture laid out in the mesh's xatlas unwrap; our front-planar
projection is a different parameterisation entirely, so the artist's art has to
be re-baked into their layout before it can be used as an init image.

Method: Paint3D already renders a UV-position map (3D xyz per texel). For each
texel we take that xyz, project it onto the front plane, and sample the drawing.
Texels whose surface faces away from the camera are left EMPTY -- that is the
whole point. It produces genuine disocclusion holes, which is both what the
inpainter needs and what our current front_planar_uv never creates (it
mirror-smears instead).

Outputs:
  seed_albedo.png  the artist's pixels in xatlas layout, holes left black
  seed_mask.png    white where the model may paint (the holes)
  UV_pos.png       Paint3D's position map, reused as the ControlNet condition
"""
import sys, os, numpy as np, cv2, torch
sys.path.insert(0, "/workspace/Paint3D")

from omegaconf import OmegaConf
from PIL import Image
from paint3d import utils
from paint3d.config.train_config_paint3d import TrainConfig
from paint3d.models.textured_mesh import TexturedMeshModel

MESH = "/workspace/char_yup.obj"
ART = "/workspace/alice_char2.png"
OUT = "/workspace/seed"
os.makedirs(OUT, exist_ok=True)

cfg = TrainConfig()
cfg.guide.shape_path = MESH
mesh_model = TexturedMeshModel(cfg=cfg, device=torch.device("cuda")).eval()

# --- 1. position map: xyz per texel, in Paint3D's atlas ---------------------
uv_pos = mesh_model.UV_pos_render()                      # (1,H,W,3) in [0,1]
utils.save_tensor_image(uv_pos.permute(0, 3, 1, 2), os.path.join(OUT, "UV_pos.png"))
pos = uv_pos[0].detach().cpu().numpy()
H, W, _ = pos.shape
print(f"atlas {W}x{H}")

# Texels with no geometry read as exactly zero in the position render.
occupied = pos.any(axis=2)
print(f"occupied texels: {occupied.mean()*100:.1f}%")

# --- 2. recover world xyz and a front-facing test ---------------------------
verts = mesh_model.mesh.vertices.detach().cpu().numpy()
lo, hi = verts.min(0), verts.max(0)
xyz = pos * (hi - lo) + lo                                # undo the [0,1] pack

# Surface normal from the position map's own gradients: cheaper and more robust
# than rasterising normals separately, and accurate enough for a facing test.
gx = np.gradient(xyz, axis=1)
gy = np.gradient(xyz, axis=0)
nrm = np.cross(gx, gy)
n = np.linalg.norm(nrm, axis=2, keepdims=True)
nrm = np.divide(nrm, np.where(n == 0, 1, n))

# char_yup.obj is Y-up with the character facing -Z (Paint3D's front camera).
FRONT = np.array([0.0, 0.0, 1.0])
facing = nrm @ FRONT
front_face = (np.abs(facing) > 0.15) & occupied     # sign is unreliable from
                                                    # gradients, magnitude is not

# --- 3. sample the drawing by front projection ------------------------------
art = np.asarray(Image.open(ART).convert("RGB"))
ah, aw = art.shape[:2]
# same aspect-preserving fit the real bake uses, so the art lands identically
span = max(hi[0] - lo[0], hi[1] - lo[1])
u = (xyz[..., 0] - (lo[0] + hi[0]) / 2) / span + 0.5
v = (xyz[..., 1] - (lo[1] + hi[1]) / 2) / span + 0.5
px = np.clip((u * (aw - 1)).astype(np.int32), 0, aw - 1)
py = np.clip(((1 - v) * (ah - 1)).astype(np.int32), 0, ah - 1)

seed = np.zeros((H, W, 3), np.uint8)
seed[front_face] = art[py[front_face], px[front_face]]

# Mask: white = the model MAY paint here (occupied geometry, no art)
mask = np.zeros((H, W), np.uint8)
mask[occupied & ~front_face] = 255
mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

cv2.imwrite(os.path.join(OUT, "seed_albedo.png"), cv2.cvtColor(seed, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "seed_mask.png"), mask)
np.save(os.path.join(OUT, "front_face.npy"), front_face)

print(f"seeded (artist's pixels): {front_face.mean()*100:.1f}% of atlas")
print(f"holes to fill           : {(mask>0).mean()*100:.1f}% of atlas")
print("BAKE_OK")
