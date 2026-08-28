"""Bake a Gaussian-splat appearance onto a mesh, without a restricted rasteriser.

TRELLIS ships `to_glb()`, which does this properly -- and unusably for us. It
does `import nvdiffrast.torch` at module scope (NVIDIA Source Code License,
research and evaluation only) and renders 100 views through Inria's
`diff-gaussian-rasterization`, whose licence forbids commercial use without
their written consent. TRELLIS being MIT does not launder either dependency,
and neither is in our image.

Decoding the Gaussian needs none of that. `formats=["mesh", "gaussian"]` runs on
spconv, which we already ship; only *rendering* the result needs a rasteriser.
So this module never renders. It averages the cloud into a sparse voxel field
and samples that at surface positions, which is enough to carry colour from the
Gaussians onto a UV atlas.

What that approximation costs, stated plainly:

  * **No view-dependent shading.** Only the degree-0 SH term is read, so a
    surface gets one colour rather than one per viewing angle.
  * **No splat footprint.** A Gaussian contributes to voxels near its centre
    rather than to the ellipsoid it actually covers, so large sparse splats
    blur more than they would under a real rasteriser.

For flat 2D artwork -- no specular, no view-dependent anything -- both are close
to free. For a photographic reference they would not be, and this would be the
wrong tool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .meshops import densify

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GaussianCloud:
    """The parts of a Gaussian splat this module reads. Plain numpy, no torch.

    Deliberately not the backbone's own Gaussian object: keeping this to arrays
    means `core` never imports the model stack, the cloud can be serialised for
    offline iteration, and a second backbone could produce one without
    inheriting TRELLIS's class.
    """

    xyz: np.ndarray          # (N, 3) world positions
    features_dc: np.ndarray  # (N, 3) degree-0 SH coefficients
    opacity: np.ndarray      # (N,)   post-sigmoid, in [0, 1]

    def __len__(self) -> int:
        return len(self.xyz)

    def extent(self) -> np.ndarray:
        return self.xyz.max(0) - self.xyz.min(0)

    def field(self, *, apply_sh: bool = True, **kwargs) -> "ColorField":
        """Opacity-weighted colour field over this cloud."""
        return ColorField(self.xyz, gaussian_rgb(self.features_dc, apply_sh=apply_sh),
                          self.opacity, **kwargs)

    def to_npz(self, vertices: np.ndarray | None = None,
               faces: np.ndarray | None = None) -> bytes:
        """Serialise for offline iteration, so re-baking never needs the GPU.

        The first probe cost a 3,150 s cold start to answer one question about
        one resolution. Handing the cloud back means every later question about
        the bake is answered locally and free.

        The mesh travels with it. Without vertices and faces the cloud cannot be
        unwrapped or baked offline, which would defeat the entire point.

        float16 for colour and opacity, float32 for position: colour is a
        display quantity that never needed 32 bits, while position feeds the
        voxel index and must not be quantised into the wrong cell.
        """
        import io as _io

        arrays = {
            "xyz": np.asarray(self.xyz, np.float32),
            "features_dc": np.asarray(self.features_dc, np.float16),
            "opacity": np.asarray(self.opacity, np.float16),
        }
        if vertices is not None:
            arrays["vertices"] = np.asarray(vertices, np.float32)
        if faces is not None:
            arrays["faces"] = np.asarray(faces, np.int32)

        buf = _io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    @classmethod
    def from_npz(cls, data: bytes):
        """Returns `(cloud, vertices, faces)`; the mesh pair may be `None`."""
        import io as _io

        z = np.load(_io.BytesIO(data))
        cloud = cls(xyz=z["xyz"].astype(np.float64),
                    features_dc=z["features_dc"].astype(np.float32),
                    opacity=z["opacity"].astype(np.float32))
        verts = z["vertices"].astype(np.float64) if "vertices" in z else None
        faces = z["faces"] if "faces" in z else None
        return cloud, verts, faces

#: 3D Gaussian Splatting stores base colour as the degree-0 spherical-harmonic
#: coefficient; every renderer in that lineage converts with this constant.
#: TRELLIS's `save_ply` writes `f_dc_*` raw and leaves the conversion to the
#: viewer, so it has to happen on this side.
SH_C0 = 0.28209479177387814


def dc_stats(features_dc: np.ndarray) -> dict[str, float]:
    """How well each reading of `f_dc` lands in [0, 1]. Diagnostic, not a guess.

    The SH convention is near-universal, but TRELLIS does not apply it in-tree,
    so rather than assume, a caller can print this and see which interpretation
    produces plausible colour. The two `in_range` values are fractions of all
    channels.
    """
    dc = np.asarray(features_dc, np.float32).reshape(len(features_dc), -1)[:, :3]
    as_sh = SH_C0 * dc + 0.5
    return {
        "sh_in_range": float(((as_sh >= 0.0) & (as_sh <= 1.0)).mean()),
        "raw_in_range": float(((dc >= 0.0) & (dc <= 1.0)).mean()),
        "dc_min": float(dc.min()),
        "dc_max": float(dc.max()),
    }


def gaussian_rgb(features_dc: np.ndarray, *, apply_sh: bool = True) -> np.ndarray:
    """SH DC coefficients -> RGB in [0, 1], one row per Gaussian."""
    dc = np.asarray(features_dc, np.float32).reshape(len(features_dc), -1)[:, :3]
    return np.clip(SH_C0 * dc + 0.5 if apply_sh else dc, 0.0, 1.0)


def texel_matched_resolution(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture_size: int,
    *,
    percentile: float = 99.9,
    floor: int = 64,
    cap: int = 4096,
) -> int:
    """Voxel resolution whose edge matches the atlas's own texel footprint.

    Picking this by hand is how the first probe was wasted: `res=256` over a
    figure 1.0 long gives a 0.12-tall head 31 voxel layers, while the atlas gave
    that same head 237 texel rows. The texture then interpolated 237 rows out of
    31 distinct values and the face came back as mush -- a limit of the sampler
    that looked exactly like a limit of the model
    (docs/benchmark/gaussian_bake/README.md).

    So derive it. Each triangle has a texel density -- its UV area in texels over
    its world area -- and the field must resolve the *dense* triangles, not the
    average one, because that is where detail lives. Hence a high percentile
    rather than a mean: a body's large flat charts would otherwise drag the
    resolution down to where faces stop working.

    The percentile is deliberately near the top. Over-resolving is graceful --
    queries that find an empty voxel fall through to a coarser level, which is
    the same blur under-resolving would have given anyway -- while
    under-resolving is unconditional. Measured on the probe mesh, the densest
    chart *is* the head, so nothing below ~p99 serves a face.

    Returns an edge count across the mesh's longest axis, so voxels are cubic.
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces)
    t = v[f]
    world = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2.0

    q = np.asarray(uv, np.float64)[f]
    # 2D cross product; np.cross on 2-vectors is deprecated in numpy 2.
    duv1, duv2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    uv_area = np.abs(duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]) / 2.0

    ok = (world > 0) & (uv_area > 0)
    if not ok.any():
        return floor
    # texels per unit of world area, then the edge length one texel covers
    density = uv_area[ok] * (texture_size ** 2) / world[ok]
    texel_edge = 1.0 / np.sqrt(np.percentile(density, percentile))

    longest = float((v.max(0) - v.min(0)).max())
    res = int(np.ceil(longest / max(texel_edge, 1e-9)))
    return int(np.clip(res, floor, cap))


class ColorField:
    """Opacity-weighted colour of a point cloud, averaged into sparse voxels.

    Voxels are **cubic in world space**: every axis is divided by the same edge,
    derived from the longest one. Normalising each axis independently would make
    a tall thin subject's voxels tall and thin too, so the resolution that suits
    its height would over-blur its width.

    Multi-resolution on purpose. A mesh vertex can sit slightly outside the
    cloud -- TRELLIS extracts the surface from the same latent, but not from the
    Gaussians themselves -- and a single-resolution lookup leaves those texels
    black. Coarser levels answer the misses, so coverage degrades into blur
    rather than into holes.

    Sparse rather than dense: at res=2048 a dense grid would be 8.6 *billion*
    cells, while the occupied set can never exceed the number of input points.
    Cost scales with the cloud, not with res^3, which is why raising the
    resolution is close to free.

    The coarsest level is always a single voxel spanning the whole cloud, which
    is occupied by construction. Without it a query clamped to a corner of the
    bounding box can miss at *every* level and come back black, reading as a
    hole in the texture rather than as the miss it is. Falling back to the
    cloud's mean colour is the right failure: blurry, not wrong.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        weight: np.ndarray | None = None,
        *,
        resolution: int = 256,
        growth: int = 4,
        pad: float = 1e-3,
    ) -> None:
        xyz = np.asarray(xyz, np.float64)
        rgb = np.asarray(rgb, np.float32)
        w = np.ones(len(xyz), np.float32) if weight is None else np.asarray(weight, np.float32).ravel()

        self._lo = xyz.min(0) - pad
        span = (xyz.max(0) + pad) - self._lo
        self._span = np.where(span > 0, span, 1.0)
        self._longest = float(self._span.max())
        self._levels: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

        res = int(max(1, resolution))
        while True:
            dims = np.maximum(np.ceil(self._span / (self._longest / res)).astype(np.int64), 1)
            keys = self._keys(xyz, dims)
            uniq, inv = np.unique(keys, return_inverse=True)
            acc = np.zeros((len(uniq), 3), np.float64)
            den = np.zeros(len(uniq), np.float64)
            np.add.at(acc, inv, rgb * w[:, None])
            np.add.at(den, inv, w)
            mean = (acc / np.maximum(den, 1e-8)[:, None]).astype(np.float32)
            self._levels.append((res, dims, uniq, mean))
            if res == 1:
                break
            res = max(1, res // growth)

        log.info(
            "colour field: %d points, resolutions %s -> occupied %s",
            len(xyz), [r for r, _, _, _ in self._levels],
            [len(k) for _, _, k, _ in self._levels],
        )

    def _keys(self, pts: np.ndarray, dims: np.ndarray) -> np.ndarray:
        unit = (pts - self._lo) / self._span
        ijk = np.clip((unit * dims).astype(np.int64), 0, dims - 1)
        return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Colour per query point, plus a mask of which points any level answered.

        `hit` is False only where no level could answer, which the final res=1
        level makes impossible -- kept as a return value so a caller can assert
        on it rather than trust the invariant.
        """
        pts = np.asarray(points, np.float64)
        out = np.zeros((len(pts), 3), np.float32)
        hit = np.zeros(len(pts), bool)

        for _res, dims, keys, colours in self._levels:
            todo = np.flatnonzero(~hit)
            if todo.size == 0:
                break
            q = self._keys(pts[todo], dims)
            pos = np.searchsorted(keys, q)
            ok = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == q)
            if not ok.any():
                continue
            idx = todo[ok]
            out[idx] = colours[pos[ok]]
            hit[idx] = True
        return out, hit


def fill_holes(rgb: np.ndarray, covered: np.ndarray, rounds: int = 32) -> np.ndarray:
    """Grow colour into uncovered texels so atlas seams do not sample black.

    A UV atlas leaves gutters between charts. Bilinear filtering at render time
    reaches into them, so an unfilled gutter shows as a dark rim along every
    chart edge -- the classic seam. Dilating a few texels past each chart costs
    nothing and removes it.
    """
    rgb = rgb.copy()
    covered = covered.copy()
    for _ in range(rounds):
        if covered.all():
            break
        acc = np.zeros_like(rgb)
        den = np.zeros(covered.shape, np.float32)
        masked = np.where(covered[..., None], rgb, 0.0)
        for axis in (0, 1):
            for shift in (1, -1):
                acc += np.roll(masked, shift, axis=axis)
                den += np.roll(covered.astype(np.float32), shift, axis=axis)
        grow = (~covered) & (den > 0)
        if not grow.any():
            break
        rgb[grow] = acc[grow] / den[grow][:, None]
        covered |= grow
    return rgb


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    field: ColorField,
    *,
    size: int = 1024,
    samples: int = 2_000_000,
) -> tuple[np.ndarray, float]:
    """Sample `field` across the surface and resolve it into a UV texture.

    Position and UV are densified in a **single** `densify` call on a stacked
    array, so every surface sample keeps its own UV. Densifying them separately
    would rely on the sampler's RNG being called identically twice -- true
    today, and not a property worth depending on.

    Returns `(rgb_uint8, coverage)`, coverage being the fraction of texels a
    surface sample landed on before hole filling. A low number means `samples`
    is too small for `size`, not that the bake failed.
    """
    verts = np.asarray(vertices, np.float64)
    uvs = np.asarray(uv, np.float64)
    dense = densify(np.hstack([verts, uvs]), faces, samples)
    pos, duv = dense[:, :3], dense[:, 3:5]

    colour, hit = field.sample(pos)
    if not hit.all():
        log.warning("%d/%d surface samples fell outside the cloud", int((~hit).sum()), len(hit))

    # glTF UV origin is bottom-left; image rows run top-down.
    px = np.clip((duv[:, 0] * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip(((1.0 - duv[:, 1]) * (size - 1)).astype(np.int32), 0, size - 1)

    acc = np.zeros((size, size, 3), np.float64)
    den = np.zeros((size, size), np.float64)
    np.add.at(acc, (py, px), colour)
    np.add.at(den, (py, px), 1.0)

    covered = den > 0
    coverage = float(covered.mean())
    rgb = np.zeros((size, size, 3), np.float32)
    rgb[covered] = (acc[covered] / den[covered][:, None]).astype(np.float32)
    rgb = fill_holes(rgb, covered)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), coverage
