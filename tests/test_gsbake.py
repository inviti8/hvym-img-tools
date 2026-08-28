"""`core.gsbake` tests — no GPU, no TRELLIS, no rasteriser.

The point of the module is that carrying colour off a Gaussian cloud needs none
of those, so its tests must not need them either. Clouds here are synthetic and
chosen so the correct answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from hvym_img_tools.core.gsbake import (
    SH_C0,
    ColorField,
    GaussianCloud,
    bake_texture,
    dc_stats,
    fill_holes,
    gaussian_rgb,
)


def _split_cloud(n: int = 40_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Points in the unit cube, red where x < 0.5 and blue where it is not."""
    rng = np.random.default_rng(seed)
    xyz = rng.random((n, 3))
    rgb = np.where(xyz[:, 0:1] < 0.5, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).astype(np.float32)
    return xyz, rgb


# --- colour conversion -----------------------------------------------------

def test_sh_dc_converts_to_mid_grey_at_zero():
    """The convention's whole point: a zero DC coefficient is 50% grey."""
    assert gaussian_rgb(np.zeros((1, 3))).tolist() == [[0.5, 0.5, 0.5]]


def test_sh_conversion_matches_the_published_constant():
    dc = np.array([[1.0, -1.0, 0.0]])
    expected = np.clip(SH_C0 * dc + 0.5, 0, 1)
    assert np.allclose(gaussian_rgb(dc), expected)


def test_raw_reading_is_available_for_when_the_convention_does_not_hold():
    dc = np.array([[0.25, 0.5, 0.75]])
    assert np.allclose(gaussian_rgb(dc, apply_sh=False), dc)


def test_dc_stats_distinguishes_the_two_readings():
    """A cloud that is plausible as SH and implausible as raw, and vice versa."""
    as_sh = dc_stats(np.array([[-1.5, 0.0, 1.5]]))
    assert as_sh["sh_in_range"] == 1.0
    assert as_sh["raw_in_range"] < 1.0

    as_raw = dc_stats(np.array([[0.2, 0.4, 0.6]]))
    assert as_raw["raw_in_range"] == 1.0


def test_extra_sh_bands_are_ignored():
    """Degree>0 coefficients are view-dependent; only the DC term is read."""
    wide = np.hstack([np.zeros((4, 3)), np.full((4, 6), 99.0)])
    assert np.allclose(gaussian_rgb(wide), 0.5)


# --- the colour field ------------------------------------------------------

def test_field_returns_the_colour_that_is_actually_there():
    field = ColorField(*_split_cloud())
    rgb, hit = field.sample(np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]]))
    assert hit.all()
    assert np.allclose(rgb[0], [1, 0, 0], atol=1e-3)
    assert np.allclose(rgb[1], [0, 0, 1], atol=1e-3)


def test_opacity_weights_the_average():
    """A transparent Gaussian must not drag the colour of a solid one."""
    xyz = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    solid_red = ColorField(xyz, rgb, np.array([1.0, 0.001], np.float32))
    out, _ = solid_red.sample(np.array([[0.5, 0.5, 0.5]]))
    assert out[0][0] > 0.99, "the opaque red should dominate"


def test_coarse_levels_answer_what_the_fine_level_cannot():
    """A query away from the cloud still resolves, because that is the design.

    A mesh vertex can sit slightly off the Gaussians -- TRELLIS extracts the
    surface from the same latent, not from the cloud -- and holes there would
    become black texels.
    """
    xyz, rgb = _split_cloud(n=2_000)
    rng = np.random.default_rng(1)
    q = rng.random((500, 3))

    field = ColorField(xyz, rgb, resolution=256)
    # A sparse cloud on a fine grid misses most queries at the finest level...
    fine_dims = field._levels[0][1]
    fine_keys = field._levels[0][2]
    hits = np.isin(field._keys(q, fine_dims), fine_keys)
    assert hits.mean() < 0.5
    # ...and the field as a whole still answers every one of them.
    assert field.sample(q)[1].all(), "the res=1 level makes a miss impossible"


def test_levels_descend_to_a_single_voxel():
    """The coarsest level must span the cloud, or a corner query can come back black."""
    xyz, rgb = _split_cloud(n=1_000)
    field = ColorField(xyz, rgb, resolution=256)
    resolutions = [r for r, _, _, _ in field._levels]
    assert resolutions[0] == 256
    assert resolutions[-1] == 1, resolutions
    assert resolutions == sorted(resolutions, reverse=True)


def test_voxels_are_cubic_not_axis_normalised():
    """A tall thin subject must not get tall thin voxels.

    Normalising each axis independently would make the resolution that suits a
    figure's height over-blur its width -- exactly the regime reangle works in,
    where subjects are ~1.0 tall and ~0.3 wide.
    """
    rng = np.random.default_rng(0)
    xyz = rng.random((20_000, 3)) * np.array([0.3, 0.2, 1.0])
    rgb = np.zeros((len(xyz), 3), np.float32)
    field = ColorField(xyz, rgb, resolution=100)
    dims = field._levels[0][1]
    edges = field._span / dims
    assert np.allclose(edges, edges[0], rtol=0.05), f"non-cubic voxels: {edges}"


def test_points_outside_the_cloud_are_clamped_not_crashed():
    field = ColorField(*_split_cloud(n=5_000))
    rgb, hit = field.sample(np.array([[-99.0, -99.0, -99.0], [99.0, 99.0, 99.0]]))
    assert hit.all()
    assert np.isfinite(rgb).all()


def test_degenerate_cloud_does_not_divide_by_zero():
    """Every Gaussian at one point: the span is zero on all three axes."""
    xyz = np.zeros((10, 3))
    rgb = np.full((10, 3), 0.25, np.float32)
    out, hit = ColorField(xyz, rgb).sample(np.zeros((1, 3)))
    assert hit.all() and np.allclose(out, 0.25)


# --- the bake --------------------------------------------------------------

def _quad():
    verts = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return verts, faces, uv


def test_bake_puts_the_right_colour_in_the_right_half():
    field = ColorField(*_split_cloud())
    tex, coverage = bake_texture(*_quad(), field, size=64, samples=200_000)
    assert tex.shape == (64, 64, 3)
    assert coverage > 0.9
    assert tex[32, 8].tolist() == [255, 0, 0], "u<0.5 samples the red half"
    assert tex[32, 56].tolist() == [0, 0, 255], "u>0.5 samples the blue half"


def test_bake_respects_the_gltf_uv_origin():
    """v runs bottom-up in glTF while image rows run top-down.

    Getting this backwards flips every texture vertically, which is the exact
    class of silent bug `detect_front_view` exists to prevent elsewhere.
    """
    xyz = np.stack(np.meshgrid(*[np.linspace(0, 1, 40)] * 3), -1).reshape(-1, 3)
    rgb = np.where(xyz[:, 1:2] < 0.5, [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]).astype(np.float32)
    tex, _ = bake_texture(*_quad(), ColorField(xyz, rgb), size=64, samples=200_000)
    # v=0 (green, low y) must land on the BOTTOM row of the image.
    assert tex[60, 32].tolist() == [0, 255, 0]
    assert tex[4, 32].tolist() == [255, 255, 0]


def test_uncovered_texels_are_filled_rather_than_left_black():
    """Unfilled gutters show as dark seams once bilinear filtering reaches them."""
    rgb = np.zeros((16, 16, 3), np.float32)
    covered = np.zeros((16, 16), bool)
    rgb[8, 8] = [1.0, 0.0, 0.0]
    covered[8, 8] = True
    out = fill_holes(rgb, covered)
    assert out[0, 0].tolist() == [1.0, 0.0, 0.0], "colour must reach the far corner"


def test_fill_holes_leaves_covered_texels_untouched():
    rgb = np.zeros((8, 8, 3), np.float32)
    covered = np.ones((8, 8), bool)
    rgb[4, 4] = [0.25, 0.5, 0.75]
    out = fill_holes(rgb, covered)
    assert np.allclose(out, rgb)


def test_bake_is_deterministic():
    """`densify` seeds its RNG, so two bakes of one input must be identical."""
    field = ColorField(*_split_cloud())
    a, _ = bake_texture(*_quad(), field, size=32, samples=50_000)
    b, _ = bake_texture(*_quad(), field, size=32, samples=50_000)
    assert np.array_equal(a, b)


# --- the cloud container ---------------------------------------------------

def test_cloud_reports_its_extent_and_builds_a_field():
    xyz, rgb = _split_cloud(n=1_000)
    cloud = GaussianCloud(
        xyz=xyz,
        features_dc=np.zeros((len(xyz), 3)),
        opacity=np.ones(len(xyz), np.float32),
    )
    assert len(cloud) == 1_000
    assert cloud.extent().shape == (3,)
    rgb_out, hit = cloud.field().sample(np.array([[0.5, 0.5, 0.5]]))
    assert hit.all()
    assert np.allclose(rgb_out, 0.5), "zero DC is mid grey through the whole path"


def test_stacked_densify_keeps_position_and_uv_in_correspondence():
    """The bake depends on one densify call carrying both, not two that agree."""
    from hvym_img_tools.core.meshops import densify

    verts, faces, uv = _quad()
    dense = densify(np.hstack([verts, uv]), faces, 5_000)
    pos, duv = dense[:, :3], dense[:, 3:5]
    # This quad's UV is its own xy, so correspondence is checkable directly.
    assert np.allclose(pos[:, 0], duv[:, 0], atol=1e-9)
    assert np.allclose(pos[:, 1], duv[:, 1], atol=1e-9)


# --- integration with the mesh pipeline ------------------------------------

class AppearanceBackbone:
    """A backbone that decodes appearance, like TRELLIS and unlike TripoSR."""

    def __init__(self):
        self._mesh = trimesh.creation.icosphere(subdivisions=3)

    def reconstruct(self, image, *, seed: int = 0, **_):
        return self._mesh.copy()

    def reconstruct_appearance(self, image, *, seed: int = 0, **_):
        mesh = self._mesh.copy()
        rng = np.random.default_rng(0)
        xyz = mesh.vertices + rng.normal(0, 0.01, mesh.vertices.shape)
        return mesh, GaussianCloud(
            xyz=np.asarray(xyz),
            features_dc=np.zeros((len(xyz), 3)),
            opacity=np.ones(len(xyz), np.float32),
        )


def test_gaussian_texture_produces_a_textured_glb():
    import io as _io

    from hvym_img_tools.tools.mesh.pipeline import run_pipeline

    buf = _io.BytesIO()
    from PIL import Image as _Image

    _Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")

    result = run_pipeline(
        buf.getvalue(),
        backbone=AppearanceBackbone(),
        target_faces=2_000,
        texture="gaussian",
        texture_size=256,
    )
    assert result.glb[:4] == b"glTF"
    assert result.coverage > 0.0

    scene = trimesh.load(trimesh.util.wrap_as_stream(result.glb), file_type="glb")
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    assert mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)
    assert mesh.visual.material.baseColorTexture is not None


def test_untextured_remains_the_default_and_is_unchanged():
    """The tool's contract is bare geometry; texturing must be opt-in."""
    import io as _io

    from PIL import Image as _Image

    from hvym_img_tools.tools.mesh.pipeline import run_pipeline

    buf = _io.BytesIO()
    _Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")
    result = run_pipeline(buf.getvalue(), backbone=AppearanceBackbone(), target_faces=2_000)
    assert result.coverage == 0.0
    scene = trimesh.load(trimesh.util.wrap_as_stream(result.glb), file_type="glb")
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    assert getattr(mesh.visual, "uv", None) is None


def test_a_geometry_only_backbone_says_so_clearly():
    """TripoSR has no appearance to bake; the error must name the reason."""
    import io as _io

    from PIL import Image as _Image

    from hvym_img_tools.tools.mesh.pipeline import run_pipeline

    class GeometryOnly:
        def reconstruct(self, image, *, seed: int = 0, **_):
            return trimesh.creation.icosphere(subdivisions=2)

    buf = _io.BytesIO()
    _Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")
    with pytest.raises(RuntimeError, match="appearance"):
        run_pipeline(buf.getvalue(), backbone=GeometryOnly(), texture="gaussian")


# --- the derived resolution, and the round trip -----------------------------

def test_resolution_is_derived_from_the_atlas_not_guessed():
    """The whole point: a figure's head must get voxels as fine as its texels.

    The first probe fixed this at 256 and under-resolved a head 7.7x. On the
    same geometry the derived value has to land far above that.
    """
    from hvym_img_tools.core.gsbake import texel_matched_resolution

    # a tall thin subject, like every reangle subject
    mesh = trimesh.creation.cylinder(radius=0.15, height=1.0, sections=64)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    uv = mesh.unwrap().visual.uv
    m2 = mesh.unwrap()
    res = texel_matched_resolution(np.asarray(m2.vertices), np.asarray(m2.faces),
                                   np.asarray(m2.visual.uv), 1024)
    assert res > 256, f"derived resolution {res} is no better than the bad default"
    assert res <= 4096


def test_resolution_scales_with_texture_size():
    """Doubling the atlas halves the texel edge, so the field must get finer."""
    from hvym_img_tools.core.gsbake import texel_matched_resolution

    m = trimesh.creation.icosphere(subdivisions=3).unwrap()
    v, f, uv = (np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.visual.uv))
    small = texel_matched_resolution(v, f, uv, 512)
    large = texel_matched_resolution(v, f, uv, 2048)
    assert large > small, f"{large} should exceed {small}"


def test_finer_field_resolves_detail_a_coarse_one_averages_away():
    """The actual claim behind the re-run, on a cloud with known fine structure."""
    rng = np.random.default_rng(0)
    xyz = rng.random((300_000, 3))
    # 40 stripes along x -- far finer than a res=8 field can represent
    stripe = (xyz[:, 0] * 40).astype(int) % 2
    rgb = np.where(stripe[:, None] == 0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).astype(np.float32)

    q = np.stack([np.linspace(0.01, 0.99, 400), np.full(400, 0.5), np.full(400, 0.5)], 1)
    coarse, _ = ColorField(xyz, rgb, resolution=8).sample(q)
    fine, _ = ColorField(xyz, rgb, resolution=256).sample(q)

    # contrast survives at high resolution and is averaged to mud at low
    assert fine[:, 0].std() > 3 * coarse[:, 0].std()


def test_cloud_round_trips_through_npz_with_its_mesh():
    """Offline iteration is the point; losing the mesh would defeat it."""
    rng = np.random.default_rng(0)
    cloud = GaussianCloud(
        xyz=rng.random((500, 3)),
        features_dc=rng.normal(0, 0.5, (500, 3)),
        opacity=rng.random(500).astype(np.float32),
    )
    m = trimesh.creation.icosphere(subdivisions=2)
    blob = cloud.to_npz(m.vertices, m.faces)

    back, verts, faces = GaussianCloud.from_npz(blob)
    assert len(back) == 500
    assert np.allclose(back.xyz, cloud.xyz, atol=1e-6)
    assert verts is not None and faces is not None
    assert len(verts) == len(m.vertices) and len(faces) == len(m.faces)
    # float16 colour is a deliberate size trade, so allow its precision
    assert np.allclose(back.features_dc, cloud.features_dc, atol=1e-2)


def test_cloud_npz_without_a_mesh_is_still_readable():
    cloud = GaussianCloud(
        xyz=np.zeros((4, 3)), features_dc=np.zeros((4, 3)), opacity=np.ones(4)
    )
    back, verts, faces = GaussianCloud.from_npz(cloud.to_npz())
    assert len(back) == 4 and verts is None and faces is None


def test_cloud_mode_returns_an_npz_not_a_glb():
    import io as _io

    from PIL import Image as _Image

    from hvym_img_tools.tools.mesh.pipeline import run_pipeline

    buf = _io.BytesIO()
    _Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")
    result = run_pipeline(buf.getvalue(), backbone=AppearanceBackbone(),
                          target_faces=2_000, texture="cloud")
    assert result.glb[:4] != b"glTF"
    back, verts, faces = GaussianCloud.from_npz(result.glb)
    assert len(back) > 0 and verts is not None and faces is not None
