"""reangle tool tests — CPU-only, no backbone, no weights.

The front-axis tests are the regression guard for a real bug: the benchmark's
first UV bake assumed the image plane was (X, Y) and so projected the artist's
art onto the mesh **from the side**. TripoSR's front axis is +X, i.e. the plane
is (Y, Z). See docs/BENCHMARK.md §5.5.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh
from PIL import Image

from hvym_img_tools.core.imageio import DEFAULT_MARGIN, DEFAULT_SIZE
from hvym_img_tools.tools.reangle.uvbake import (
    bake_glb,
    detect_front_view,
    front_planar_uv,
)

# A character-shaped slab: thin in X (depth), medium in Y (width), tall in Z.
# Viewed along +X it gives a tall narrow silhouette, like a standing figure.
X_HALF, Y_HALF, Z_HALF = 0.1, 0.3, 1.0


@pytest.fixture
def slab():
    mesh = trimesh.creation.box(extents=(X_HALF * 2, Y_HALF * 2, Z_HALF * 2))
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)


@pytest.fixture
def slab_alpha():
    """Silhouette of the slab seen along X, matching fit_to_frame's placement."""
    long_side = (Z_HALF * 2) * (1 + 2 * DEFAULT_MARGIN)
    scale = (DEFAULT_SIZE - 1) / long_side
    half_w = Y_HALF * scale
    half_h = Z_HALF * scale
    centre = (DEFAULT_SIZE - 1) / 2
    alpha = np.zeros((DEFAULT_SIZE, DEFAULT_SIZE), np.uint8)
    r0, r1 = int(centre - half_h), int(centre + half_h)
    c0, c1 = int(centre - half_w), int(centre + half_w)
    alpha[r0:r1 + 1, c0:c1 + 1] = 255
    return alpha


def test_detects_x_as_the_depth_axis(slab, slab_alpha):
    """The bug: assuming (X, Y) projects the art from the side."""
    vertices, faces = slab
    view = detect_front_view(vertices, slab_alpha, faces)
    assert view.d_axis == 0, "depth must be X for a slab that is thin in X"
    assert view.plane == (1, 2), "image plane must be (Y, Z), not (X, Y)"
    assert view.silhouette_iou > 0.9


def test_front_view_beats_the_side_view(slab, slab_alpha):
    """Scoring must actually discriminate, not just pick the first candidate."""
    vertices, faces = slab
    best = detect_front_view(vertices, slab_alpha, faces)

    # Score the wrong (X, Y) plane the way detect_front_view does internally.
    from hvym_img_tools.tools.reangle.uvbake import _splat_mask

    wrong = _splat_mask(vertices[:, [0, 1]], DEFAULT_SIZE, DEFAULT_MARGIN)
    silhouette = slab_alpha > 12
    wrong_iou = (wrong & silhouette).sum() / (wrong | silhouette).sum()
    assert best.silhouette_iou > wrong_iou * 1.5


def test_uv_in_unit_range_and_spans_the_atlas(slab, slab_alpha):
    vertices, faces = slab
    uv = front_planar_uv(vertices, detect_front_view(vertices, slab_alpha, faces))
    assert uv.shape == (len(vertices), 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    # the tall axis must use most of the atlas; the narrow one much less
    assert np.ptp(uv[:, 1]) > 0.8
    assert np.ptp(uv[:, 0]) < np.ptp(uv[:, 1])


def test_uv_is_stable_under_uniform_scale(slab, slab_alpha):
    """Scale-invariance matters: the backbone's absolute units are arbitrary."""
    vertices, faces = slab
    view = detect_front_view(vertices, slab_alpha, faces)
    a = front_planar_uv(vertices, view)
    b = front_planar_uv(vertices * 7.5, view)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_bake_glb_is_loadable_and_textured(slab, slab_alpha):
    vertices, faces = slab
    art = Image.new("RGBA", (DEFAULT_SIZE, DEFAULT_SIZE), (200, 30, 40, 255))
    uv = front_planar_uv(vertices, detect_front_view(vertices, slab_alpha, faces))
    blob = bake_glb(vertices, faces, uv, art)

    assert blob[:4] == b"glTF", "must be a binary glTF container"
    scene = trimesh.load(trimesh.util.wrap_as_stream(blob), file_type="glb", process=False)
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    assert len(mesh.vertices) == len(vertices)
    assert mesh.visual.uv is not None, "texture must survive the round trip"


def test_tool_contract_is_registered():
    from hvym_img_tools.core import registry

    registry.discover()
    tool_cls = registry.get("reangle")
    assert tool_cls.file_fields() == ("image",)
    assert tool_cls.returns_media() is True
    # 0.3.0 defaults to TRELLIS. TripoSR stays *declared* (so it remains a
    # rollback lever) but is not warmed, because an image may carry only one.
    assert set(tool_cls().models_needed()) == {"isnet", "trellis"}
    assert set(tool_cls().model_loaders()) == {"isnet", "trellis", "triposr"}


def test_texture_size_is_exposed_and_defaults_to_2048():
    """512 softened the linework once magnified in-app; the default is the fix."""
    from hvym_img_tools.core.imageio import DEFAULT_TEXTURE_SIZE
    from hvym_img_tools.tools.reangle.tool import ReangleInput, ReangleTool

    field = ReangleInput.model_fields["texture_size"]
    assert field.default == DEFAULT_TEXTURE_SIZE == 2048
    assert ReangleInput(image=b"x").texture_size == 2048

    # The version is part of the cache key, so a texture change must bump it or
    # cached 512-pixel results would be served for requests that now want 2048.
    assert ReangleTool.version != "0.1.0"


def test_texture_size_changes_the_cache_key():
    from hvym_img_tools.tools.reangle.tool import ReangleInput, ReangleTool

    tool = ReangleTool()
    a = tool.cache_key_parts(ReangleInput(image=b"same", texture_size=512))
    b = tool.cache_key_parts(ReangleInput(image=b"same", texture_size=2048))
    assert a != b, "different texture sizes must not share a cached result"


def test_front_detection_accepts_any_matte_resolution(slab):
    """Regression: raising texture_size to 2048 broke the live worker.

    `detect_front_view` splats the mesh at `size` (512) and compares it against
    the matte's alpha. Once the matte was baked at 2048 the two no longer had
    the same shape:

        ValueError: operands could not be broadcast together
                    with shapes (512,512) (2048,2048)

    Every unit test passed because they all handed it a 512 alpha. The fix
    normalises the alpha, so this checks a range of sizes AND that the detected
    axis does not change with resolution.
    """
    import numpy as np

    from hvym_img_tools.tools.reangle.uvbake import detect_front_view

    verts, faces = slab
    ref = None
    for size in (256, 512, 1024, 2048):
        alpha = np.zeros((size, size), np.uint8)
        lo, hi = int(size * 0.2), int(size * 0.8)
        alpha[lo:hi, lo:hi] = 255
        view = detect_front_view(verts, alpha, faces)
        axes = (view.h_axis, view.v_axis, view.d_axis)
        if ref is None:
            ref = axes
        assert axes == ref, f"front axis changed at matte size {size}"


# --- backbone handoff ------------------------------------------------------
# The pipeline used to composite the matte onto grey before every backbone.
# That is TripoSR's convention and it destroys the alpha TRELLIS reads: handed
# an opaque image, TRELLIS re-segments with its own rembg and builds geometry
# against a silhouette the UV bake below does not agree with.


def _png(size=(64, 64)) -> bytes:
    import io as _io

    buf = _io.BytesIO()
    Image.new("RGB", size, (180, 60, 40)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeIsnet:
    """Minimal onnxruntime stand-in returning a tall, centred blob."""

    class _Inp:
        name = "input"
        shape = [1, 3, 256, 256]

    def get_inputs(self):
        return [self._Inp()]

    def run(self, _outputs, _feed):
        mask = np.zeros((1, 1, 256, 256), np.float32)
        mask[:, :, 40:216, 96:160] = 1.0
        return [mask]


class _CapturingBackbone:
    def __init__(self, subdivisions: int = 4):
        self.seen: dict = {}
        self._subdivisions = subdivisions

    def reconstruct(self, image, **kwargs):
        self.seen["mode"] = image.mode
        self.seen["alpha"] = (
            image.getchannel("A").getextrema() if image.mode == "RGBA" else None
        )
        self.seen["kwargs"] = kwargs
        return trimesh.creation.icosphere(subdivisions=self._subdivisions)


def test_pipeline_hands_the_backbone_the_matte_with_its_alpha_intact():
    from hvym_img_tools.tools.reangle.pipeline import run_pipeline

    backbone = _CapturingBackbone()
    run_pipeline(_png(), isnet_session=_FakeIsnet(), backbone=backbone, texture_size=128)

    assert backbone.seen["mode"] == "RGBA", "the matte must reach the backbone unflattened"
    assert backbone.seen["alpha"][0] == 0, "a real silhouette, not an opaque rectangle"


def test_pipeline_passes_both_backbones_knobs():
    """It does not know which backbone it holds, so it sends both and each ignores
    what it does not use."""
    from hvym_img_tools.tools.reangle.pipeline import run_pipeline

    backbone = _CapturingBackbone()
    run_pipeline(
        _png(), isnet_session=_FakeIsnet(), backbone=backbone,
        texture_size=128, mc_resolution=192, seed=7,
    )
    assert backbone.seen["kwargs"] == {"mc_resolution": 192, "seed": 7}


def test_pipeline_decimates_before_the_uv_bake():
    """UVs are per-vertex, so decimating after the bake would discard them."""
    from hvym_img_tools.tools.reangle.pipeline import run_pipeline

    res = run_pipeline(
        _png(), isnet_session=_FakeIsnet(), backbone=_CapturingBackbone(),
        texture_size=128, target_faces=2_000,
    )
    assert res.faces_raw > res.faces, "a dense backbone result should have shrunk"
    assert res.faces <= 2_000

    scene = trimesh.load(trimesh.util.wrap_as_stream(res.glb), file_type="glb", process=False)
    baked = trimesh.util.concatenate(tuple(scene.geometry.values()))
    assert len(baked.vertices) == res.vertices
    assert len(baked.visual.uv) == res.vertices, "every surviving vertex needs a UV"


def test_pipeline_leaves_the_triposr_path_uncapped():
    """`None` is the default: TripoSR's ~31k-face proxy must pass through whole."""
    from hvym_img_tools.tools.reangle.pipeline import run_pipeline

    res = run_pipeline(
        _png(), isnet_session=_FakeIsnet(), backbone=_CapturingBackbone(), texture_size=128,
    )
    assert res.faces == res.faces_raw


def test_densify_handles_uvs_as_well_as_vertices():
    """The probe scores a delivered .glb by splatting its UVs, and must densify
    the same way detection does or the two masks are not comparable -- bare
    vertices scored 0.095 against the pipeline's 0.344 on identical geometry."""
    from hvym_img_tools.tools.reangle.uvbake import densify

    mesh = trimesh.creation.icosphere(subdivisions=2)
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, float)
    uv = verts[:, :2]

    dense_3d = densify(verts, faces, 40_000)
    dense_2d = densify(uv, faces, 40_000)
    assert dense_3d.shape[1] == 3 and dense_2d.shape[1] == 2
    assert len(dense_2d) == len(dense_3d) >= 40_000
    # the originals stay at the front, so vertex-indexed data still lines up
    np.testing.assert_allclose(dense_2d[: len(uv)], uv)


def test_densify_leaves_a_dense_mesh_alone():
    from hvym_img_tools.tools.reangle.uvbake import densify

    pts = np.zeros((50_000, 3))
    assert densify(pts, np.array([[0, 1, 2]]), 40_000) is pts


# --- 0.3.0: TRELLIS by default, TripoSR still reachable --------------------
# The backbone is expected to be switched back if it does not hold up in
# Inkternity, so each rollback lever gets a test. A lever nobody exercises is a
# lever that quietly stops working.


def test_default_backbone_is_trellis():
    from hvym_img_tools.tools.reangle.tool import ReangleInput, SHIPPING_BACKBONE

    assert SHIPPING_BACKBONE == "trellis"
    assert ReangleInput(image=b"x").backbone == "trellis"


def test_env_var_switches_the_default_back(monkeypatch):
    """Lever 2: an operator flips a worker without a client change or rebuild."""
    from hvym_img_tools.tools.reangle.tool import BACKBONE_ENV, ReangleInput, ReangleTool

    monkeypatch.setenv(BACKBONE_ENV, "triposr")
    assert ReangleInput(image=b"x").backbone == "triposr"
    assert set(ReangleTool().models_needed()) == {"isnet", "triposr"}, \
        "the env default must decide what gets warmed, or the flip costs a cold load"


def test_request_can_still_ask_for_triposr():
    """Lever 3: per-request override, for A/B from the client."""
    from hvym_img_tools.tools.reangle.tool import ReangleInput

    assert ReangleInput(image=b"x", backbone="triposr").backbone == "triposr"


def test_version_bumped_so_triposr_results_are_not_served_for_trellis():
    """The cache key carries the version; without a bump every cached .glb from
    0.2.0 would answer requests that now mean a different backbone entirely."""
    from hvym_img_tools.tools.reangle.tool import ReangleTool

    assert ReangleTool.version == "0.3.0"


def test_backbone_and_new_knobs_change_the_cache_key():
    from hvym_img_tools.tools.reangle.tool import ReangleInput, ReangleTool

    tool = ReangleTool()
    base = tool.cache_key_parts(ReangleInput(image=b"same"))
    for changed in (
        ReangleInput(image=b"same", backbone="triposr"),
        ReangleInput(image=b"same", target_faces=50_000),
        ReangleInput(image=b"same", seed=7),
    ):
        assert tool.cache_key_parts(changed) != base


def test_target_faces_follows_the_backbone_when_unset():
    """TripoSR's depth proxy must stay uncapped or 0.2.0's output changes."""
    from hvym_img_tools.backbones import default_target_faces_for
    from hvym_img_tools.core import registry
    from hvym_img_tools.tools.reangle.tool import ReangleInput

    registry.discover()
    assert ReangleInput(image=b"x").target_faces is None, "unset means 'ask the backbone'"
    assert default_target_faces_for("trellis") == 20_000
    assert default_target_faces_for("triposr") is None


def test_target_faces_is_bounded_when_given():
    from hvym_img_tools.tools.reangle.tool import ReangleInput

    with pytest.raises(Exception):
        ReangleInput(image=b"x", target_faces=500)
    with pytest.raises(Exception):
        ReangleInput(image=b"x", target_faces=500_000)


def test_missing_weights_name_the_backbone_and_the_way_out():
    """A reangle-only image asked for TRELLIS must say so, not fail inside
    someone else's library three frames deep."""
    from hvym_img_tools.core.tool import Context
    from hvym_img_tools.tools.reangle.tool import ReangleInput, ReangleTool

    class _NoTrellis:
        def get(self, key):
            if key == "trellis":
                raise KeyError("no loader registered for model 'trellis'")
            return object()

    ctx = Context(models=_NoTrellis(), cache=None, workspace=None, config=None)
    with pytest.raises(RuntimeError) as exc:
        ReangleTool().run(ReangleInput(image=b"x", backbone="trellis"), ctx)
    message = str(exc.value)
    assert "trellis" in message
    assert "triposr" in message, "the error must name what this worker CAN do"
