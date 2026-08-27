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
    from PIL import Image

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
    assert set(tool_cls().models_needed()) == {"isnet", "triposr"}


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
