import itertools

import numpy as np
import pytest

from rosterfit import geometry

ORDER = ["offense", "playmaking", "portability", "defense"]


def vals(o, pm, port, d):
    return {"offense": o, "playmaking": pm, "portability": port, "defense": d}


def test_full_shape_is_the_square():
    assert geometry.area_fraction(vals(1, 1, 1, 1), ORDER) == pytest.approx(1.0)
    poly = geometry.to_polygon(geometry.shape_points(vals(1, 1, 1, 1), ORDER))
    assert poly.area / geometry.SQUARE_AREA == pytest.approx(1.0)


def test_corner_order_sets_vertex_positions():
    pts = geometry.shape_points(vals(1, 0.5, 0.25, 0.75), ORDER)
    np.testing.assert_allclose(pts, [[-1, 1], [0.5, 0.5], [0.25, -0.25], [-0.75, -0.75]])


@pytest.mark.parametrize("v", list(itertools.product([0.1, 0.4, 0.9], repeat=4))[::7])
def test_area_formula_matches_polygon_and_opposite_pair_identity(v):
    shape = dict(zip(ORDER, v))
    analytic = geometry.area_fraction(shape, ORDER)
    poly = geometry.to_polygon(geometry.shape_points(shape, ORDER))
    assert poly.area / geometry.SQUARE_AREA == pytest.approx(analytic)
    o, pm, port, d = v  # with this order offense/portability and playmaking/defense are opposite
    assert analytic == pytest.approx((o + port) * (pm + d) / 4)


def test_minutes_scale_shrinks_area_quadratically():
    shape = vals(0.8, 0.6, 0.4, 0.7)
    assert geometry.area_fraction(shape, ORDER, scale=0.5) == pytest.approx(
        0.25 * geometry.area_fraction(shape, ORDER))


def test_opposite_corners_only_is_a_line_with_no_area():
    shape = vals(1, 0, 1, 0)
    assert geometry.area_fraction(shape, ORDER) == 0
    assert geometry.to_polygon(geometry.shape_points(shape, ORDER)) is None


def test_overlap_of_identical_shapes_equals_one_copy():
    p = geometry.to_polygon(geometry.shape_points(vals(0.5, 0.5, 0.5, 0.5), ORDER))
    total, union, overlap = geometry.overlap_fraction([p, p])
    assert union == pytest.approx(0.25)
    assert total == pytest.approx(0.5)
    assert overlap == pytest.approx(0.25)


def test_union_of_shapes_in_different_corners():
    a = geometry.to_polygon(geometry.shape_points(vals(1, 1, 0, 0), ORDER))  # top triangle, area 1/4
    b = geometry.to_polygon(geometry.shape_points(vals(0, 0, 1, 1), ORDER))  # bottom triangle, 1/4
    total, union, overlap = geometry.overlap_fraction([a, b, None])
    assert union == pytest.approx(0.5)
    assert overlap == pytest.approx(0.0, abs=1e-12)


def test_missing_value_is_rejected():
    with pytest.raises(ValueError):
        geometry.area_fraction(vals(0.5, float("nan"), 0.5, 0.5), ORDER)


def test_labels_for_corner_points_stay_inside_the_drawing():
    from rosterfit.plot import _place_labels

    # a long name on a point in the bottom-left corner, plus crowding near it
    candidates = [("Victor Wembanyama", (-0.97, -0.97)), ("Luke Kornet", (-0.6, -0.62)),
                  ("Devin Vassell", (-0.85, 0.85)), ("Stephon Castle", (0.99, 0.99))]
    for name, _, (lx, ly, ha) in _place_labels(candidates):
        width = 0.021 * len(name) + 0.01
        left = lx if ha == "left" else lx - width
        assert -1.17 - 1e-9 <= left and left + width <= 1.17 + 1e-9, name
        assert -1.0 - 1e-9 <= ly - 0.0275 and ly + 0.0275 <= 1.0 + 1e-9, name


def test_labels_keep_clear_of_other_players_dots():
    from rosterfit.plot import _place_labels

    # Hayes Pike's dot sits where Finn Alder's first-choice label would go
    candidates = [("Finn Alder", (0.45, -0.58)), ("Hayes Pike", (0.55, -0.63))]
    lx, ly, ha = _place_labels(candidates)[0][2]
    width, half_h, dot = 0.021 * len("Finn Alder") + 0.01, 0.0275, 0.025
    left = lx if ha == "left" else lx - width
    dx, dy = candidates[1][1]
    assert left > dx + dot or left + width < dx - dot or ly - half_h > dy + dot or ly + half_h < dy - dot
