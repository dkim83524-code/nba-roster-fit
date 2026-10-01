import math

import pytest

from rosterfit.sources.formulas import box_creation, three_point_proficiency


def test_three_point_proficiency_sigmoid():
    assert three_point_proficiency(0, 0.5) == pytest.approx(0.0)
    assert three_point_proficiency(2, 0.4) == pytest.approx((2 / (1 + math.exp(-2)) - 1) * 0.4)
    assert three_point_proficiency(10, 0.4) == pytest.approx(0.4, abs=1e-3)


def test_box_creation_matches_published_formula():
    ast, pts, tov, fg3a, pct = 10.0, 30.0, 4.0, 8.0, 0.37
    prof = (2 / (1 + math.exp(-fg3a)) - 1) * pct
    expected = (ast * 0.1843 + (pts + tov) * 0.0969 - 2.3021 * prof
                + 0.0582 * (ast * (pts + tov) * prof) - 1.1942)
    assert float(box_creation(ast, pts, tov, fg3a, pct)) == pytest.approx(expected)
