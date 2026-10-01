import numpy as np
import pandas as pd
import pytest

from rosterfit.metrics.pool import composite, percentile_against, qualified_mask, zscore


def test_percentile_is_share_at_or_below():
    pool = [1, 2, 3, 4]
    np.testing.assert_allclose(percentile_against([0, 1, 2.5, 4, 9], pool), [0, 25, 50, 100, 100])
    assert np.isnan(percentile_against([np.nan], pool)[0])


def test_zscore_uses_pool_only():
    values = pd.Series([0.0, 2.0, 4.0, 100.0])
    pool = pd.Series([True, True, True, False])
    z = zscore(values, pool)
    sd = np.std([0, 2, 4])
    assert z.iloc[1] == pytest.approx(0.0)
    assert z.iloc[3] == pytest.approx((100 - 2) / sd)


def test_qualified_mask():
    m = qualified_mask(pd.Series([30, 30, 10]), pd.Series([20, 10, 30]), 25, 15)
    assert m.tolist() == [True, False, False]


def test_composite_renormalizes_and_requires_enough_weight():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, np.nan, np.nan]})
    pool = pd.Series([True] * 4)
    out = composite(df, {"a": 0.75, "b": 0.25}, pool, min_weight_present=0.5)
    za = zscore(df["a"], pool)
    # rows 2-3 lack 'b' but still have 75% of the weight: they score as z(a) alone
    assert out.iloc[2] == pytest.approx(za.iloc[2])
    out_strict = composite(df, {"a": 0.25, "b": 0.75}, pool, min_weight_present=0.5)
    assert np.isnan(out_strict.iloc[2])


def test_composite_negative_orientation():
    df = pd.DataFrame({"tov": [1.0, 2.0, 3.0]})
    out = composite(df, {"tov": 1.0}, pd.Series([True] * 3), negative=("tov",))
    assert out.iloc[0] > out.iloc[2]


def test_rank_scaling_stops_one_extreme_stat_from_dominating():
    # player 0 is a center: off-the-charts OREB, poor at the other three components
    df = pd.DataFrame({
        "oreb": [40.0] + [float(i % 7) for i in range(1, 60)],
        "shoot": [0.0] + [float(i % 11) for i in range(1, 60)],
        "versatility": [0.0] + [float(i % 5) for i in range(1, 60)],
        "screens": [0.0] + [float(i % 3) for i in range(1, 60)],
    })
    pool = pd.Series([True] * len(df))
    w = {"oreb": 0.25, "shoot": 0.25, "versatility": 0.25, "screens": 0.25}
    z_rank = percentile_against(composite(df, w, pool), composite(df, w, pool))[0]
    r = composite(df, w, pool, scaling="rank")
    r_rank = percentile_against(r, r)[0]
    assert z_rank > 60          # plain z: the one outlier lifts the center well up the list
    assert r_rank < 20          # rank-based: one great stat can't hide three bad ones
    assert np.nanmax(np.abs(r)) < 3.5
