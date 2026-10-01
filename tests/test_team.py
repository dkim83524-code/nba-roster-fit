import numpy as np
import pytest

from rosterfit import geometry
from rosterfit.config import CORNERS
from rosterfit.team import PLAYOFF_MODE, evaluate_team, roster_sums


def test_league_builds_complete_pool(league):
    assert league.pool_seasons == ["2023-24", "2024-25", "2025-26"]
    assert not league.incomplete
    p = league.players["2025-26"]
    q = p[p["qualified"]]
    for c in CORNERS:
        assert q[f"{c}_pct"].between(0, 100).all()
        assert q[f"{c}_z"].mean() == pytest.approx(0.0, abs=1e-9)


def test_minutes_weights_sum_to_five(league):
    sums = roster_sums(league, "DEM", "2025-26", "regular")
    assert sums.roster["weight"].sum() == pytest.approx(5.0)


def test_floor_applies_only_to_capped_corners(league):
    r = roster_sums(league, "DEM", "2025-26", "regular").roster
    assert (r["playmaking_contrib"] >= 0).all()
    assert (r["portability_contrib"] >= 0).all()
    assert (r["offense_contrib"] < 0).any() or (r["defense_contrib"] < 0).any()


def test_pool_percentile_cap_binds_for_expected_share(league):
    for corner in league.cfg.team.capped_corners:
        share = league.cap_binding_share(corner, "regular")
        assert share == pytest.approx(0.2, abs=0.06)


def test_capped_value_and_redundancy_add_up(league):
    res = evaluate_team(league, "DEM", "2025-26")
    for c in league.cfg.team.capped_corners:
        assert res.values[c] <= res.cap[c] + 1e-12
        assert res.values[c] + res.redundancy[c] == pytest.approx(res.uncapped[c])
    for c in ("offense", "defense"):
        assert res.values[c] == pytest.approx(res.uncapped[c])


def test_players_worth_mode(league):
    league.cfg.team.cap_mode = "players_worth"
    league._raw_pools.clear()
    res = evaluate_team(league, "DEM", "2025-26")
    c = "playmaking"
    one = league.one_worth(c, "2025-26")
    assert res.cap[c] == pytest.approx(league.cfg.team.cap_players_worth * one)
    assert res.redundancy_players[c] == pytest.approx(res.redundancy[c] / one)


def test_coverage_is_area_of_team_shape(league):
    res = evaluate_team(league, "DEM", "2025-26")
    v = {c: res.pct[c] / 100 for c in CORNERS}
    assert res.coverage == pytest.approx(geometry.area_fraction(v, league.cfg.corners.order))
    assert 0 <= res.union <= 1
    assert res.overlap == pytest.approx(res.sum_areas - res.union)


def test_minutes_leader_drawn_full_size(league):
    res = evaluate_team(league, "DEM", "2025-26")
    drawn = res.roster[res.roster["drawn"]]
    assert drawn["scale"].max() == pytest.approx(1.0)
    leader = drawn.loc[drawn["MIN"].idxmax()]
    assert leader["scale"] == pytest.approx(1.0)


def test_playoff_rotation_is_top_eight(league):
    res = evaluate_team(league, "DEM", "2025-26", PLAYOFF_MODE)
    assert len(res.roster) == 8
    assert res.roster["weight"].sum() == pytest.approx(5.0)
    # the playoff pool holds only playoff teams' top-8s
    assert res.pool_size == sum(len(league.teams(s, PLAYOFF_MODE)) for s in league.pool_seasons)


def test_unknown_team_is_a_clear_error(league):
    with pytest.raises(ValueError, match="no regular minutes for ZZZ"):
        evaluate_team(league, "ZZZ", "2025-26")


def test_team_percentiles_in_range(league):
    for team in league.teams("2025-26"):
        res = evaluate_team(league, team, "2025-26")
        assert all(0 < res.pct[c] <= 100 for c in CORNERS)
        assert not np.isnan(res.coverage)
