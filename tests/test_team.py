import numpy as np
import pytest

from rosterfit import geometry
from rosterfit.cache import Cache
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


def test_player_colors_match_between_regular_and_playoff_charts(league):
    from rosterfit.plot import assign_colors

    palette = list("ABCDEFGH")
    reg = evaluate_team(league, "DEM", "2025-26")
    po = evaluate_team(league, "DEM", "2025-26", PLAYOFF_MODE)
    assert reg.color_order == po.color_order
    reg_colors = assign_colors(list(reg.roster["PLAYER_ID"][:8]), reg.color_order, palette)
    po_colors = assign_colors(list(po.roster["PLAYER_ID"]), po.color_order, palette)
    for pid in set(reg_colors) & set(po_colors):
        assert reg_colors[pid] == po_colors[pid]
    assert len(set(po_colors.values())) == len(po_colors)  # no two players share a color


def test_assign_colors_gives_newcomers_the_free_slots():
    from rosterfit.plot import assign_colors

    colors = assign_colors([2, 99, 1], order=[1, 2, 3], palette=["a", "b", "c"])
    assert colors == {1: "a", 2: "b", 99: "c"}


def test_player_missing_from_tracking_tables_counts_as_zero_not_unknown(cfg, synthetic_dirs, tmp_path):
    """A center with no catch-and-shoot attempts isn't in NBA.com's C&S table at all. That must read
    as a non-shooter (0), not as missing data that gets averaged away."""
    from rosterfit.metrics.players import build_player_table, load_season

    cache_dir, impact_dir = synthetic_dirs
    cfg.impact.dir = str(impact_dir)
    cfg.qualified.min_gp = 10
    data = load_season(Cache(cache_dir), "2025-26")
    big = int(data.advanced["PLAYER_ID"].iloc[0])
    data.catch_shoot = data.catch_shoot[data.catch_shoot["PLAYER_ID"] != big]
    data.hustle = data.hustle[data.hustle["PLAYER_ID"] != big]
    from rosterfit.sources.darko import load_impact
    impact, _ = load_impact(cfg, {"2025-26": data.gamelog[["PLAYER_ID", "PLAYER_NAME"]]})
    players = build_player_table(data, impact, cfg).set_index("PLAYER_ID")
    assert players.loc[big, "port_cs3_proficiency"] == 0
    assert players.loc[big, "port_screen_ast_per100"] == 0
    assert players["port_cs3_proficiency"].notna().sum() == len(players.dropna(subset=["POSS"]))
