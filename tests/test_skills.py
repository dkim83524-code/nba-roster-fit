import numpy as np
import pandas as pd
import pytest

from rosterfit.metrics.players import download_tables, required_tables, skill_tables
from rosterfit.metrics.skills import RIM_PRIOR_ATTEMPTS, rim_points_saved, skill_components
from rosterfit.seasons import PLAYOFFS, REGULAR
from rosterfit.sources.manual_nba import identify, request_url


def test_three_d_tables_are_optional_downloads(cfg):
    core = required_tables(cfg)
    assert not {"scoring", "drives", "rim_defense"} & set(core)  # a missing one never drops a season
    assert skill_tables(cfg) == ["drives", "rim_defense", "scoring"]
    assert set(download_tables(cfg)) == set(core) | {"drives", "rim_defense", "scoring"}
    cfg.skills.enabled = False
    assert skill_tables(cfg) == []


def test_rim_points_saved_shrinks_small_samples():
    players = pd.DataFrame({"PLAYER_ID": [1, 2, 3]})
    rim = pd.DataFrame({"CLOSE_DEF_PERSON_ID": [1, 2], "D_FGA": [400.0, 20.0], "D_FGM": [220.0, 8.0],
                        "NORMAL_FG_PCT": [0.62, 0.62]})
    saved = rim_points_saved(players, rim)
    # 2 x (0.62 x 400 - 220) = 56 points, x 400 / 500
    assert saved[0] == pytest.approx(56 * 400 / (400 + RIM_PRIOR_ATTEMPTS))
    assert saved[1] == pytest.approx(2 * (0.62 * 20 - 8) * 20 / (20 + RIM_PRIOR_ATTEMPTS))
    assert saved[2] == 0  # never defended a shot at the rim


def test_skills_on_the_synthetic_league(league, cfg):
    p = league.players["2025-26"]
    q = p[p["qualified"]]
    for skill in cfg.skills.order:
        pct = q[f"skill_{skill}_pct"]
        assert pct.notna().all() and pct.between(0, 100).all() and pct.max() == 100
    assert q["impact_pct"].between(0, 100).all()
    # the made-up bigs protect the rim and the made-up creators create
    corr = q[[f"skill_{s}_raw" for s in cfg.skills.order]].corr()
    assert corr.loc["skill_screening_raw", "skill_rim_protection_raw"] > 0.2
    assert corr.loc["skill_creation_raw", "skill_rim_protection_raw"] < 0


def test_missing_table_leaves_its_components_empty(league, cfg):
    class Season:  # only the game-log stats
        scoring = drives = hustle = rim_defense = None
    players = pd.DataFrame({"PLAYER_ID": [1, 2], "POSS": [1000.0, 0.0], "FGM": [300, 0], "FTA": [100, 0],
                            "STL": [30, 0], "BLK": [10, 0]})
    blank = pd.DataFrame(np.nan, index=players.index, columns=["potential_ast_per100", "ast_pts_created_per100",
                                                                "three_volume", "three_accuracy", "oreb_pct",
                                                                "screen_ast_per100"])
    comps = skill_components(players, Season(), blank, blank)
    assert comps["unassisted_fgm_per100"].isna().all() and comps["rim_contests_per100"].isna().all()
    assert comps.loc[0, "fta_per100"] == pytest.approx(10.0)
    assert np.isnan(comps.loc[1, "stl_per100"])  # no possessions: no rate


def test_new_tables_are_recognized():
    def resp(resource, params):
        return {"resource": resource, "parameters": {"Season": "2025-26", "SeasonType": "Playoffs", **params},
                "resultSets": [{"headers": ["X"], "rowSet": []}]}
    assert identify(resp("leaguedashplayerstats", {"MeasureType": "Scoring"}))[:3] == ("scoring", "2025-26", PLAYOFFS)
    assert identify(resp("leaguedashptstats", {"PtMeasureType": "Drives"}))[0] == "drives"
    assert identify(resp("leaguedashptdefend", {"DefenseCategory": "Less Than 6Ft"}))[0] == "rim_defense"
    url = request_url("rim_defense", "2025-26", REGULAR)
    assert "/stats/leaguedashptdefend?" in url and "DefenseCategory=Less+Than+6Ft" in url
    assert "LastNGames=0" in url and "PerMode=Totals" in url


def test_rim_table_reads_the_distance_category_names():
    from rosterfit.metrics.skills import rim_table
    overall = pd.DataFrame({"CLOSE_DEF_PERSON_ID": [1], "D_FGM": [50], "D_FGA": [100], "NORMAL_FG_PCT": [0.6]})
    lt6 = pd.DataFrame({"CLOSE_DEF_PERSON_ID": [1], "FGM_LT_06": [50], "FGA_LT_06": [100], "NS_LT_06_PCT": [0.6]})
    no_normal = pd.DataFrame({"CLOSE_DEF_PERSON_ID": [1], "FGM_LT_06": [50], "FGA_LT_06": [100],
                              "LT_06_PCT": [0.5], "PLUSMINUS": [-0.1]})
    for df in (overall, lt6, no_normal):
        t = rim_table(df)
        assert list(t.iloc[0]) == pytest.approx([1, 50, 100, 0.6])
    assert rim_table(pd.DataFrame({"CLOSE_DEF_PERSON_ID": [1], "WHO_KNOWS": [3]})) is None
    assert rim_table(None) is None
