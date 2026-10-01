import numpy as np
import pandas as pd
import pytest
import yaml

from rosterfit.cache import Cache
from rosterfit.config import load_config
from rosterfit.metrics.offense_defense import offense_defense
from rosterfit.metrics.pool import percentile_against, zscore
from rosterfit.sources.darko import load_impact
from rosterfit.team import build_league

from .conftest import ROOT, SEASONS

ROSTER = {"2025-26": pd.DataFrame({"PLAYER_ID": [1, 2, 3, 4],
                                   "PLAYER_NAME": ["Nikola Jokić", "Stephon Castle", "Rudy Gobert", "Rookie"]})}


def write_lebron(cfg, rows, name="lebron-data-2026.csv"):
    folder = cfg.path(cfg.impact.extra_sources["LEBRON"].dir)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / name, index=False)


def test_reads_lebron_export_by_id_with_end_year_and_roles(cfg):
    write_lebron(cfg, {"nba_id": [1, 2], "Player": ["Nikola Jokic", "Stephon Castle"], "Season": [2026, 2026],
                       "Team": ["DEN", "SAS"], "LEBRON": [9.0, 1.0], "O-LEBRON": [7.5, 0.2], "D-LEBRON": [1.5, 0.8],
                       "OffRole": ["Shot Creator", "Primary Ball Handler"], "DefRole": ["Helper", "Point of Attack"]})
    impact, report = load_impact(cfg, ROSTER, source="LEBRON")
    row = impact.set_index("PLAYER_ID").loc[2]
    assert row["season"] == "2025-26"                  # 2026 is the year 2025-26 ends
    assert (row["offense"], row["defense"]) == (0.2, 0.8)
    assert (row["off_role"], row["def_role"]) == ("Primary Ball Handler", "Point of Attack")
    assert report.name == "LEBRON" and report.seasons == ["2025-26"]


def _players(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"PLAYER_ID": np.arange(1, n + 1)}), pd.Series(True, index=range(n)), rng


def test_blend_weights_sources_and_falls_back_to_what_a_player_has(cfg):
    players, pool, rng = _players()
    ids = players["PLAYER_ID"]
    darko = pd.DataFrame({"season": "2025-26", "PLAYER_ID": ids, "offense": rng.normal(0, 2, 40),
                          "defense": rng.normal(0, 1, 40), "off_role": None, "def_role": None})
    lebron = darko.assign(offense=rng.normal(0, 3, 40), defense=rng.normal(0, 1.5, 40),
                          off_role="Connector", def_role="Helper")
    lebron = lebron[lebron["PLAYER_ID"] != 40]         # player 40 has DARKO only
    out = offense_defense(players, {"DARKO": darko, "LEBRON": lebron}, "2025-26", cfg, pool)

    zl = zscore(players["PLAYER_ID"].map(lebron.set_index("PLAYER_ID")["defense"]), pool)
    zd = zscore(players["PLAYER_ID"].map(darko.set_index("PLAYER_ID")["defense"]), pool)
    assert out.loc[0, "defense"] == pytest.approx((2 * zl[0] + 0.5 * zd[0]) / 2.5)
    assert out.loc[39, "defense"] == pytest.approx(zd[39])   # DARKO alone is 20% of the weight: enough
    assert out.loc[0, "def_role"] == "Helper" and pd.isna(out.loc[39, "def_role"])
    assert out.loc[0, "defense_src_LEBRON"] == lebron.iloc[0]["defense"]

    cfg.impact.min_weight_present = 0.5                # now DARKO alone isn't enough
    out = offense_defense(players, {"DARKO": darko, "LEBRON": lebron}, "2025-26", cfg, pool)
    assert np.isnan(out.loc[39, "defense"])


def test_main_source_alone_gives_the_same_percentiles_as_before(cfg):
    players, pool, rng = _players()
    darko = pd.DataFrame({"season": "2025-26", "PLAYER_ID": players["PLAYER_ID"],
                          "offense": rng.normal(0, 2, 40), "defense": rng.normal(0, 1, 40)})
    cfg.impact.weights = {}
    out = offense_defense(players, darko, "2025-26", cfg, pool)
    raw = darko["defense"]
    assert np.allclose(percentile_against(out["defense"], out["defense"]), percentile_against(raw, raw))


def test_config_rejects_unknown_sources_and_corners(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["impact"]["weights"]["defense"] = {"EPM": 1.0}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown source"):
        load_config(path)
    raw["impact"]["weights"] = {"playmaking": {"DARKO": 1.0}}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="isn't offense or defense"):
        load_config(path)


def test_season_without_a_weighted_source_leaves_the_pool(cfg, synthetic_dirs):
    from rosterfit.synthetic import use_synthetic_sources

    cache_dir, impact_dir = synthetic_dirs
    use_synthetic_sources(cfg, impact_dir)
    cfg.seasons.window = [SEASONS[0], SEASONS[-1]]
    cfg.qualified.min_gp = 10
    lebron_dir = impact_dir.parent / "lebron"
    keep = sorted(lebron_dir.glob("*.csv"))[1:]          # drop the first season's LEBRON file
    cfg.impact.extra_sources["LEBRON"].dir = str(cfg.path("lebron_subset"))
    cfg.path("lebron_subset").mkdir()
    for f in keep:
        (cfg.path("lebron_subset") / f.name).write_text(f.read_text())
    league = build_league(cfg, Cache(cache_dir), SEASONS)
    assert league.incomplete == {SEASONS[0]: ["LEBRON CSV"]}
    assert league.pool_seasons == SEASONS[1:]


def test_league_players_carry_sources_and_roles(league):
    p = league.players["2025-26"]
    q = p[p["qualified"]]
    assert q["defense_src_LEBRON"].notna().all() and q["defense_src_DARKO"].notna().all()
    assert set(q["def_role"]) <= {"Point of Attack", "Chaser", "Wing Stopper", "Anchor Big", "Helper", "Low Activity"}
    assert q["defense_pct"].between(0, 100).all()


def test_explain_shows_both_sources_for_defense(league):
    from rosterfit.explain import ingredients

    table = ingredients(league.players["2025-26"], league.minutes["regular"]["2025-26"], "DEM", "defense",
                        league.cfg, top=5)
    assert list(table.columns) == ["PLAYER_NAME", "MIN", "LEBRON", "DARKO", "corner"]
    assert len(table) == 5
