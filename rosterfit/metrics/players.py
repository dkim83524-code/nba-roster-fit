"""Build the per-season player table: raw value, z-score and percentile for every corner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cache import Cache
from ..config import CORNERS, Config
from ..seasons import PLAYOFFS, REGULAR
from ._util import pick
from .offense_defense import offense_defense
from .playmaking import NEGATIVE as PM_NEGATIVE
from .playmaking import playmaking_components
from .pool import composite, parts_sum, percentile_against, qualified_mask, zscore
from .portability import portability_components
from .skills import SKILL_TABLES, skill_scores

# Which cached table each weighted component depends on.
_COMPONENT_TABLE = {
    "potential_ast_per100": "passing",
    "ast_pts_created_per100": "passing",
    "cs3_proficiency": "catch_shoot",
    "three_volume": "catch_shoot",
    "three_accuracy": "catch_shoot",
    "screen_ast_per100": "hustle",
    "versatility": "matchups",
}


def required_tables(cfg: Config) -> list[str]:
    """Tables the four-corner square needs; a season missing one is left out of the team pool."""
    needed = {"gamelog", "advanced"}
    for weights in (cfg.playmaking.weights, cfg.portability.component_weights()):
        needed |= {_COMPONENT_TABLE[k] for k, w in weights.items() if w > 0 and k in _COMPONENT_TABLE}
    return sorted(needed)


def skill_tables(cfg: Config) -> list[str]:
    """Extra tables only the 3D view uses. Optional: without them the 3D view has gaps, nothing else changes."""
    if not cfg.skills.enabled:
        return []
    used = {c for weights in cfg.skills.axes.values() for c, w in weights.items() if w > 0}
    return sorted({t for c, t in SKILL_TABLES.items() if c in used} - set(required_tables(cfg)))


def download_tables(cfg: Config) -> list[str]:
    """Everything to fetch or download: the square's tables plus the 3D view's."""
    return sorted(set(required_tables(cfg)) | set(skill_tables(cfg)))


@dataclass
class SeasonData:
    season: str
    gamelog: pd.DataFrame
    gamelog_playoffs: pd.DataFrame | None = None
    advanced: pd.DataFrame | None = None
    passing: pd.DataFrame | None = None
    catch_shoot: pd.DataFrame | None = None
    hustle: pd.DataFrame | None = None
    matchups: pd.DataFrame | None = None
    scoring: pd.DataFrame | None = None
    drives: pd.DataFrame | None = None
    rim_defense: pd.DataFrame | None = None

    @property
    def regular_season_end(self) -> pd.Timestamp | None:
        if "GAME_DATE" not in self.gamelog.columns or self.gamelog.empty:
            return None
        return pd.to_datetime(self.gamelog["GAME_DATE"]).max()


def load_season(cache: Cache, season: str) -> SeasonData | None:
    gamelog = cache.read_or_none(season, REGULAR, "gamelog")
    if gamelog is None:
        return None
    return SeasonData(
        season=season,
        gamelog=gamelog,
        gamelog_playoffs=cache.read_or_none(season, PLAYOFFS, "gamelog"),
        advanced=cache.read_or_none(season, REGULAR, "advanced"),
        passing=cache.read_or_none(season, REGULAR, "passing"),
        catch_shoot=cache.read_or_none(season, REGULAR, "catch_shoot"),
        hustle=cache.read_or_none(season, REGULAR, "hustle"),
        matchups=cache.read_or_none(season, REGULAR, "matchups"),
        scoring=cache.read_or_none(season, REGULAR, "scoring"),
        drives=cache.read_or_none(season, REGULAR, "drives"),
        rim_defense=cache.read_or_none(season, REGULAR, "rim_defense"),
    )


def missing_tables(data: SeasonData, cfg: Config) -> list[str]:
    return [t for t in required_tables(cfg) if t != "gamelog" and getattr(data, t) is None]


def minutes(series: pd.Series) -> pd.Series:
    """Game-log minutes as float; accepts numbers or 'MM:SS' strings."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).fillna(0.0)

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        s = str(v)
        if ":" in s:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60.0
        return float(s) if s.strip() else 0.0
    return series.map(one).astype(float)


# box-score columns the 3D view uses; added when the game log has them
_OPTIONAL_BOX = ("FGM", "FTA", "STL", "BLK")


def player_totals(gamelog: pd.DataFrame) -> pd.DataFrame:
    """Season totals per player across all their teams."""
    g = gamelog.assign(MIN=minutes(gamelog["MIN"]))
    g = g[g["MIN"] > 0]
    sums = {c: (c, "sum") for c in ("AST", "PTS", "TOV", "FG3A", "FG3M")}
    sums.update({c: (c, "sum") for c in _OPTIONAL_BOX if c in g.columns})
    agg = g.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "last"),
        GP=("MIN", "size"),
        MIN=("MIN", "sum"),
        **sums,
    ).reset_index()
    agg["MPG"] = agg["MIN"] / agg["GP"]
    return agg


def team_minutes(gamelog: pd.DataFrame) -> pd.DataFrame:
    """Minutes and games per (team, player): a traded player counts only their time with each team."""
    if gamelog is None or gamelog.empty:
        return pd.DataFrame(columns=["TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN", "GP"])
    g = gamelog.assign(MIN=minutes(gamelog["MIN"]))
    g = g[g["MIN"] > 0]
    return (g.groupby(["TEAM_ABBREVIATION", "PLAYER_ID"])
             .agg(PLAYER_NAME=("PLAYER_NAME", "last"), MIN=("MIN", "sum"), GP=("MIN", "size"))
             .reset_index())


def build_player_table(data: SeasonData, impact: dict[str, pd.DataFrame] | pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """impact: source name -> that source's table from sources.darko.load_impacts (or one table:
    the main source)."""
    players = player_totals(data.gamelog)
    if data.advanced is not None and len(data.advanced):
        players["POSS"] = pick(players, data.advanced, "POSS")
    else:
        players["POSS"] = np.nan
    pool = qualified_mask(players["GP"], players["MPG"], cfg.qualified.min_gp, cfg.qualified.min_mpg)
    players["qualified"] = pool

    od = offense_defense(players, impact, data.season, cfg, pool)
    pm = playmaking_components(players, data.passing)
    port = portability_components(
        players, data.advanced, data.catch_shoot, data.hustle, data.matchups,
        cfg.portability.cs3_pct_prior_attempts, cfg.portability.versatility_min_partial_poss,
        cfg.portability.catch_shoot_weight)
    port_cfg = cfg.portability
    if port_cfg.combine == "parts":
        portability = parts_sum(port, port_cfg.parts, pool, port_cfg.min_weight_present, port_cfg.component_scaling)
    else:
        portability = composite(port, port_cfg.weights, pool, (), port_cfg.min_weight_present,
                                port_cfg.component_scaling)

    raw = {
        "offense": od["offense"],
        "defense": od["defense"],
        "playmaking": composite(pm, cfg.playmaking.weights, pool, PM_NEGATIVE,
                                cfg.playmaking.min_weight_present, cfg.playmaking.component_scaling),
        "portability": portability,
    }
    for corner in CORNERS:
        values = raw[corner].astype(float)
        players[f"{corner}_raw"] = values
        players[f"{corner}_z"] = zscore(values, pool)
        players[f"{corner}_pct"] = percentile_against(values, values[pool])
    if cfg.skills.enabled:
        skills = skill_scores(players, data, pm, port, cfg.skills, pool)
        for col in skills.columns:
            players[col] = skills[col]
        # shape size in the 3D view: overall impact, offense z + defense z
        impact = players["offense_z"] + players["defense_z"]
        players["impact_pct"] = percentile_against(impact, impact[pool])
    for col in od.columns:
        if col not in ("offense", "defense"):
            players[col] = od[col]
    for name in pm.columns:
        players[f"pm_{name}"] = pm[name]
    for name in port.columns:
        players[f"port_{name}"] = port[name]
    players.insert(0, "season", data.season)
    return players
