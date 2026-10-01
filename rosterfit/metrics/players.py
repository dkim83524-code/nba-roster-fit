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
from .pool import composite, percentile_against, qualified_mask, zscore
from .portability import portability_components

# Which cached table each weighted component depends on.
_COMPONENT_TABLE = {
    "potential_ast_per100": "passing",
    "ast_pts_created_per100": "passing",
    "cs3_proficiency": "catch_shoot",
    "screen_ast_per100": "hustle",
    "versatility": "matchups",
}


def required_tables(cfg: Config) -> list[str]:
    needed = {"gamelog", "advanced"}
    for weights in (cfg.playmaking.weights, cfg.portability.weights):
        needed |= {_COMPONENT_TABLE[k] for k, w in weights.items() if w > 0 and k in _COMPONENT_TABLE}
    return sorted(needed)


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


def player_totals(gamelog: pd.DataFrame) -> pd.DataFrame:
    """Season totals per player across all their teams."""
    g = gamelog.assign(MIN=minutes(gamelog["MIN"]))
    g = g[g["MIN"] > 0]
    agg = g.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "last"),
        GP=("MIN", "size"),
        MIN=("MIN", "sum"),
        AST=("AST", "sum"),
        PTS=("PTS", "sum"),
        TOV=("TOV", "sum"),
        FG3A=("FG3A", "sum"),
        FG3M=("FG3M", "sum"),
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


def build_player_table(data: SeasonData, impact: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    players = player_totals(data.gamelog)
    if data.advanced is not None and len(data.advanced):
        players["POSS"] = pick(players, data.advanced, "POSS")
    else:
        players["POSS"] = np.nan
    pool = qualified_mask(players["GP"], players["MPG"], cfg.qualified.min_gp, cfg.qualified.min_mpg)
    players["qualified"] = pool

    od = offense_defense(players, impact, data.season)
    pm = playmaking_components(players, data.passing)
    port = portability_components(
        players, data.advanced, data.catch_shoot, data.hustle, data.matchups,
        cfg.portability.cs3_pct_prior_attempts, cfg.portability.versatility_min_partial_poss)

    raw = {
        "offense": od["offense"],
        "defense": od["defense"],
        "playmaking": composite(pm, cfg.playmaking.weights, pool, PM_NEGATIVE,
                                cfg.playmaking.min_weight_present),
        "portability": composite(port, cfg.portability.weights, pool, (),
                                 cfg.portability.min_weight_present),
    }
    for corner in CORNERS:
        values = raw[corner].astype(float)
        players[f"{corner}_raw"] = values
        players[f"{corner}_z"] = zscore(values, pool)
        players[f"{corner}_pct"] = percentile_against(values, values[pool])
    for name in pm.columns:
        players[f"pm_{name}"] = pm[name]
    for name in port.columns:
        players[f"port_{name}"] = port[name]
    players.insert(0, "season", data.season)
    return players
