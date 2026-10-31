"""The 3D view's six skills, each a league percentile among qualified players.

Components, per 100 possessions unless noted (NBA.com tables; a player who played but isn't in a
tracking table had none of those events, so he counts 0; a table that isn't downloaded leaves the
component empty):
  creation           unassisted_fgm_per100   made shots nobody assisted (FGM x scoring's PCT_UAST_FGM)
                     drives_per100           drives to the basket (drives tracking)
                     fta_per100              free-throw attempts (game log)
  playmaking         potential_ast_per100, ast_pts_created_per100   (same as the playmaking corner)
  spacing            three_volume, three_accuracy                   (same as "Fits next to stars")
  screening          oreb_pct, screen_ast_per100                    (same as "Fits next to stars")
  perimeter_defense  stl_per100              steals (game log)
                     deflections_per100      deflections (hustle)
                     contested3_per100       contested 3-point shots (hustle)
  rim_protection     rim_points_saved_per100 2 x (expected makes - makes) on shots he defended
                                             within 6 ft, expected from the shooters' normal FG%,
                                             shrunk toward 0 for small samples
                     rim_contests_per100     shots defended within 6 ft
                     blk_per100              blocks (game log)

Each skill is the weighted mean of its rank-standardized components (metrics.pool.composite).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._util import per100, pick
from .pool import composite, percentile_against

# which table each component comes from (game-log ones need no extra download)
SKILL_TABLES = {
    "unassisted_fgm_per100": "scoring",
    "drives_per100": "drives",
    "potential_ast_per100": "passing",
    "ast_pts_created_per100": "passing",
    "three_volume": "catch_shoot",
    "three_accuracy": "catch_shoot",
    "oreb_pct": "advanced",
    "screen_ast_per100": "hustle",
    "deflections_per100": "hustle",
    "contested3_per100": "hustle",
    "rim_points_saved_per100": "rim_defense",
    "rim_contests_per100": "rim_defense",
}
# Rim points saved is (normal FG% - FG% allowed) x attempts; with few attempts that gap is mostly
# luck, so it's multiplied by attempts / (attempts + this).
RIM_PRIOR_ATTEMPTS = 100.0
RIM_ID = "CLOSE_DEF_PERSON_ID"
# NBA.com names the columns by defense category: "Overall" sends D_FGM / D_FGA / NORMAL_FG_PCT, the
# distance categories send their own (FGM_LT_06 ... for "Less Than 6Ft"). Accept either.
_RIM_ALIASES = {
    RIM_ID: ("CLOSE_DEF_PERSON_ID", "PLAYER_ID"),
    "D_FGM": ("D_FGM", "FGM_LT_06"),
    "D_FGA": ("D_FGA", "FGA_LT_06"),
    "NORMAL_FG_PCT": ("NORMAL_FG_PCT", "NS_LT_06_PCT"),
    "allowed_pct": ("D_FG_PCT", "LT_06_PCT"),       # FG% allowed
    "plusminus": ("PCT_PLUSMINUS", "PLUSMINUS"),     # allowed - normal
}
log = logging.getLogger(__name__)
_warned: set[tuple[str, ...]] = set()


def rim_table(rim: pd.DataFrame | None) -> pd.DataFrame | None:
    """The rim-defense table with standard columns (CLOSE_DEF_PERSON_ID, D_FGM, D_FGA,
    NORMAL_FG_PCT), whichever names NBA.com used; None when it's missing or unreadable."""
    if rim is None or rim.empty:
        return None
    found = {std: next((c for c in names if c in rim.columns), None) for std, names in _RIM_ALIASES.items()}
    out = pd.DataFrame({std: rim[col] for std, col in found.items() if col and std in (RIM_ID, "D_FGM", "D_FGA")})
    if found["NORMAL_FG_PCT"]:
        out["NORMAL_FG_PCT"] = rim[found["NORMAL_FG_PCT"]]
    elif found["allowed_pct"] and found["plusminus"]:
        out["NORMAL_FG_PCT"] = rim[found["allowed_pct"]].astype(float) - rim[found["plusminus"]].astype(float)
    needed = (RIM_ID, "D_FGM", "D_FGA", "NORMAL_FG_PCT")
    if all(c in out.columns for c in needed):
        return out
    key = tuple(rim.columns)
    if key not in _warned:  # once per layout, not once per season
        _warned.add(key)
        log.warning("rim_defense: can't find shots defended / made / normal FG%% in its columns %s; "
                    "Protects the rim stays empty. Please send these column names to fix the mapping.",
                    list(rim.columns))
    return None


def _present(table: pd.DataFrame | None, *columns: str) -> bool:
    return table is not None and len(table) > 0 and all(c in table.columns for c in columns)


def _box_rate(players: pd.DataFrame, column: str) -> pd.Series:
    if column not in players.columns:
        return pd.Series(np.nan, index=players.index)
    return per100(players[column].fillna(0), players["POSS"])


def _tracked_rate(players: pd.DataFrame, table: pd.DataFrame | None, column: str,
                  id_col: str = "PLAYER_ID") -> pd.Series:
    if not _present(table, id_col, column):
        return pd.Series(np.nan, index=players.index)
    return per100(pick(players, table, column, id_col=id_col).fillna(0), players["POSS"])


def rim_points_saved(players: pd.DataFrame, rim: pd.DataFrame) -> pd.Series:
    """Points saved at the rim per player (not per 100), shrunk for small samples."""
    att = rim["D_FGA"].astype(float)
    saved = 2.0 * (rim["NORMAL_FG_PCT"].astype(float) * att - rim["D_FGM"].astype(float))
    by = pd.DataFrame({RIM_ID: rim[RIM_ID], "saved": saved, "att": att}).groupby(RIM_ID).sum()
    shrunk = by["saved"] * by["att"] / (by["att"] + RIM_PRIOR_ATTEMPTS)
    return pd.Series(players["PLAYER_ID"].map(shrunk).fillna(0).to_numpy(dtype=float), index=players.index)


def skill_components(players: pd.DataFrame, data, playmaking: pd.DataFrame,
                     portability: pd.DataFrame) -> pd.DataFrame:
    """data: a players.SeasonData. playmaking / portability: those corners' component tables."""
    out = pd.DataFrame(index=players.index)
    nan = pd.Series(np.nan, index=players.index)
    if _present(data.scoring, "PLAYER_ID", "PCT_UAST_FGM") and "FGM" in players.columns:
        share = pick(players, data.scoring, "PCT_UAST_FGM", how="first")
        out["unassisted_fgm_per100"] = per100(players["FGM"].fillna(0) * share.fillna(0), players["POSS"])
    else:
        out["unassisted_fgm_per100"] = nan
    out["drives_per100"] = _tracked_rate(players, data.drives, "DRIVES")
    out["fta_per100"] = _box_rate(players, "FTA")
    for name in ("potential_ast_per100", "ast_pts_created_per100"):
        out[name] = playmaking[name]
    for name in ("three_volume", "three_accuracy", "oreb_pct", "screen_ast_per100"):
        out[name] = portability[name]
    out["stl_per100"] = _box_rate(players, "STL")
    out["deflections_per100"] = _tracked_rate(players, data.hustle, "DEFLECTIONS")
    out["contested3_per100"] = _tracked_rate(players, data.hustle, "CONTESTED_SHOTS_3PT")
    rim = rim_table(data.rim_defense)
    if rim is not None:
        out["rim_points_saved_per100"] = per100(rim_points_saved(players, rim), players["POSS"])
        out["rim_contests_per100"] = _tracked_rate(players, rim, "D_FGA", id_col=RIM_ID)
    else:
        out["rim_points_saved_per100"] = nan
        out["rim_contests_per100"] = nan
    out["blk_per100"] = _box_rate(players, "BLK")
    return out


def skill_scores(players: pd.DataFrame, data, playmaking: pd.DataFrame, portability: pd.DataFrame,
                 cfg, pool: pd.Series) -> pd.DataFrame:
    """Columns skill_<name>_raw and skill_<name>_pct for each configured skill, plus sk_<component>.
    cfg: a config.SkillsCfg."""
    comps = skill_components(players, data, playmaking, portability)
    out = pd.DataFrame(index=players.index)
    for skill in cfg.order:
        score = composite(comps, cfg.axes[skill], pool, (), cfg.min_weight_present, "rank")
        out[f"skill_{skill}_raw"] = score
        out[f"skill_{skill}_pct"] = percentile_against(score, score[pool])
    for name in comps.columns:
        out[f"sk_{name}"] = comps[name]
    return out
