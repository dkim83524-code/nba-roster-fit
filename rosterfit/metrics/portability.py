"""Portability corner: value a player keeps without the ball.

Components:
  cs3_proficiency    Taylor's 3-pt proficiency applied to catch-and-shoot 3s: a sigmoid on C&S 3PA
                     per 100 times C&S 3P% (shrunk toward league average for small samples)
  oreb_pct           offensive rebound percentage
  screen_ast_per100  screen assists per 100 possessions
  versatility        how evenly the player's defensive matchup possessions spread across
                     guards, forwards and centers (normalized entropy: 0 = one position, 1 = even)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..sources.formulas import three_point_proficiency
from ._util import entropy_share, per100, pick

POSITIONS = ("G", "F", "C")


def cs3_proficiency(players: pd.DataFrame, catch_shoot: pd.DataFrame, prior_attempts: float) -> pd.Series:
    made = pick(players, catch_shoot, "CATCH_SHOOT_FG3M")
    att = pick(players, catch_shoot, "CATCH_SHOOT_FG3A")
    total_att = catch_shoot["CATCH_SHOOT_FG3A"].sum()
    league_pct = catch_shoot["CATCH_SHOOT_FG3M"].sum() / total_att if total_att else 0.36
    shrunk = (made.fillna(0) + prior_attempts * league_pct) / (att.fillna(0) + prior_attempts)
    rate = per100(att.fillna(0), players["POSS"])
    prof = pd.Series(three_point_proficiency(rate, shrunk), index=players.index)
    return prof.where(rate.notna() & att.notna())


def versatility(matchups: pd.DataFrame, min_partial_poss: float) -> pd.Series:
    """Normalized entropy of matchup possessions by offensive position, indexed by defender id."""
    df = matchups.copy()
    df["pos"] = df["POSITION"].astype(str).str.strip().str[:1].str.upper()
    df = df[df["pos"].isin(POSITIONS)]
    weight_col = "PARTIAL_POSS" if "PARTIAL_POSS" in df.columns else "PERCENT_OF_TIME"
    by = df.groupby(["DEF_PLAYER_ID", "pos"])[weight_col].sum().unstack(fill_value=0.0)
    by = by.reindex(columns=list(POSITIONS), fill_value=0.0)
    totals = by.sum(axis=1)
    shares = by.div(totals.where(totals > 0), axis=0)
    ent = shares.apply(lambda r: entropy_share(r.to_numpy(dtype=float)), axis=1) / math.log(len(POSITIONS))
    if weight_col == "PARTIAL_POSS":
        ent = ent.where(totals >= min_partial_poss)
    return ent


def portability_components(players: pd.DataFrame, advanced: pd.DataFrame | None,
                           catch_shoot: pd.DataFrame | None, hustle: pd.DataFrame | None,
                           matchups: pd.DataFrame | None, prior_attempts: float,
                           min_partial_poss: float) -> pd.DataFrame:
    out = pd.DataFrame(index=players.index)
    nan = pd.Series(np.nan, index=players.index)
    out["cs3_proficiency"] = (cs3_proficiency(players, catch_shoot, prior_attempts)
                              if catch_shoot is not None and len(catch_shoot) else nan)
    out["oreb_pct"] = (pick(players, advanced, "OREB_PCT", how="first")
                       if advanced is not None and len(advanced) else nan)
    out["screen_ast_per100"] = (per100(pick(players, hustle, "SCREEN_ASSISTS"), players["POSS"])
                                if hustle is not None and len(hustle) else nan)
    if matchups is not None and len(matchups):
        vers = versatility(matchups, min_partial_poss)
        out["versatility"] = players["PLAYER_ID"].map(vers).to_numpy(dtype=float)
    else:
        out["versatility"] = nan
    return out
