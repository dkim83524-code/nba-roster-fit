"""Portability corner: value a player keeps without the ball.

Components:
  three_volume       3-point attempts per 100 possessions, all 3s, with a catch-and-shoot 3 counting
                     catch_shoot_weight times a pull-up or other 3: off-ball spacing gravity
  three_accuracy     3P% on the same weighted attempts, shrunk toward league average for small samples
  cs3_proficiency    (older formula) Taylor's 3-pt proficiency on catch-and-shoot 3s only: a sigmoid
                     on C&S 3PA per 100 times C&S 3P%; it stops caring about volume past ~4 per 100
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
    # NBA.com only lists players who took a catch-and-shoot shot: absent means none, i.e. 0.
    return prof.where(rate.notna())


def three_point_spacing(players: pd.DataFrame, catch_shoot: pd.DataFrame | None, prior_attempts: float,
                        cs_weight: float) -> tuple[pd.Series, pd.Series]:
    """(volume, accuracy) on all 3s, a catch-and-shoot 3 counting `cs_weight` times any other 3.

    volume: weighted attempts per 100 possessions. accuracy: weighted makes / weighted attempts,
    shrunk toward the league's weighted 3P% by `prior_attempts`.
    """
    att = players["FG3A"].astype(float).fillna(0)
    made = players["FG3M"].astype(float).fillna(0)
    if catch_shoot is not None and len(catch_shoot):
        # NBA.com only lists players who took a catch-and-shoot shot: absent means none
        cs_att = pick(players, catch_shoot, "CATCH_SHOOT_FG3A").fillna(0).clip(upper=att)
        cs_made = pick(players, catch_shoot, "CATCH_SHOOT_FG3M").fillna(0).clip(upper=made)
    else:
        cs_att = cs_made = pd.Series(0.0, index=players.index)
    w_att = att + (cs_weight - 1.0) * cs_att
    w_made = made + (cs_weight - 1.0) * cs_made
    league_pct = w_made.sum() / w_att.sum() if w_att.sum() else 0.36
    accuracy = (w_made + prior_attempts * league_pct) / (w_att + prior_attempts)
    volume = per100(w_att, players["POSS"])
    return volume, accuracy.where(volume.notna())


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
                           min_partial_poss: float, cs_weight: float = 1.5) -> pd.DataFrame:
    out = pd.DataFrame(index=players.index)
    nan = pd.Series(np.nan, index=players.index)
    out["three_volume"], out["three_accuracy"] = three_point_spacing(players, catch_shoot, prior_attempts, cs_weight)
    out["cs3_proficiency"] = (cs3_proficiency(players, catch_shoot, prior_attempts)
                              if catch_shoot is not None and len(catch_shoot) else nan)
    out["oreb_pct"] = (pick(players, advanced, "OREB_PCT", how="first")
                       if advanced is not None and len(advanced) else nan)
    out["screen_ast_per100"] = (per100(pick(players, hustle, "SCREEN_ASSISTS").fillna(0), players["POSS"])
                                if hustle is not None and len(hustle) else nan)
    if matchups is not None and len(matchups):
        vers = versatility(matchups, min_partial_poss)
        out["versatility"] = players["PLAYER_ID"].map(vers).to_numpy(dtype=float)
    else:
        out["versatility"] = nan
    return out
