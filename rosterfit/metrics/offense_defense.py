"""Offense and defense corners: a weighted blend of plus-minus sources (LEBRON and DARKO by default).

Each source is z-scored within the season's qualified pool and the results are averaged with the
weights under impact.weights, renormalized over the sources a player has (see pool.composite).
"""
from __future__ import annotations

import pandas as pd

from ..config import IMPACT_CORNERS, Config
from ..sources.darko import LABELS
from .pool import composite


def offense_defense(players: pd.DataFrame, impacts: dict[str, pd.DataFrame] | pd.DataFrame, season: str,
                    cfg: Config, pool: pd.Series) -> pd.DataFrame:
    """Blended offense/defense for each player (index aligned with `players`), plus each source's
    raw value (`offense_src_LEBRON`, ...) and role/position labels when a source has them."""
    if isinstance(impacts, pd.DataFrame):  # a single table: the main source
        impacts = {cfg.impact.source_name: impacts}
    out = pd.DataFrame(index=players.index)
    for label in LABELS:
        out[label] = None
    sides: dict[str, dict[str, pd.Series]] = {side: {} for side in IMPACT_CORNERS}
    for name, imp in impacts.items():
        rows = imp[imp["season"] == season].drop(columns="season")
        merged = players[["PLAYER_ID"]].merge(rows, on="PLAYER_ID", how="left")
        merged.index = players.index
        for side in IMPACT_CORNERS:
            sides[side][name] = merged[side].astype(float)
            out[f"{side}_src_{name}"] = merged[side].astype(float)
        for label in LABELS:
            if label in merged:
                out[label] = out[label].where(out[label].notna(), merged[label])
    for side in IMPACT_CORNERS:
        weights = cfg.impact.weights_for(side)
        comps = pd.DataFrame({name: sides[side].get(name, pd.Series(float("nan"), index=players.index))
                              for name in weights})
        out[side] = composite(comps, weights, pool, (), cfg.impact.min_weight_present, "z")
    return out
