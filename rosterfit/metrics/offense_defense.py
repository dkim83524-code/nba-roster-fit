"""Offense and defense corners: the two halves of one plus-minus metric (DARKO by default)."""
from __future__ import annotations

import pandas as pd


def offense_defense(players: pd.DataFrame, impact: pd.DataFrame, season: str) -> pd.DataFrame:
    """Raw offense/defense values for each player (index aligned with `players`)."""
    imp = impact[impact["season"] == season][["PLAYER_ID", "offense", "defense"]]
    merged = players[["PLAYER_ID"]].merge(imp, on="PLAYER_ID", how="left")
    merged.index = players.index
    return merged[["offense", "defense"]]
