"""Playmaking corner: creating shots for teammates (self-scoring belongs in offense).

Components, per 100 possessions:
  potential_ast_per100    passes that lead to a shot attempt that would be an assist if made
  ast_pts_created_per100  points produced by the player's assists
  turnovers_per100        optional penalty (weight 0 by default)
  box_creation            Ben Taylor's Box Creation (weight 0 by default; includes scorer gravity)
"""
from __future__ import annotations

import pandas as pd

from ..sources.formulas import box_creation
from ._util import first_column, per100, pick

NEGATIVE = ("turnovers_per100",)


def playmaking_components(players: pd.DataFrame, passing: pd.DataFrame | None) -> pd.DataFrame:
    """players must carry PLAYER_ID, POSS and box totals (AST, PTS, TOV, FG3A, FG3M)."""
    out = pd.DataFrame(index=players.index)
    poss = players["POSS"]
    if passing is not None and len(passing):
        pot = pick(players, passing, "POTENTIAL_AST")
        created = pick(players, passing, first_column(passing, ["AST_PTS_CREATED", "AST_POINTS_CREATED"]))
        out["potential_ast_per100"] = per100(pot, poss)
        out["ast_pts_created_per100"] = per100(created, poss)
    else:
        out["potential_ast_per100"] = float("nan")
        out["ast_pts_created_per100"] = float("nan")
    out["turnovers_per100"] = per100(players["TOV"], poss)
    fg3_pct = (players["FG3M"] / players["FG3A"]).where(players["FG3A"] > 0, 0.0)
    out["box_creation"] = box_creation(
        per100(players["AST"], poss), per100(players["PTS"], poss), per100(players["TOV"], poss),
        per100(players["FG3A"], poss), fg3_pct)
    return out
