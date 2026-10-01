"""Break a corner into its ingredients for one team: the stats behind playmaking and
portability, or the plus-minus sources blended into offense and defense.

Shows each player's percentile on every weighted ingredient, among the season's qualified
players, next to the corner percentile those ingredients add up to.
"""
from __future__ import annotations

import pandas as pd

from .config import IMPACT_CORNERS, Config
from .metrics.playmaking import NEGATIVE as PM_NEGATIVE
from .metrics.pool import percentile_against

LABELS = {
    "potential_ast_per100": "Potential AST",
    "ast_pts_created_per100": "AST pts created",
    "turnovers_per100": "Few turnovers",
    "box_creation": "Box Creation",
    "cs3_proficiency": "C&S 3s",
    "oreb_pct": "OREB%",
    "screen_ast_per100": "Screen AST",
    "versatility": "Versatility",
}
_SECTIONS = {"playmaking": ("pm_", PM_NEGATIVE), "portability": ("port_", ()),
             "offense": ("offense_src_", ()), "defense": ("defense_src_", ())}


def corner_weights(cfg: Config, corner: str) -> dict[str, float]:
    if corner in IMPACT_CORNERS:
        return cfg.impact.weights_for(corner)
    return getattr(cfg, corner).weights


def corner_scaling(cfg: Config, corner: str) -> str:
    return "z" if corner in IMPACT_CORNERS else getattr(cfg, corner).component_scaling


def ingredients(players: pd.DataFrame, team_minutes: pd.DataFrame, team: str, corner: str,
                cfg: Config, top: int = 10) -> pd.DataFrame:
    prefix, negative = _SECTIONS[corner]
    weights = corner_weights(cfg, corner)
    active = [k for k, w in weights.items() if w > 0 and prefix + k in players]
    pool = players["qualified"]
    roster = (team_minutes[team_minutes["TEAM_ABBREVIATION"] == team]
              .nlargest(top, "MIN")[["PLAYER_ID", "MIN"]])
    cols = ["PLAYER_ID", "PLAYER_NAME", f"{corner}_pct"] + [prefix + k for k in active]
    out = roster.merge(players[cols], on="PLAYER_ID", how="left")
    table = out[["PLAYER_NAME", "MIN"]].copy()
    for k in active:
        pct = percentile_against(out[prefix + k], players.loc[pool, prefix + k])
        table[LABELS.get(k, k)] = 100 - pct if k in negative else pct
    table["corner"] = out[f"{corner}_pct"]
    return table


def format_table(table: pd.DataFrame, title: str, weights: dict[str, float], scaling: str) -> str:
    cols = [c for c in table.columns if c not in ("PLAYER_NAME", "MIN")]
    widths = {c: max(len(c), 5) + 2 for c in cols}
    lines = [title, f"{'player':<24}{'MIN':>7}" + "".join(f"{c:>{widths[c]}}" for c in cols)]
    for _, row in table.iterrows():
        cells = "".join(f"{'-' if pd.isna(row[c]) else f'{row[c]:.0f}':>{widths[c]}}" for c in cols)
        lines.append(f"{str(row['PLAYER_NAME'])[:23]:<24}{row['MIN']:>7,.0f}{cells}")
    used = ", ".join(f"{LABELS.get(k, k)} {w:g}" for k, w in weights.items() if w > 0)
    lines.append(f"weights: {used}; scaling: {scaling}. 'corner' is the result shown on the chart.")
    return "\n".join(lines)
