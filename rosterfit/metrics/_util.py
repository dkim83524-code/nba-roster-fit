from __future__ import annotations

import numpy as np
import pandas as pd


def per100(count: pd.Series, poss: pd.Series) -> pd.Series:
    return (100.0 * count.astype(float) / poss.astype(float)).where(poss > 0)


def first_column(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def pick(players: pd.DataFrame, table: pd.DataFrame, column: str,
         id_col: str = "PLAYER_ID", how: str = "sum") -> pd.Series:
    """Per-player value of `column` from a league table, aligned with `players`.

    how="sum" adds up any per-team rows (counting stats); how="first" is for rates.
    """
    grouped = table.groupby(id_col)[column]
    values = grouped.sum(min_count=1) if how == "sum" else grouped.first()
    return pd.Series(players["PLAYER_ID"].map(values).to_numpy(dtype=float), index=players.index)


def entropy_share(shares: np.ndarray) -> float:
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum())
