"""Local parquet cache, one file per (season, season type, table).

Layout: <cache_dir>/<season>/<regular|playoffs>/<table>.parquet
Completed seasons never change, so nothing here expires; use `fetch --refresh` to redownload.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .seasons import PLAYOFFS, REGULAR

_SLUG = {REGULAR: "regular", PLAYOFFS: "playoffs"}


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, season: str, season_type: str, table: str) -> Path:
        return self.root / season / _SLUG[season_type] / f"{table}.parquet"

    def has(self, season: str, season_type: str, table: str) -> bool:
        return self.path(season, season_type, table).exists()

    def read(self, season: str, season_type: str, table: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(season, season_type, table))

    def read_or_none(self, season: str, season_type: str, table: str) -> pd.DataFrame | None:
        return self.read(season, season_type, table) if self.has(season, season_type, table) else None

    def write(self, df: pd.DataFrame, season: str, season_type: str, table: str) -> Path:
        path = self.path(season, season_type, table)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path
