"""Offense/defense impact from CSVs you download yourself (DARKO by default).

darko.app offers CSV downloads on its leaderboard and team pages. Save them into
`impact.dir` (default data/manual/darko/). Nothing here scrapes the site.

Accepted layouts, mixed freely:
  * one file per season, season in the file name:   darko_2025-26.csv
  * any file with a season column (2025-26 or 2026)
  * daily/history exports with a date column: the last row on or before the end of the
    regular season is used, i.e. the end-of-regular-season value.

Players are matched on an NBA.com player-id column when one exists, otherwise by name.
Any other plus-minus with offense/defense halves works the same way: point `impact.dir`
at its CSVs and add its column names to `impact.columns`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..seasons import previous, season_from_end_year, season_from_start, start_year
from .names import normalize_column, normalize_name

_FILE_SEASON = re.compile(r"(\d{4}-\d{2})")


@dataclass
class ImpactReport:
    files: list[str] = field(default_factory=list)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)  # file -> role -> column
    seasons: list[str] = field(default_factory=list)
    unmatched: dict[str, list[str]] = field(default_factory=dict)     # season -> names


def _resolve(columns: dict[str, str], aliases: list[str]) -> str | None:
    for alias in aliases:
        key = normalize_column(alias)
        if key in columns:
            return columns[key]
    return None


def _season_values(raw: pd.Series, end_year: bool) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if _FILE_SEASON.fullmatch(s):
            start_year(s)
            return s
        try:
            year = int(float(s))
        except ValueError:
            raise ValueError(f"can't read season value '{v}'") from None
        return season_from_end_year(year) if end_year else season_from_start(year)
    return raw.map(one)


def read_impact_files(cfg: Config) -> tuple[pd.DataFrame, ImpactReport]:
    """All CSVs in the impact dir as one long table: season, player_id, player_name, offense, defense, date."""
    folder = cfg.path(cfg.impact.dir)
    report = ImpactReport()
    frames = []
    aliases = cfg.impact.columns
    for path in sorted(Path(folder).glob("*.csv")) if folder.exists() else []:
        raw = pd.read_csv(path)
        cols = {normalize_column(c): c for c in raw.columns}
        found = {role: _resolve(cols, aliases.get(role, [])) for role in
                 ("player_id", "player_name", "offense", "defense", "season", "date")}
        report.files.append(path.name)
        report.columns[path.name] = {k: v for k, v in found.items() if v}
        missing = [r for r in ("offense", "defense") if not found[r]]
        if missing:
            raise ValueError(
                f"{path.name}: no column found for {missing}. Columns in file: {list(raw.columns)}. "
                "Add the right names under impact.columns in config.yaml."
            )
        if not (found["player_id"] or found["player_name"]):
            raise ValueError(f"{path.name}: needs a player id or player name column; has {list(raw.columns)}")
        out = pd.DataFrame({
            "offense": pd.to_numeric(raw[found["offense"]], errors="coerce"),
            "defense": pd.to_numeric(raw[found["defense"]], errors="coerce"),
        })
        out["player_id"] = (pd.to_numeric(raw[found["player_id"]], errors="coerce")
                            if found["player_id"] else np.nan)
        out["player_name"] = raw[found["player_name"]].astype(str) if found["player_name"] else ""
        if found["season"]:
            out["season"] = _season_values(raw[found["season"]], cfg.impact.numeric_season_is_end_year)
        else:
            m = _FILE_SEASON.search(path.stem)
            if not m:
                raise ValueError(f"{path.name}: no season column and no season like 2025-26 in the file name")
            start_year(m.group(1))
            out["season"] = m.group(1)
        out["date"] = pd.to_datetime(raw[found["date"]], errors="coerce") if found["date"] else pd.NaT
        out["file"] = path.name
        frames.append(out)
    if not frames:
        empty = pd.DataFrame(columns=["season", "player_id", "player_name", "offense", "defense", "date", "file"])
        return empty, report
    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["season"])
    if not cfg.impact.defense_higher_is_better:
        long["defense"] = -long["defense"]
    report.seasons = sorted(long["season"].unique())
    return long, report


def _snapshot(rows: pd.DataFrame, season_end: pd.Timestamp | None) -> pd.DataFrame:
    """Reduce to one value per player key.

    Dated rows: the last row on or before the end of the regular season; a player with no row
    by then gets their first row after it (e.g. a leaderboard downloaded in the offseason).
    Undated rows: the mean of duplicates.
    """
    dated = rows["date"].notna()
    parts = []
    if dated.any():
        d = rows[dated].sort_values("date")
        if season_end is not None:
            before = d[d["date"] <= season_end].drop_duplicates("key", keep="last")
            after = d[(d["date"] > season_end) & ~d["key"].isin(before["key"])]
            d = pd.concat([before, after.drop_duplicates("key", keep="first")])
        else:
            d = d.drop_duplicates("key", keep="last")
        parts.append(d)
    if (~dated).any():
        u = rows[~dated].groupby("key", as_index=False).agg(
            offense=("offense", "mean"), defense=("defense", "mean"),
            player_id=("player_id", "first"), player_name=("player_name", "first"))
        parts.append(u)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("key", keep="first")


def load_impact(cfg: Config, players: dict[str, pd.DataFrame],
                season_end: dict[str, pd.Timestamp] | None = None) -> tuple[pd.DataFrame, ImpactReport]:
    """Impact per (season, PLAYER_ID).

    players: season -> DataFrame with PLAYER_ID and PLAYER_NAME (from the game logs), used for name matching.
    season_end: season -> last regular-season game date, used to pick end-of-regular-season values.
    """
    long, report = read_impact_files(cfg)
    season_end = season_end or {}
    overrides = {normalize_name(k): int(v) for k, v in cfg.impact.name_overrides.items()}
    out = []
    for season, rows in long.groupby("season"):
        rows = rows.copy()
        has_id = rows["player_id"].notna()
        rows["key"] = np.where(has_id, "id:" + rows["player_id"].astype("Int64").astype(str),
                               "name:" + rows["player_name"].map(normalize_name))
        snap = _snapshot(rows, season_end.get(season))

        lookup: dict[str, int] = {}
        roster = players.get(season)
        if roster is not None and len(roster):
            names = roster.drop_duplicates("PLAYER_ID").assign(norm=lambda d: d["PLAYER_NAME"].map(normalize_name))
            counts = names["norm"].value_counts()
            lookup = {n: int(pid) for n, pid in zip(names["norm"], names["PLAYER_ID"]) if counts[n] == 1}
        lookup.update(overrides)

        ids = []
        unmatched = []
        for pid, name in zip(snap["player_id"], snap["player_name"]):
            if pd.notna(pid):
                ids.append(int(pid))
                continue
            norm = normalize_name(name)
            if norm in lookup:
                ids.append(lookup[norm])
            else:
                ids.append(None)
                unmatched.append(name)
        snap["PLAYER_ID"] = pd.array(ids, dtype="Int64")
        if unmatched:
            report.unmatched[season] = sorted(set(unmatched))
        snap = snap.dropna(subset=["PLAYER_ID"]).drop_duplicates("PLAYER_ID")
        out.append(snap.assign(season=season)[["season", "PLAYER_ID", "offense", "defense"]])

    if not out:
        return pd.DataFrame(columns=["season", "PLAYER_ID", "offense", "defense"]), report
    impact = pd.concat(out, ignore_index=True)
    impact["PLAYER_ID"] = impact["PLAYER_ID"].astype("int64")
    if cfg.impact.defense_multi_year.enabled:
        impact = blend_defense(impact, cfg.impact.defense_multi_year.weights)
    return impact, report


def blend_defense(impact: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    """Defense for season t = weighted mean of seasons t, t-1, ... (weights renormalized over what exists)."""
    by_key = {(s, p): d for s, p, d in zip(impact["season"], impact["PLAYER_ID"], impact["defense"])}
    blended = []
    for season, pid in zip(impact["season"], impact["PLAYER_ID"]):
        num = den = 0.0
        for lag, w in enumerate(weights):
            d = by_key.get((previous(season, lag), pid))
            if d is not None and not pd.isna(d):
                num += w * d
                den += w
        blended.append(num / den if den else np.nan)
    return impact.assign(defense=blended)
