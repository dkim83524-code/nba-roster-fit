"""Offense/defense impact from CSVs you download yourself: DARKO, LEBRON, or any plus-minus with
offense and defense halves (one folder per source, set up under `impact` in config.yaml).

darko.app offers CSV downloads on its leaderboard and team pages; BBall Index's LEBRON database
has a CSV button. Save them into each source's folder. Nothing here scrapes a site.

Accepted layouts, mixed freely:
  * one file per season, season in the file name:   darko_2025-26.csv
  * any file with a season column (2025-26 or 2026)
  * daily/history exports with a date column: the last row on or before the end of the
    regular season is used, i.e. the end-of-regular-season value.

Players are matched on an NBA.com player-id column when one exists, otherwise by name. Optional
off_role / def_role columns (LEBRON's role labels) are carried through for the website.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, SourceCfg
from ..seasons import previous, season_from_end_year, season_from_start, start_year
from .names import match_names, normalize_column, normalize_name

_FILE_SEASON = re.compile(r"(\d{4}-\d{2})")
ROLES = ("off_role", "def_role")
OUT_COLUMNS = ["season", "PLAYER_ID", "offense", "defense", *ROLES]


@dataclass
class ImpactReport:
    name: str = ""
    dir: str = ""
    files: list[str] = field(default_factory=list)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)  # file -> role -> column
    seasons: list[str] = field(default_factory=list)
    unmatched: dict[str, list[str]] = field(default_factory=dict)     # season -> names with no NBA.com match
    loose: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # season -> (name, NBA.com name)


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


def _source(cfg: Config, name: str | None) -> tuple[str, SourceCfg]:
    name = name or cfg.impact.source_name
    sources = cfg.impact.sources()
    if name not in sources:
        raise ValueError(f"no impact source named {name!r}; have {sorted(sources)}")
    return name, sources[name]


def read_impact_files(cfg: Config, source: str | None = None) -> tuple[pd.DataFrame, ImpactReport]:
    """All CSVs in a source's folder as one long table: season, player_id, player_name, offense,
    defense, date, and role labels when the files have them."""
    name, src = _source(cfg, source)
    folder = cfg.path(src.dir)
    report = ImpactReport(name=name, dir=str(folder))
    frames = []
    aliases = src.columns
    for path in sorted(Path(folder).glob("*.csv")) if folder.exists() else []:
        raw = pd.read_csv(path)
        cols = {normalize_column(c): c for c in raw.columns}
        found = {role: _resolve(cols, aliases.get(role, [])) for role in
                 ("player_id", "player_name", "offense", "defense", "season", "date", *ROLES)}
        report.files.append(path.name)
        report.columns[path.name] = {k: v for k, v in found.items() if v}
        missing = [r for r in ("offense", "defense") if not found[r]]
        if missing:
            raise ValueError(
                f"{path.name}: no column found for {missing}. Columns in file: {list(raw.columns)}. "
                f"Add the right names under the {name} source's columns in config.yaml."
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
        for role in ROLES:
            out[role] = raw[found[role]].where(raw[found[role]].notna(), None) if found[role] else None
        if found["season"]:
            out["season"] = _season_values(raw[found["season"]], src.numeric_season_is_end_year)
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
        empty = pd.DataFrame(columns=["season", "player_id", "player_name", "offense", "defense", "date",
                                      *ROLES, "file"])
        return empty, report
    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["season"])
    if not src.defense_higher_is_better:
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
            player_id=("player_id", "first"), player_name=("player_name", "first"),
            **{role: (role, "first") for role in ROLES})
        parts.append(u)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("key", keep="first")


def load_impact(cfg: Config, players: dict[str, pd.DataFrame],
                season_end: dict[str, pd.Timestamp] | None = None,
                source: str | None = None) -> tuple[pd.DataFrame, ImpactReport]:
    """One source's impact per (season, PLAYER_ID); the main source (DARKO) unless `source` names another.

    players: season -> DataFrame with PLAYER_ID and PLAYER_NAME (from the game logs), used for name matching.
    season_end: season -> last regular-season game date, used to pick end-of-regular-season values.
    """
    _, src = _source(cfg, source)
    long, report = read_impact_files(cfg, source)
    season_end = season_end or {}
    out = []
    for season, rows in long.groupby("season"):
        rows = rows.copy()
        has_id = rows["player_id"].notna()
        rows["key"] = np.where(has_id, "id:" + rows["player_id"].astype("Int64").astype(str),
                               "name:" + rows["player_name"].map(normalize_name))
        snap = _snapshot(rows, season_end.get(season))

        roster = players.get(season)
        if roster is None:
            roster = pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME"])
        by_id = {int(p) for p in snap["player_id"].dropna()}
        to_match = [n for p, n in zip(snap["player_id"], snap["player_name"]) if pd.isna(p)]
        mapping, loose = match_names(to_match, roster, src.name_overrides, taken=by_id)
        snap["PLAYER_ID"] = pd.array(
            [int(p) if pd.notna(p) else mapping.get(n) for p, n in zip(snap["player_id"], snap["player_name"])],
            dtype="Int64")
        unmatched = sorted({n for n in to_match if n not in mapping})
        if unmatched:
            report.unmatched[season] = unmatched
        if loose:
            report.loose[season] = loose
        snap = snap.dropna(subset=["PLAYER_ID"]).drop_duplicates("PLAYER_ID")
        out.append(snap.assign(season=season)[OUT_COLUMNS])

    if not out:
        return pd.DataFrame(columns=OUT_COLUMNS), report
    impact = pd.concat(out, ignore_index=True)
    impact["PLAYER_ID"] = impact["PLAYER_ID"].astype("int64")
    if cfg.impact.defense_multi_year.enabled:
        impact = blend_defense(impact, cfg.impact.defense_multi_year.weights)
    return impact, report


def load_impacts(cfg: Config, players: dict[str, pd.DataFrame],
                 season_end: dict[str, pd.Timestamp] | None = None
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, ImpactReport]]:
    """Every source that carries weight in the offense or defense corner, by source name."""
    impacts, reports = {}, {}
    for name in cfg.impact.active_sources():
        impacts[name], reports[name] = load_impact(cfg, players, season_end, name)
    return impacts, reports


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
