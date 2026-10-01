"""Team scoring: minutes weights, caps, redundancy, team percentiles, coverage and overlap.

Minutes weight: w = player minutes / (team minutes / 5), so weights sum to 5 and a player
on the floor for every minute counts 1.0.

Corner sum = sum of w x z over the roster.
  * offense, defense: uncapped; negative z counts against the team.
  * playmaking, portability: negative z floored at 0 (configurable), then capped.
    cap_mode "pool_percentile": cap = that percentile of all team-season sums in the window.
    cap_mode "players_worth":   cap = players_worth x (reference_mpg / 48) x z(reference percentile).
    Anything above the cap is redundancy, reported in "players' worth"
    (one player's worth = a reference-percentile player at reference minutes).

Team corner percentile = share of all team-seasons in the window at or below that (capped) value.
Coverage = area of the team shape drawn from those four percentiles, as a share of the square.
Overlap = sum of player-shape areas - area of their union (minutes-scaled shapes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import geometry
from .cache import Cache
from .config import CORNERS, Config
from .metrics.players import (SeasonData, build_player_table, load_season, missing_tables,
                              team_minutes)
from .metrics.pool import percentile_against
from .sources.darko import ImpactReport, load_impact

log = logging.getLogger(__name__)

REGULAR_MODE = "regular"
PLAYOFF_MODE = "playoffs"


# -- league -------------------------------------------------------------------------------------

@dataclass
class League:
    cfg: Config
    players: dict[str, pd.DataFrame]
    minutes: dict[str, dict[str, pd.DataFrame]]  # mode -> season -> team minutes
    pool_seasons: list[str]
    incomplete: dict[str, list[str]]
    impact_report: ImpactReport
    _raw_pools: dict[str, pd.DataFrame] = field(default_factory=dict)

    def teams(self, season: str, mode: str = REGULAR_MODE) -> list[str]:
        tm = self.minutes[mode].get(season)
        return sorted(tm["TEAM_ABBREVIATION"].unique()) if tm is not None and len(tm) else []

    def raw_pool(self, mode: str) -> pd.DataFrame:
        """Uncapped corner sums for every team-season in the pool."""
        if mode not in self._raw_pools:
            rows = []
            for season in self.pool_seasons:
                for team in self.teams(season, mode):
                    sums = roster_sums(self, team, season, mode)
                    rows.append({"season": season, "team": team, **sums.uncapped})
            self._raw_pools[mode] = pd.DataFrame(rows, columns=["season", "team", *CORNERS])
        return self._raw_pools[mode]

    def one_worth(self, corner: str, season: str) -> float:
        """One player's worth: a reference-percentile player at reference minutes, in w x z units."""
        players = self.players[season]
        z = players.loc[players["qualified"], f"{corner}_z"].dropna()
        if z.empty:
            return float("nan")
        z_ref = float(np.quantile(z, self.cfg.team.cap_reference_percentile / 100.0))
        return self.cfg.team.cap_reference_mpg / 48.0 * z_ref

    def cap(self, corner: str, season: str, mode: str) -> float:
        t = self.cfg.team
        if t.cap_mode == "players_worth":
            return t.cap_players_worth * self.one_worth(corner, season)
        pool = self.raw_pool(mode)[corner].dropna()
        if pool.empty:
            return float("nan")
        return float(np.quantile(pool, t.cap_pool_percentile / 100.0))

    def capped(self, corner: str, value: float, season: str, mode: str) -> tuple[float, float]:
        """(coverage value, excess above the cap)."""
        if corner not in self.cfg.team.capped_corners:
            return value, 0.0
        cap = self.cap(corner, season, mode)
        if np.isnan(cap):
            return value, 0.0
        return min(value, cap), max(value - cap, 0.0)

    def pool(self, mode: str) -> pd.DataFrame:
        """Capped corner values for every team-season in the pool."""
        raw = self.raw_pool(mode).copy()
        for corner in self.cfg.team.capped_corners:
            raw[corner] = [self.capped(corner, v, s, mode)[0] for v, s in zip(raw[corner], raw["season"])]
        return raw

    def cap_binding_share(self, corner: str, mode: str) -> float:
        raw = self.raw_pool(mode)
        if raw.empty:
            return float("nan")
        over = [v > self.cap(corner, s, mode) for v, s in zip(raw[corner], raw["season"])]
        return float(np.mean(over))


def build_league(cfg: Config, cache: Cache, seasons: list[str]) -> League:
    """Load cached tables and impact CSVs and build player tables for `seasons`."""
    data: dict[str, SeasonData] = {}
    incomplete: dict[str, list[str]] = {}
    for season in seasons:
        d = load_season(cache, season)
        if d is None:
            incomplete[season] = ["gamelog"]
            continue
        data[season] = d
        missing = missing_tables(d, cfg)
        if missing:
            incomplete[season] = missing

    rosters = {s: d.gamelog[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID") for s, d in data.items()}
    ends = {s: d.regular_season_end for s, d in data.items()}
    impact, report = load_impact(cfg, rosters, ends)
    have_impact = set(impact["season"].unique())
    for season in data:
        if season not in have_impact:
            incomplete.setdefault(season, []).append(f"{cfg.impact.source_name} CSV")

    players = {s: build_player_table(d, impact, cfg) for s, d in data.items()}
    minutes = {
        REGULAR_MODE: {s: team_minutes(d.gamelog) for s, d in data.items()},
        PLAYOFF_MODE: {s: team_minutes(d.gamelog_playoffs) for s, d in data.items()},
    }
    window = set(cfg.window_seasons)
    pool_seasons = sorted(s for s in data if s in window and s not in incomplete)
    return League(cfg, players, minutes, pool_seasons, incomplete, report)


# -- one roster ---------------------------------------------------------------------------------

@dataclass
class Sums:
    roster: pd.DataFrame
    uncapped: dict[str, float]
    missing_minutes: dict[str, float]


def select_roster(league: League, team: str, season: str, mode: str) -> pd.DataFrame:
    tm = league.minutes[mode].get(season)
    if tm is None or tm.empty:
        raise ValueError(f"no {mode} minutes cached for {season}")
    roster = tm[tm["TEAM_ABBREVIATION"] == team]
    if roster.empty:
        known = ", ".join(league.teams(season, mode))
        hint = " (did they make the playoffs?)" if mode == PLAYOFF_MODE else ""
        raise ValueError(f"no {mode} minutes for {team} in {season}{hint}. Teams: {known}")
    if mode == PLAYOFF_MODE:
        roster = roster.nlargest(league.cfg.playoffs.rotation_size, "MIN")
    return roster.sort_values("MIN", ascending=False).reset_index(drop=True)


def roster_sums(league: League, team: str, season: str, mode: str) -> Sums:
    cfg = league.cfg
    players = league.players[season]
    roster = select_roster(league, team, season, mode)
    cols = ["PLAYER_ID"] + [f"{c}_{k}" for c in CORNERS for k in ("z", "pct")]
    roster = roster.merge(players[cols], on="PLAYER_ID", how="left")
    total_min = roster["MIN"].sum()
    roster["weight"] = roster["MIN"] / (total_min / cfg.team.on_court_slots)

    uncapped, missing = {}, {}
    for corner in CORNERS:
        z = roster[f"{corner}_z"]
        missing[corner] = float(roster.loc[z.isna(), "MIN"].sum() / total_min)
        if corner in cfg.team.floor_negative_corners:
            z = z.clip(lower=0.0).fillna(0.0)
        else:
            z = z.fillna(cfg.team.missing_value_z)
        roster[f"{corner}_contrib"] = roster["weight"] * z
        uncapped[corner] = float(roster[f"{corner}_contrib"].sum())
    return Sums(roster, uncapped, missing)


# -- full evaluation ----------------------------------------------------------------------------

@dataclass
class TeamResult:
    team: str
    team_name: str
    season: str
    mode: str
    roster: pd.DataFrame
    values: dict[str, float]
    uncapped: dict[str, float]
    pct: dict[str, float]
    cap: dict[str, float]
    one_worth: dict[str, float]
    redundancy: dict[str, float]
    redundancy_players: dict[str, float]
    cap_binding_share: dict[str, float]
    missing_minutes: dict[str, float]
    coverage: float
    sum_areas: float
    union: float
    overlap: float
    pool_size: int
    pool_seasons: list[str]
    qualified_n: int
    not_drawn: list[str]
    color_order: list[int] = field(default_factory=list)  # team's regular-season minutes order
    synthetic: bool = False


def team_name(abbr: str) -> str:
    try:
        from nba_api.stats.static import teams
        found = teams.find_team_by_abbreviation(abbr)
        if found:
            return found["full_name"]
    except Exception:
        pass
    return abbr


def evaluate_team(league: League, team: str, season: str, mode: str = REGULAR_MODE) -> TeamResult:
    cfg = league.cfg
    if season not in league.players:
        raise ValueError(f"no cached data for {season}; run `rosterfit fetch --seasons {season}` first")
    if season in league.incomplete:
        log.warning("%s is missing %s; affected corners fall back to missing-data rules",
                    season, ", ".join(league.incomplete[season]))
    sums = roster_sums(league, team, season, mode)

    values, redundancy, caps, one_worth, red_players, binding = {}, {}, {}, {}, {}, {}
    for corner in CORNERS:
        values[corner], excess = league.capped(corner, sums.uncapped[corner], season, mode)
        if corner in cfg.team.capped_corners:
            caps[corner] = league.cap(corner, season, mode)
            one_worth[corner] = league.one_worth(corner, season)
            redundancy[corner] = excess
            red_players[corner] = excess / one_worth[corner] if one_worth[corner] > 0 else float("nan")
            binding[corner] = league.cap_binding_share(corner, mode)

    pool = league.pool(mode)
    if pool.empty:
        log.warning("team pool is empty (no complete seasons in the window); percentiles use this team only")
        pool = pd.DataFrame([values])
    pct = {c: float(percentile_against([values[c]], pool[c])[0]) for c in CORNERS}
    order = cfg.corners.order
    coverage = geometry.area_fraction({c: pct[c] / 100 for c in CORNERS}, order)

    roster = sums.roster
    complete = roster[[f"{c}_pct" for c in CORNERS]].notna().all(axis=1)
    drawable = complete if mode == PLAYOFF_MODE else complete & (roster["MIN"] >= cfg.drawing.min_minutes_to_draw)
    roster["drawn"] = drawable
    ref = roster.loc[drawable, "MIN"].max() if drawable.any() else 1.0
    if cfg.drawing.minutes_scaling == "area":
        roster["scale"] = np.sqrt(roster["MIN"] / ref).clip(upper=1.0)
    else:
        roster["scale"] = 1.0
    polys, areas = [], []
    for _, row in roster.iterrows():
        if not row["drawn"]:
            polys.append(None)
            areas.append(np.nan)
            continue
        vals = {c: row[f"{c}_pct"] / 100 for c in CORNERS}
        polys.append(geometry.to_polygon(geometry.shape_points(vals, order, row["scale"])))
        areas.append(geometry.area_fraction(vals, order, row["scale"]))
    roster["area"] = areas
    sum_areas, union, overlap = geometry.overlap_fraction(polys)
    not_drawn = roster.loc[~complete & (roster["MIN"] >= cfg.drawing.min_minutes_to_draw),
                           "PLAYER_NAME"].tolist()

    return TeamResult(
        team=team, team_name=team_name(team), season=season, mode=mode, roster=roster,
        values=values, uncapped=sums.uncapped, pct=pct, cap=caps, one_worth=one_worth,
        redundancy=redundancy, redundancy_players=red_players, cap_binding_share=binding,
        missing_minutes=sums.missing_minutes, coverage=coverage, sum_areas=sum_areas,
        union=union, overlap=overlap, pool_size=len(pool), pool_seasons=list(league.pool_seasons),
        qualified_n=int(league.players[season]["qualified"].sum()), not_drawn=not_drawn,
        color_order=regular_order(league, team, season),
    )


def regular_order(league: League, team: str, season: str) -> list[int]:
    """The team's players by regular-season minutes, so colors match across regular/playoff charts."""
    tm = league.minutes[REGULAR_MODE].get(season)
    if tm is None or tm.empty:
        return []
    rows = tm[tm["TEAM_ABBREVIATION"] == team].sort_values("MIN", ascending=False)
    return [int(p) for p in rows["PLAYER_ID"]]


def summary_table(result: TeamResult) -> pd.DataFrame:
    """The player rows behind the picture (the plot's table view)."""
    cols = ["PLAYER_NAME", "MIN", "GP", "weight", "scale", "area", "drawn"]
    for c in CORNERS:
        cols += [f"{c}_pct", f"{c}_z", f"{c}_contrib"]
    out = result.roster[cols].copy()
    out.insert(0, "team", result.team)
    out.insert(1, "season", result.season)
    out.insert(2, "mode", result.mode)
    return out
