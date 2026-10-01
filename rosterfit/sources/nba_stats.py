"""NBA.com stats tables via nba_api, cached locally.

NBA.com blocks most cloud/datacenter IPs, so run `rosterfit fetch` from your own machine.
Every downstream step reads only the cache.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..cache import Cache
from ..seasons import PLAYOFFS, REGULAR, at_least

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Table:
    name: str
    first_season: str
    season_types: tuple[str, ...]
    what: str


TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        Table("gamelog", "1996-97", (REGULAR, PLAYOFFS), "player game logs: minutes per team, GP, box score"),
        Table("advanced", "1996-97", (REGULAR,), "advanced totals: possessions, OREB%"),
        Table("passing", "2013-14", (REGULAR,), "passing tracking: potential assists, assist points created"),
        Table("catch_shoot", "2013-14", (REGULAR,), "catch-and-shoot tracking: C&S 3PA / 3PM"),
        Table("hustle", "2016-17", (REGULAR,), "hustle stats: screen assists"),
        Table("matchups", "2017-18", (REGULAR,), "defensive matchup time by opponent position"),
    )
}


def available(table: str, season: str) -> bool:
    return at_least(season, TABLES[table].first_season)


def team_ids() -> list[int]:
    from nba_api.stats.static import teams
    return [t["id"] for t in teams.get_teams()]


def request_spec(table: str, season: str, season_type: str,
                 def_team_id: int | None = None) -> tuple[Callable, dict]:
    """The nba_api endpoint class and arguments for one table: the single definition of each request.

    Used by the Python fetcher and by the browser download script, so both ask for the same thing.
    """
    from nba_api.stats import endpoints as ep

    if table == "gamelog":
        return ep.LeagueGameLog, dict(season=season, season_type_all_star=season_type,
                                      player_or_team_abbreviation="P")
    if table == "advanced":
        return ep.LeagueDashPlayerStats, dict(season=season, season_type_all_star=season_type,
                                              measure_type_detailed_defense="Advanced",
                                              per_mode_detailed="Totals")
    if table in ("passing", "catch_shoot"):
        measure = {"passing": "Passing", "catch_shoot": "CatchShoot"}[table]
        return ep.LeagueDashPtStats, dict(season=season, season_type_all_star=season_type,
                                          pt_measure_type=measure, per_mode_simple="Totals",
                                          player_or_team="Player")
    if table == "hustle":
        return ep.LeagueHustleStatsPlayer, dict(season=season, season_type_all_star=season_type,
                                                per_mode_time="Totals")
    if table == "matchups":
        kwargs = dict(season=season, season_type_playoffs=season_type, per_mode_simple="Totals")
        if def_team_id is not None:
            kwargs["def_team_id_nullable"] = str(def_team_id)
        return ep.MatchupsRollup, kwargs
    raise KeyError(f"unknown table '{table}'")


class NBAStatsFetcher:
    """Thin, polite wrapper around nba_api: one request at a time, a pause between, retries."""

    def __init__(self, cache: Cache, sleep_seconds: float = 1.0, timeout: int = 60, retries: int = 3):
        self.cache = cache
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.retries = retries
        self._last_call = 0.0

    # -- low level -----------------------------------------------------------------------
    def _call(self, endpoint_cls: Callable, **params) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            wait = self.sleep_seconds - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                endpoint = endpoint_cls(**params, timeout=self.timeout)
                return endpoint.get_data_frames()[0]
            except Exception as err:  # network errors, timeouts, bad JSON
                last_err = err
                log.warning("%s failed (attempt %d/%d): %s", endpoint_cls.__name__, attempt, self.retries, err)
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds * 2 ** attempt)
        raise RuntimeError(
            f"{endpoint_cls.__name__} failed after {self.retries} attempts: {last_err}. "
            "NBA.com drops requests from most cloud servers; run fetch from a home connection."
        ) from last_err

    # -- tables --------------------------------------------------------------------------
    def _table(self, table: str, season: str, season_type: str) -> pd.DataFrame:
        cls, kwargs = request_spec(table, season, season_type)
        if table != "matchups":
            return self._call(cls, **kwargs)
        try:
            df = self._call(cls, **kwargs)
        except RuntimeError as err:
            log.info("league-wide matchup rollup failed for %s (%s)", season, err)
            df = pd.DataFrame()
        if len(df):
            return df
        # Some seasons only answer per defending team; fall back to 30 smaller requests.
        log.info("league-wide matchup rollup empty for %s; fetching per team", season)
        parts = []
        for team_id in team_ids():
            cls, kwargs = request_spec(table, season, season_type, def_team_id=team_id)
            part = self._call(cls, **kwargs)
            if len(part):
                parts.append(part.assign(DEF_TEAM_ID=team_id))
        return pd.concat(parts, ignore_index=True) if parts else df

    def fetch(self, table: str, season: str, season_type: str = REGULAR,
              refresh: bool = False) -> pd.DataFrame | None:
        """Return the table from cache, downloading it first if needed. None if not tracked that season."""
        if not available(table, season):
            return None
        if not refresh and self.cache.has(season, season_type, table):
            return self.cache.read(season, season_type, table)
        log.info("downloading %s %s (%s)", table, season, season_type)
        df = self._table(table, season, season_type)
        if df is None or df.empty:
            if season_type == PLAYOFFS:
                df = pd.DataFrame()  # season without playoffs yet: cache the empty answer
            else:
                raise RuntimeError(f"NBA.com returned no rows for {table} {season} {season_type}")
        self.cache.write(df, season, season_type, table)
        return df
