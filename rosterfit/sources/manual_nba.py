"""NBA.com data downloaded by hand, for when `rosterfit fetch` can't reach stats.nba.com.

Drop files into `fetch.manual_dir` (default data/manual/nba/). Two kinds are accepted:

  * bundles written by the browser download script (`rosterfit browser-script`), and
  * single raw NBA.com API responses saved from the browser's developer tools
    (Network tab -> the stats.nba.com request -> Save / Copy response).

Each response is recognized from its own "resource" and "parameters" fields (or, failing that,
from a file name like passing_2025-26.json / gamelog_2025-26_playoffs.json) and copied into the
same cache that `fetch` fills, so everything downstream works unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from ..cache import Cache
from ..seasons import PLAYOFFS, REGULAR, start_year
from .nba_stats import TABLES, available, request_spec, team_ids

BASE_URL = "https://stats.nba.com/stats/"
_FILE_HINT = re.compile(r"(gamelog|advanced|passing|catch_shoot|hustle|matchups)[_-](\d{4}-\d{2})([_-]playoffs)?",
                        re.IGNORECASE)
# Counting columns we use, per table; per-game downloads are multiplied back up to season totals.
_COUNT_COLUMNS = {
    "advanced": ["POSS"],
    "passing": ["POTENTIAL_AST", "AST_PTS_CREATED", "AST_POINTS_CREATED"],
    "catch_shoot": ["CATCH_SHOOT_FG3A", "CATCH_SHOOT_FG3M"],
    "hustle": ["SCREEN_ASSISTS"],
    "matchups": ["PARTIAL_POSS"],
}


# -- requests -> URLs (for the browser script) ---------------------------------------------------

def request_url(table: str, season: str, season_type: str, def_team_id: int | None = None) -> str:
    """The exact URL nba_api would request: same parameters, sorted, None values dropped."""
    cls, kwargs = request_spec(table, season, season_type, def_team_id)
    endpoint = cls(**kwargs, get_request=False)
    params = sorted((k, v) for k, v in endpoint.parameters.items() if v is not None)
    query = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params)
    return f"{BASE_URL}{cls.endpoint}?{query}"


def download_jobs(tables: list[str], seasons: list[str], cache: Cache | None = None) -> list[dict]:
    jobs = []
    for season in seasons:
        for table in tables:
            if not available(table, season):
                continue
            for season_type in TABLES[table].season_types:
                if cache is not None and cache.has(season, season_type, table):
                    continue
                jobs.append({"table": table, "season": season, "season_type": season_type,
                             "url": request_url(table, season, season_type)})
    return jobs


def matchup_team_urls(season: str, season_type: str) -> dict[str, str]:
    return {str(t): request_url("matchups", season, season_type, t) for t in team_ids()}


_SCRIPT = r"""// rosterfit: download NBA.com stats tables from your own browser.
// 1. Open https://www.nba.com/stats in Chrome, Edge or Firefox and stay on that tab.
// 2. Open the console (Mac: Cmd+Option+J, Windows: Ctrl+Shift+J). Chrome may ask you to type
//    "allow pasting" first.
// 3. Paste this whole file and press Enter. Allow multiple downloads if the browser asks.
// 4. Move the downloaded nba_stats_<season>.json files into data/manual/nba/ and run
//    `python -m rosterfit check`.
(async () => {
  const JOBS = __JOBS__;
  const MATCHUP_TEAM_URLS = __TEAM_URLS__;
  const PAUSE_MS = 1200;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const rowCount = (data) => {
    const rs = data.resultSets ?? data.resultSet;
    const first = Array.isArray(rs) ? rs[0] : rs;
    return first && first.rowSet ? first.rowSet.length : 0;
  };
  async function get(url) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch(url, { credentials: "omit" });
        if (!res.ok) throw new Error("HTTP " + res.status);
        return await res.json();
      } catch (err) {
        console.warn(`  attempt ${attempt} failed: ${err}`);
        await sleep(3000 * attempt);
      }
    }
    throw new Error("gave up after 3 attempts");
  }
  const bySeason = {};
  const failed = [];
  for (const [i, job] of JOBS.entries()) {
    const tag = `${i + 1}/${JOBS.length} ${job.table} ${job.season} ${job.season_type}`;
    try {
      const data = await get(job.url);
      const items = (bySeason[job.season] ??= []);
      if (job.table === "matchups" && rowCount(data) === 0) {
        const urls = MATCHUP_TEAM_URLS[job.season + "|" + job.season_type] || {};
        console.log(`${tag}: empty league-wide, fetching ${Object.keys(urls).length} teams`);
        let total = 0;
        for (const [teamId, url] of Object.entries(urls)) {
          await sleep(PAUSE_MS);
          const part = await get(url);
          total += rowCount(part);
          items.push({ ...job, url, def_team_id: Number(teamId), response: part });
        }
        console.log(`${tag}: ${total} rows from team requests`);
      } else {
        items.push({ ...job, response: data });
        console.log(`${tag}: ${rowCount(data)} rows`);
      }
    } catch (err) {
      failed.push(job);
      console.error(`${tag}: FAILED (${err})`);
    }
    await sleep(PAUSE_MS);
  }
  for (const [season, items] of Object.entries(bySeason)) {
    const blob = new Blob([JSON.stringify({ rosterfit_bundle: 1, items })], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `nba_stats_${season}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    await sleep(1000);
  }
  console.log(failed.length ? `Done, ${failed.length} request(s) failed; paste the script again to retry them.`
                            : "Done. Move the nba_stats_*.json downloads into data/manual/nba/.");
})();
"""


def browser_script(jobs: list[dict]) -> str:
    team_urls = {}
    for job in jobs:
        if job["table"] == "matchups":
            team_urls[f"{job['season']}|{job['season_type']}"] = matchup_team_urls(job["season"], job["season_type"])
    return (_SCRIPT.replace("__JOBS__", json.dumps(jobs, indent=1))
                   .replace("__TEAM_URLS__", json.dumps(team_urls)))


# -- responses -> cache --------------------------------------------------------------------------

def parse_result(resp: dict) -> pd.DataFrame:
    """First result set of an NBA.com stats response as a DataFrame."""
    rs = resp.get("resultSets", resp.get("resultSet"))
    if isinstance(rs, list):
        rs = rs[0] if rs else None
    if not rs or "headers" not in rs:
        raise ValueError("no resultSets/headers in this response")
    return pd.DataFrame(rs.get("rowSet") or [], columns=rs["headers"])


def _parameters(resp: dict) -> dict:
    params = resp.get("parameters") or {}
    if isinstance(params, list):  # some endpoints send a list of one-key dicts
        merged = {}
        for d in params:
            if isinstance(d, dict):
                merged.update(d)
        params = merged
    return params


def identify(resp: dict, filename: str = "") -> tuple[str, str, str, str]:
    """(table, season, season_type, per_mode) for a raw response."""
    resource = str(resp.get("resource", "")).lower()
    p = _parameters(resp)
    table = None
    if resource == "leaguegamelog" and str(p.get("PlayerOrTeam", "P")).upper().startswith("P"):
        table = "gamelog"
    elif resource == "leaguedashplayerstats" and p.get("MeasureType") == "Advanced":
        table = "advanced"
    elif resource == "leaguedashptstats" and str(p.get("PlayerOrTeam", "Player")).lower().startswith("p"):
        table = {"Passing": "passing", "CatchShoot": "catch_shoot"}.get(p.get("PtMeasureType"))
    elif resource == "leaguehustlestatsplayer":
        table = "hustle"
    elif resource == "matchupsrollup":
        table = "matchups"
    season, season_type = p.get("Season"), p.get("SeasonType")
    hint = _FILE_HINT.search(filename)
    if hint:
        table = table or hint.group(1).lower()
        season = season or hint.group(2)
        season_type = season_type or (PLAYOFFS if hint.group(3) else REGULAR)
    if not table:
        raise ValueError(f"not a table rosterfit uses (resource '{resource or '?'}', parameters {p or '?'})")
    if not season:
        raise ValueError("can't tell the season; name the file like passing_2025-26.json")
    start_year(season)
    return table, season, season_type or REGULAR, str(p.get("PerMode", "Totals"))


def to_totals(df: pd.DataFrame, table: str, per_mode: str) -> pd.DataFrame:
    """Turn a per-game download back into season totals for the counting columns we use."""
    if per_mode == "Totals" or table not in _COUNT_COLUMNS or df.empty:
        return df
    if per_mode != "PerGame":
        raise ValueError(f"{table} was saved with PerMode={per_mode}; switch the page to Totals and save again")
    games = "GP" if "GP" in df.columns else "G"
    out = df.copy()
    for col in _COUNT_COLUMNS[table]:
        if col in out.columns:
            out[col] = out[col].astype(float) * out[games].astype(float)
    return out


def import_manual(manual_dir: Path, cache: Cache) -> list[str]:
    """Copy every recognized response in manual_dir into the cache. Returns messages for the user."""
    messages: list[str] = []
    if not manual_dir.exists():
        return messages
    groups: dict[tuple[str, str, str], list[tuple[pd.DataFrame, Path]]] = {}
    for path in sorted(manual_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            messages.append(f"skipped {path.name}: not readable JSON ({err})")
            continue
        items = data["items"] if isinstance(data, dict) and "items" in data else [{"response": data}]
        for n, item in enumerate(items):
            resp = item.get("response", item) if isinstance(item, dict) else item
            where = path.name if len(items) == 1 else f"{path.name} item {n + 1}"
            meta = item if isinstance(item, dict) else {}
            try:
                try:
                    table, season, season_type, per_mode = identify(resp, path.name)
                except ValueError:
                    if not (meta.get("table") and meta.get("season")):
                        raise
                    table, season, season_type = meta["table"], meta["season"], REGULAR
                    per_mode = str(_parameters(resp).get("PerMode", "Totals"))
                # bundle labels, when present, win over what the response says about itself
                table = meta.get("table", table)
                season = meta.get("season", season)
                season_type = meta.get("season_type", season_type)
                if season_type not in (REGULAR, PLAYOFFS):
                    messages.append(f"skipped {where}: season type '{season_type}' isn't used")
                    continue
                df = to_totals(parse_result(resp), table, per_mode)
            except ValueError as err:
                messages.append(f"skipped {where}: {err}")
                continue
            groups.setdefault((season, season_type, table), []).append((df, path))

    for (season, season_type, table), parts in sorted(groups.items()):
        target = cache.path(season, season_type, table)
        newest = max(p.stat().st_mtime for _, p in parts)
        if target.exists() and target.stat().st_mtime >= newest:
            continue  # already imported and nothing newer since
        frames = [df for df, _ in parts if len(df)]
        df = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else parts[0][0]
        if df.empty and season_type == REGULAR:
            messages.append(f"skipped {table} {season}: the saved response has no rows")
            continue
        cache.write(df.reset_index(drop=True), season, season_type, table)
        files = sorted({p.name for _, p in parts})
        messages.append(f"imported {table:<12} {season} {season_type:<15} {len(df):>6} rows  ({', '.join(files)})")
    return messages
