"""Export everything the Roster Fit website needs into one data file and one self-contained page.

The page (app.html + core.js) recomputes team numbers in the browser with the same rules as
team.py, so the Builder can score made-up lineups. tests/test_web.py checks the two agree.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from ..config import CORNERS, Config
from ..sources.contracts import SOURCE_NAME, SOURCE_URL
from ..sources.names import match_names
from ..team import PLAYOFF_MODE, REGULAR_MODE, League, select_roster, team_name

HERE = Path(__file__).parent
_DATA_SLOT = "/*__ROSTERFIT_DATA__*/"
_CORE_SLOT = "/*__ROSTERFIT_CORE__*/"
MODES = (REGULAR_MODE, PLAYOFF_MODE)


def _num(x):
    """JSON-safe number: NaN/None -> null."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _text(x):
    """JSON-safe label: NaN/None/empty -> null."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return str(x) or None


def _primary_teams(league: League, season: str) -> dict[int, str]:
    tm = league.minutes[REGULAR_MODE].get(season)
    if tm is None or tm.empty:
        return {}
    by_player = tm.sort_values("MIN", ascending=False).groupby("PLAYER_ID")["TEAM_ABBREVIATION"]
    return {int(pid): "/".join(teams) for pid, teams in by_player}


def build_web_data(league: League, cfg: Config, synthetic: bool = False,
                   names: dict[str, str] | None = None) -> dict:
    """names: optional abbreviation -> display name (the synthetic league's teams have none)."""
    names = names or {}
    seasons = sorted(league.players)
    players = {}
    for season in seasons:
        table = league.players[season]
        teams = _primary_teams(league, season)
        rows = []
        for r in table.itertuples(index=False):
            pid = int(r.PLAYER_ID)
            rows.append([pid, str(r.PLAYER_NAME), teams.get(pid, ""), int(r.GP), _num(r.MIN), int(bool(r.qualified))]
                        + [_num(getattr(r, f"{c}_pct")) for c in CORNERS]
                        + [_num(getattr(r, f"{c}_z")) for c in CORNERS]
                        + [_text(getattr(r, "off_role", None)), _text(getattr(r, "def_role", None)),
                           _text(getattr(r, "position", None))])
        players[season] = rows

    teams: dict[str, dict] = {}
    for season in seasons:
        teams[season] = {}
        for abbr in league.teams(season, REGULAR_MODE):
            entry = {"name": names.get(abbr) or team_name(abbr)}
            for mode in MODES:
                try:
                    roster = select_roster(league, abbr, season, mode)
                except ValueError:
                    roster = pd.DataFrame(columns=["PLAYER_ID", "MIN"])
                entry[mode] = [[int(p), _num(m)] for p, m in zip(roster["PLAYER_ID"], roster["MIN"])]
            teams[season][abbr] = entry

    pools = {}
    caps = {}
    for mode in MODES:
        pool = league.pool(mode)
        pools[mode] = {c: sorted(float(v) for v in pool[c].dropna()) for c in CORNERS}
        caps[mode] = {s: {c: _num(league.cap(c, s, mode)) for c in cfg.team.capped_corners} for s in seasons}
    one_worth = {s: {c: _num(league.one_worth(c, s)) for c in cfg.team.capped_corners} for s in seasons}

    return {
        "meta": {
            "generated": dt.date.today().isoformat(),
            "synthetic": synthetic,
            "seasons": seasons,
            "pool_seasons": list(league.pool_seasons),
            "pool_sizes": {mode: len(league.pool(mode)) for mode in MODES},
            "qualified": {s: int(league.players[s]["qualified"].sum()) for s in seasons},
            "corners": list(CORNERS),
            "order": list(cfg.corners.order),
            "labels": {c: cfg.label(c) for c in CORNERS},
            "descriptions": {c: cfg.corners.descriptions.get(c, "") for c in CORNERS},
            "slots": cfg.team.on_court_slots,
            "capped": list(cfg.team.capped_corners),
            "floor": list(cfg.team.floor_negative_corners),
            "missing_value_z": cfg.team.missing_value_z,
            "minutes_scaling": cfg.drawing.minutes_scaling,
            "scale": cfg.drawing.scale,
            "min_minutes_to_draw": cfg.drawing.min_minutes_to_draw,
            "max_colored": cfg.drawing.max_colored_players,
            "rotation_size": cfg.playoffs.rotation_size,
            "min_gp": cfg.qualified.min_gp,
            "min_mpg": cfg.qualified.min_mpg,
            "impact_source": " + ".join(cfg.impact.active_sources()),
            "impact_sources": [{"name": n, "url": src.url,
                                "offense": cfg.impact.weights_for("offense").get(n, 0),
                                "defense": cfg.impact.weights_for("defense").get(n, 0)}
                               for n, src in cfg.impact.sources().items() if n in cfg.impact.active_sources()],
        },
        "players": players,
        "teams": teams,
        "pools": pools,
        "caps": caps,
        "one_worth": one_worth,
    }


def add_game(data: dict, contract_season: str, contracts: pd.DataFrame, cap: float,
             min_minutes: float = 500) -> dict:
    """Add the cap game's deck: every player with a contract that season and a full shape from the
    latest stats season (min_minutes or more), with his position there (LEBRON's) when known. Contracts are matched to players by name, or by a
    PLAYER_ID column when there is one. Returns counts, plus the contract names that matched no
    player, biggest salaries first.
    """
    stats_season = data["meta"]["seasons"][-1]
    rows = {r[0]: r for r in data["players"][stats_season]}
    roster = pd.DataFrame({"PLAYER_ID": list(rows), "PLAYER_NAME": [r[1] for r in rows.values()]})
    given = "PLAYER_ID" in contracts.columns  # ids already known (the demo); otherwise match names
    matched = {} if given else match_names(contracts["name"], roster)[0]
    n = len(CORNERS)
    deck, unmatched = [], []
    for c in contracts.itertuples(index=False):
        r = rows.get(int(c.PLAYER_ID) if given else matched.get(c.name))
        if r is None:
            unmatched.append(c.name)
        elif (r[4] or 0) >= min_minutes and all(v is not None for v in r[6:6 + 2 * n]):
            deck.append([r[0], int(c.salary), c.team, r[8 + 2 * n] if len(r) > 8 + 2 * n else None])
    data["game"] = {
        "contract_season": contract_season,
        "stats_season": stats_season,
        "cap": float(cap),
        "min_minutes": min_minutes,
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "deck": sorted(deck, key=lambda d: -d[1]),
    }
    return {"contracts": len(contracts), "matched": len(contracts) - len(unmatched), "deck": len(deck),
            "unmatched": unmatched}


def render_page(data: dict) -> str:
    """The app as a page fragment (title, styles, markup, scripts) with the data inlined."""
    page = (HERE / "app.html").read_text(encoding="utf-8")
    core = (HERE / "core.js").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return page.replace(_CORE_SLOT, core).replace(_DATA_SLOT, payload)


def standalone(fragment: str) -> str:
    """Wrap the fragment so it also opens straight from disk with the right encoding."""
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n")


def write_web(data: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment = render_page(data)
    paths = {
        "page": out_dir / "roster-fit-lab.html",
        "fragment": out_dir / "roster-fit-lab.fragment.html",
        "data": out_dir / "roster-fit-data.json",
    }
    paths["page"].write_text(standalone(fragment), encoding="utf-8")
    paths["fragment"].write_text(fragment, encoding="utf-8")
    paths["data"].write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return paths
