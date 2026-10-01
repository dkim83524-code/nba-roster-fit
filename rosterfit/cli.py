"""Command line: fetch | check | plot | demo.

    python -m rosterfit fetch                       # download NBA.com tables for the window (run locally)
    python -m rosterfit check                       # what's cached, what's missing, DARKO matching
    python -m rosterfit plot --team NYK             # chart + CSV for one team, default season
    python -m rosterfit plot --team NYK --playoffs  # top-8 playoff rotation
    python -m rosterfit compare --team NYK --vs SAS --playoffs   # two teams side by side
    python -m rosterfit explain --team NYK          # ingredients behind playmaking and portability
    python -m rosterfit explain --team DEN --corner defense   # LEBRON and DARKO behind "Stops"
    python -m rosterfit demo                        # synthetic league, no network needed
    python -m rosterfit export-web                  # the website: one HTML page + its data file
    python -m rosterfit browser-script              # if fetch can't connect: download from your browser

Files saved into data/manual/nba/ (NBA.com) and data/manual/darko/ (DARKO) are picked up by
check and plot automatically.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from .cache import Cache
from .config import Config, load_config
from .seasons import PLAYOFFS, REGULAR, parse_season_arg
from .sources.manual_nba import browser_script, download_jobs, import_manual
from .sources.names import normalize_name
from .sources.nba_stats import TABLES, NBAStatsFetcher, available
from .team import PLAYOFF_MODE, REGULAR_MODE, TeamResult, build_league, evaluate_team, summary_table

log = logging.getLogger("rosterfit")


def _seasons(cfg: Config, arg: str | None) -> list[str]:
    if arg:
        return parse_season_arg(arg)
    seasons = cfg.window_seasons
    return seasons if cfg.seasons.target in seasons else seasons + [cfg.seasons.target]


def _open_cache(cfg: Config) -> Cache:
    """The local cache, after copying in anything saved by hand under fetch.manual_dir."""
    cache = Cache(cfg.path(cfg.fetch.cache_dir))
    for line in import_manual(cfg.path(cfg.fetch.manual_dir), cache):
        print(line)
    return cache


# Hosted dev environments whose servers NBA.com silently ignores (requests just time out).
_HOSTED = {"CODESPACES": "GitHub Codespaces", "GITPOD_WORKSPACE_ID": "Gitpod",
           "CLOUD_SHELL": "Google Cloud Shell", "REPL_ID": "Replit"}


def hosted_environment() -> str | None:
    for var, name in _HOSTED.items():
        if os.environ.get(var):
            return name
    return None


def cmd_fetch(cfg: Config, args) -> int:
    from .metrics.players import required_tables

    hosted = hosted_environment()
    if hosted and not args.anyway:
        print(f"This is running in {hosted}, on a cloud server. NBA.com ignores requests from cloud\n"
              "servers, so fetch would only time out. Either run fetch on your own computer, or download\n"
              "from your browser (your browser runs on your computer, so it works):\n\n"
              "  python -m rosterfit browser-script\n\n"
              "(Use `fetch --anyway` to try regardless.)", file=sys.stderr)
        return 2
    cache = _open_cache(cfg)
    fetcher = NBAStatsFetcher(cache, cfg.fetch.sleep_seconds, cfg.fetch.timeout, cfg.fetch.retries)
    tables = args.tables.split(",") if args.tables else required_tables(cfg)
    for season in _seasons(cfg, args.seasons):
        for table in tables:
            for season_type in TABLES[table].season_types:
                if not available(table, season):
                    print(f"  {season} {table:<12} not tracked before {TABLES[table].first_season}")
                    continue
                cached = cache.has(season, season_type, table)
                if cached and not args.refresh:
                    continue
                try:
                    df = fetcher.fetch(table, season, season_type, refresh=args.refresh)
                except RuntimeError as err:
                    print(f"\nFAILED {season} {table} ({season_type}): {err}", file=sys.stderr)
                    print("\nIf this keeps failing, download from your browser instead:\n"
                          "  python -m rosterfit browser-script", file=sys.stderr)
                    return 2
                print(f"  {season} {table:<12} {season_type:<15} {len(df):>6} rows")
    print(f"\nCached under {cache.root}")
    return 0


def _check_source(cfg: Config, league, seasons: list[str], name: str, rep, min_minutes: float) -> None:
    src = cfg.impact.sources()[name]
    print(f"\n{name} files in {rep.dir}: {len(rep.files)}  (seasons: {', '.join(rep.seasons) or 'none'})")
    for file in rep.files[:3]:
        print(f"  {file}: columns used {rep.columns[file]}")
    if len(rep.files) > 3:
        print(f"  ... and {len(rep.files) - 3} more")
    if rep.loose:
        print("  Matched by last name or word order (worth a glance):")
        for season, pairs in sorted(rep.loose.items()):
            print(f"    {season}: " + ", ".join(f"{a} -> {b}" for a, b in pairs))
    col = f"offense_src_{name}"
    gaps = []
    for season in seasons:
        p = league.players.get(season)
        if p is None or season not in rep.seasons or col not in p:
            continue
        gap = p[p[col].isna() & (p["MIN"] >= min_minutes)].sort_values("MIN", ascending=False)
        gaps += [(season, n, m) for n, m in zip(gap["PLAYER_NAME"], gap["MIN"])]
    if gaps:
        print(f"  Played {min_minutes:g}+ minutes but have no {name} value: {len(gaps)}")
        for season, player, mins in gaps[:10]:
            last = normalize_name(player).split()[-1:]
            maybe = [c for c in rep.unmatched.get(season, []) if normalize_name(c).split()[-1:] == last]
            hint = f"   {name} has: {', '.join(maybe)}" if maybe else ""
            print(f"    {season}  {player} ({mins:,.0f} min){hint}")
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more")
        if not src.columns.get("player_id") or rep.unmatched:
            print(f"  A name mismatch is fixed under the {name} source's name_overrides in config.yaml:\n"
                  f"    \"Name In The {name} CSV\": \"Name On NBA.com\"")
    elif rep.seasons:
        print(f"  Every player with {min_minutes:g}+ minutes has a {name} value.")
    unmatched = sum(len(v) for v in rep.unmatched.values())
    if unmatched:
        print(f"  ({unmatched} {name} rows across all seasons are players with no NBA.com minutes "
              "that season, e.g. injured all year; they're ignored.)")


def cmd_check(cfg: Config, args) -> int:
    from .metrics.players import required_tables

    cache = _open_cache(cfg)
    seasons = _seasons(cfg, args.seasons)
    need = required_tables(cfg)
    print(f"Required tables: {', '.join(need)}  (regular season and playoffs)")
    print(f"{'season':<9}" + "".join(f"{t:<13}" for t in need) + "playoffs     impact")
    league = build_league(cfg, cache, seasons)
    reports = league.impact_reports
    for season in seasons:
        cells = []
        for t in need:
            if not available(t, season):
                cells.append("n/a")
            else:
                cells.append("ok" if cache.has(season, REGULAR, t) else "MISSING")
        po_tables = [t for t in need if PLAYOFFS in TABLES[t].season_types and available(t, season)]
        po_have = [t for t in po_tables if cache.has(season, PLAYOFFS, t)]
        po = ("-" if "gamelog" not in po_have else "ok" if len(po_have) == len(po_tables)
              else f"{len(po_have)}/{len(po_tables)} tables")
        lacking = [n for n, r in reports.items() if season not in r.seasons]
        imp = "ok" if not lacking else "MISSING " + ", ".join(lacking)
        print(f"{season:<9}" + "".join(f"{c:<13}" for c in cells) + f"{po:<13}{imp}")
    print("(table columns are the regular season; 'playoffs' counts the same tables for the playoffs, "
          "and `browser-script` downloads any that are missing)")
    weights = {c: cfg.impact.weights_for(c) for c in ("offense", "defense")}
    print("\nOffense and defense blend: " + "; ".join(
        f"{c} " + " + ".join(f"{n} x{w:g}" for n, w in ws.items() if w > 0) for c, ws in weights.items()))
    for name, rep in reports.items():
        _check_source(cfg, league, seasons, name, rep, args.min_minutes)
    print(f"\nTeam pool: {len(league.pool_seasons)} complete season(s) in the window: "
          f"{', '.join(league.pool_seasons) or 'none'}")
    for season, missing in sorted(league.incomplete.items()):
        print(f"  {season} left out: missing {', '.join(missing)}")
    if league.pool_seasons:
        for corner in cfg.team.capped_corners:
            share = league.cap_binding_share(corner, REGULAR_MODE)
            print(f"  cap on {cfg.label(corner)!r} binds for {share * 100:.0f}% of team-seasons")
    return 0


def describe(result: TeamResult, cfg: Config) -> str:
    lines = [
        f"{result.team_name} ({result.team}) - {result.season} {result.mode}",
        f"Coverage: {result.coverage * 100:.1f}% of the square filled by the players' shapes; "
        f"overlap {result.overlap:.2f} squares (sum of shape areas {result.sum_areas:.2f})",
        f"Depth: {result.depth * 100:.1f}% of the square inside the team outline "
        f"(team outline vs {result.pool_size} team-seasons)",
    ]
    for c in cfg.team.capped_corners:
        lines.append(f"Redundancy, {cfg.label(c)}: +{result.redundancy_players[c]:.2f} players' worth "
                     f"(cap binds for {result.cap_binding_share[c] * 100:.0f}% of team-seasons)")
    lines.append("")
    lines.append(f"{'corner':<20}{'team value':>11}{'uncapped':>10}{'team pct':>10}{'missing min':>13}")
    for c in cfg.corners.order:
        lines.append(f"{cfg.label(c):<20}{result.values[c]:>11.2f}{result.uncapped[c]:>10.2f}"
                     f"{result.pct[c]:>10.0f}{result.missing_minutes[c] * 100:>12.0f}%")
    lines.append("")
    r = result.roster
    head = f"{'player':<26}{'MIN':>7}{'weight':>8}" + "".join(f"{cfg.label(c)[:10]:>12}" for c in cfg.corners.order)
    lines.append(head + "   (league percentiles)")
    for _, row in r.iterrows():
        pcts = "".join(f"{row[f'{c}_pct']:>12.0f}" if row[f"{c}_pct"] == row[f"{c}_pct"] else f"{'-':>12}"
                       for c in cfg.corners.order)
        lines.append(f"{str(row['PLAYER_NAME'])[:25]:<26}{row['MIN']:>7.0f}{row['weight']:>8.2f}{pcts}")
    if result.not_drawn:
        lines.append(f"\nNot drawn (missing data): {', '.join(result.not_drawn)}")
    return "\n".join(lines)


def _render(result: TeamResult, cfg: Config, out: Path | None, theme: str | None) -> None:
    from .plot import render

    out = out or cfg.path("outputs") / f"{result.team}_{result.season}_{result.mode}.png"
    png = render(result, cfg, out, theme)
    csv = png.with_suffix(".csv")
    summary_table(result).to_csv(csv, index=False)
    print(f"\nWrote {png}\n      {csv}")


def cmd_plot(cfg: Config, args) -> int:
    season = args.season or cfg.seasons.target
    cache = _open_cache(cfg)
    seasons = sorted(set(cfg.window_seasons) | {season})
    league = build_league(cfg, cache, seasons)
    mode = PLAYOFF_MODE if args.playoffs else REGULAR_MODE
    try:
        result = evaluate_team(league, args.team.upper(), season, mode)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(describe(result, cfg))
    _render(result, cfg, Path(args.out) if args.out else None, args.theme)
    return 0


def cmd_browser_script(cfg: Config, args) -> int:
    from .metrics.players import required_tables

    cache = _open_cache(cfg)
    tables = args.tables.split(",") if args.tables else required_tables(cfg)
    jobs = download_jobs(tables, _seasons(cfg, args.seasons), None if args.all else cache)
    if not jobs:
        print("Everything the config needs is already cached; nothing to download.")
        return 0
    out = Path(args.out) if args.out else cfg.path(cfg.fetch.manual_dir) / "download_nba_stats.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(browser_script(jobs))
    hosted = hosted_environment()
    move = (f"In {hosted}: drag the files from your Downloads folder onto {cfg.fetch.manual_dir} in the\n"
            "   Explorer panel (or right-click that folder > Upload...)." if hosted else
            f"Move the downloaded nba_stats_<season>.json files into {cfg.path(cfg.fetch.manual_dir)}")
    minutes = len(jobs) * 3 / 60  # pause + response time; game logs are several MB each
    print(f"""Wrote {out} ({len(jobs)} requests, roughly {max(1, round(minutes))} min).

1. Open {out.name} in the editor, select all and copy it.
2. In a new browser tab open https://www.nba.com/stats, then its console:
   Cmd+Option+J (Mac) or Ctrl+Shift+J (Windows). Red errors already in the console are nba.com's
   own ads and trackers; ignore them. Type rosterfit in the console's Filter box to hide them.
3. Paste and press Enter. Only if Chrome warns about pasting: type allow pasting, press Enter,
   then paste again. Stay on the tab until it says Done; allow multiple downloads if asked.
4. {move}
5. Run: python -m rosterfit check""")
    return 0


def cmd_explain(cfg: Config, args) -> int:
    from .explain import corner_scaling, corner_weights, format_table, ingredients

    season = args.season or cfg.seasons.target
    league = build_league(cfg, _open_cache(cfg), [season])
    if season not in league.players:
        print(f"error: no cached data for {season}", file=sys.stderr)
        return 2
    team = args.team.upper()
    minutes = league.minutes[REGULAR_MODE][season]
    if team not in set(minutes["TEAM_ABBREVIATION"]):
        print(f"error: no {season} minutes for {team}", file=sys.stderr)
        return 2
    n = int(league.players[season]["qualified"].sum())
    corners = {"both": ["playmaking", "portability"], "all": list(cfg.corners.order)}.get(args.corner, [args.corner])
    for corner in corners:
        table = ingredients(league.players[season], minutes, team, corner, cfg, args.top)
        title = (f"\n{cfg.label(corner)} ({corner}) - {team} {season}: percentile on each ingredient "
                 f"among {n} qualified players")
        print(format_table(table, title, corner_weights(cfg, corner), corner_scaling(cfg, corner)))
    return 0


def describe_compare(a: TeamResult, b: TeamResult, cfg: Config) -> str:
    rows = [("Coverage (players' shapes)", lambda r: f"{r.coverage * 100:.0f}%"),
            ("Depth (team outline)", lambda r: f"{r.depth * 100:.0f}%"),
            ("Overlap (squares)", lambda r: f"{r.overlap:.2f}")]
    rows += [(f"Redundancy: {cfg.label(c)}", lambda r, c=c: f"+{r.redundancy_players.get(c, 0):.2f}")
             for c in cfg.team.capped_corners]
    rows += [(f"{cfg.label(c)} (team pct)", lambda r, c=c: f"{r.pct[c]:.0f}") for c in cfg.corners.order]
    lines = [f"{a.team_name} vs {b.team_name} - {a.season} {a.mode}",
             f"{'':<34}{a.team:>8}{b.team:>8}"]
    lines += [f"{name:<34}{fmt(a):>8}{fmt(b):>8}" for name, fmt in rows]
    return "\n".join(lines)


def cmd_compare(cfg: Config, args) -> int:
    import pandas as pd

    from .plot import render_compare

    season = args.season or cfg.seasons.target
    league = build_league(cfg, _open_cache(cfg), sorted(set(cfg.window_seasons) | {season}))
    mode = PLAYOFF_MODE if args.playoffs else REGULAR_MODE
    try:
        a = evaluate_team(league, args.team.upper(), season, mode)
        b = evaluate_team(league, args.vs.upper(), season, mode)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(describe_compare(a, b, cfg))
    out = Path(args.out) if args.out else cfg.path("outputs") / f"{a.team}_vs_{b.team}_{season}_{mode}.png"
    png = render_compare(a, b, cfg, out, args.theme)
    pd.concat([summary_table(a), summary_table(b)]).to_csv(png.with_suffix(".csv"), index=False)
    print(f"\nWrote {png}\n      {png.with_suffix('.csv')}")
    return 0


def cmd_demo(cfg: Config, args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        league = _demo_league(cfg, Path(tmp))
        seasons = sorted(league.players)
        mode = PLAYOFF_MODE if args.playoffs else REGULAR_MODE
        result = evaluate_team(league, "DEM", seasons[-1], mode)
        result.synthetic = True
        result.team_name = "Demo Team"
        print(describe(result, cfg))
        out = Path(args.out) if args.out else cfg.path("outputs") / f"demo_{mode}.png"
        _render(result, cfg, out, args.theme)
    return 0


def _demo_league(cfg: Config, tmp: Path):
    from .synthetic import make_league, use_synthetic_sources

    seasons = ["2023-24", "2024-25", "2025-26"]
    cache_dir, impact_dir = make_league(tmp, seasons)
    use_synthetic_sources(cfg, impact_dir)
    cfg.seasons.window = [seasons[0], seasons[-1]]
    return build_league(cfg, Cache(cache_dir), seasons)


def _add_cap_game(cfg: Config, args, data: dict) -> int:
    from .sources.contracts import SALARY_CAPS, find_contracts_file, read_contracts
    from .synthetic import make_contracts
    from .web import add_game

    if args.demo:
        season, contracts = "2026-27", make_contracts(data)
    else:
        folder = cfg.path("data/manual/contracts")
        path = Path(args.contracts) if args.contracts else find_contracts_file(folder)
        if path is None:
            print(f"Cap game left out: no contracts table in {folder}/ (see README).")
            return 0
        try:
            season, contracts = read_contracts(path)
        except (OSError, ValueError) as err:
            print(f"error: can't read contracts from {path}: {err}", file=sys.stderr)
            return 2
    cap = args.cap or SALARY_CAPS.get(season)
    if cap is None:
        print(f"error: no official salary cap on file for {season}; pass --cap in dollars", file=sys.stderr)
        return 2
    info = add_game(data, season, contracts, cap)
    print(f"Cap game: {season} contracts, cap ${cap / 1e6:.3f}M. {info['deck']} of {info['contracts']} "
          f"players are in the deck; {len(info['unmatched'])} played no {data['game']['stats_season']} games")
    if info["unmatched"]:
        print("  (rookies, injuries, or a name that didn't match): " + ", ".join(info["unmatched"][:12])
              + (" ..." if len(info["unmatched"]) > 12 else ""))
    return 0


def cmd_export_web(cfg: Config, args) -> int:
    from .web import build_web_data, write_web

    out_dir = Path(args.out) if args.out else cfg.path("outputs") / "web"
    if args.demo:
        with tempfile.TemporaryDirectory() as tmp:
            league = _demo_league(cfg, Path(tmp))
            names = {t: ("Demo Team" if t == "DEM" else f"Team {t[1:]}")
                     for s in league.players for t in league.teams(s, REGULAR_MODE)}
            data = build_web_data(league, cfg, synthetic=True, names=names)
    else:
        league = build_league(cfg, _open_cache(cfg), _seasons(cfg, args.seasons))
        if league.incomplete:
            print("warning: some seasons are incomplete; run `check` to see what's missing", file=sys.stderr)
        data = build_web_data(league, cfg)
    status = _add_cap_game(cfg, args, data)
    if status:
        return status
    paths = write_web(data, out_dir)
    n_teams = sum(len(t) for t in data["teams"].values())
    print(f"{len(data['meta']['seasons'])} seasons, {n_teams} team-seasons")
    print(f"Wrote {paths['page']}   (open in a browser)")
    print(f"      {paths['data']}   (upload this to Claude to update the shared page)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rosterfit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download NBA.com tables into the local cache")
    p.add_argument("--seasons", help="'2025-26' or '2017-18:2025-26' (default: the window + target)")
    p.add_argument("--tables", help=f"comma list from: {', '.join(TABLES)} (default: what the config needs)")
    p.add_argument("--refresh", action="store_true", help="redownload even if cached")
    p.add_argument("--anyway", action="store_true", help="try even in a hosted environment like Codespaces")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("browser-script",
                       help="write a script that downloads the NBA.com tables from your browser")
    p.add_argument("--seasons", help="'2025-26' or '2017-18:2025-26' (default: the window + target)")
    p.add_argument("--tables", help="comma list (default: what the config needs)")
    p.add_argument("--all", action="store_true", help="include tables that are already cached")
    p.add_argument("--out", help="where to write the script")
    p.set_defaults(func=cmd_browser_script)

    p = sub.add_parser("compare", help="two teams side by side")
    p.add_argument("--team", required=True, help="first team, e.g. NYK")
    p.add_argument("--vs", required=True, help="second team, e.g. SAS")
    p.add_argument("--season", help="default: seasons.target in config.yaml")
    p.add_argument("--playoffs", action="store_true", help="compare top playoff rotations")
    p.add_argument("--theme", choices=["light", "dark"])
    p.add_argument("--no-outline", action="store_true", help="hide the team depth outlines")
    p.add_argument("--out", help="output PNG path (a CSV with the same name is written next to it)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("explain", help="show the ingredients behind a team's corners")
    p.add_argument("--team", required=True)
    p.add_argument("--season", help="default: seasons.target in config.yaml")
    p.add_argument("--corner", choices=["playmaking", "portability", "offense", "defense", "both", "all"],
                   default="both", help="both = playmaking and portability; all = every corner")
    p.add_argument("--top", type=int, default=10, help="how many players, by minutes")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("export-web", help="write the Roster Fit website (one HTML page) and its data file")
    p.add_argument("--seasons", help="'2017-18:2025-26' (default: the window + target)")
    p.add_argument("--demo", action="store_true", help="made-up league, no downloads needed")
    p.add_argument("--contracts", help="Basketball-Reference contracts table for the cap game "
                                       "(default: the newest file in data/manual/contracts/)")
    p.add_argument("--cap", type=float, help="salary cap in dollars (default: the official cap for the contracts' season)")
    p.add_argument("--out", help="output folder (default: outputs/web)")
    p.set_defaults(func=cmd_export_web)

    p = sub.add_parser("check", help="report cached tables, impact CSVs and the team pool")
    p.add_argument("--seasons")
    p.add_argument("--min-minutes", type=float, default=100,
                   help="list players with at least this many minutes who lack an impact value")
    p.set_defaults(func=cmd_check)

    for name, func, helptext in (("plot", cmd_plot, "draw one team"),
                                 ("demo", cmd_demo, "draw a synthetic team (no network)")):
        p = sub.add_parser(name, help=helptext)
        if name == "plot":
            p.add_argument("--team", required=True, help="NBA.com abbreviation, e.g. NYK")
            p.add_argument("--season", help="default: seasons.target in config.yaml")
        p.add_argument("--playoffs", action="store_true", help="top playoff rotation instead of full roster")
        p.add_argument("--theme", choices=["light", "dark"])
        p.add_argument("--no-outline", action="store_true", help="hide the team depth outline")
        p.add_argument("--out", help="output PNG path (a CSV with the same name is written next to it)")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if getattr(args, "no_outline", False):
        cfg.drawing.team_outline = False
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
