"""Command line: fetch | check | plot | demo.

    python -m rosterfit fetch                       # download NBA.com tables for the window (run locally)
    python -m rosterfit check                       # what's cached, what's missing, DARKO matching
    python -m rosterfit plot --team NYK             # chart + CSV for one team, default season
    python -m rosterfit plot --team NYK --playoffs  # top-8 playoff rotation
    python -m rosterfit demo                        # synthetic league, no network needed
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


def cmd_check(cfg: Config, args) -> int:
    from .metrics.players import required_tables

    cache = _open_cache(cfg)
    seasons = _seasons(cfg, args.seasons)
    need = required_tables(cfg)
    print(f"Required tables: {', '.join(need)}  (+ playoff game logs for --playoffs)")
    print(f"{'season':<9}" + "".join(f"{t:<13}" for t in need) + "playoffs     impact")
    league = build_league(cfg, cache, seasons)
    impact_seasons = set(league.impact_report.seasons)
    for season in seasons:
        cells = []
        for t in need:
            if not available(t, season):
                cells.append("n/a")
            else:
                cells.append("ok" if cache.has(season, REGULAR, t) else "MISSING")
        po = "ok" if cache.has(season, PLAYOFFS, "gamelog") else "-"
        imp = "ok" if season in impact_seasons else "MISSING"
        print(f"{season:<9}" + "".join(f"{c:<13}" for c in cells) + f"{po:<13}{imp}")
    rep = league.impact_report
    print(f"\n{cfg.impact.source_name} files in {cfg.path(cfg.impact.dir)}: {len(rep.files)}")
    for name in rep.files:
        print(f"  {name}: columns used {rep.columns[name]}")
    for season, names in sorted(rep.unmatched.items()):
        shown = ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")
        print(f"  {season}: {len(names)} names not matched to NBA.com players: {shown}")
    if rep.unmatched:
        print("  (add them under impact.name_overrides in config.yaml if any matter)")
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
        f"Coverage: {result.coverage * 100:.1f}% of the square "
        f"(team outline vs {result.pool_size} team-seasons)",
        f"Player shapes combined: {result.union * 100:.1f}% of the square; overlap {result.overlap:.2f} squares "
        f"(sum of shape areas {result.sum_areas:.2f})",
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


def cmd_demo(cfg: Config, args) -> int:
    from .synthetic import make_league

    seasons = ["2023-24", "2024-25", "2025-26"]
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir, impact_dir = make_league(Path(tmp), seasons)
        cfg.impact.dir = str(impact_dir)
        cfg.seasons.window = [seasons[0], seasons[-1]]
        league = build_league(cfg, Cache(cache_dir), seasons)
        mode = PLAYOFF_MODE if args.playoffs else REGULAR_MODE
        result = evaluate_team(league, "DEM", seasons[-1], mode)
        result.synthetic = True
        result.team_name = "Demo Team"
        print(describe(result, cfg))
        out = Path(args.out) if args.out else cfg.path("outputs") / f"demo_{mode}.png"
        _render(result, cfg, out, args.theme)
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

    p = sub.add_parser("check", help="report cached tables, impact CSVs and the team pool")
    p.add_argument("--seasons")
    p.set_defaults(func=cmd_check)

    for name, func, helptext in (("plot", cmd_plot, "draw one team"),
                                 ("demo", cmd_demo, "draw a synthetic team (no network)")):
        p = sub.add_parser(name, help=helptext)
        if name == "plot":
            p.add_argument("--team", required=True, help="NBA.com abbreviation, e.g. NYK")
            p.add_argument("--season", help="default: seasons.target in config.yaml")
        p.add_argument("--playoffs", action="store_true", help="top playoff rotation instead of full roster")
        p.add_argument("--theme", choices=["light", "dark"])
        p.add_argument("--out", help="output PNG path (a CSV with the same name is written next to it)")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
