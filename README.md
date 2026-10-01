# nba-roster-fit

A stats-based take on Taylormetrics' "Wyman Diagram": a team is a square, each player is a shape,
and a well-built roster fills the square without piling everyone into the same corners.

![Synthetic demo](docs/demo.png)
*The demo uses a made-up league, so the names and numbers are not real.*

## The four corners

| Corner (clockwise from top-left) | Label | Measure |
|---|---|---|
| Offense | Scores | O-DPM from DARKO |
| Playmaking | Sets up others | 50/50 blend of potential assists and assist points created, per 100 possessions (NBA.com passing tracking) |
| Portability | Fits next to stars | Equal blend of catch-and-shoot 3-point proficiency, OREB%, screen assists per 100, and defensive versatility |
| Defense | Stops | D-DPM from DARKO |

Every component is z-scored within the season's qualified pool (≥25 GP, ≥15 MPG). Each player
gets a percentile per corner, and their vertex sits that far along the diagonal from the center
to the corner.

**Corner order matters.** Only neighboring corners multiply in a shape's area:
`area = (offense + portability) × (playmaking + defense) / 4`. This order rewards two-way
players (offense next to defense), scorer-creators (offense next to playmaking) and 3-and-D glue
(portability next to defense).

Details on each component:
- **C&S 3-point proficiency** is Ben Taylor's 3-pt proficiency curve applied to catch-and-shoot
  3s: a sigmoid on attempts per 100 times C&S 3P%. The percentage is shrunk toward league average
  by 50 attempts.
- **Versatility** is the normalized entropy of a defender's matchup possessions against guards,
  forwards and centers. NBA.com only splits those three ways, so it's coarse.
- **Box Creation** (Taylor's public formula) is available as a playmaking component with weight 0.
  It's left out by default because its points term credits scoring gravity.

## Team numbers

- **Minutes weight:** `w = player minutes / (team minutes / 5)`. Weights sum to 5, so a player
  who plays every minute counts 1.0.
- **Corner sum:** `Σ w × z`.
  - Offense and defense are uncapped, and negative values count against the team.
  - In playmaking and portability, below-average players add 0 instead of a negative, and the sum
    is capped. Anything above the cap is **redundancy**, reported in "players' worth" (one player's
    worth = an 85th-percentile player at 34 MPG).
- **Team corner percentile:** where the (capped) sum ranks among all team-seasons in the window
  (default 2017-18 to 2025-26, about 270 team-seasons).
- **Coverage:** area of the team outline drawn from those four percentiles, as a share of the square.
- **Player shapes combined / overlap:** each player's shape is scaled so its area is proportional
  to minutes relative to the team's minutes leader. Overlap = sum of shape areas − area of their
  union. It can exceed 1 square when many shapes stack.
- **Playoff mode** (`--playoffs`): the top 8 by playoff minutes, using regular-season metrics,
  compared against other playoff teams' top 8s.

### The cap: `team.cap_mode`

- `pool_percentile` (default) fills a corner once a team reaches the 80th percentile of
  team-seasons, so it always binds for the top 20%.
- `players_worth` uses `cap_players_worth × one player's worth`. With the zero floor, an average
  roster already adds up to about two players' worth. A cap of 1.75 therefore binds for about
  three quarters of teams, and those corners read "full" almost everywhere.

`rosterfit check` prints how often the cap binds.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m rosterfit demo            # synthetic league, no network; writes outputs/demo_regular.png
pytest
```

## Getting real data

**1. NBA.com tables. Run this on your own machine.** stats.nba.com drops requests from cloud
servers, so this step won't work from a hosted environment.

```bash
python -m rosterfit fetch           # 2017-18 to 2025-26: game logs, advanced, passing, C&S, hustle, matchups
```

This is about 60 requests at one per second (more if the matchup endpoint needs per-team calls).
Everything is cached in `data/cache/` and never re-downloaded unless you pass `--refresh`.

**If `fetch` can't connect, download from your browser instead.** NBA.com answers your browser
even when it refuses Python. Generate a download script, paste it into the browser console on
nba.com, and drop the files it saves into `data/manual/nba/`:

```bash
python -m rosterfit browser-script   # writes data/manual/nba/download_nba_stats.js (only what's missing)
```

1. Open https://www.nba.com/stats and open the console: Cmd+Option+J (Mac) or Ctrl+Shift+J
   (Windows). Chrome may ask you to type `allow pasting` first.
2. Paste the whole script and press Enter. Stay on the tab until it prints `Done`, and allow
   multiple downloads if asked.
3. Move the downloaded `nba_stats_<season>.json` files into `data/manual/nba/`.

The script makes exactly the requests `fetch` would. Single responses saved by hand also work:
in the Network tab, find the `stats.nba.com` request, use "Save response" or "Copy response", and
save it as a `.json` file in `data/manual/nba/`. The importer recognizes which table and season
each response is from (or name the file like `passing_2025-26.json`). Per-game downloads are
converted back to season totals.

Anything in `data/manual/nba/` is copied into the cache every time you run `check`, `plot` or
`fetch`, so there's no separate import step.

**2. DARKO CSVs. Download these by hand; nothing scrapes darko.app.** Use the CSV download on
darko.app's leaderboard and save one file per season into `data/manual/darko/`, for example
`darko_2025-26.csv`. A single history file with a season or date column also works.

- With a date column, the value on or before the last regular-season game is used.
- Players are matched by an NBA.com id column if the file has one, otherwise by name.
- `rosterfit check` lists any names it couldn't match. Fix those with `impact.name_overrides`.
- If DARKO's column names differ from the defaults, add them under `impact.columns`.

Seasons without a DARKO file are left out of the team pool. With only `darko_2025-26.csv`, team
percentiles compare against the 30 teams of that season only.

**3. Check and plot**

```bash
python -m rosterfit check
python -m rosterfit plot --team NYK                  # 2025-26 regular season, full roster
python -m rosterfit plot --team NYK --playoffs       # top-8 playoff rotation
python -m rosterfit plot --team NYK --theme dark --out outputs/knicks_dark.png
```

Each plot also writes a CSV next to the PNG with every number behind the picture.

## Configuration

Every weight, cap, threshold, corner order, label and season window lives in
[`config.yaml`](config.yaml), with comments. Unknown keys are rejected, so typos fail loudly.

## Data sources and terms

- **NBA.com stats** via [nba_api](https://github.com/swar/nba_api): free and unofficial. Requests
  are paced at one per second and cached. Check NBA.com's terms of use yourself; this is meant for
  personal, non-commercial analysis.
- **DARKO** ([darko.app](https://www.darko.app)): free, with CSV downloads on the site.
- **Not used:** EPM (paid subscription and API tier) and LEBRON (free to view, but no bulk
  download is offered). Third-party GitHub mirrors of LEBRON redistribute it without a license.
- Downloaded data is git-ignored. Don't commit it to a public repo.

## Package layout

```
rosterfit/
  config.py           load + validate config.yaml
  cache.py            parquet cache: data/cache/<season>/<regular|playoffs>/<table>.parquet
  sources/            nba_stats.py (nba_api), manual_nba.py (files saved from the browser),
                      darko.py (your CSVs), formulas.py (Taylor's formulas)
  metrics/            offense_defense, playmaking, portability, pool (z / percentiles), players
  team.py             weights, caps, redundancy, team percentiles, coverage, overlap
  geometry.py         shapes, areas, union (shapely)
  plot.py             matplotlib chart
  synthetic.py        made-up league for the demo and tests
  cli.py              fetch | browser-script | check | plot | demo
```

## Next

- Side-by-side comparison of two teams.
- A court-map panel (interior/perimeter × offense/defense) under the square.
- Clutch study: team corner coverage vs clutch offensive rating and clutch assist rate.
