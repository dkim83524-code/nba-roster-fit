"""A made-up league in the exact cache/CSV formats, for tests and `rosterfit demo`.

Nothing here is real: team "DEM" and every player name are invented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cache import Cache
from .seasons import PLAYOFFS, REGULAR

ARCHETYPES = {
    # name: (offense, defense, passing, c&s shooting, oreb, screens, versatility) in z-ish units
    "creator": (1.6, -0.4, 2.0, 0.0, -0.6, -0.5, 0.2),
    "scorer": (1.2, -0.5, 0.3, 0.6, -0.4, -0.4, 0.0),
    "wing3d": (0.2, 0.9, -0.3, 1.4, -0.3, -0.3, 1.0),
    "big": (0.3, 0.8, -0.6, -1.5, 1.8, 1.8, -1.0),
    "connector": (0.1, 0.3, 0.9, 0.5, 0.0, 0.2, 0.8),
    "bench": (-0.6, -0.3, -0.2, -0.2, -0.1, -0.1, -0.2),
}
TEMPLATE = ["creator", "scorer", "wing3d", "big", "connector", "wing3d", "big",
            "bench", "creator", "bench", "bench", "bench", "bench", "bench"]
MPG = [35, 33, 31, 29, 26, 22, 18, 16, 15, 10, 8, 6, 4, 3]
_FIRST = ["Avery", "Blake", "Cam", "Dev", "Eli", "Finn", "Gray", "Hayes", "Ira", "Jules", "Kai",
          "Lane", "Milo", "Noel", "Oak", "Parker", "Quinn", "Reese", "Sage", "Tate"]
_LAST = ["Alder", "Brook", "Cove", "Dale", "Ember", "Frost", "Glen", "Heath", "Isle", "Jett",
         "Knoll", "Lark", "Moss", "North", "Oriel", "Pike", "Quill", "Reed", "Stone", "Thorne"]


def _teams(n: int) -> list[str]:
    return ["DEM"] + [f"T{i:02d}" for i in range(1, n)]


def make_league(root: Path, seasons: list[str], n_teams: int = 30, games: int = 82,
                seed: int = 7) -> tuple[Path, Path]:
    """Write a synthetic league. Returns (cache_dir, impact_dir)."""
    rng = np.random.default_rng(seed)
    cache = Cache(root / "cache")
    impact_dir = root / "darko"
    impact_dir.mkdir(parents=True, exist_ok=True)
    teams = _teams(n_teams)
    names = [f"{f} {l}" for f in _FIRST for l in _LAST]
    rng.shuffle(names)

    pid = 1000
    for s_idx, season in enumerate(seasons):
        rows_players = []
        for t_idx, team in enumerate(teams):
            shift = rng.integers(0, len(TEMPLATE))
            for slot, mpg in enumerate(MPG):
                arche = TEMPLATE[(slot + shift) % len(TEMPLATE)] if slot >= 2 else TEMPLATE[slot]
                base = np.array(ARCHETYPES[arche]) + rng.normal(0, 0.45, 7)
                quality = rng.normal(0, 0.5) + (0.5 if slot < 5 else -0.2)
                rows_players.append({
                    "PLAYER_ID": pid, "PLAYER_NAME": names[(pid - 1000) % len(names)],
                    "TEAM_ID": 1610612700 + t_idx, "TEAM_ABBREVIATION": team,
                    "mpg": max(1.0, mpg + rng.normal(0, 2.0)), "traits": base, "quality": quality,
                })
                pid += 1
        players = pd.DataFrame(rows_players)
        start = pd.Timestamp(f"{2000 + int(season[2:4])}-10-20")

        # regular-season game logs: one row per player per game played
        logs = []
        for p in players.itertuples():
            gp = int(min(games, max(5, rng.normal(games * 0.85, games * 0.12))))
            t = p.traits
            per_min = {
                "AST": max(0.02, 0.09 + 0.05 * t[2]), "PTS": max(0.2, 0.45 + 0.12 * t[0]),
                "TOV": max(0.02, 0.05 + 0.02 * t[2]), "FG3A": max(0.0, 0.15 + 0.08 * t[3]),
            }
            mins = np.clip(rng.normal(p.mpg, 4.0, gp), 1, 46).round()
            fg3a = rng.poisson(per_min["FG3A"] * mins)
            logs.append(pd.DataFrame({
                "PLAYER_ID": p.PLAYER_ID, "PLAYER_NAME": p.PLAYER_NAME, "TEAM_ID": p.TEAM_ID,
                "TEAM_ABBREVIATION": p.TEAM_ABBREVIATION,
                "GAME_ID": [f"{season}-{g}" for g in range(gp)],
                "GAME_DATE": [(start + pd.Timedelta(days=2 * g)).strftime("%Y-%m-%d") for g in range(gp)],
                "MIN": mins, "AST": rng.poisson(per_min["AST"] * mins),
                "PTS": rng.poisson(per_min["PTS"] * mins), "TOV": rng.poisson(per_min["TOV"] * mins),
                "FG3A": fg3a, "FG3M": rng.binomial(fg3a, float(np.clip(0.35 + 0.02 * t[3], 0.2, 0.45))),
            }))
        gamelog = pd.concat(logs, ignore_index=True)
        cache.write(gamelog, season, REGULAR, "gamelog")

        totals = gamelog.groupby("PLAYER_ID")["MIN"].sum()
        players["MIN"] = players["PLAYER_ID"].map(totals).fillna(0)
        players["POSS"] = players["MIN"] * 2.1
        tr = np.vstack(players["traits"].to_numpy())

        cache.write(pd.DataFrame({"PLAYER_ID": players["PLAYER_ID"], "POSS": players["POSS"],
                                  "OREB_PCT": np.clip(0.04 + 0.03 * tr[:, 4], 0.0, 0.2)}),
                    season, REGULAR, "advanced")
        pot = np.clip(0.06 + 0.035 * tr[:, 2], 0.005, None) * players["MIN"]
        cache.write(pd.DataFrame({"PLAYER_ID": players["PLAYER_ID"], "POTENTIAL_AST": pot.round(),
                                  "AST_PTS_CREATED": (pot * 1.15).round()}), season, REGULAR, "passing")
        cs_att = np.clip(0.07 + 0.05 * tr[:, 3], 0.0, None) * players["MIN"]
        cs_pct = np.clip(0.36 + 0.025 * tr[:, 3], 0.2, 0.46)
        cache.write(pd.DataFrame({"PLAYER_ID": players["PLAYER_ID"], "CATCH_SHOOT_FG3A": cs_att.round(),
                                  "CATCH_SHOOT_FG3M": (cs_att * cs_pct).round()}), season, REGULAR, "catch_shoot")
        cache.write(pd.DataFrame({"PLAYER_ID": players["PLAYER_ID"],
                                  "SCREEN_ASSISTS": (np.clip(0.03 + 0.03 * tr[:, 5], 0, None)
                                                     * players["MIN"]).round()}), season, REGULAR, "hustle")
        mrows = []
        for p, v in zip(players.itertuples(), tr[:, 6]):
            spread = 1 / (1 + np.exp(-v))  # 0 = specialist, 1 = even
            main = rng.integers(0, 3)
            shares = np.full(3, spread / 3)
            shares[main] += 1 - spread
            for pos, share in zip("GFC", shares):
                mrows.append({"DEF_PLAYER_ID": p.PLAYER_ID, "POSITION": pos,
                              "PARTIAL_POSS": share * p.MIN * 1.0, "PERCENT_OF_TIME": share})
        cache.write(pd.DataFrame(mrows), season, REGULAR, "matchups")

        # playoffs: 16 teams, about 10 games, top-9 players
        strength = players.groupby("TEAM_ABBREVIATION")["quality"].sum().sort_values(ascending=False)
        po_teams = list(strength.index[:16])
        if "DEM" not in po_teams:
            po_teams[-1] = "DEM"
        po = []
        po_start = start + pd.Timedelta(days=2 * games + 5)
        for team in po_teams:
            roster = players[players["TEAM_ABBREVIATION"] == team].nlargest(9, "mpg")
            for g in range(10):
                for p in roster.itertuples():
                    po.append((p.PLAYER_ID, p.PLAYER_NAME, p.TEAM_ID, team, f"{season}-po-{team}-{g}",
                               (po_start + pd.Timedelta(days=g)).strftime("%Y-%m-%d"),
                               float(np.clip(rng.normal(p.mpg * 1.08, 3), 1, 46)), 0, 0, 0, 0, 0))
        cache.write(pd.DataFrame(po, columns=gamelog.columns), season, PLAYOFFS, "gamelog")

        # DARKO-style CSV, written the way a user would save it
        o = 0.9 * tr[:, 0] + players["quality"] + rng.normal(0, 0.4, len(players))
        d = 0.8 * tr[:, 1] + 0.4 * players["quality"] + rng.normal(0, 0.4, len(players))
        pd.DataFrame({"nba_id": players["PLAYER_ID"], "Player": players["PLAYER_NAME"],
                      "O-DPM": o.round(2), "D-DPM": d.round(2), "DPM": (o + d).round(2)}) \
            .to_csv(impact_dir / f"darko_{season}.csv", index=False)
    return root / "cache", impact_dir


def make_contracts(data: dict, seed: int = 7) -> pd.DataFrame:
    """Made-up contracts for the website's cap game, in read_contracts' format: better shapes
    cost more, with noise so there are bargains."""
    rng = np.random.default_rng(seed)
    season = data["meta"]["seasons"][-1]
    rows = []
    for r in data["players"][season]:
        pct = [v for v in r[6:10] if v is not None]
        if len(pct) < 4 or not r[2]:
            continue
        quality = (sum(pct) / 400) ** 2.2
        salary = 1.3e6 + 48e6 * quality * rng.uniform(0.55, 1.35)
        rows.append({"PLAYER_ID": r[0], "bbref_id": f"demo{r[0]}", "name": r[1], "team": r[2].split("/")[0],
                     "salary": int(min(salary, 0.35 * 164_961_000)), "guaranteed": float("nan")})
    return pd.DataFrame(rows).sort_values("salary", ascending=False).reset_index(drop=True)
