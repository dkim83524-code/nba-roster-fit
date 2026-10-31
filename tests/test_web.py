import json
import shutil
import subprocess

import pytest

from rosterfit import geometry
from rosterfit.config import CORNERS
from rosterfit.team import PLAYOFF_MODE, REGULAR_MODE, evaluate_team
from rosterfit.synthetic import make_contracts
from rosterfit.web import HERE, add_game, build_web_data, render_page, write_web


@pytest.fixture
def data(league, cfg):
    return build_web_data(league, cfg, synthetic=True, names={"DEM": "Demo Team"})


def test_export_is_strict_json_with_the_expected_shape(data):
    text = json.dumps(data, allow_nan=False)  # browsers can't parse NaN
    assert len(text) > 10_000
    m = data["meta"]
    assert m["seasons"] == ["2023-24", "2024-25", "2025-26"]
    assert [x["name"] for x in m["impact_sources"]] == ["DARKO", "LEBRON"]
    assert m["impact_sources"][1]["defense"] == 2.0
    for season in m["seasons"]:
        row = data["players"][season][0]
        assert len(row) == 9 + 2 * len(CORNERS)  # ... percentiles, z-scores, off. role, def. role, position
        assert any(r[-2] for r in data["players"][season])
        assert {r[-1] for r in data["players"][season]} <= {"PG", "SG", "SF", "PF", "C", None}
        assert data["teams"][season]["DEM"]["name"] == "Demo Team"
        po = data["teams"][season]["DEM"]["playoffs"]
        assert 0 < len(po) <= m["rotation_size"]
    for mode in (REGULAR_MODE, PLAYOFF_MODE):
        for c in CORNERS:
            pool = data["pools"][mode][c]
            assert pool == sorted(pool) and len(pool) == m["pool_sizes"][mode]


def test_page_inlines_core_and_data(data, tmp_path):
    page = render_page(data)
    assert "/*__ROSTERFIT_DATA__*/" not in page and "/*__ROSTERFIT_CORE__*/" not in page
    assert "RosterFitCore" in page and "Demo Team" in page
    assert not page.lstrip().lower().startswith("<!doctype")  # a fragment; standalone() adds the shell
    paths = write_web(data, tmp_path)
    assert paths["page"].read_text().startswith("<!doctype html>")
    assert json.loads(paths["data"].read_text())["meta"]["synthetic"] is True


_NODE_SCRIPT = """
const RF = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const index = RF.playerIndex(data);
const out = {};
for (const [season, teams] of Object.entries(data.teams)) {
  for (const abbr of Object.keys(teams)) {
    for (const mode of ["regular", "playoffs"]) {
      if (!teams[abbr][mode].length) continue;
      const r = RF.evaluateTeam(data, index, season, abbr, mode);
      out[[season, abbr, mode].join("|")] = {
        coverage: r.coverage, depth: r.depth, overlap: r.overlap, sumAreas: r.sumAreas,
        pct: r.pct, values: r.values, redundancy: r.redundancy,
      };
    }
  }
}
const season = data.meta.seasons[data.meta.seasons.length - 1];
const picks = data.teams[season].DEM.regular.slice(0, 5).map(([pid]) => ({ season, pid }));
const lineup = RF.evaluateLineup(data, index, picks);
out.lineup = { picks, coverage: lineup.coverage, sumAreas: lineup.sumAreas,
  weights: lineup.players.map((p) => p.weight) };
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_math_matches_python(league, data, tmp_path):
    script = tmp_path / "parity.js"
    script.write_text(_NODE_SCRIPT)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    run = subprocess.run(["node", str(script), str(HERE / "core.js"), str(data_path)],
                         capture_output=True, text=True, check=True)
    js = json.loads(run.stdout)

    checked = 0
    for key, r in js.items():
        if key == "lineup":
            continue
        season, abbr, mode = key.split("|")
        py = evaluate_team(league, abbr, season, mode)
        assert r["coverage"] == pytest.approx(py.coverage, abs=2e-3), key  # sampled vs exact union
        assert r["sumAreas"] == pytest.approx(py.sum_areas, abs=1e-9), key
        assert r["overlap"] == pytest.approx(py.overlap, abs=2e-3), key
        assert r["depth"] == pytest.approx(py.depth, abs=1e-9), key
        for c in CORNERS:
            assert r["pct"][c] == pytest.approx(py.pct[c], abs=1e-9), (key, c)
            assert r["values"][c] == pytest.approx(py.values[c], abs=1e-9), (key, c)
        for c in league.cfg.team.capped_corners:
            assert r["redundancy"][c] == pytest.approx(py.redundancy[c], abs=1e-9), (key, c)
        checked += 1
    assert checked >= 2 * len(data["teams"]["2025-26"])

    # Builder: five full-size starters, weight 1 each; coverage is the exact union of their shapes.
    lineup = js["lineup"]
    assert lineup["weights"] == [1, 1, 1, 1, 1]
    order = league.cfg.corners.order
    table = league.players[lineup["picks"][0]["season"]].set_index("PLAYER_ID")
    n = int(table["qualified"].sum())
    polys = []
    for pick in lineup["picks"]:
        row = table.loc[pick["pid"]]
        vals = {c: geometry.radius(row[f"{c}_pct"], n, league.cfg.drawing.scale) for c in CORNERS}
        polys.append(geometry.to_polygon(geometry.shape_points(vals, order)))
    assert lineup["coverage"] == pytest.approx(geometry.union_fraction(polys), abs=2e-3)


def test_cap_game_deck_holds_only_players_with_full_shapes(data):
    contracts = make_contracts(data)
    info = add_game(data, "2026-27", contracts, 164_961_000, min_minutes=500)
    game = data["game"]
    assert game["stats_season"] == "2025-26" and game["cap"] == 164_961_000
    assert info["deck"] == len(game["deck"]) > 0
    rows = {r[0]: r for r in data["players"]["2025-26"]}
    for pid, salary, team, pos in game["deck"]:
        r = rows[pid]
        assert r[4] >= 500 and all(v is not None for v in r[6:14])
        assert salary > 0 and team in data["teams"]["2025-26"]
        assert pos == r[-1] and pos in {"PG", "SG", "SF", "PF", "C"}
    assert [d[1] for d in game["deck"]] == sorted((d[1] for d in game["deck"]), reverse=True)
    json.dumps(data, allow_nan=False)


def test_cap_game_matches_contract_names(data):
    rows = data["players"]["2025-26"]
    names = {}
    for r in rows:  # synthetic names repeat; keep the unique ones
        names[r[1]] = None if r[1] in names else r
    unique = [r for r in names.values() if r and r[4] >= 500 and None not in r[6:14]][:3]
    import pandas as pd
    contracts = pd.DataFrame({"bbref_id": ["a", "b", "c", "d"], "name": [r[1] for r in unique] + ["Rookie Nobody"],
                              "team": ["DEM"] * 4, "salary": [3, 2, 1, 9], "guaranteed": [0.0] * 4})
    info = add_game(data, "2026-27", contracts, 100)
    assert info["unmatched"] == ["Rookie Nobody"]
    assert sorted(d[0] for d in data["game"]["deck"]) == sorted(r[0] for r in unique)


_DEAL_SCRIPT = """
const RF = require(process.argv[2]);
const teams = ["SAS", "NYK", "BOS", "DEN", "OKC", "LAL"];
const a = RF.dealOrder(teams, RF.hashSeed("2026-27|2026-10-01"));
const b = RF.dealOrder(teams.slice().reverse(), RF.hashSeed("2026-27|2026-10-01"));
const c = RF.dealOrder(teams, RF.hashSeed("2026-27|2026-10-02"));
const floor = [RF.signingFloor([[9, 2, 5], [3, 4], [], [7]], 2), RF.signingFloor([[9, 2, 5], [3, 4], [], [7]], 5)];
console.log(JSON.stringify({ a, b, c, floor, afford: [RF.canAfford(10, 14, 4), RF.canAfford(10, 13.9, 4)] }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_daily_deal_is_the_same_for_everyone(tmp_path):
    script = tmp_path / "deal.js"
    script.write_text(_DEAL_SCRIPT)
    run = subprocess.run(["node", str(script), str(HERE / "core.js")], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    assert sorted(out["a"]) == sorted(["SAS", "NYK", "BOS", "DEN", "OKC", "LAL"])
    assert out["a"] == out["b"]  # input order doesn't matter, only the seed
    assert out["a"] != out["c"]
    assert out["afford"] == [True, False]
    assert out["floor"] == [3, 7]  # 2 teams have someone at or under 3; with too few teams, the priciest minimum


_NODE_3D = """
const RF = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const dirs = RF.sphereDirections(4000);
const one = [0.9, 0.2, 0.5, 0.7, 0.3, 0.6];
const sk = RF.skillIndex(data);
const season = data.meta.seasons[data.meta.seasons.length - 1];
const entries = data.teams[season].DEM.regular.map(([pid, min]) => ({ pid, min, ...sk[season].get(pid) }));
const team = RF.evaluate3d(entries, { scale: "percentile", n: data.meta.qualified[season],
  sizeFloor: data.skills.size_floor, minutes: true, minMinutes: data.meta.min_minutes_to_draw, dirs });
console.log(JSON.stringify({
  unit: RF.unionVolumeShare([[1, 1, 1, 1, 1, 1]], dirs),
  single: RF.unionVolumeShare([one], dirs), singleExact: RF.octaVolumeShare(one),
  nested: RF.unionVolumeShare([one, one.map((v) => v / 2)], dirs),
  team: { coverage: team.coverage, sum: team.sumVolumes, best: team.best,
          shapes: team.players.filter((p) => p.drawn).map((p) => p.r) },
}));
"""


def test_3d_export(data, cfg, tmp_path):
    sk = data["skills"]
    assert sk["order"] == cfg.skills.order and len(sk["pairs"]) == 3
    for season in data["meta"]["seasons"]:
        assert sk["available"][season] == cfg.skills.order
        rows = sk["players"][season]
        assert len(rows) == len(data["players"][season]) and all(len(r) == 8 for r in rows)
    json.dumps(sk, allow_nan=False)
    paths = write_web(data, tmp_path)
    page = paths["page_3d"].read_text()
    assert page.startswith("<!doctype html>") and "evaluate3d" in page and '"pools"' not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_3d_volume_math(data, tmp_path):
    import numpy as np

    script = tmp_path / "vol.js"
    script.write_text(_NODE_3D)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    run = subprocess.run(["node", str(script), str(HERE / "core.js"), str(data_path)],
                         capture_output=True, text=True, check=True)
    js = json.loads(run.stdout)
    assert js["unit"] == pytest.approx(1.0, abs=1e-3)                  # the full shape is 100%
    assert js["single"] == pytest.approx(js["singleExact"], abs=1e-3)  # (a+b)(c+d)(e+f)/8
    assert js["nested"] == pytest.approx(js["single"], abs=1e-9)       # a shape inside another adds nothing

    # the team's union volume, checked independently: random points in the cube [-1, 1]^3 (8 units),
    # inside a shape when |x|/X + |y|/Y + |z|/Z <= 1 with that octant's radii; the full shape is 4/3
    team = js["team"]
    shapes = np.array(team["shapes"])
    assert len(shapes) >= 5
    pts = np.random.default_rng(0).uniform(-1, 1, (200_000, 3))
    inside = np.zeros(len(pts), dtype=bool)
    for r in shapes:
        radii = np.where(pts >= 0, r[[0, 2, 4]], r[[1, 3, 5]])
        inside |= (np.abs(pts) / radii).sum(axis=1) <= 1
    assert team["coverage"] == pytest.approx(inside.mean() * 8 / (4 / 3), abs=3e-3)
    assert team["coverage"] <= team["sum"] + 1e-9
    for i, b in enumerate(team["best"]):
        assert b["value"] == pytest.approx(shapes[:, i].max())
