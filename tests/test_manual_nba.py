import json
import os
import shutil
import subprocess
from urllib.parse import parse_qsl, urlparse

import pandas as pd
import pytest

from rosterfit.cache import Cache
from rosterfit.seasons import PLAYOFFS, REGULAR
from rosterfit.sources.manual_nba import (browser_script, download_jobs, identify, import_manual,
                                          request_url, to_totals)


def response(resource, params, headers, rows, key="resultSets"):
    rs = {"name": resource, "headers": headers, "rowSet": rows}
    return {"resource": resource, "parameters": params, key: [rs] if key == "resultSets" else rs}


PASSING = response("leaguedashptstats",
                   {"Season": "2025-26", "SeasonType": "Regular Season", "PtMeasureType": "Passing",
                    "PerMode": "Totals", "PlayerOrTeam": "Player"},
                   ["PLAYER_ID", "PLAYER_NAME", "GP", "POTENTIAL_AST", "AST_PTS_CREATED"],
                   [[1, "A", 50, 400, 600], [2, "B", 40, 100, 150]])


def test_request_url_matches_nba_api_parameters():
    url = request_url("passing", "2025-26", REGULAR)
    parsed = urlparse(url)
    assert parsed.netloc == "stats.nba.com" and parsed.path == "/stats/leaguedashptstats"
    keys = [k for k, _ in parse_qsl(parsed.query, keep_blank_values=True)]
    assert keys == sorted(keys)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    assert q["PtMeasureType"] == "Passing" and q["PerMode"] == "Totals" and q["Season"] == "2025-26"
    # None-valued parameters are dropped, like requests does
    assert "DefTeamID" not in dict(parse_qsl(urlparse(request_url("matchups", "2025-26", REGULAR)).query))
    assert "DefTeamID=1610612752" in request_url("matchups", "2025-26", REGULAR, def_team_id=1610612752)


def test_download_jobs_skip_cached_and_untracked(tmp_path):
    cache = Cache(tmp_path)
    cache.write(pd.DataFrame({"x": [1]}), "2025-26", REGULAR, "passing")
    jobs = download_jobs(["passing", "hustle", "gamelog"], ["2015-16", "2025-26"], cache)
    keys = {(j["table"], j["season"], j["season_type"]) for j in jobs}
    assert ("passing", "2025-26", REGULAR) not in keys          # already cached
    assert ("hustle", "2015-16", REGULAR) not in keys           # not tracked yet in 2015-16
    assert ("gamelog", "2025-26", PLAYOFFS) in keys
    assert ("passing", "2015-16", REGULAR) in keys


def test_identify_from_response_and_from_file_name():
    assert identify(PASSING) == ("passing", "2025-26", REGULAR, "Totals")
    listy = dict(PASSING, parameters=[{k: v} for k, v in PASSING["parameters"].items()])
    assert identify(listy)[0] == "passing"
    bare = {"resultSets": PASSING["resultSets"]}
    assert identify(bare, "gamelog_2024-25_playoffs.json") == ("gamelog", "2024-25", PLAYOFFS, "Totals")
    with pytest.raises(ValueError, match="not a table"):
        identify({"resource": "playerindex", "parameters": {"Season": "2025-26"}})


def test_per_game_download_is_turned_back_into_totals():
    df = pd.DataFrame({"GP": [10, 20], "POTENTIAL_AST": [5.0, 2.5], "AST_PTS_CREATED": [8.0, 1.0]})
    out = to_totals(df, "passing", "PerGame")
    assert out["POTENTIAL_AST"].tolist() == [50.0, 50.0]
    with pytest.raises(ValueError, match="Totals"):
        to_totals(df, "passing", "Per100Possessions")


def test_import_single_response_and_bundle(tmp_path):
    manual, cache = tmp_path / "manual", Cache(tmp_path / "cache")
    manual.mkdir()
    (manual / "anything.json").write_text(json.dumps(PASSING))
    part = lambda ids: response("matchupsrollup", {"Season": "2025-26", "SeasonType": "Regular Season",  # noqa: E731
                                                   "PerMode": "Totals"},
                                ["DEF_PLAYER_ID", "POSITION", "PARTIAL_POSS"],
                                [[i, "G", 100.0 + i] for i in ids], key="resultSet")
    bundle = {"rosterfit_bundle": 1, "items": [
        {"table": "matchups", "season": "2025-26", "season_type": REGULAR, "response": part([1, 2])},
        {"table": "matchups", "season": "2025-26", "season_type": REGULAR, "response": part([3])},
        {"table": "gamelog", "season": "2025-26", "season_type": PLAYOFFS,
         "response": response("leaguegamelog", {}, ["PLAYER_ID", "MIN"], [])},
    ]}
    (manual / "nba_stats_2025-26.json").write_text(json.dumps(bundle))
    (manual / "notes.json").write_text("{not json")

    messages = import_manual(manual, cache)
    assert any(m.startswith("imported passing") for m in messages)
    assert any("skipped notes.json" in m for m in messages)
    assert cache.read("2025-26", REGULAR, "passing")["POTENTIAL_AST"].tolist() == [400, 100]
    assert sorted(cache.read("2025-26", REGULAR, "matchups")["DEF_PLAYER_ID"]) == [1, 2, 3]
    assert cache.has("2025-26", PLAYOFFS, "gamelog")  # an empty playoff log is still recorded

    assert not [m for m in import_manual(manual, cache) if m.startswith("imported")]  # nothing new
    later = os.path.getmtime(cache.path("2025-26", REGULAR, "passing")) + 10
    os.utime(manual / "anything.json", (later, later))
    assert any(m.startswith("imported passing") for m in import_manual(manual, cache))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_script_is_valid_javascript(tmp_path):
    jobs = download_jobs(["passing", "matchups"], ["2025-26"])
    path = tmp_path / "dl.js"
    path.write_text(browser_script(jobs))
    subprocess.run(["node", "--check", str(path)], check=True)
