import pandas as pd
import pytest
import yaml

from rosterfit.cli import main
from rosterfit.config import load_config

from .conftest import ROOT


def test_demo_writes_png_and_csv(tmp_path):
    out = tmp_path / "demo.png"
    assert main(["--config", str(ROOT / "config.yaml"), "demo", "--out", str(out)]) == 0
    assert out.stat().st_size > 50_000
    table = pd.read_csv(out.with_suffix(".csv"))
    assert {"PLAYER_NAME", "MIN", "offense_pct", "playmaking_contrib"} <= set(table.columns)


def test_demo_playoffs_dark(tmp_path):
    out = tmp_path / "po.png"
    assert main(["--config", str(ROOT / "config.yaml"), "demo", "--playoffs", "--theme", "dark",
                 "--out", str(out)]) == 0
    assert out.exists()


def test_config_rejects_unknown_keys(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["team"]["cap_player_worth"] = 2  # typo
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_config_rejects_bad_corner_order(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["corners"]["order"] = ["offense", "offense", "portability", "defense"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="permutation"):
        load_config(path)


def test_fetch_stops_early_in_codespaces(monkeypatch, capsys):
    monkeypatch.setenv("CODESPACES", "true")
    assert main(["--config", str(ROOT / "config.yaml"), "fetch"]) == 2
    assert "browser-script" in capsys.readouterr().err


def test_config_local_overlays_and_is_validated(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    (tmp_path / "config.local.yaml").write_text(yaml.safe_dump(
        {"playmaking": {"component_scaling": "rank", "weights": {"turnovers_per100": 0.2}}}))
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.playmaking.component_scaling == "rank"
    assert cfg.playmaking.weights["turnovers_per100"] == 0.2
    assert cfg.playmaking.weights["potential_ast_per100"] == 0.5  # untouched keys keep their value
    (tmp_path / "config.local.yaml").write_text(yaml.safe_dump({"team": {"cap_moed": "x"}}))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(tmp_path / "config.yaml")


def test_explain_lists_ingredients(league):
    from rosterfit.explain import format_table, ingredients

    players = league.players["2025-26"]
    minutes = league.minutes["regular"]["2025-26"]
    table = ingredients(players, minutes, "DEM", "portability", league.cfg, top=5)
    assert list(table.columns) == ["PLAYER_NAME", "MIN", "C&S 3s", "OREB%", "Screen AST", "Versatility",
                                   "corner"]
    assert len(table) == 5 and table["MIN"].is_monotonic_decreasing
    assert table[["C&S 3s", "OREB%"]].stack().between(0, 100).all()
    text = format_table(table, "title", league.cfg.portability.weights, "rank")
    assert "weights: C&S 3s 0.25" in text and "scaling: rank" in text


def test_compare_two_teams(league, tmp_path):
    from rosterfit.cli import describe_compare
    from rosterfit.plot import render_compare
    from rosterfit.team import evaluate_team

    other = [t for t in league.teams("2025-26") if t != "DEM"][0]
    a = evaluate_team(league, "DEM", "2025-26")
    b = evaluate_team(league, other, "2025-26")
    text = describe_compare(a, b, league.cfg)
    assert "Coverage" in text and f"{a.coverage * 100:.0f}%" in text and other in text
    assert "Depth" in text and f"{a.depth * 100:.0f}%" in text
    out = render_compare(a, b, league.cfg, tmp_path / "cmp.png")
    assert out.stat().st_size > 50_000


def test_no_outline_flag(tmp_path):
    out = tmp_path / "demo.png"
    assert main(["--config", str(ROOT / "config.yaml"), "demo", "--no-outline", "--out", str(out)]) == 0
    assert out.exists()
