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
