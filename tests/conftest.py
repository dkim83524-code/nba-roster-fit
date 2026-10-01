from pathlib import Path

import pytest

from rosterfit.cache import Cache
from rosterfit.config import load_config
from rosterfit.synthetic import make_league, use_synthetic_sources
from rosterfit.team import build_league

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ["2023-24", "2024-25", "2025-26"]


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c.base_dir = tmp_path
    return c


@pytest.fixture(scope="session")
def synthetic_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("league")
    return make_league(root, SEASONS, n_teams=12, games=30, seed=3)


@pytest.fixture
def league(cfg, synthetic_dirs):
    cache_dir, impact_dir = synthetic_dirs
    use_synthetic_sources(cfg, impact_dir)
    cfg.seasons.window = [SEASONS[0], SEASONS[-1]]
    cfg.qualified.min_gp = 10
    return build_league(cfg, Cache(cache_dir), SEASONS)
