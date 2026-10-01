"""Load and validate config.yaml into typed dataclasses.

Unknown keys raise an error so a typo in the config fails loudly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .seasons import season_range

CORNERS = ("offense", "playmaking", "portability", "defense")


@dataclass
class SeasonsCfg:
    target: str = "2025-26"
    window: list[str] = field(default_factory=lambda: ["2017-18", "2025-26"])


@dataclass
class QualifiedCfg:
    min_gp: int = 25
    min_mpg: float = 15.0


@dataclass
class CornersCfg:
    order: list[str] = field(default_factory=lambda: list(CORNERS))
    labels: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class MultiYearCfg:
    enabled: bool = False
    weights: list[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])


@dataclass
class SourceCfg:
    """One plus-minus source with offense and defense halves, from CSVs saved by hand."""
    dir: str = ""
    url: str = ""
    columns: dict[str, list[str]] = field(default_factory=dict)
    numeric_season_is_end_year: bool = True
    defense_higher_is_better: bool = True
    name_overrides: dict[str, int | str] = field(default_factory=dict)


IMPACT_CORNERS = ("offense", "defense")


@dataclass
class ImpactCfg:
    # the main source (DARKO), kept at the top level so older config.local.yaml files still load
    source_name: str = "DARKO"
    dir: str = "data/manual/darko"
    url: str = "https://www.darko.app"
    columns: dict[str, list[str]] = field(default_factory=dict)
    numeric_season_is_end_year: bool = True
    defense_higher_is_better: bool = True
    defense_multi_year: MultiYearCfg = field(default_factory=MultiYearCfg)
    name_overrides: dict[str, int | str] = field(default_factory=dict)
    extra_sources: dict[str, SourceCfg] = field(default_factory=dict)
    # corner -> source name -> weight; a corner left out uses the main source alone
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    min_weight_present: float = 0.2

    def sources(self) -> dict[str, SourceCfg]:
        main = SourceCfg(dir=self.dir, url=self.url, columns=self.columns,
                         numeric_season_is_end_year=self.numeric_season_is_end_year,
                         defense_higher_is_better=self.defense_higher_is_better,
                         name_overrides=self.name_overrides)
        return {self.source_name: main, **self.extra_sources}

    def weights_for(self, corner: str) -> dict[str, float]:
        return dict(self.weights.get(corner) or {self.source_name: 1.0})

    def active_sources(self) -> list[str]:
        """Sources with weight in either corner, in config order."""
        used = {n for c in IMPACT_CORNERS for n, w in self.weights_for(c).items() if w > 0}
        return [n for n in self.sources() if n in used]


@dataclass
class PlaymakingCfg:
    weights: dict[str, float] = field(default_factory=dict)
    min_weight_present: float = 0.5
    component_scaling: str = "z"


@dataclass
class PortabilityCfg:
    weights: dict[str, float] = field(default_factory=dict)
    min_weight_present: float = 0.5
    component_scaling: str = "z"
    cs3_pct_prior_attempts: float = 50
    versatility_min_partial_poss: float = 300


@dataclass
class TeamCfg:
    on_court_slots: int = 5
    capped_corners: list[str] = field(default_factory=lambda: ["playmaking", "portability"])
    floor_negative_corners: list[str] = field(default_factory=lambda: ["playmaking", "portability"])
    cap_mode: str = "pool_percentile"
    cap_pool_percentile: float = 80
    cap_players_worth: float = 1.75
    cap_reference_percentile: float = 85
    cap_reference_mpg: float = 34
    missing_value_z: float = 0.0


@dataclass
class PlayoffsCfg:
    rotation_size: int = 8


@dataclass
class DrawingCfg:
    theme: str = "light"
    minutes_scaling: str = "area"
    team_outline: bool = True
    min_minutes_to_draw: float = 100
    max_colored_players: int = 8
    fill_alpha: float = 0.16
    dpi: int = 200


@dataclass
class FetchCfg:
    cache_dir: str = "data/cache"
    manual_dir: str = "data/manual/nba"
    sleep_seconds: float = 1.0
    timeout: int = 60
    retries: int = 3


@dataclass
class Config:
    seasons: SeasonsCfg = field(default_factory=SeasonsCfg)
    qualified: QualifiedCfg = field(default_factory=QualifiedCfg)
    corners: CornersCfg = field(default_factory=CornersCfg)
    impact: ImpactCfg = field(default_factory=ImpactCfg)
    playmaking: PlaymakingCfg = field(default_factory=PlaymakingCfg)
    portability: PortabilityCfg = field(default_factory=PortabilityCfg)
    team: TeamCfg = field(default_factory=TeamCfg)
    playoffs: PlayoffsCfg = field(default_factory=PlayoffsCfg)
    drawing: DrawingCfg = field(default_factory=DrawingCfg)
    fetch: FetchCfg = field(default_factory=FetchCfg)
    # Directory that relative paths (cache_dir, impact.dir) resolve against.
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def window_seasons(self) -> list[str]:
        return season_range(self.seasons.window[0], self.seasons.window[1])

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base_dir / p

    def label(self, corner: str) -> str:
        return self.corners.labels.get(corner, corner.title())


def _build(cls: type, data: dict[str, Any] | None, where: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"config section '{where}' must be a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.name != "base_dir"}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        path = f"{where}.{name}" if where else name
        args = get_args(hint)
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, path)
        elif get_origin(hint) is dict and len(args) == 2 and dataclasses.is_dataclass(args[1]):
            if not isinstance(value or {}, dict):
                raise ValueError(f"config section '{path}' must be a mapping")
            kwargs[name] = {k: _build(args[1], v, f"{path}.{k}") for k, v in (value or {}).items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    order = cfg.corners.order
    if sorted(order) != sorted(CORNERS):
        raise ValueError(f"corners.order must be a permutation of {list(CORNERS)}, got {order}")
    for corner in cfg.team.capped_corners + cfg.team.floor_negative_corners:
        if corner not in CORNERS:
            raise ValueError(f"unknown corner '{corner}' in team settings")
    if len(cfg.seasons.window) != 2:
        raise ValueError("seasons.window must be [first_season, last_season]")
    cfg.window_seasons  # validates the season strings
    if cfg.drawing.minutes_scaling not in ("area", "none"):
        raise ValueError("drawing.minutes_scaling must be 'area' or 'none'")
    if cfg.drawing.theme not in ("light", "dark"):
        raise ValueError("drawing.theme must be 'light' or 'dark'")
    if not 0 < cfg.team.cap_reference_percentile < 100:
        raise ValueError("team.cap_reference_percentile must be between 0 and 100")
    if cfg.team.cap_mode not in ("pool_percentile", "players_worth"):
        raise ValueError("team.cap_mode must be 'pool_percentile' or 'players_worth'")
    if not 0 < cfg.team.cap_pool_percentile <= 100:
        raise ValueError("team.cap_pool_percentile must be in (0, 100]")
    sources = cfg.impact.sources()
    for corner, weights in cfg.impact.weights.items():
        if corner not in IMPACT_CORNERS:
            raise ValueError(f"impact.weights: '{corner}' isn't offense or defense")
        unknown = set(weights) - set(sources)
        if unknown:
            raise ValueError(f"impact.weights.{corner}: unknown source(s) {sorted(unknown)}; "
                             f"sources are {sorted(sources)} (add others under impact.extra_sources)")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"impact.weights.{corner}: weights must be >= 0 with at least one > 0")
    if not 0 < cfg.impact.min_weight_present <= 1:
        raise ValueError("impact.min_weight_present must be in (0, 1]")
    for section in (cfg.playmaking, cfg.portability):
        if section.component_scaling not in ("z", "rank"):
            raise ValueError("component_scaling must be 'z' or 'rank'")
        if any(w < 0 for w in section.weights.values()):
            raise ValueError("component weights must be >= 0 (orientation is handled in code)")
        if sum(section.weights.values()) <= 0:
            raise ValueError("at least one component weight must be > 0")


def _merge(base: dict, extra: dict) -> dict:
    """Recursively overlay `extra` on `base` (mappings merge, everything else replaces)."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def local_path(path: str | Path) -> Path:
    """config.yaml -> config.local.yaml: personal settings that git pull never touches."""
    path = Path(path)
    return path.with_name(f"{path.stem}.local{path.suffix}")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read config.yaml, then overlay config.local.yaml next to it if that file exists."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    local = local_path(path)
    if local.exists():
        with open(local) as fh:
            raw = _merge(raw, yaml.safe_load(fh) or {})
    cfg = _build(Config, raw, "")
    cfg.base_dir = path.resolve().parent
    validate(cfg)
    return cfg
