"""Season-string helpers. Seasons are written the NBA.com way: "2025-26"."""
from __future__ import annotations

import re

_SEASON = re.compile(r"^(\d{4})-(\d{2})$")

REGULAR = "Regular Season"
PLAYOFFS = "Playoffs"


def start_year(season: str) -> int:
    m = _SEASON.match(season)
    if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
        raise ValueError(f"bad season '{season}', expected e.g. '2025-26'")
    return int(m.group(1))


def season_from_start(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def season_from_end_year(year: int) -> str:
    return season_from_start(year - 1)


def season_range(first: str, last: str) -> list[str]:
    a, b = start_year(first), start_year(last)
    if b < a:
        raise ValueError(f"season range {first}..{last} is reversed")
    return [season_from_start(y) for y in range(a, b + 1)]


def parse_season_arg(arg: str) -> list[str]:
    """'2025-26' -> one season; '2017-18:2025-26' -> inclusive range."""
    if ":" in arg:
        first, last = arg.split(":", 1)
        return season_range(first.strip(), last.strip())
    start_year(arg)
    return [arg]


def previous(season: str, n: int = 1) -> str:
    return season_from_start(start_year(season) - n)


def at_least(season: str, first: str) -> bool:
    return start_year(season) >= start_year(first)
