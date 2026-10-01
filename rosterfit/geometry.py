"""Square geometry.

The square spans [-1, 1] x [-1, 1] (area 4). Corners clockwise from top-left: TL, TR, BR, BL.
A shape puts a vertex on each center-to-corner diagonal at distance r (0 to 1) of the way to that
corner, where r comes from the percentile through `radius` (the rank scale by default). Its area
as a share of the square is

    (v1*v2 + v2*v3 + v3*v4 + v4*v1) / 4  =  (v1 + v3) * (v2 + v4) / 4

so only neighboring corners multiply, and the two opposite pairs act as two factors.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import shapely
from shapely.geometry import Polygon

CORNER_XY = np.array([(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])  # TL, TR, BR, BL
SQUARE_AREA = 4.0
_EPS = 1e-12
SCALES = ("rank", "percentile")
RING_RANKS = (5, 20, 50)


def radius(pct, n: float, scale: str = "rank"):
    """Distance from the center (0 to 1) for a percentile among n players (or team-seasons).

    "percentile": pct / 100, so the 95th and the 99.7th percentile land almost together.
    "rank": 1 - ln(rank) / ln(1 + n), with rank = 1 + players ahead = 1 + n * (1 - pct / 100).
    The best reaches the corner and every time the rank halves (20th -> 10th -> 5th) the vertex
    moves the same step outward: ln 2 / ln(1 + n), about 12% of the way among 330 players. Talent
    at the top is spread out like that, so the 4th-best (0.76) is well clear of the 20th (0.48).
    NaN stays NaN.
    """
    p = np.clip(np.asarray(pct, dtype=float) / 100.0, 0.0, 1.0)
    if scale == "percentile":
        out = p
    else:
        m = max(float(n), 1.0)
        out = 1.0 - np.log1p(m * (1.0 - p)) / np.log1p(m)
    return float(out) if np.ndim(out) == 0 else out


def rings(n: float, scale: str = "rank") -> list[tuple[float, str]]:
    """Guide rings as (distance, label): top 5 / 20 / 50 on the rank scale, 25 / 50 / 75 otherwise."""
    if scale == "percentile":
        return [(p, f"{p * 100:.0f}") for p in (0.25, 0.5, 0.75)]
    m = max(float(n), 1.0)
    return [(1.0 - np.log1p(k - 1) / np.log1p(m), f"top {k}") for k in RING_RANKS]


def corner_vectors(order: Sequence[str]) -> dict[str, np.ndarray]:
    return {corner: CORNER_XY[i] for i, corner in enumerate(order)}


def _vector(values: Mapping[str, float], order: Sequence[str]) -> np.ndarray:
    v = np.array([float(values[c]) for c in order])
    if np.any(np.isnan(v)):
        raise ValueError(f"shape has a missing corner value: {dict(values)}")
    return np.clip(v, 0.0, 1.0)


def shape_points(values: Mapping[str, float], order: Sequence[str], scale: float = 1.0) -> np.ndarray:
    """Vertices (4 x 2) for corner values in [0, 1], optionally shrunk toward the center."""
    return CORNER_XY * (_vector(values, order) * scale)[:, None]


def area_fraction(values: Mapping[str, float], order: Sequence[str], scale: float = 1.0) -> float:
    v = _vector(values, order)
    return float(scale ** 2 * (v * np.roll(v, -1)).sum() / 4.0)


def to_polygon(points: np.ndarray) -> Polygon | None:
    poly = Polygon(points)
    if poly.area < _EPS:
        return None  # a line or a point: no area to add to a union
    return poly if poly.is_valid else shapely.make_valid(poly)


def union_fraction(polygons: Iterable[Polygon | None]) -> float:
    polys = [p for p in polygons if p is not None]
    if not polys:
        return 0.0
    return float(shapely.union_all(polys).area / SQUARE_AREA)


def overlap_fraction(polygons: Sequence[Polygon | None]) -> tuple[float, float, float]:
    """(sum of areas, union area, overlap = sum - union), each as a share of the square."""
    polys = [p for p in polygons if p is not None]
    total = sum(p.area for p in polys) / SQUARE_AREA
    union = union_fraction(polys)
    return total, union, max(total - union, 0.0)
