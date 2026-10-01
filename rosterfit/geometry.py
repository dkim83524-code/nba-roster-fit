"""Square geometry.

The square spans [-1, 1] x [-1, 1] (area 4). Corners clockwise from top-left: TL, TR, BR, BL.
A shape puts a vertex on each center-to-corner diagonal at distance (percentile / 100) of the way
to that corner. Its area as a share of the square is

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
