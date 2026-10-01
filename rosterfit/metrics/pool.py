"""The qualified pool, z-scores and percentiles.

Percentile convention used everywhere (players and teams): the share of the pool at or below a
value, times 100. The best value in the pool scores 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qualified_mask(gp: pd.Series, mpg: pd.Series, min_gp: int, min_mpg: float) -> pd.Series:
    return (gp >= min_gp) & (mpg >= min_mpg)


def zscore(values: pd.Series, pool: pd.Series) -> pd.Series:
    """Standardize with the mean and SD of the pool rows (NaNs ignored)."""
    ref = values[pool & values.notna()]
    if len(ref) < 2 or ref.std(ddof=0) == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - ref.mean()) / ref.std(ddof=0)


def percentile_against(values, pool_values) -> np.ndarray:
    """100 x share of pool_values <= each value. NaN stays NaN."""
    ref = np.sort(np.asarray(pool_values, dtype=float))
    ref = ref[~np.isnan(ref)]
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan)
    if len(ref) == 0:
        return out
    ok = ~np.isnan(vals)
    out[ok] = 100.0 * np.searchsorted(ref, vals[ok], side="right") / len(ref)
    return out


def composite(components: pd.DataFrame, weights: dict[str, float], pool: pd.Series,
              negative: tuple[str, ...] = (), min_weight_present: float = 0.5) -> pd.Series:
    """Weighted mean of pool-standardized components, renormalized over what each player has.

    A player missing more than (1 - min_weight_present) of the total weight gets NaN.
    """
    active = {k: w for k, w in weights.items() if w > 0}
    total = sum(active.values())
    num = pd.Series(0.0, index=components.index)
    den = pd.Series(0.0, index=components.index)
    for name, w in active.items():
        z = zscore(components[name].astype(float), pool)
        if name in negative:
            z = -z
        has = z.notna()
        num = num + (z.fillna(0.0) * w)
        den = den + (has * w)
    out = num / den.where(den > 0)
    return out.where(den >= min_weight_present * total)
