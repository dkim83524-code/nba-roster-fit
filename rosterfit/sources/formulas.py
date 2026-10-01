"""Ben Taylor's published box-score formulas (Nylon Calculus, 2017). All inputs are per 100 possessions."""
from __future__ import annotations

import numpy as np


def three_point_proficiency(fg3a_per100, fg3_pct):
    """Volume-aware 3-point skill: a sigmoid on attempts times accuracy.

    Barely shooting 3s scores near zero however accurate; past ~4 attempts per 100 it's
    essentially the shooting percentage.
    """
    fg3a_per100 = np.asarray(fg3a_per100, dtype=float)
    return (2.0 / (1.0 + np.exp(-fg3a_per100)) - 1.0) * np.asarray(fg3_pct, dtype=float)


def box_creation(ast, pts, tov, fg3a, fg3_pct):
    """Estimated open shots created for teammates per 100 possessions.

    Note the (PTS + TOV) term: it credits a scorer's gravity, so this is not a pure passing measure.
    """
    ast, pts, tov = (np.asarray(x, dtype=float) for x in (ast, pts, tov))
    prof = three_point_proficiency(fg3a, fg3_pct)
    return (ast * 0.1843 + (pts + tov) * 0.0969 - 2.3021 * prof
            + 0.0582 * (ast * (pts + tov) * prof) - 1.1942)
