import numpy as np
import pandas as pd
import pytest

from rosterfit.config import PartCfg
from rosterfit.metrics.pool import composite, parts_sum, rank_normal
from rosterfit.metrics.portability import three_point_spacing


def test_spacing_counts_all_threes_with_catch_and_shoot_weighted():
    players = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "FG3A": [100.0, 100.0, 0.0], "FG3M": [40.0, 40.0, 0.0],
                            "POSS": [1000.0, 1000.0, 1000.0]})
    # player 1: all 100 threes catch-and-shoot; player 2: none (pull-ups); player 3 never shoots
    cs = pd.DataFrame({"PLAYER_ID": [1], "CATCH_SHOOT_FG3A": [100.0], "CATCH_SHOOT_FG3M": [40.0]})
    volume, accuracy = three_point_spacing(players, cs, prior_attempts=0, cs_weight=1.5)
    assert volume.tolist() == pytest.approx([15.0, 10.0, 0.0])   # 150 and 100 weighted attempts per 1000 poss
    assert accuracy[0] == pytest.approx(0.4) and accuracy[1] == pytest.approx(0.4)
    # shrinkage pulls a tiny sample toward the league's weighted 3P%
    v2, a2 = three_point_spacing(players, cs, prior_attempts=50, cs_weight=1.5)
    league = (60 + 40) / (150 + 100)
    assert a2[2] == pytest.approx(league)


def _parts():
    return {"spacing": PartCfg(0.4, {"shoot": 1.0}), "screening": PartCfg(0.4, {"oreb": 0.5, "screens": 0.5}),
            "versatility": PartCfg(0.2, {"vers": 1.0})}


def test_parts_sum_ignores_a_weak_part_and_rewards_both():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"shoot": rng.normal(0, 1, n), "oreb": rng.normal(0, 1, n),
                       "screens": rng.normal(0, 1, n), "vers": rng.normal(0, 1, n)})
    df.loc[0] = [3.0, -3.0, -3.0, 0.0]     # pure sniper: elite shooting, no interior game
    df.loc[1] = [-3.0, 3.0, 3.0, 0.0]      # pure rim-runner
    df.loc[2] = [1.5, 1.5, 1.5, 0.0]       # shooting big: good at both
    df.loc[3] = [0.3, 0.3, 0.3, 0.3]       # a bit above average everywhere
    pool = pd.Series(True, index=df.index)
    out = parts_sum(df, _parts(), pool, scaling="rank")

    # the sniper's empty interior counts 0, not a negative: same as if he were average there
    assert out[0] == pytest.approx(0.4 * rank_normal(df["shoot"], pool)[0] + 0.2 * rank_normal(df["vers"], pool)[0].clip(0))
    assert abs(out[0] - out[1]) < 0.15                 # sniper and rim-runner land together
    assert out[2] > max(out[0], out[1])                # the shooting big earns both halves
    assert out[0] > out[3] and out[1] > out[3]         # specialists beat a jack of all trades
    # under the plain average the sniper sinks below the jack of all trades
    flat = composite(df, {"shoot": 0.4, "oreb": 0.2, "screens": 0.2, "vers": 0.2}, pool, scaling="rank")
    assert flat[0] < flat[3]


def test_parts_sum_is_nan_only_without_any_part():
    df = pd.DataFrame({"shoot": [1.0, np.nan, 0.5, -1.0], "oreb": [np.nan] * 4, "screens": [np.nan] * 4,
                       "vers": [np.nan] * 4})
    out = parts_sum(df, _parts(), pd.Series(True, index=df.index), scaling="z")
    assert np.isnan(out[1]) and not np.isnan(out[0]) and out[3] == 0.0
