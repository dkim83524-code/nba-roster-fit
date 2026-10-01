import pandas as pd
import pytest

from rosterfit.sources.darko import blend_defense, load_impact
from rosterfit.sources.names import normalize_name

PLAYERS = {"2025-26": pd.DataFrame({"PLAYER_ID": [1, 2, 3],
                                    "PLAYER_NAME": ["Nikola Jokić", "Jaren Jackson Jr.", "Kyle Anderson"]})}


def write(cfg, name, df):
    folder = cfg.path(cfg.impact.dir)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / name, index=False)


def test_name_normalization():
    assert normalize_name("Nikola Jokić") == "nikola jokic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("D'Angelo Russell") == "dangelo russell"


def test_season_from_filename_and_name_matching(cfg):
    write(cfg, "darko_2025-26.csv", pd.DataFrame({
        "Player": ["Nikola Jokic", "Jaren Jackson", "Nobody Here"],
        "O-DPM": [8.0, 0.5, 1.0], "D-DPM": [1.0, 2.0, 0.0]}))
    impact, report = load_impact(cfg, PLAYERS)
    assert sorted(impact["PLAYER_ID"]) == [1, 2]
    assert impact.set_index("PLAYER_ID").loc[1, "offense"] == 8.0
    assert report.unmatched["2025-26"] == ["Nobody Here"]
    assert report.columns["darko_2025-26.csv"]["offense"] == "O-DPM"


def test_numeric_season_column_is_end_year_and_ids_win(cfg):
    write(cfg, "history.csv", pd.DataFrame({
        "nba_id": [3, 3], "player_name": ["K. Anderson", "K. Anderson"], "season": [2026, 2025],
        "o_dpm": [1.0, 9.0], "d_dpm": [0.5, 9.0]}))
    impact, _ = load_impact(cfg, PLAYERS)
    row = impact[(impact["season"] == "2025-26")].iloc[0]
    assert (row["PLAYER_ID"], row["offense"]) == (3, 1.0)
    assert set(impact["season"]) == {"2025-26", "2024-25"}


def test_dated_rows_use_end_of_regular_season(cfg):
    write(cfg, "daily_2025-26.csv", pd.DataFrame({
        "nba_id": [1, 1, 1, 2], "date": ["2026-03-01", "2026-04-12", "2026-05-20", "2026-06-30"],
        "o_dpm": [1.0, 2.0, 3.0, 4.0], "d_dpm": [0.0, 0.0, 0.0, 0.0]}))
    impact, _ = load_impact(cfg, PLAYERS, {"2025-26": pd.Timestamp("2026-04-12")})
    by_id = impact.set_index("PLAYER_ID")["offense"]
    assert by_id[1] == 2.0   # last value on or before the season end
    assert by_id[2] == 4.0   # only an offseason row exists: use it


def test_name_override(cfg):
    cfg.impact.name_overrides = {"Big Honey": 2}
    write(cfg, "darko_2025-26.csv", pd.DataFrame({"Player": ["Big Honey"], "O-DPM": [1.0], "D-DPM": [1.0]}))
    impact, report = load_impact(cfg, PLAYERS)
    assert impact["PLAYER_ID"].tolist() == [2]
    assert not report.unmatched


def test_missing_offense_column_is_a_clear_error(cfg):
    write(cfg, "darko_2025-26.csv", pd.DataFrame({"Player": ["X"], "DPM": [1.0]}))
    with pytest.raises(ValueError, match="no column found"):
        load_impact(cfg, PLAYERS)


def test_defense_blend_renormalizes():
    impact = pd.DataFrame({"season": ["2025-26", "2024-25", "2025-26"], "PLAYER_ID": [1, 1, 2],
                           "offense": [0, 0, 0], "defense": [2.0, 4.0, 1.0]})
    out = blend_defense(impact, [0.6, 0.3, 0.1]).set_index(["season", "PLAYER_ID"])["defense"]
    assert out[("2025-26", 1)] == pytest.approx((0.6 * 2 + 0.3 * 4) / 0.9)
    assert out[("2025-26", 2)] == pytest.approx(1.0)


def test_name_variants_match_by_last_name_word_order_and_renames():
    from rosterfit.sources.names import match_names

    roster = pd.DataFrame({"PLAYER_ID": [1, 2, 3, 4, 5, 6, 7], "PLAYER_NAME": [
        "Nic Claxton", "Bones Hyland", "Yongxi Cui", "Enes Freedom", "KJ Martin", "Jalen Williams",
        "Jaylin Williams"]})
    darko = ["Nicolas Claxton", "Nah'Shon Hyland", "Cui Yongxi", "Enes Kanter", "Kenyon Martin Jr.",
             "Kevin Durant",            # didn't play: no NBA.com row, stays unmatched
             "Jeenathan Williams"]      # two unmatched Williamses on NBA.com: too ambiguous to guess
    mapping, loose = match_names(darko, roster)
    assert mapping == {"Nicolas Claxton": 1, "Nah'Shon Hyland": 2, "Cui Yongxi": 3, "Enes Kanter": 4,
                       "Kenyon Martin Jr.": 5}
    assert ("Cui Yongxi", "Yongxi Cui") in loose and ("Nicolas Claxton", "Nic Claxton") in loose
    assert ("Kenyon Martin Jr.", "KJ Martin") in loose
    loose_names = {name for name, _ in loose}
    assert not loose_names & {"Enes Kanter", "Nah'Shon Hyland"}  # known renames count as exact


def test_last_name_rule_skips_players_already_matched():
    from rosterfit.sources.names import match_names

    roster = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["LaMelo Ball", "Lonzo Ball"]})
    mapping, loose = match_names(["LaMelo Ball", "Lonzo Ball Sr."], roster)
    assert mapping == {"LaMelo Ball": 1, "Lonzo Ball Sr.": 2}
    # Lonzo sat out the season and LaMelo has no row of his own: never hand Lonzo's value to LaMelo
    mapping, loose = match_names(["Lonzo Ball"], roster.iloc[:1])
    assert mapping == {} and loose == []


def test_override_by_nba_name_or_id():
    from rosterfit.sources.names import match_names

    roster = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Herbert Jones", "Tari Eason"]})
    mapping, _ = match_names(["Herb Jones", "T. Eason"], roster,
                             {"Herb Jones": "Herbert Jones", "T. Eason": 2})
    assert mapping == {"Herb Jones": 1, "T. Eason": 2}
