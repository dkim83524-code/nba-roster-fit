import pytest

from rosterfit.sources.contracts import find_contracts_file, read_contracts

HTML = """<html><body><table><thead>
<tr class="over_header"><th aria-label="" data-stat="" colspan="3"></th><th data-stat="header_salary" colspan="6">Salary</th></tr>
<tr><th data-stat="ranker" scope="col">Rk</th><th data-stat="player" scope="col">Player</th>
<th data-stat="team_id" scope="col">Tm</th><th data-stat="y1" scope="col">2026-27</th><th data-stat="y2" scope="col">2027-28</th>
<th data-stat="remain_gtd" scope="col">Guaranteed</th></tr></thead><tbody>
<tr data-row="0"><th scope="row" data-stat="ranker">1</th><td data-append-csv="jokicni01" data-stat="player">Nikola Jokić</td>
<td data-stat="team_id">DEN</td><td class="right " data-stat="y1">$59,033,114</td><td data-stat="y2"></td><td data-stat="remain_gtd">$59,033,114</td></tr>
<tr data-row="1"><th scope="row" data-stat="ranker">2</th><td data-append-csv="bealbr01" data-stat="player">Bradley Beal</td>
<td data-stat="team_id">PHO</td><td class="right salary-pl" data-stat="y1">$25,807,810</td><td data-stat="y2"></td><td data-stat="remain_gtd">$77,532,040</td></tr>
<tr data-row="2"><th scope="row" data-stat="ranker">3</th><td data-append-csv="bealbr01" data-stat="player">Bradley Beal</td>
<td data-stat="team_id">LAC</td><td data-stat="y1">$25,807,810</td><td data-stat="y2"></td><td data-stat="remain_gtd">$6,424,800</td></tr>
<tr data-row="3"><th scope="row" data-stat="ranker">4</th><td data-append-csv="bridgmi01" data-stat="player">Mikal Bridges</td>
<td data-stat="team_id">NYK</td><td data-stat="y1">$24,900,000</td><td data-stat="y2"></td><td data-stat="remain_gtd"></td></tr>
<tr data-row="4"><th scope="row" data-stat="ranker">5</th><td data-append-csv="claxtni01" data-stat="player">Nic Claxton</td>
<td data-stat="team_id">BRK</td><td data-stat="y1">$25,352,272</td><td data-stat="y2"></td><td data-stat="remain_gtd"></td></tr>
</tbody></table></body></html>"""

CSV = """,,,Salary,Salary,,,
Rk,Player,Tm,2026-27,2027-28,Guaranteed,-9999
1,Nikola Jokić,DEN,"$59,033,114",,"$59,033,114",jokicni01
2,Nic Claxton,BRK,"$25,352,272",,,claxtni01
Rk,Player,Tm,2026-27,2027-28,Guaranteed,-9999
3,Two Way,CHO,,,,twowa01
"""


def test_reads_the_excel_download(tmp_path):
    path = tmp_path / "sportsref_download.xls"
    path.write_text(HTML, encoding="utf-8")
    season, df = read_contracts(path)
    assert season == "2026-27"
    assert list(df["name"]) == ["Nikola Jokić", "Bradley Beal", "Nic Claxton", "Mikal Bridges"]
    beal = df[df["bbref_id"] == "bealbr01"].iloc[0]
    assert beal["team"] == "LAC"  # waived by PHO, which still owes him; his new team owes less
    assert df.set_index("name").loc["Nic Claxton", "team"] == "BKN"  # NBA.com abbreviation
    assert df["salary"].dtype.kind == "i"


def test_reads_the_csv_download(tmp_path):
    path = tmp_path / "contracts.csv"
    path.write_text(CSV, encoding="utf-8")
    season, df = read_contracts(path)
    assert season == "2026-27"
    assert list(df["bbref_id"]) == ["jokicni01", "claxtni01"]  # repeated header and no-salary row dropped
    assert list(df["team"]) == ["DEN", "BKN"]


def test_rejects_a_table_without_seasons(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Rk,Player,Tm,Salary\n1,A,DEN,$1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_contracts(path)


def test_finds_the_newest_saved_table(tmp_path):
    assert find_contracts_file(tmp_path / "missing") is None
    (tmp_path / "notes.txt").write_text("x")
    old = tmp_path / "a.csv"
    old.write_text(CSV)
    new = tmp_path / "b.xls"
    new.write_text(HTML)
    import os
    os.utime(old, (1, 1))
    assert find_contracts_file(tmp_path) == new
