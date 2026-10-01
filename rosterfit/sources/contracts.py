"""Player contracts from Basketball-Reference's contracts page, for the website's cap game.

Save the table at basketball-reference.com/contracts/players.html into data/manual/contracts/ with
Share & Export -> "Get as Excel Workbook" (an HTML file named .xls) or "Get table as CSV". Sports
Reference asks to be cited with a link wherever its data is used; the website does that.
"""
from __future__ import annotations

import csv
import io
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

SOURCE_NAME = "Basketball-Reference"
SOURCE_URL = "https://www.basketball-reference.com/contracts/players.html"

# Official salary caps, from the NBA's announcements (pr.nba.com). Add a season once it is set.
SALARY_CAPS = {"2026-27": 164_961_000}

# Basketball-Reference team abbreviations that differ from NBA.com's
BBREF_TO_NBA = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}

_SEASON = re.compile(r"^\d{4}-\d{2}$")
COLUMNS = ["bbref_id", "name", "team", "salary", "guaranteed"]


def _money(text: str) -> float:
    digits = re.sub(r"[^\d]", "", text or "")
    return float(digits) if digits else float("nan")


class _TableParser(HTMLParser):
    """Rows of the contracts table, keyed by each cell's data-stat."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self.headers: dict[str, str] = {}
        self._row: dict | None = None
        self._cell: str | None = None
        self._head: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr" and "data-row" in a:
            self._row = {}
        elif tag in ("td", "th") and a.get("data-stat"):
            if self._row is not None:
                self._cell = a["data-stat"]
                self._row[self._cell] = ""
                if a.get("data-append-csv"):
                    self._row["bbref_id"] = a["data-append-csv"]
            elif a.get("scope") == "col":
                self._head = a["data-stat"]
                self.headers[self._head] = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._cell = self._head = None
        elif tag == "tr" and self._row is not None:
            if self._row.get("player"):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell:
            self._row[self._cell] += data
        elif self._head:
            self.headers[self._head] += data


def _from_html(text: str) -> tuple[str, pd.DataFrame]:
    p = _TableParser()
    p.feed(text)
    season = p.headers.get("y1", "").strip()
    rows = [{"bbref_id": r.get("bbref_id", ""), "name": r["player"].strip(), "team": r.get("team_id", "").strip(),
             "salary": _money(r.get("y1", "")), "guaranteed": _money(r.get("remain_gtd", ""))} for r in p.rows]
    return season, pd.DataFrame(rows, columns=COLUMNS)


def _from_csv(text: str) -> tuple[str, pd.DataFrame]:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Player" in line and "Tm" in line), None)
    if start is None:
        raise ValueError("no header row with Player and Tm")
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(reader)
    seasons = [i for i, h in enumerate(header) if _SEASON.match(h.strip())]
    if not seasons:
        raise ValueError("no season column like 2026-27")
    col = {h.strip(): i for i, h in enumerate(header)}
    id_col = next((col[k] for k in ("-9999", "Player-additional", "BBRef ID") if k in col), None)
    rows = []
    for rec in reader:
        if len(rec) <= seasons[0] or not rec[col["Player"]].strip() or rec[col["Player"]] == "Player":
            continue
        name, _, embedded_id = rec[col["Player"]].partition("\\")
        rows.append({"bbref_id": rec[id_col].strip() if id_col is not None else embedded_id.strip(),
                     "name": name.strip(), "team": rec[col["Tm"]].strip(), "salary": _money(rec[seasons[0]]),
                     "guaranteed": _money(rec[col["Guaranteed"]]) if "Guaranteed" in col else float("nan")})
    return header[seasons[0]].strip(), pd.DataFrame(rows, columns=COLUMNS)


def read_contracts(path: str | Path) -> tuple[str, pd.DataFrame]:
    """(season, one row per player): bbref_id, name, team (NBA.com abbreviation), salary that season.

    A player listed under two teams was waived by one, which still owes him money, and signed by
    the other; the row with less guaranteed money left is his new team.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    season, df = _from_html(text) if text.lstrip().startswith("<") else _from_csv(text)
    if not _SEASON.match(season):
        raise ValueError(f"{path}: can't tell which season the salaries are for")
    df = df[df["salary"] > 0].copy()
    df["team"] = df["team"].replace(BBREF_TO_NBA)
    df["key"] = df["bbref_id"].where(df["bbref_id"] != "", df["name"])
    df = (df.assign(_g=df["guaranteed"].fillna(0)).sort_values(["key", "_g"])
            .drop_duplicates("key").drop(columns=["key", "_g"]))
    df["salary"] = df["salary"].astype(int)
    return season, df.sort_values("salary", ascending=False).reset_index(drop=True)


def find_contracts_file(folder: Path) -> Path | None:
    """The newest contracts table saved in folder, if any."""
    files = [p for p in folder.glob("*") if p.suffix.lower() in (".xls", ".html", ".htm", ".csv")] if folder.is_dir() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None
