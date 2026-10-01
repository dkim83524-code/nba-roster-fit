"""Player-name normalization for matching CSVs that lack NBA.com player IDs."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", "").replace(".", ""))
    parts = [p for p in text.split() if p not in _SUFFIXES]
    return " ".join(parts)


def normalize_column(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


# Name changes and nicknames that no matching rule could guess. NBA.com shows its current name
# for every season. Keys and values are normalized names.
KNOWN_RENAMES = {
    "enes kanter": "enes freedom",
    "nahshon hyland": "bones hyland",
    "carlton carrington": "bub carrington",
    "jeenathan williams": "nate williams",
}


def _same_first_name(a: str, b: str) -> bool:
    """Nic/Nicolas, Alex/Alexandre, Charlie/Charles, KJ/Kenyon, but not Lonzo/LaMelo."""
    if a.startswith(b) or b.startswith(a):
        return True
    if min(len(a), len(b)) <= 2:  # initials like KJ, CJ, PJ
        return a[0] == b[0]
    return a[:3] == b[:3]


def match_names(names, roster: pd.DataFrame, overrides: dict[str, int | str] | None = None,
                taken: set[int] | None = None) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Map player names from another source (e.g. DARKO) to NBA.com PLAYER_IDs for one season.

    roster: that season's NBA.com players (PLAYER_ID, PLAYER_NAME), i.e. everyone who played.
    Rules, in order:
      1. config override (to an NBA.com id or name), or the same name after normalization (accents, punctuation, Jr./III,
         known renames)
      2. the same words in a different order ("Cui Yongxi" / "Yongxi Cui")
      3. the same last name and a compatible first name, when exactly one unmatched name and one
         unmatched NBA.com player share that last name ("Nicolas Claxton" / "Nic Claxton",
         "Kenyon Martin Jr." / "KJ Martin")
    Returns ({name: PLAYER_ID}, [(name, NBA.com name)] for matches made by rules 2-3).
    """
    roster = roster.drop_duplicates("PLAYER_ID")
    nba = {int(p): str(n) for p, n in zip(roster["PLAYER_ID"], roster["PLAYER_NAME"])}
    nba_norm = {p: normalize_name(n) for p, n in nba.items()}
    counts = Counter(nba_norm.values())
    exact = {n: p for p, n in nba_norm.items() if counts[n] == 1}
    # overrides map a name to an NBA.com player id or to the player's NBA.com name
    resolved = {}
    for k, v in (overrides or {}).items():
        pid = int(v) if str(v).strip().isdigit() else exact.get(normalize_name(v))
        if pid is not None:
            resolved[normalize_name(k)] = pid
    overrides = resolved
    taken = set(taken or ())
    result: dict[str, int] = {}
    loose: list[tuple[str, str]] = []

    for name in dict.fromkeys(names):
        norm = normalize_name(name)
        renamed = KNOWN_RENAMES.get(norm, norm)
        if norm in overrides:
            result[name] = overrides[norm]
        elif renamed in exact:
            result[name] = exact[renamed]
    taken |= set(result.values())

    def loose_pass(key, compatible=lambda a, b: True):
        pending = [n for n in dict.fromkeys(names) if n not in result]
        by_name, by_id = defaultdict(list), defaultdict(list)
        for n in pending:
            k = key(normalize_name(n))
            if k:
                by_name[k].append(n)
        for p, n in nba_norm.items():
            if p not in taken:
                k = key(n)
                if k:
                    by_id[k].append(p)
        for k, found in by_name.items():
            ids = by_id.get(k, [])
            if len(found) == 1 and len(ids) == 1 and compatible(normalize_name(found[0]), nba_norm[ids[0]]):
                result[found[0]] = ids[0]
                taken.add(ids[0])
                loose.append((found[0], nba[ids[0]]))

    loose_pass(lambda s: " ".join(sorted(s.split())))
    loose_pass(lambda s: s.split()[-1] if s else "",
               lambda a, b: len(a.split()) > 1 and len(b.split()) > 1
               and _same_first_name(a.split()[0], b.split()[0]))
    return result, loose
