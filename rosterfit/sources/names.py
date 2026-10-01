"""Player-name normalization for matching CSVs that lack NBA.com player IDs."""
from __future__ import annotations

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", "").replace(".", ""))
    parts = [p for p in text.split() if p not in _SUFFIXES]
    return " ".join(parts)


def normalize_column(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
