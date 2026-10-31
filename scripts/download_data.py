"""Download the Chinook SQLite file (public, ~1 MB)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
DST = Path(__file__).resolve().parents[1] / "data" / "Chinook_Sqlite.sqlite"

if __name__ == "__main__":
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        print("already present:", DST)
    else:
        urllib.request.urlretrieve(URL, DST)
        print("downloaded", DST, DST.stat().st_size, "bytes")
