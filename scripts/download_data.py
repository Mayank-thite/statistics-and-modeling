"""Download the two UCI datasets used in this repo into data/raw/.

Bank Marketing (UCI id 222) and Bike Sharing (UCI id 275), both CC BY 4.0.
The script is idempotent: files that already exist are left alone.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    "bank": [
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    ],
    "bike": [
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    ],
}

# Files we keep, and where they live inside the UCI archives.
BANK_FILE = "bank-additional-full.csv"
BIKE_FILES = ["day.csv", "hour.csv"]


def _fetch(urls: list[str]) -> bytes:
    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as err:  # try the next mirror
            print(f"  failed {url}: {err}")
            last_err = err
    raise RuntimeError(f"all downloads failed: {last_err}")


def _find_in_zip(zf: zipfile.ZipFile, name: str) -> bytes | None:
    """Search a zip (and any zips nested inside it) for a file by basename."""
    for info in zf.infolist():
        base = Path(info.filename).name
        if "__MACOSX" in info.filename:
            continue
        if base == name:
            return zf.read(info)
        if base.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            found = _find_in_zip(inner, name)
            if found is not None:
                return found
    return None


def download_bank() -> None:
    out = RAW / BANK_FILE
    if out.exists():
        print(f"exists: {out.name}")
        return
    print("downloading bank marketing")
    zf = zipfile.ZipFile(io.BytesIO(_fetch(SOURCES["bank"])))
    data = _find_in_zip(zf, BANK_FILE)
    if data is None:
        raise RuntimeError(f"{BANK_FILE} not found in archive")
    out.write_bytes(data)
    print(f"wrote {out.name} ({len(data):,} bytes)")


def download_bike() -> None:
    if all((RAW / f).exists() for f in BIKE_FILES):
        print("exists: " + ", ".join(BIKE_FILES))
        return
    print("downloading bike sharing")
    zf = zipfile.ZipFile(io.BytesIO(_fetch(SOURCES["bike"])))
    for name in BIKE_FILES:
        data = _find_in_zip(zf, name)
        if data is None:
            raise RuntimeError(f"{name} not found in archive")
        (RAW / name).write_bytes(data)
        print(f"wrote {name} ({len(data):,} bytes)")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    download_bank()
    download_bike()
    return 0


if __name__ == "__main__":
    sys.exit(main())
