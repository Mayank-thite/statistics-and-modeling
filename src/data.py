"""Loaders for the two UCI datasets.

Both loaders return tidy pandas DataFrames with readable column names.
Run scripts/download_data.py first.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

WEATHER_LABELS = {1: "clear", 2: "mist", 3: "light rain/snow", 4: "heavy rain/snow"}
SEASON_LABELS = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}


def load_bank(path: Path | str | None = None, drop_duration: bool = True) -> pd.DataFrame:
    """Load UCI Bank Marketing (bank-additional-full.csv).

    Adds a 0/1 target column ``subscribed`` and drops the original ``y``.
    ``duration`` is the length of the call, which is only known after the
    call ends and is almost a proxy for the outcome, so it is dropped by default.
    ``pdays == 999`` means "never contacted before", so it is split into a flag
    and a cleaned numeric column.
    """
    path = Path(path) if path else RAW / "bank-additional-full.csv"
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df["subscribed"] = (df.pop("y") == "yes").astype(int)
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != 999)
    if drop_duration:
        df = df.drop(columns="duration")
    return df


def _clean_bike(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    # UCI ships these normalised. Undo it so coefficients are in real units.
    df["temp_c"] = df.pop("temp") * 41
    df["feels_like_c"] = df.pop("atemp") * 50
    df["humidity"] = df.pop("hum") * 100
    df["windspeed"] = df["windspeed"] * 67
    df["season_name"] = df["season"].map(SEASON_LABELS)
    df["weather"] = df["weathersit"].map(WEATHER_LABELS)
    return df.drop(columns="instant")


def load_bike_day(path: Path | str | None = None) -> pd.DataFrame:
    """Load the daily Bike Sharing table (731 days, 2011 to 2012)."""
    path = Path(path) if path else RAW / "day.csv"
    return _clean_bike(pd.read_csv(path))


def load_bike_hour(path: Path | str | None = None) -> pd.DataFrame:
    """Load the hourly Bike Sharing table (17,379 hours)."""
    path = Path(path) if path else RAW / "hour.csv"
    df = _clean_bike(pd.read_csv(path))
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df
