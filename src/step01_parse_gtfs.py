"""
Step 1 — Parse and Clean the GTFS Feed
=======================================
Loads the CARTA GTFS static feed (summer 2025: gtfs_20250817) from
data/gtfs/gtfs_20250817/, cleans all tables, parses GTFS times (including
values past 24:00:00), filters to weekday service, and produces a single
analysis-ready stop_events DataFrame.

Outputs
-------
data/gtfs/stop_events.parquet     — one row per (trip, stop)
data/gtfs/stops_clean.parquet     — stop metadata with shade classification
data/gtfs/routes_clean.parquet    — route metadata
data/gtfs/trips_clean.parquet     — trip metadata (weekday only)
data/gtfs/calendar_clean.parquet  — service calendar
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
GTFS_DIR = ROOT / "data" / "gtfs" / "gtfs_20250817"
OUT_DIR  = ROOT / "data" / "gtfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEEKDAY_SERVICE_ID = 2

SHELTER_KEYWORDS = [
    "station", "terminal", "transit center", "transfer center",
    "depot", "hub", "center", "mall", "downtown",
]


def _read_csv(filename: str) -> pd.DataFrame:
    path = GTFS_DIR / filename
    df   = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    log.info("  Loaded %-25s  %6d rows  %d cols", filename, len(df), df.shape[1])
    return df


def _gtfs_time_to_sec(series: pd.Series) -> pd.Series:
    """
    Convert HH:MM:SS to integer seconds since midnight.
    GTFS permits values past 24:00:00. Raises ValueError on bad input.
    """
    def _parse(t: str) -> int:
        parts = str(t).strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed GTFS time: {t!r}")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return series.apply(_parse)


def _is_sheltered(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in SHELTER_KEYWORDS)


def load_agency() -> pd.DataFrame:
    df = _read_csv("agency.txt")
    df["agency_id"] = df["agency_id"].astype(int)
    return df


def load_calendar() -> pd.DataFrame:
    df = _read_csv("calendar.txt")
    for c in ["service_id", "monday", "tuesday", "wednesday",
              "thursday", "friday", "saturday", "sunday"]:
        df[c] = df[c].astype(int)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y%m%d")
    df["end_date"]   = pd.to_datetime(df["end_date"],   format="%Y%m%d")
    return df


def load_routes() -> pd.DataFrame:
    df = _read_csv("routes.txt")
    df["agency_id"]        = df["agency_id"].astype(int)
    df["route_type"]       = df["route_type"].astype(int)
    df["route_id"]         = df["route_id"].str.strip()
    df["route_short_name"] = df["route_short_name"].str.strip()
    df["route_long_name"]  = df["route_long_name"].str.strip()
    return df


def load_trips(weekday_only: bool = True) -> pd.DataFrame:
    df = _read_csv("trips.txt")
    df["trip_id"]       = df["trip_id"].astype(int)
    df["service_id"]    = df["service_id"].astype(int)
    df["direction_id"]  = df["direction_id"].astype(int)
    df["block_id"]      = df["block_id"].astype(int)
    df["route_id"]      = df["route_id"].str.strip()
    df["trip_headsign"] = df["trip_headsign"].str.strip()
    if weekday_only:
        n_before = len(df)
        df = df[df["service_id"] == WEEKDAY_SERVICE_ID].copy()
        log.info("  Trips: kept %d / %d weekday trips (service_id=%d)",
                 len(df), n_before, WEEKDAY_SERVICE_ID)
    return df


def load_stop_times() -> pd.DataFrame:
    df = _read_csv("stop_times.txt")
    df["trip_id"]       = df["trip_id"].astype(int)
    df["stop_id"]       = df["stop_id"].astype(int)
    df["stop_sequence"] = df["stop_sequence"].astype(int)
    df["pickup_type"]   = df["pickup_type"].astype(int)
    df["drop_off_type"] = df["drop_off_type"].astype(int)
    df["timepoint"]     = df["timepoint"].astype(int)
    df["arrival_sec"]   = _gtfs_time_to_sec(df["arrival_time"])
    df["departure_sec"] = _gtfs_time_to_sec(df["departure_time"])
    df["clock_hour"]    = (df["departure_sec"] % 86400 // 3600).astype(int)
    df["minute_of_day"] = (df["departure_sec"] % 86400 // 60).astype(int)
    assert (df["arrival_sec"] <= df["departure_sec"]).all(), \
        "stop_times: arrival_sec > departure_sec"
    return df


def load_stops() -> pd.DataFrame:
    df = _read_csv("stops.txt")
    df["stop_id"]   = df["stop_id"].astype(int)
    df["stop_lat"]  = df["stop_lat"].astype(float)
    df["stop_lon"]  = df["stop_lon"].astype(float)
    df["stop_name"] = df["stop_name"].str.strip()
    in_box = (df["stop_lat"].between(34.9, 35.2) &
              df["stop_lon"].between(-85.5, -85.1))
    assert in_box.all(), f"{(~in_box).sum()} stops outside Chattanooga bbox"
    df["is_sheltered"] = df["stop_name"].apply(_is_sheltered)
    df["shade_factor"] = np.where(df["is_sheltered"], 0.7, 1.0)
    log.info("  Stops: %d sheltered / %d total  (%.1f%%)",
             df["is_sheltered"].sum(), len(df), 100 * df["is_sheltered"].mean())
    return df


def build_stop_events(trips, stop_times, routes, stops) -> pd.DataFrame:
    log.info("Building stop_events …")
    weekday_ids = set(trips["trip_id"])
    st = stop_times[stop_times["trip_id"].isin(weekday_ids)].copy()
    log.info("  stop_times after weekday filter: %d rows", len(st))

    st = st.merge(trips[["trip_id", "route_id", "service_id",
                          "direction_id", "trip_headsign", "block_id"]],
                  on="trip_id", how="inner")
    st = st.merge(routes[["route_id", "route_short_name", "route_long_name"]],
                  on="route_id", how="inner")
    st = st.merge(stops[["stop_id", "stop_name", "stop_lat", "stop_lon",
                          "is_sheltered", "shade_factor"]],
                  on="stop_id", how="inner")

    assert st["stop_lat"].notna().all(),        "Unmatched stop_ids after merge"
    assert st["route_short_name"].notna().all(), "Unmatched route_ids after merge"

    st = st.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    log.info("  Final stop_events: %d rows, %d trips, %d stops",
             len(st), st["trip_id"].nunique(), st["stop_id"].nunique())
    return st


def print_summary(stop_events, routes, calendar) -> None:
    log.info("=" * 60)
    log.info("GTFS SUMMARY (weekday summer service)")
    log.info("Routes: %d  |  Trips: %d  |  Stops: %d  |  Events: %d",
             routes["route_id"].nunique(),
             stop_events["trip_id"].nunique(),
             stop_events["stop_id"].nunique(),
             len(stop_events))
    for _, row in calendar[calendar["service_id"] == WEEKDAY_SERVICE_ID].iterrows():
        log.info("  Service: %s -> %s  (Mon=%d Tue=%d Wed=%d Thu=%d Fri=%d)",
                 row["start_date"].date(), row["end_date"].date(),
                 row["monday"], row["tuesday"], row["wednesday"],
                 row["thursday"], row["friday"])
    route_summary = (
        stop_events.groupby(["route_id", "route_short_name", "route_long_name"])
        .agg(trips=("trip_id", "nunique"), events=("stop_id", "count"))
        .reset_index().sort_values("route_short_name")
    )
    for _, row in route_summary.iterrows():
        log.info("  Route %-5s  %-40s  %3d trips  %5d events",
                 row["route_short_name"], row["route_long_name"],
                 row["trips"], row["events"])


def main() -> dict:
    log.info("Step 1 - Parsing GTFS feed from: %s", GTFS_DIR)

    agency   = load_agency()
    calendar = load_calendar()
    routes   = load_routes()
    trips    = load_trips(weekday_only=True)
    st       = load_stop_times()
    stops    = load_stops()

    stop_events = build_stop_events(trips, st, routes, stops)
    print_summary(stop_events, routes, calendar)

    stop_events.to_parquet(OUT_DIR / "stop_events.parquet",    index=False)
    stops.to_parquet(      OUT_DIR / "stops_clean.parquet",    index=False)
    routes.to_parquet(     OUT_DIR / "routes_clean.parquet",   index=False)
    trips.to_parquet(      OUT_DIR / "trips_clean.parquet",    index=False)
    calendar.to_parquet(   OUT_DIR / "calendar_clean.parquet", index=False)

    log.info("All outputs written. Step 1 complete.")
    return dict(stop_events=stop_events, stops=stops,
                routes=routes, trips=trips, calendar=calendar)


if __name__ == "__main__":
    main()
