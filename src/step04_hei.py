"""
Step 4 — Define and Compute the Heat Exposure Index (HEI)
==========================================================
Assigns a quantitative heat-exposure score to every transfer pair
identified in Step 2, using the WBGT diurnal profile from Step 3.

HEI Formula
-----------
For a single transfer event:

    HEI = WBGT(h) × wait_minutes × shade_factor

where
  WBGT(h)      : mean summer wet-bulb globe temperature at clock hour h
                 (°C), from the ERA5-Land diurnal profile (Step 3)
  wait_minutes : scheduled passenger wait time at the transfer stop (min)
  shade_factor : 0.7 if the stop is sheltered, 1.0 otherwise (Step 1)

Units: °C·min — a "thermal dose" proportional to both intensity (WBGT)
and duration (wait time).

Passenger Frequency Proxy
--------------------------
The paper's optimisation objective is

    f₂(δ) = Σ HEI(stop, time + δ) × passenger_frequency

In the absence of ridership counts in the GTFS feed, we proxy
passenger demand at each transfer by the normalised weekday trip
frequency of the feeder route:

    freq_proxy = n_trips_feeder / max(n_trips_across_all_routes)

Trip frequency is a standard GTFS-available surrogate for route-level
demand (Cervero 1990; El-Geneidy et al. 2014). We also provide a
uniform-weight version (freq_proxy = 1) as a robustness variant and
report both in the results.

WBGT Interpolation for the Optimiser
-------------------------------------
When the optimiser applies a time-shift δ to a trip, the transfer
moves to a different clock hour, changing the effective WBGT.  We
therefore expose two lookup objects from this module:

  WBGT_BY_HOUR   : np.ndarray shape (24,)  — mean WBGT at each hour
  WBGT_P90_BY_HOUR : np.ndarray shape (24,) — 90th-pct WBGT at each hour

The optimiser in Steps 5–6 uses linear interpolation between adjacent
hours when a fractional-hour shift results in a non-integer clock hour.

Outputs
-------
data/gtfs/transfer_hei.parquet          — full per-transfer table
data/weather/wbgt_lookup.npy            — (24,) WBGT mean array (for optimizer)
data/weather/wbgt_p90_lookup.npy        — (24,) WBGT p90 array
results/tables/hei_summary.csv          — system-level HEI statistics
results/tables/top_hei_transfers.csv    — top 30 worst transfer pairs
results/tables/hei_by_hour.csv          — hourly HEI aggregates
results/tables/hei_by_route_pair.csv    — route-pair HEI aggregates
results/tables/hei_by_stop.csv          — stop-level HEI aggregates
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
GTFS_DIR    = ROOT / "data" / "gtfs"
WEATHER_DIR = ROOT / "data" / "weather"
TABLE_DIR   = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WBGT lookup construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_wbgt_lookup(profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the two 24-element WBGT arrays used throughout the optimisation.

    The profile DataFrame must have one row per clock_hour (0–23).
    We sort by clock_hour to guarantee index alignment, then extract
    wbgt_mean and wbgt_p90 as plain numpy arrays.

    Returns
    -------
    wbgt_mean_arr : shape (24,)  — mean summer WBGT per clock hour
    wbgt_p90_arr  : shape (24,)  — 90th-pct summer WBGT per clock hour
    """
    p = profile.sort_values("clock_hour").reset_index(drop=True)

    if len(p) != 24:
        raise ValueError(f"Expected 24 rows in profile, got {len(p)}")
    if not (p["clock_hour"] == np.arange(24)).all():
        raise ValueError("Profile clock_hour column is not 0–23 in order")

    wbgt_mean_arr = p["wbgt_mean"].to_numpy(dtype=float)
    wbgt_p90_arr  = p["wbgt_p90"].to_numpy(dtype=float)

    log.info("WBGT lookup arrays built:")
    log.info("  Mean  — min=%.2f°C (h%02d)  max=%.2f°C (h%02d)",
             wbgt_mean_arr.min(), int(wbgt_mean_arr.argmin()),
             wbgt_mean_arr.max(), int(wbgt_mean_arr.argmax()))
    log.info("  p90   — min=%.2f°C (h%02d)  max=%.2f°C (h%02d)",
             wbgt_p90_arr.min(), int(wbgt_p90_arr.argmin()),
             wbgt_p90_arr.max(), int(wbgt_p90_arr.argmax()))

    return wbgt_mean_arr, wbgt_p90_arr


def wbgt_at_shifted_hour(arrival_sec: np.ndarray,
                          delta_sec: np.ndarray,
                          wbgt_lookup: np.ndarray) -> np.ndarray:
    """
    Return the WBGT value at the shifted feeder arrival time.

    Uses linear interpolation between adjacent clock hours so the
    optimiser can handle fractional-hour shifts smoothly.

    Parameters
    ----------
    arrival_sec : scheduled feeder arrival time, seconds since midnight
    delta_sec   : time shift applied to the feeder trip (seconds, may be
                  negative for earlier departure)
    wbgt_lookup : (24,) WBGT array indexed by clock hour

    Returns
    -------
    wbgt : WBGT at the shifted arrival time, same shape as arrival_sec
    """
    shifted_sec  = arrival_sec + delta_sec
    # Map to [0, 86400) — GTFS >24h times wrap back to the same climate hour
    shifted_sec  = shifted_sec % 86400.0
    # Fractional clock hour
    frac_hour    = shifted_sec / 3600.0

    # Extend the lookup array by wrapping h=23 and h=0 for interp continuity
    extended     = np.append(wbgt_lookup, wbgt_lookup[0])   # shape (25,)
    hours        = np.arange(25, dtype=float)

    wbgt_vals    = np.interp(frac_hour, hours, extended)
    return wbgt_vals


# ═══════════════════════════════════════════════════════════════════════════════
# Passenger frequency proxy
# ═══════════════════════════════════════════════════════════════════════════════

def build_frequency_proxy(trips: pd.DataFrame) -> pd.Series:
    """
    Compute a per-route frequency proxy from weekday trip counts.

    Returns a pd.Series indexed by route_id with values in (0, 1],
    where 1.0 is assigned to the route with the most weekday trips.

    freq_proxy(r) = n_trips(r) / max_n_trips
    """
    route_trips = trips.groupby("route_id").size()
    proxy       = route_trips / route_trips.max()
    log.info("Route frequency proxies:")
    for rid, val in proxy.sort_values(ascending=False).items():
        log.info("  Route %-5s  %4d trips  proxy=%.3f",
                 rid, int(route_trips[rid]), val)
    return proxy


# ═══════════════════════════════════════════════════════════════════════════════
# Core HEI computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hei(pairs: pd.DataFrame,
                wbgt_mean_arr: np.ndarray,
                wbgt_p90_arr:  np.ndarray,
                freq_proxy:    pd.Series) -> pd.DataFrame:
    """
    Attach HEI columns to the transfer pairs DataFrame.

    New columns added
    -----------------
    wbgt_mean           : WBGT (mean profile) interpolated at fractional
                          feeder arrival clock hour — matches the optimizer's
                          WBGT evaluation exactly (linear interpolation)
    wbgt_p90            : WBGT (90th-pct profile) likewise interpolated
    hei_mean            : WBGT_mean × wait_min × shade_factor
    hei_p90             : WBGT_p90  × wait_min × shade_factor
    freq_proxy          : normalised feeder route trip frequency ∈ (0,1]
    hei_weighted_mean   : hei_mean × freq_proxy  (primary f₂ component)
    hei_weighted_p90    : hei_p90  × freq_proxy
    hei_uniform         : hei_mean × 1.0 (uniform-weight robustness check)

    Note: WBGT is evaluated at the fractional feeder arrival hour (e.g.
    15.75 h for an arrival at 15:45), not the integer clock hour.  This
    matches the optimiser's treatment and gives ~0.5% higher precision
    than integer-hour lookup.
    """
    df = pairs.copy()

    # ── WBGT at fractional feeder arrival hour (linear interpolation) ────
    # Extend lookup array to allow wrap-around interpolation at 23h → 0h
    hours_ext    = np.arange(25, dtype=float)
    wbgt_mean_ext = np.append(wbgt_mean_arr, wbgt_mean_arr[0])
    wbgt_p90_ext  = np.append(wbgt_p90_arr,  wbgt_p90_arr[0])

    frac_hour = (df["feeder_arrival_sec"].to_numpy(dtype=float) % 86400.0) / 3600.0

    df["wbgt_mean"] = np.interp(frac_hour, hours_ext, wbgt_mean_ext)
    df["wbgt_p90"]  = np.interp(frac_hour, hours_ext, wbgt_p90_ext)

    # ── Raw HEI (per transfer, no ridership weighting) ───────────────────
    df["hei_mean"] = df["wbgt_mean"] * df["wait_min"] * df["shade_factor"]
    df["hei_p90"]  = df["wbgt_p90"]  * df["wait_min"] * df["shade_factor"]

    # ── Frequency proxy ───────────────────────────────────────────────────
    df["freq_proxy"] = df["feeder_route_id"].map(freq_proxy)
    assert df["freq_proxy"].notna().all(), \
        f"freq_proxy is NaN for routes: {df[df['freq_proxy'].isna()]['feeder_route_id'].unique()}"

    # ── Weighted HEI ─────────────────────────────────────────────────────
    df["hei_weighted_mean"] = df["hei_mean"] * df["freq_proxy"]
    df["hei_weighted_p90"]  = df["hei_p90"]  * df["freq_proxy"]
    df["hei_uniform"]       = df["hei_mean"] * 1.0   # uniform weighting

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_hei(df: pd.DataFrame) -> None:
    """Defensive checks on the HEI-annotated table."""
    log.info("Validating HEI columns …")

    assert (df["hei_mean"] >= 0).all(),   "Negative hei_mean found"
    assert (df["hei_p90"]  >= 0).all(),   "Negative hei_p90 found"
    assert (df["freq_proxy"] > 0).all(),  "Zero or negative freq_proxy found"
    assert (df["freq_proxy"] <= 1.0).all(), "freq_proxy > 1.0 found"

    # HEI must be >= wait_min * shade_factor * min_wbgt (which is positive)
    assert (df["hei_mean"] >= df["wait_min"] * df["shade_factor"] * 10).all(), \
        "HEI implausibly low — WBGT < 10°C in summer? Check lookup."

    # p90 must be >= mean
    assert (df["hei_p90"] >= df["hei_mean"] - 1e-9).all(), \
        "hei_p90 < hei_mean at some rows"

    # Weighted HEI must be <= unweighted (since freq_proxy ∈ (0,1])
    assert (df["hei_weighted_mean"] <= df["hei_mean"] + 1e-9).all(), \
        "Weighted HEI exceeds unweighted HEI"

    log.info("  All HEI validation checks passed ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_hei_summary(df: pd.DataFrame) -> None:
    """Log a comprehensive HEI summary for the paper."""

    n = len(df)
    log.info("=" * 65)
    log.info("HEI SUMMARY  (baseline: δ = 0, n = %d transfer pairs)", n)
    log.info("=" * 65)

    # ── System-level totals ───────────────────────────────────────────────
    total_hei       = df["hei_mean"].sum()
    total_wait_min  = df["wait_min"].sum()
    total_whei      = df["hei_weighted_mean"].sum()

    log.info("System totals:")
    log.info("  Total HEI (unweighted)  : %10.1f  °C·min", total_hei)
    log.info("  Total HEI (freq-wtd)    : %10.1f  °C·min", total_whei)
    log.info("  Total wait time         : %10.1f  min  (%d h)",
             total_wait_min, int(total_wait_min // 60))
    log.info("  Mean HEI per transfer   : %10.3f  °C·min", df["hei_mean"].mean())
    log.info("  Median HEI per transfer : %10.3f  °C·min", df["hei_mean"].median())

    # ── HEI distribution ──────────────────────────────────────────────────
    log.info("")
    log.info("HEI distribution (°C·min):")
    for pct, lbl in [(10,'p10'), (25,'p25'), (50,'p50'),
                     (75,'p75'), (90,'p90'), (95,'p95'), (99,'p99')]:
        log.info("  %-4s  %7.3f", lbl, np.percentile(df["hei_mean"], pct))
    log.info("  max   %7.3f  (transfer_id=%d)",
             df["hei_mean"].max(), int(df.loc[df["hei_mean"].idxmax(), "transfer_id"]))

    # ── Decomposition: WBGT vs wait-time contributions ───────────────────
    log.info("")
    log.info("HEI decomposition:")
    log.info("  Corr(WBGT, HEI)      : %+.4f", df["wbgt_mean"].corr(df["hei_mean"]))
    log.info("  Corr(wait_min, HEI)  : %+.4f", df["wait_min"].corr(df["hei_mean"]))
    log.info("  (Both inputs drive HEI; stronger correlation indicates dominant driver)")

    # ── HEI by clock hour ─────────────────────────────────────────────────
    log.info("")
    log.info("HEI by clock hour (total unweighted HEI, °C·min):")
    hour_hei = df.groupby("transfer_clock_hour")["hei_mean"].agg(["sum","mean","count"])
    for h, row in hour_hei.iterrows():
        pct_total = 100 * row["sum"] / total_hei
        bar = "█" * int(pct_total / 0.5)
        log.info("  %02dh  sum=%7.1f  mean=%6.3f  n=%4d  (%4.1f%%)  %s",
                 h, row["sum"], row["mean"], int(row["count"]), pct_total, bar)

    # ── HEI by shade ──────────────────────────────────────────────────────
    log.info("")
    log.info("HEI by shade category:")
    for sheltered, grp in df.groupby("is_sheltered"):
        label = "Sheltered (shade=0.7)" if sheltered else "Exposed  (shade=1.0)"
        log.info("  %-25s  n=%5d  total_HEI=%8.1f  mean_HEI=%.3f",
                 label, len(grp), grp["hei_mean"].sum(), grp["hei_mean"].mean())
    theoretical_if_all_exposed = (df["hei_mean"] / df["shade_factor"]).sum()
    shade_saving = theoretical_if_all_exposed - total_hei
    log.info("  Shade reduces total HEI by %.1f °C·min (%.2f%%) at sheltered stops",
             shade_saving, 100 * shade_saving / theoretical_if_all_exposed)

    # ── Top 15 worst transfer pairs ───────────────────────────────────────
    log.info("")
    log.info("Top 15 worst transfer pairs (by hei_mean):")
    log.info("  %-6s  %-5s→%-5s  %-28s  wait  wbgt  shade  HEI",
             "xfr_id", "feed", "conn", "stop_name")
    top = df.nlargest(15, "hei_mean")
    for _, row in top.iterrows():
        log.info("  %6d  %-5s→%-5s  %-28s  %4.1fm  %5.2f  %.1f  %7.3f",
                 row["transfer_id"],
                 row["feeder_route_short"], row["connector_route_short"],
                 row["stop_name"][:28],
                 row["wait_min"], row["wbgt_mean"], row["shade_factor"],
                 row["hei_mean"])

    # ── HEI by route pair ─────────────────────────────────────────────────
    log.info("")
    log.info("Top 10 route pairs by total HEI (°C·min):")
    rp = (df.groupby(["feeder_route_short", "connector_route_short"])
            .agg(total_hei=("hei_mean", "sum"),
                 n_xfr    =("hei_mean", "count"),
                 mean_hei =("hei_mean", "mean"),
                 mean_wait=("wait_min",  "mean"))
            .reset_index()
            .sort_values("total_hei", ascending=False))
    for _, row in rp.head(10).iterrows():
        log.info("  %-5s→%-5s  total=%8.1f  n=%4d  mean_hei=%.3f  mean_wait=%.1fm",
                 row["feeder_route_short"], row["connector_route_short"],
                 row["total_hei"], int(row["n_xfr"]),
                 row["mean_hei"], row["mean_wait"])

    # ── HEI hotspot stops ─────────────────────────────────────────────────
    log.info("")
    log.info("Top 10 HEI hotspot stops (total HEI, °C·min):")
    sp = (df.groupby(["stop_id", "stop_name", "is_sheltered"])
            .agg(total_hei=("hei_mean", "sum"),
                 n_xfr    =("hei_mean", "count"),
                 mean_wait=("wait_min",  "mean"),
                 mean_wbgt=("wbgt_mean", "mean"))
            .reset_index()
            .sort_values("total_hei", ascending=False))
    for _, row in sp.head(10).iterrows():
        shade_tag = "[S]" if row["is_sheltered"] else "   "
        log.info("  %s stop %5d  %-32s  HEI=%8.1f  n=%4d  wait=%.1fm",
                 shade_tag, row["stop_id"], row["stop_name"][:32],
                 row["total_hei"], int(row["n_xfr"]), row["mean_wait"])

    # ── Peak-hour contribution ────────────────────────────────────────────
    log.info("")
    peak_hours_hei = df[df["transfer_clock_hour"].between(12, 17)]["hei_mean"].sum()
    log.info("Peak heat hours (12–17h) account for %.1f%% of total system HEI",
             100 * peak_hours_hei / total_hei)


# ═══════════════════════════════════════════════════════════════════════════════
# Save aggregates for paper tables
# ═══════════════════════════════════════════════════════════════════════════════

def save_aggregates(df: pd.DataFrame) -> None:
    """Write CSV summary tables for downstream use in the paper."""

    # ── HEI by hour ───────────────────────────────────────────────────────
    hour_agg = (
        df.groupby("transfer_clock_hour")
        .agg(
            n_transfers   = ("hei_mean",   "count"),
            total_hei     = ("hei_mean",   "sum"),
            total_hei_p90 = ("hei_p90",    "sum"),
            mean_hei      = ("hei_mean",   "mean"),
            mean_wbgt     = ("wbgt_mean",  "mean"),
            mean_wait_min = ("wait_min",   "mean"),
            total_wait_min= ("wait_min",   "sum"),
        )
        .reset_index()
        .rename(columns={"transfer_clock_hour": "clock_hour"})
    )
    hour_agg.to_csv(TABLE_DIR / "hei_by_hour.csv", index=False)

    # ── HEI by route pair ─────────────────────────────────────────────────
    rp_agg = (
        df.groupby(["feeder_route_short", "connector_route_short",
                    "feeder_route_long",  "connector_route_long"])
        .agg(
            n_transfers   = ("hei_mean",  "count"),
            total_hei     = ("hei_mean",  "sum"),
            mean_hei      = ("hei_mean",  "mean"),
            mean_wait_min = ("wait_min",  "mean"),
            total_wait_min= ("wait_min",  "sum"),
        )
        .reset_index()
        .sort_values("total_hei", ascending=False)
    )
    rp_agg.to_csv(TABLE_DIR / "hei_by_route_pair.csv", index=False)

    # ── HEI by stop ───────────────────────────────────────────────────────
    stop_agg = (
        df.groupby(["stop_id", "stop_name", "stop_lat", "stop_lon",
                    "is_sheltered", "shade_factor"])
        .agg(
            n_transfers   = ("hei_mean",  "count"),
            total_hei     = ("hei_mean",  "sum"),
            total_hei_p90 = ("hei_p90",   "sum"),
            mean_hei      = ("hei_mean",  "mean"),
            mean_wait_min = ("wait_min",  "mean"),
            mean_wbgt     = ("wbgt_mean", "mean"),
        )
        .reset_index()
        .sort_values("total_hei", ascending=False)
    )
    stop_agg.to_csv(TABLE_DIR / "hei_by_stop.csv", index=False)
    # Also write top-30 version with canonical name (read by step08 fig05)
    stop_agg.head(30).to_csv(TABLE_DIR / "tab07_top_hei_stops.csv", index=False)

    # ── Top 30 worst individual transfer pairs ────────────────────────────
    top30_cols = [
        "transfer_id", "stop_id", "stop_name", "is_sheltered", "shade_factor",
        "feeder_route_short", "connector_route_short",
        "transfer_clock_hour", "wait_min",
        "wbgt_mean", "wbgt_p90",
        "hei_mean", "hei_p90", "freq_proxy",
        "hei_weighted_mean",
    ]
    top30 = df.nlargest(30, "hei_mean")[top30_cols]
    top30.to_csv(TABLE_DIR / "top_hei_transfers.csv", index=False)

    # ── System summary scalar ─────────────────────────────────────────────
    summary = pd.DataFrame([{
        "n_transfer_pairs"    : len(df),
        "n_transfer_stops"    : df["stop_id"].nunique(),
        "n_exposed_stops"     : (~df["is_sheltered"]).sum(),
        "n_sheltered_stops"   : df["is_sheltered"].sum(),
        "total_wait_min"      : df["wait_min"].sum(),
        "mean_wait_min"       : df["wait_min"].mean(),
        "total_hei"           : df["hei_mean"].sum(),
        "total_hei_p90"       : df["hei_p90"].sum(),
        "total_hei_weighted"  : df["hei_weighted_mean"].sum(),
        "mean_hei_per_xfer"   : df["hei_mean"].mean(),
        "max_hei_single_xfer" : df["hei_mean"].max(),
        "pct_hei_peak_12_17h" : 100 * df[df["transfer_clock_hour"].between(12,17)]["hei_mean"].sum() / df["hei_mean"].sum(),
        "peak_wbgt_hour"      : int(df.groupby("transfer_clock_hour")["wbgt_mean"].mean().idxmax()),
        "peak_hei_hour"       : int(df.groupby("transfer_clock_hour")["hei_mean"].sum().idxmax()),
    }])
    summary.to_csv(TABLE_DIR / "hei_summary.csv", index=False)
    log.info("Summary tables saved to %s", TABLE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    log.info("Step 4 — Computing Heat Exposure Index (HEI)")

    # ── Load inputs ───────────────────────────────────────────────────────
    pairs   = pd.read_parquet(GTFS_DIR    / "transfer_pairs.parquet")
    profile = pd.read_parquet(WEATHER_DIR / "wbgt_profile_summer.parquet")
    trips   = pd.read_parquet(GTFS_DIR    / "trips_clean.parquet")

    log.info("Loaded %d transfer pairs, profile with %d hours, %d trips",
             len(pairs), len(profile), len(trips))

    # ── Build lookup arrays ───────────────────────────────────────────────
    wbgt_mean_arr, wbgt_p90_arr = build_wbgt_lookup(profile)

    np.save(WEATHER_DIR / "wbgt_lookup.npy",     wbgt_mean_arr)
    np.save(WEATHER_DIR / "wbgt_p90_lookup.npy", wbgt_p90_arr)
    log.info("WBGT lookup arrays saved to %s", WEATHER_DIR)

    # ── Build frequency proxy ─────────────────────────────────────────────
    freq_proxy = build_frequency_proxy(trips)

    # ── Compute HEI ───────────────────────────────────────────────────────
    log.info("Computing HEI for %d transfer pairs …", len(pairs))
    df = compute_hei(pairs, wbgt_mean_arr, wbgt_p90_arr, freq_proxy)

    # ── Validate ──────────────────────────────────────────────────────────
    validate_hei(df)

    # ── Print full summary ────────────────────────────────────────────────
    print_hei_summary(df)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_parquet(GTFS_DIR / "transfer_hei.parquet", index=False)
    log.info("Saved transfer_hei.parquet (%d rows, %d cols)", len(df), df.shape[1])

    save_aggregates(df)

    log.info("Step 4 complete.")

    return {
        "transfer_hei"  : df,
        "wbgt_mean_arr" : wbgt_mean_arr,
        "wbgt_p90_arr"  : wbgt_p90_arr,
        "freq_proxy"    : freq_proxy,
    }


if __name__ == "__main__":
    result = main()
