"""
Step 3 — Fetch Weather Data and Compute WBGT
=============================================
Fetches ERA5-Land reanalysis data from the Open-Meteo archive API for
Chattanooga, TN (35.05°N, 85.31°W) and computes the Wet Bulb Globe
Temperature (WBGT) for every hourly observation.

Data is always fetched live from the Open-Meteo API (free, no API key
required) and saved to data/weather/raw_{season}.csv for reproducibility.

Three seasonal windows are fetched to support the primary analysis
(summer) and the sensitivity analysis in Step 7 (spring, fall):

  Summer : 2025-06-01 → 2025-08-31   (primary — matches GTFS schedule)
  Spring : 2025-03-01 → 2025-05-31   (sensitivity)
  Fall   : 2024-09-01 → 2024-11-30   (sensitivity)

WBGT Formula  (ISO 7243 / NWS outdoor approximation)
-------------
  WBGT = 0.7·T_wet + 0.2·T_globe + 0.1·T_dry

T_wet   — Stull (2011) isobaric wet-bulb approximation from T and RH
T_globe — Black-globe temperature from Dimiceli et al. (2011)
          simplified regression of the Liljegren energy-balance model
T_dry   — 2-m air temperature (ERA5-Land, °C)

References
----------
Stull R (2011). "Wet-Bulb Temperature from Relative Humidity and Air
  Temperature." J. Appl. Meteorol. Climatol. 50:2267–2269.

Dimiceli VE, Piltz SF, Amburn SA (2011). "Estimation of Black Globe
  Temperature for Calculation of the WBGT Index." WCECS Proc.

Liljegren JC et al. (2008). "Modeling Wet Bulb Globe Temperature Using
  Standard Meteorological Measurements." J. Occup. Environ. Hyg.
  5(10):645–655.

Outputs
-------
data/weather/raw_summer_2025.csv
data/weather/raw_spring_2025.csv
data/weather/raw_fall_2024.csv
data/weather/wbgt_hourly_summer.parquet   — per-hour WBGT with full context
data/weather/wbgt_profile_summer.parquet  — mean & p90 WBGT by clock hour
data/weather/wbgt_profile_spring.parquet
data/weather/wbgt_profile_fall.parquet
"""

import json
import logging
import urllib.request
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
WEATHER_DIR = ROOT / "data" / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

# ── station ───────────────────────────────────────────────────────────────────
LAT      = 35.05
LON      = -85.31
TIMEZONE = "America/New_York"

# ── seasons to fetch ──────────────────────────────────────────────────────────
SEASONS = {
    "summer": ("2025-06-01", "2025-08-31"),
    "spring": ("2025-03-01", "2025-05-31"),
    "fall":   ("2024-09-01", "2024-11-30"),
}

# Variables requested from the API
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",       # returned in km/h — converted to m/s below
    "shortwave_radiation",  # W/m²
]


# ═══════════════════════════════════════════════════════════════════════════════
# API fetch
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_open_meteo(start: str, end: str) -> dict:
    """
    Fetch hourly ERA5-Land reanalysis from the Open-Meteo archive API.
    Raises urllib.error.URLError or json.JSONDecodeError on failure.
    """
    base   = "https://archive-api.open-meteo.com/v1/archive"
    params = (
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        f"&hourly={','.join(HOURLY_VARS)}"
        f"&timezone={TIMEZONE.replace('/', '%2F')}"
    )
    url = base + params
    log.info("  GET %s", url)

    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())

    assert "hourly" in payload, "Missing 'hourly' key in API response"
    for var in HOURLY_VARS:
        assert var in payload["hourly"], f"Missing variable '{var}' in API response"

    n = len(payload["hourly"]["time"])
    log.info("  Received %d hourly records (%s → %s)",
             n, payload["hourly"]["time"][0], payload["hourly"]["time"][-1])
    return payload


def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    Convert a raw Open-Meteo JSON payload to a tidy DataFrame with
    unit-corrected columns.

    Unit corrections applied here:
      wind_speed_10m : km/h → m/s  (÷ 3.6)
    """
    h  = payload["hourly"]
    df = pd.DataFrame({
        "time"           : pd.to_datetime(h["time"]),
        "temperature_2m" : np.array(h["temperature_2m"],      dtype=float),
        "rel_humidity"   : np.array(h["relative_humidity_2m"], dtype=float),
        "wind_speed_ms"  : np.array(h["wind_speed_10m"],       dtype=float) / 3.6,
        "solar_rad_wm2"  : np.array(h["shortwave_radiation"],  dtype=float),
    })

    df["date"]       = df["time"].dt.date
    df["clock_hour"] = df["time"].dt.hour
    df["month"]      = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")

    assert df[["temperature_2m", "rel_humidity",
               "wind_speed_ms", "solar_rad_wm2"]].notna().all().all(), \
        "NaN values in raw weather data from Open-Meteo API"

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WBGT physics
# ═══════════════════════════════════════════════════════════════════════════════

def stull_wet_bulb(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """
    Compute isobaric wet-bulb temperature (°C) from dry-bulb temperature
    (°C) and relative humidity (%) using the Stull (2011) polynomial
    approximation.

    Equation (1) from:
      Stull R (2011). J. Appl. Meteorol. Climatol. 50:2267–2269.

    Valid range: −20 ≤ T ≤ 50 °C,  5 ≤ RH ≤ 99 %.
    At very low RH the polynomial can produce T_wet marginally above T_dry,
    which is physically impossible (wet-bulb ≤ dry-bulb by definition).
    We enforce this thermodynamic bound by capping at T_dry.  This is a
    physical domain constraint, not a data-correction fallback.

    Parameters
    ----------
    T  : dry-bulb temperature, °C
    RH : relative humidity, %

    Returns
    -------
    T_wet : wet-bulb temperature, °C  (guaranteed ≤ T_dry)
    """
    T  = np.asarray(T,  dtype=float)
    RH = np.asarray(RH, dtype=float)

    T_wet = (
        T  * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
        +    np.arctan(T + RH)
        -    np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )

    # Thermodynamic upper bound: wet-bulb cannot exceed dry-bulb
    return np.minimum(T_wet, T)


def dimiceli_globe_temp(T: np.ndarray,
                         solar: np.ndarray,
                         wind_ms: np.ndarray) -> np.ndarray:
    """
    Estimate black-globe temperature (°C) from air temperature, solar
    irradiance, and wind speed using the Dimiceli et al. (2011)
    regression approximation to the Liljegren energy-balance model.

    T_globe = T_air + 0.01498 * solar^0.7036 / wind^0.1805

    The formula is valid for wind > 0 m/s.  ERA5-Land hourly averages
    occasionally reach 0 m/s due to floating-point rounding of very
    low winds.  Following NIOSH (2016) and ISO 7933 occupational heat
    stress standards, 0.1 m/s is used as the physical calm-wind minimum —
    the lowest meaningful convective transfer rate the formula can represent.
    This is a model domain bound, not a data-correction fallback.

    The equation collapses to T_globe ≈ T_air at night (solar = 0).

    Parameters
    ----------
    T       : air temperature, °C
    solar   : shortwave irradiance, W/m²
    wind_ms : wind speed at 10 m, m/s  (ERA5: occasionally 0.0)

    Returns
    -------
    T_globe : black-globe temperature, °C
    """
    T       = np.asarray(T,       dtype=float)
    solar   = np.asarray(solar,   dtype=float)
    wind_ms = np.asarray(wind_ms, dtype=float)

    # Apply the model's minimum: 0.1 m/s (calm-wind lower bound, not a fallback)
    wind_pos = np.where(wind_ms > 0.1, wind_ms, 0.1)

    solar_term = np.where(solar > 0,
                          0.01498 * solar ** 0.7036 / wind_pos ** 0.1805,
                          0.0)

    return T + solar_term


def compute_wbgt(T: np.ndarray,
                 RH: np.ndarray,
                 solar: np.ndarray,
                 wind_ms: np.ndarray) -> dict:
    """
    Compute full outdoor WBGT and its three components.

    WBGT = 0.7 * T_wet + 0.2 * T_globe + 0.1 * T_dry

    Parameters
    ----------
    T       : dry-bulb temperature, °C
    RH      : relative humidity, %
    solar   : shortwave irradiance, W/m²
    wind_ms : wind speed, m/s

    Returns
    -------
    dict with keys: T_dry, T_wet, T_globe, WBGT
    All values are np.ndarray of same shape as T.
    """
    T       = np.asarray(T,       dtype=float)
    RH      = np.asarray(RH,      dtype=float)
    solar   = np.asarray(solar,   dtype=float)
    wind_ms = np.asarray(wind_ms, dtype=float)

    T_wet    = stull_wet_bulb(T, RH)
    T_globe  = dimiceli_globe_temp(T, solar, wind_ms)
    wbgt_arr = 0.7 * T_wet + 0.2 * T_globe + 0.1 * T

    return {
        "T_dry"   : T,
        "T_wet"   : T_wet,
        "T_globe" : T_globe,
        "WBGT"    : wbgt_arr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WBGT validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

# WBGT action limits from OSHA / ISO 7243 (light work, acclimatised worker)
WBGT_THRESHOLDS = {
    "Low risk"      : (0,    25),   # < 25 °C
    "Moderate risk" : (25,   28),   # 25–28 °C
    "High risk"     : (28,   30),   # 28–30 °C
    "Very high"     : (30,   32.2), # 30–32.2 °C
    "Extreme"       : (32.2, 100),  # > 32.2 °C (OSHA limit for light work)
}


def wbgt_risk_category(wbgt: float) -> str:
    for label, (lo, hi) in WBGT_THRESHOLDS.items():
        if lo <= wbgt < hi:
            return label
    return "Extreme"


def validate_wbgt(df: pd.DataFrame, season: str) -> None:
    """Sanity-check the computed WBGT values."""
    log.info("Validating WBGT for %s …", season)

    assert (df["WBGT"] >= -5).all(),  "WBGT below −5 °C — formula error"
    assert (df["WBGT"] <= 50).all(),  "WBGT above 50 °C — formula error"
    assert (df["T_wet"] <= df["T_dry"]).all(), \
        "T_wet exceeds T_dry — thermodynamic bound violated"
    assert (df["T_globe"] >= df["T_dry"] - 0.01).all(), \
        "T_globe below T_dry at non-zero solar — Dimiceli formula error"

    n = len(df)
    log.info("  WBGT  min=%.1f  mean=%.1f  max=%.1f  (all %d records)",
             df["WBGT"].min(), df["WBGT"].mean(), df["WBGT"].max(), n)
    log.info("  T_dry min=%.1f  mean=%.1f  max=%.1f",
             df["T_dry"].min(), df["T_dry"].mean(), df["T_dry"].max())
    log.info("  T_wet min=%.1f  mean=%.1f  max=%.1f",
             df["T_wet"].min(), df["T_wet"].mean(), df["T_wet"].max())
    log.info("  T_glo min=%.1f  mean=%.1f  max=%.1f",
             df["T_globe"].min(), df["T_globe"].mean(), df["T_globe"].max())

    # Risk distribution
    if season == "summer":
        df["risk"] = df["WBGT"].apply(wbgt_risk_category)
        for label in WBGT_THRESHOLDS:
            cnt = (df["risk"] == label).sum()
            log.info("  %-20s  %4d h  (%.1f%%)", label, cnt, 100 * cnt / n)
    log.info("  Validation passed ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Profile computation
# ═══════════════════════════════════════════════════════════════════════════════

def build_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the raw hourly WBGT time-series into a representative
    diurnal profile used as the weather lookup in Step 4.

    For each clock hour (0–23) we compute:
      - wbgt_mean   : mean WBGT across all days in the window
      - wbgt_p90    : 90th-percentile WBGT (design-day / hot conditions)
      - wbgt_p10    : 10th-percentile WBGT (cool conditions reference)
      - t_dry_mean  : mean air temperature
      - t_wet_mean  : mean wet-bulb temperature
      - t_globe_mean: mean globe temperature
      - solar_mean  : mean shortwave radiation

    The mean profile is the primary lookup for Step 4; p90 and p10 are
    used in the sensitivity analysis of Step 7.
    """
    profile = (
        df.groupby("clock_hour")
        .agg(
            wbgt_mean    = ("WBGT",        "mean"),
            wbgt_p90     = ("WBGT",        lambda x: np.percentile(x, 90)),
            wbgt_p10     = ("WBGT",        lambda x: np.percentile(x, 10)),
            wbgt_std     = ("WBGT",        "std"),
            t_dry_mean   = ("T_dry",       "mean"),
            t_wet_mean   = ("T_wet",       "mean"),
            t_globe_mean = ("T_globe",     "mean"),
            solar_mean   = ("solar_rad_wm2", "mean"),
            n_obs        = ("WBGT",        "count"),
        )
        .reset_index()
    )

    # Add risk category labels for the mean profile
    profile["risk_mean"] = profile["wbgt_mean"].apply(wbgt_risk_category)

    return profile


def build_monthly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as build_hourly_profile but stratified by month.
    Used in the seasonal sensitivity section of Step 7.
    """
    profile = (
        df.groupby(["month", "month_name", "clock_hour"])
        .agg(
            wbgt_mean    = ("WBGT",   "mean"),
            wbgt_p90     = ("WBGT",   lambda x: np.percentile(x, 90)),
            t_dry_mean   = ("T_dry",  "mean"),
            solar_mean   = ("solar_rad_wm2", "mean"),
            n_obs        = ("WBGT",   "count"),
        )
        .reset_index()
    )
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_summer_profile(profile: pd.DataFrame) -> None:
    """Log the 24-hour mean summer WBGT profile for quick inspection."""
    log.info("── Summer WBGT diurnal profile (mean / p90) ──")
    log.info("  Hour  T_dry  T_wet  T_glob  WBGT_mean  WBGT_p90  Risk")
    for _, row in profile.iterrows():
        log.info("  %02dh   %5.1f  %5.1f  %6.1f      %5.2f     %5.2f   %s",
                 row["clock_hour"], row["t_dry_mean"], row["t_wet_mean"],
                 row["t_globe_mean"], row["wbgt_mean"], row["wbgt_p90"],
                 row["risk_mean"])

    peak_hour = profile.loc[profile["wbgt_mean"].idxmax(), "clock_hour"]
    peak_wbgt = profile["wbgt_mean"].max()
    log.info("  Peak mean WBGT: %.2f °C at %02dh", peak_wbgt, int(peak_hour))

    # Count transfer-heavy hours (09–18) vs all hours
    transit_hours = profile[profile["clock_hour"].between(9, 18)]
    log.info("  Mean WBGT during transit hours (09–18h): %.2f °C",
             transit_hours["wbgt_mean"].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline for one season
# ═══════════════════════════════════════════════════════════════════════════════

def process_season(season: str, start: str, end: str) -> tuple:
    """
    Full pipeline for one season: fetch from Open-Meteo → compute WBGT → profiles.

    Always fetches fresh data from the Open-Meteo archive API and saves the
    raw CSV to data/weather/raw_{season}.csv for reproducibility.

    Parameters
    ----------
    season : label string ("summer", "spring", "fall")
    start  : ISO date string  e.g. "2025-06-01"
    end    : ISO date string  e.g. "2025-08-31"

    Returns
    -------
    (hourly_df, profile_df, monthly_df)
    """
    raw_path = WEATHER_DIR / f"raw_{season}.csv"

    log.info("[%s] Fetching %s → %s from Open-Meteo …", season, start, end)
    payload = fetch_open_meteo(start, end)
    df      = payload_to_dataframe(payload)
    df.to_csv(raw_path, index=False)
    log.info("[%s] Raw data saved to %s", season, raw_path)

    # ── Compute WBGT ──────────────────────────────────────────────────────
    log.info("[%s] Computing WBGT …", season)
    wbgt_components = compute_wbgt(
        T       = df["temperature_2m"].values,
        RH      = df["rel_humidity"].values,
        solar   = df["solar_rad_wm2"].values,
        wind_ms = df["wind_speed_ms"].values,
    )

    df["T_dry"]   = wbgt_components["T_dry"]
    df["T_wet"]   = wbgt_components["T_wet"]
    df["T_globe"] = wbgt_components["T_globe"]
    df["WBGT"]    = wbgt_components["WBGT"]

    validate_wbgt(df, season)

    hourly_profile  = build_hourly_profile(df)
    monthly_profile = build_monthly_profiles(df)

    if season == "summer":
        print_summer_profile(hourly_profile)

    return df, hourly_profile, monthly_profile


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    log.info("Step 3 — Fetch weather data and compute WBGT")

    results = {}

    for season, (start, end) in SEASONS.items():
        log.info("=" * 60)
        log.info("Season: %s  (%s → %s)", season, start, end)
        log.info("=" * 60)

        hourly_df, profile_df, monthly_df = process_season(
            season, start, end)

        # ── Save outputs ──────────────────────────────────────────────────
        hourly_df.to_parquet(
            WEATHER_DIR / f"wbgt_hourly_{season}.parquet", index=False)
        profile_df.to_parquet(
            WEATHER_DIR / f"wbgt_profile_{season}.parquet", index=False)
        monthly_df.to_parquet(
            WEATHER_DIR / f"wbgt_monthly_{season}.parquet", index=False)

        log.info("[%s] Saved hourly (%d rows), profile (24 rows), monthly (%d rows)",
                 season, len(hourly_df), len(monthly_df))

        results[season] = {
            "hourly" : hourly_df,
            "profile": profile_df,
            "monthly": monthly_df,
        }

    # ── Cross-season comparison ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("Cross-season WBGT comparison (mean over 09–18h transit window)")
    log.info("=" * 60)
    for season, res in results.items():
        transit = res["profile"][res["profile"]["clock_hour"].between(9, 18)]
        log.info("  %-8s  mean WBGT=%.2f°C  peak=%.2f°C at h=%02d",
                 season,
                 transit["wbgt_mean"].mean(),
                 res["profile"]["wbgt_mean"].max(),
                 int(res["profile"].loc[res["profile"]["wbgt_mean"].idxmax(),
                                        "clock_hour"]))

    log.info("Step 3 complete.")
    return results


if __name__ == "__main__":
    results = main()
