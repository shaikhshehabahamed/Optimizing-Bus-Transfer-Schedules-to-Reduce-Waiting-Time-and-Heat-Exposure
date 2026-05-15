"""
Step 7 — Sensitivity Analysis and Metrics Summary
==================================================
Evaluates the optimisation solutions, runs seasonal and headway
sensitivity analyses, and logs all key metrics cited in the paper.

Figure generation is handled entirely by step08_paper_outputs.py.

Outputs (legacy sensitivity CSVs — step08 produces the canonical tables)
-------------------------------------------------------------------------
results/tables/sensitivity_seasonal.csv
results/tables/sensitivity_headways.csv
results/tables/sensitivity_route_breakdown.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

sys.path.insert(0, str(Path(__file__).parent))
from step05_formulation import (
    build_trip_index, build_transfer_arrays, build_headway_arrays,
    build_wbgt_interp_arrays, TransferSyncProblem, DELTA_MAX_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
GTFS_DIR = ROOT / "data" / "gtfs"
WX_DIR   = ROOT / "data" / "weather"
SOL_DIR  = ROOT / "results" / "solutions"
TAB_DIR  = ROOT / "results" / "tables"


# ─────────────────────────────────────────────────────────────────────────────
# Problem reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_problem():
    """Rebuild the TransferSyncProblem from saved parquet/npy files."""
    se      = pd.read_parquet(GTFS_DIR / "stop_events.parquet")
    hei_df  = pd.read_parquet(GTFS_DIR / "transfer_hei.parquet")
    hw_df   = pd.read_parquet(ROOT / "data" / "problem" / "headway_pairs.parquet")
    wbgt    = np.load(WX_DIR / "wbgt_lookup.npy")

    ti      = build_trip_index(se)
    ta      = build_transfer_arrays(hei_df, ti)
    ha      = build_headway_arrays(se, ti)
    he, we  = build_wbgt_interp_arrays(wbgt)
    prob    = TransferSyncProblem(ta, ha, he, we, len(ti), DELTA_MAX_SEC)

    log.info("Problem reconstructed: %d trips, %d transfers, %d headway pairs",
             prob.n_var, prob.n_transfers, prob.n_headways)
    return prob, hei_df, hw_df


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(prob, hei_df, hw_df):
    log.info("Sensitivity analysis")

    pareto_X  = np.load(SOL_DIR / "nsga2_pareto_X.npy")
    pareto_F  = np.load(SOL_DIR / "nsga2_pareto_F.npy")
    delta_gf1 = np.load(SOL_DIR / "greedy_f1_delta.npy")
    delta_gf2 = np.load(SOL_DIR / "greedy_f2_delta.npy")

    bf1 = prob.wait_sec.sum() / 60.0
    bf2 = (np.interp((prob.arr_sec % 86400) / 3600, prob.hours_ext, prob.wbgt_ext)
           * prob.wait_sec / 60.0 * prob.shade * prob.freq).sum()

    f1n         = pareto_F[:, 0] / bf1
    f2n         = pareto_F[:, 1] / bf2
    knee_delta  = pareto_X[np.sqrt(f1n**2 + f2n**2).argmin()]
    minf2_delta = pareto_X[pareto_F[:, 1].argmin()]
    minf1_delta = pareto_X[pareto_F[:, 0].argmin()]

    solutions = {
        "Baseline"       : np.zeros(prob.n_var),
        "Greedy-f1"      : delta_gf1,
        "Greedy-f2"      : delta_gf2,
        "NSGA-II min-f1" : minf1_delta,
        "NSGA-II min-f2" : minf2_delta,
        "NSGA-II knee"   : knee_delta,
    }

    # ── Table S1: seasonal re-scoring ─────────────────────────────────────
    log.info("  Table S1: seasonal re-scoring")
    rows = []
    for season in ["summer", "spring", "fall"]:
        prof = (pd.read_parquet(WX_DIR / f"wbgt_profile_{season}.parquet")
                .sort_values("clock_hour"))
        wa   = np.append(prof["wbgt_mean"].values, prof["wbgt_mean"].values[0])
        he2  = np.arange(25, dtype=float)
        for sol_name, delta in solutions.items():
            fh  = ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600
            wv  = np.interp(fh, he2, wa)
            nwm = np.maximum(0, prob.wait_sec + delta[prob.conn_idx]
                             - delta[prob.feeder_idx]) / 60.0
            f2s = (wv * nwm * prob.shade * prob.freq).sum()
            rows.append({"season": season, "solution": sol_name,
                         "f1": nwm.sum(), "f2_season": f2s})
    s1 = pd.DataFrame(rows)
    pivot = s1.pivot_table(index="solution", columns="season",
                           values=["f1", "f2_season"], aggfunc="first")
    pivot.to_csv(TAB_DIR / "sensitivity_seasonal.csv")
    log.info("    Seasonal re-scoring table saved")
    for _, row in s1[s1["season"] == "summer"].iterrows():
        log.info("    %-22s  summer f2=%.1f", row["solution"], row["f2_season"])

    # ── Table S2: headway stability ────────────────────────────────────────
    log.info("  Table S2: headway stability statistics")
    hw_rows = []
    for sol_name, delta in solutions.items():
        new_hw = prob.hw_sec + (delta[prob.next_idx] - delta[prob.prev_idx])
        hw_rows.append({
            "solution"          : sol_name,
            "n_pairs"           : len(prob.hw_sec),
            "n_violated"        : int((new_hw < 0.8 * prob.hw_sec - 1e-6).sum()),
            "min_hw_ratio"      : round(float((new_hw / prob.hw_sec).min()), 4),
            "mean_hw_ratio"     : round(float((new_hw / prob.hw_sec).mean()), 4),
            "pct_hw_compressed" : round(float(100 * (new_hw < prob.hw_sec).mean()), 1),
        })
    s2 = pd.DataFrame(hw_rows)
    s2.to_csv(TAB_DIR / "sensitivity_headways.csv", index=False)
    log.info("    Headway stability table saved")
    for _, row in s2.iterrows():
        log.info("    %-22s  n_viol=%d  min_ratio=%.3f  pct_compressed=%.1f%%",
                 row["solution"], row["n_violated"],
                 row["min_hw_ratio"], row["pct_hw_compressed"])

    # ── Table S3: per-route breakdown for knee solution ────────────────────
    log.info("  Table S3: per-route breakdown for knee solution")
    base_wm   = prob.wait_sec / 60.0
    base_wbgt = np.interp((prob.arr_sec % 86400) / 3600, prob.hours_ext, prob.wbgt_ext)
    base_hei  = base_wbgt * base_wm * prob.shade * prob.freq

    rk     = prob.evaluate_single(knee_delta)
    knee_wm= np.maximum(0, rk["new_wait_sec"]) / 60.0
    knee_wa= np.interp(((prob.arr_sec + knee_delta[prob.feeder_idx]) % 86400) / 3600,
                        prob.hours_ext, prob.wbgt_ext)
    knee_hei = knee_wa * knee_wm * prob.shade * prob.freq

    route_rows = []
    for rte, grp in hei_df.groupby("feeder_route_short"):
        idx = grp.index
        saved_wait = (base_wm[idx] - knee_wm[idx]).sum()
        saved_hei  = (base_hei[idx] - knee_hei[idx]).sum()
        route_rows.append({
            "route"          : rte,
            "n_transfers"    : len(idx),
            "wait_saved_min" : round(saved_wait, 2),
            "wait_saved_pct" : round(100 * saved_wait / base_wm[idx].sum(), 2),
            "hei_saved"      : round(saved_hei, 2),
            "hei_saved_pct"  : round(100 * saved_hei / base_hei[idx].sum(), 2),
        })
        log.info("    Route %-5s  n=%4d  wait_saved=%.1f min (%.1f%%)  hei_saved=%.1f (%.1f%%)",
                 rte, len(idx), saved_wait,
                 100 * saved_wait / base_wm[idx].sum(),
                 saved_hei, 100 * saved_hei / base_hei[idx].sum())
    s3 = pd.DataFrame(route_rows)
    s3.to_csv(TAB_DIR / "sensitivity_route_breakdown.csv", index=False)
    log.info("    Per-route breakdown saved")

    return s1, s2, s3


# ─────────────────────────────────────────────────────────────────────────────
# Paper metrics summary
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_metrics(prob, hei_df):
    """Log all key numbers cited in the paper — all computed dynamically."""
    pareto_X = np.load(SOL_DIR / "nsga2_pareto_X.npy")
    pareto_F = np.load(SOL_DIR / "nsga2_pareto_F.npy")

    bf1 = prob.wait_sec.sum() / 60.0
    bf2 = (np.interp((prob.arr_sec % 86400) / 3600, prob.hours_ext, prob.wbgt_ext)
           * prob.wait_sec / 60.0 * prob.shade * prob.freq).sum()

    f1n         = pareto_F[:, 0] / bf1
    f2n         = pareto_F[:, 1] / bf2
    knee_delta  = pareto_X[np.sqrt(f1n**2 + f2n**2).argmin()]
    minf2_delta = pareto_X[pareto_F[:, 1].argmin()]

    Fn = pareto_F / np.array([bf1, bf2])
    hv = float(HV(ref_point=np.array([1.01, 1.01]))(Fn))

    se          = pd.read_parquet(GTFS_DIR / "stop_events.parquet")
    rc          = pd.read_parquet(GTFS_DIR / "routes_clean.parquet")
    wbgt_arr    = np.load(WX_DIR / "wbgt_lookup.npy")

    peak_hei_hour = int(
        hei_df.groupby("transfer_clock_hour")["hei_weighted_mean"]
        .sum().idxmax())
    pct_12_17 = 100 * (
        hei_df[hei_df["transfer_clock_hour"].between(12, 17)]["hei_weighted_mean"].sum()
        / hei_df["hei_weighted_mean"].sum())

    log.info("=" * 70)
    log.info("KEY METRICS FOR PAPER (all computed dynamically from current data)")
    log.info("=" * 70)
    log.info("Network: %d routes total (%d weekday) | %d trips | %d transfer pairs",
             rc["route_short_name"].nunique(), se["route_id"].nunique(),
             se["trip_id"].nunique(), len(hei_df))
    log.info("  Stops: %d unique | %d exposed | %d sheltered",
             se["stop_id"].nunique(),
             int((~hei_df["is_sheltered"]).sum()),
             int(hei_df["is_sheltered"].sum()))
    log.info("  Transfer window: 1-15 min | mean=%.2f min | median=%.2f min",
             hei_df["wait_min"].mean(), hei_df["wait_min"].median())
    log.info("")
    log.info("Baseline (delta=0):")
    log.info("  f1 = %.2f min  (%.1f h total passenger wait)", bf1, bf1 / 60)
    log.info("  f2 = %.2f degrees_C*min  (freq-weighted HEI)", bf2)
    log.info("  Peak WBGT hour: %dh (%.2f degrees_C mean)", int(wbgt_arr.argmax()), wbgt_arr.max())
    log.info("  Peak HEI hour: %dh  (%.1f%% of total HEI)", peak_hei_hour, pct_12_17)
    log.info("")
    for sol_name, delta in [
        ("Greedy-f1",       np.load(SOL_DIR / "greedy_f1_delta.npy")),
        ("Greedy-f2",       np.load(SOL_DIR / "greedy_f2_delta.npy")),
        ("NSGA-II min-f1",  pareto_X[pareto_F[:, 0].argmin()]),
        ("NSGA-II min-f2",  minf2_delta),
        ("NSGA-II knee",    knee_delta),
    ]:
        r = prob.evaluate_single(delta)
        log.info("%-22s  f1=%.1f (-%.2f%%)  f2=%.1f (-%.2f%%)  feasible=%s  missed=%d",
                 sol_name, r["f1"], 100 * (1 - r["f1"] / bf1),
                 r["f2"], 100 * (1 - r["f2"] / bf2),
                 r["is_feasible"], r["n_missed_connections"])
    log.info("")
    log.info("Pareto front: %d solutions | HV=%.6f (normalised, ref=1.01x baseline)",
             len(pareto_F), hv)
    conv = pd.read_csv(TAB_DIR / "tab08_convergence.csv")
    log.info("NSGA-II: %d gens x pop=200 | runtime %.0f s",
             int(conv["gen"].max()), float(conv["elapsed"].max()))
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Step 7 - Sensitivity Analysis and Metrics Summary")

    prob, hei_df, hw_df = _reconstruct_problem()
    sensitivity_analysis(prob, hei_df, hw_df)
    print_paper_metrics(prob, hei_df)

    log.info("Step 7 complete.")
    log.info("Run step08_paper_outputs.py to generate all figures and canonical tables.")


if __name__ == "__main__":
    main()
