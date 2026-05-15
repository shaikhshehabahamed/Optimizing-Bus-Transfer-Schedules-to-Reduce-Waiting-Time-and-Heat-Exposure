"""
Step 5 — Bi-Objective Optimisation Problem Formulation
========================================================
Defines the TransferSyncProblem class — a pymoo-compatible Problem
subclass — and constructs all the precomputed data structures that
make batch objective evaluation fully vectorised.

Problem Statement
-----------------
Decision variables
    δ ∈ ℝⁿ,   n = number of weekday trips (1137)
    δᵢ ∈ [−Δ, +Δ] seconds   (Δ = 300 s = 5 min)
    δᵢ is the number of seconds trip i's departure is slid forward
    (positive) or backward (negative) from its scheduled time.

Objectives (both minimised)
    f₁(δ) = Σ_k  max(0, w_k + δ[conn_k] − δ[feed_k]) / 60
             total passenger transfer wait time  [min]

    f₂(δ) = Σ_k  WBGT(h_k(δ)) × max(0, w_k + δ[conn_k] − δ[feed_k]) / 60
                 × shade_k × freq_k
             total frequency-weighted heat exposure index  [°C·min]

    where
      w_k              : scheduled wait for transfer k  [s]
      conn_k, feed_k   : trip indices of the connector and feeder legs
      WBGT(h_k(δ))     : mean summer WBGT at the shifted feeder arrival
                          hour, from the 24-h diurnal profile; evaluated
                          via linear interpolation across clock hours
      shade_k          : 0.7 (sheltered stop) or 1.0 (exposed)
      freq_k           : feeder-route trip-frequency proxy ∈ (0,1]

Constraints (inequality, g ≤ 0 in pymoo convention)
    For each consecutive intra-block trip pair (p, q) on the same
    route-block-direction, the scheduled block headway hw_pq must not
    shrink below 80% of its scheduled value:

        g_{pq} = −(δ[q] − δ[p]) − 0.2 × hw_{pq} ≤ 0

    i.e.,   δ[q] − δ[p] ≥ −0.2 × hw_{pq}

    New headway = hw_{pq} + (δ[q] − δ[p]) ≥ 0.8 × hw_{pq}.

    There are 928 such constraints (one per intra-block consecutive pair).

Note on missed connections
    When δ[conn_k] − δ[feed_k] < −w_k (the connector now departs before
    the feeder arrives), max(0, ·) clamps the contribution to zero.
    This soft treatment under-penalises missed connections; we document
    it as a model limitation and note that the baseline schedule has no
    such violations by construction.

Outputs
-------
data/problem/headway_pairs.parquet  — intra-block headway constraint pairs
                                       (used by step07 sensitivity analysis)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

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
PROB_DIR    = ROOT / "data" / "problem"
TABLE_DIR   = ROOT / "results" / "tables"
PROB_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── problem constants ─────────────────────────────────────────────────────────
DELTA_MAX_SEC   = 300.0   # ±5 minutes maximum shift
HEADWAY_FACTOR  = 0.20    # headway may shrink by at most 20%


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Build precomputed arrays
# ═══════════════════════════════════════════════════════════════════════════════

def build_trip_index(stop_events: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a dense integer index (0 … n_trips-1) to every unique trip_id.
    This index is the position of that trip's δ in the decision vector.

    Returns a DataFrame with columns [trip_id, trip_idx].
    """
    trip_ids = np.sort(stop_events["trip_id"].unique())
    trip_index = pd.DataFrame({
        "trip_id" : trip_ids,
        "trip_idx": np.arange(len(trip_ids), dtype=int),
    })
    log.info("Trip index: %d trips  (idx 0 … %d)", len(trip_index), len(trip_index) - 1)
    return trip_index


def build_transfer_arrays(hei_df: pd.DataFrame,
                           trip_index: pd.DataFrame) -> dict:
    """
    Build vectorised numpy arrays for the objective functions.

    For each of the N_T = 15038 transfer pairs we store:
      feeder_idx         : (N_T,) int   — δ-vector index of feeder trip
      conn_idx           : (N_T,) int   — δ-vector index of connector trip
      scheduled_wait_sec : (N_T,) float — baseline wait w_k  [s]
      feeder_arrival_sec : (N_T,) float — baseline feeder arrival  [s since midnight]
      shade              : (N_T,) float — 0.7 or 1.0
      freq               : (N_T,) float — frequency proxy ∈ (0,1]
    """
    t2i = trip_index.set_index("trip_id")["trip_idx"]

    feeder_idx  = hei_df["feeder_trip_id"].map(t2i).to_numpy(dtype=int)
    conn_idx    = hei_df["connector_trip_id"].map(t2i).to_numpy(dtype=int)

    n_unmapped = np.isnan(feeder_idx.astype(float)).sum() + \
                 np.isnan(conn_idx.astype(float)).sum()
    if n_unmapped > 0:
        raise ValueError(f"{n_unmapped} transfer trips not found in trip_index")

    arrays = {
        "feeder_idx"         : feeder_idx,
        "conn_idx"           : conn_idx,
        "scheduled_wait_sec" : hei_df["wait_sec"].to_numpy(dtype=float),
        "feeder_arrival_sec" : hei_df["feeder_arrival_sec"].to_numpy(dtype=float),
        "shade"              : hei_df["shade_factor"].to_numpy(dtype=float),
        "freq"               : hei_df["freq_proxy"].to_numpy(dtype=float),
    }

    log.info("Transfer arrays built:  %d pairs × %d fields",
             len(feeder_idx), len(arrays))
    log.info("  feeder_idx range: %d – %d", feeder_idx.min(), feeder_idx.max())
    log.info("  conn_idx   range: %d – %d", conn_idx.min(),   conn_idx.max())
    log.info("  scheduled_wait_sec: mean=%.1f  min=%.1f  max=%.1f",
             arrays["scheduled_wait_sec"].mean(),
             arrays["scheduled_wait_sec"].min(),
             arrays["scheduled_wait_sec"].max())
    return arrays


def build_headway_arrays(stop_events: pd.DataFrame,
                          trip_index: pd.DataFrame) -> dict:
    """
    Build vectorised numpy arrays for the intra-block headway constraints.

    Consecutive trips within the same (route, direction, block) tuple are
    identified by sorting on first-stop departure time within each group.
    The constraint for pair (prev, next) is:

        g = −(δ[next_idx] − δ[prev_idx]) − 0.2 × hw_sec ≤ 0

    Returns arrays:
      prev_idx    : (N_H,) int    — δ-vector index of the preceding trip
      next_idx    : (N_H,) int    — δ-vector index of the following trip
      headway_sec : (N_H,) float  — scheduled intra-block headway  [s]
    """
    t2i = trip_index.set_index("trip_id")["trip_idx"]

    # First-stop departure time for each trip
    first_dep = (
        stop_events.sort_values("stop_sequence")
        .groupby("trip_id")[["departure_sec", "route_id",
                              "direction_id", "block_id"]]
        .first()
        .reset_index()
    )

    records = []
    for (route_id, direction_id, block_id), grp in first_dep.groupby(
            ["route_id", "direction_id", "block_id"]):
        grp = grp.sort_values("departure_sec").reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(1, len(grp)):
            hw = grp.loc[i, "departure_sec"] - grp.loc[i - 1, "departure_sec"]
            if hw <= 0:
                continue
            records.append({
                "route_id"    : route_id,
                "direction_id": int(direction_id),
                "block_id"    : int(block_id),
                "prev_trip_id": int(grp.loc[i - 1, "trip_id"]),
                "next_trip_id": int(grp.loc[i,     "trip_id"]),
                "headway_sec" : float(hw),
            })

    hw_df = pd.DataFrame(records)

    prev_idx = hw_df["prev_trip_id"].map(t2i).to_numpy(dtype=int)
    next_idx = hw_df["next_trip_id"].map(t2i).to_numpy(dtype=int)

    arrays = {
        "prev_idx"   : prev_idx,
        "next_idx"   : next_idx,
        "headway_sec": hw_df["headway_sec"].to_numpy(dtype=float),
        "hw_df"      : hw_df,   # kept for diagnostics only
    }

    log.info("Headway arrays built:  %d intra-block pairs", len(prev_idx))
    log.info("  headway_sec: mean=%.1f s  min=%.1f s  max=%.1f s",
             arrays["headway_sec"].mean(),
             arrays["headway_sec"].min(),
             arrays["headway_sec"].max())
    log.info("  min allowed (δ_next − δ_prev): %.1f s  (tightest constraint)",
             (-HEADWAY_FACTOR * arrays["headway_sec"]).max())
    return arrays


def build_wbgt_interp_arrays(wbgt_mean_arr: np.ndarray) -> tuple:
    """
    Build the extended (25-element) arrays needed for np.interp with
    hour-wrapping.  The 25th element is wbgt[0] to allow smooth
    interpolation across the 23h → 0h boundary.

    Returns (hours_ext, wbgt_ext) both shape (25,).
    """
    hours_ext = np.arange(25, dtype=float)
    wbgt_ext  = np.append(wbgt_mean_arr, wbgt_mean_arr[0])
    return hours_ext, wbgt_ext


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — pymoo Problem class
# ═══════════════════════════════════════════════════════════════════════════════

class TransferSyncProblem(Problem):
    """
    Bi-objective transit transfer synchronisation problem.

    Variables   : δ ∈ [−Δ, +Δ]ⁿ  seconds  (one per weekday trip)

    Objectives
    ----------
    f₁ = Σ_k max(0, new_wait_k) / 60        [total passenger wait, min]
    f₂ = Σ_k WBGT(h_k) × max(0, new_wait_k)/60 × shade_k × freq_k
                                              [freq-weighted HEI, °C·min]

    Constraints  (928 intra-block headway pairs)
    ----------
    g_{pq} ≤ 0 : (δ[prev] − δ[next]) − 0.2×hw ≤ 0

    Missed connections (new_wait < 60 s) are NOT hard-constrained — they
    are counted post-hoc in evaluate_single and reported as a transparency
    metric.  This is consistent with the slip-shifting literature where
    feasibility of passenger connections is handled by planners at the
    implementation stage; the optimiser explores the full objective space.
    """

    MIN_WAIT_SEC = 60.0

    def __init__(self,
                 transfer_arrays: dict,
                 headway_arrays:  dict,
                 hours_ext:       np.ndarray,
                 wbgt_ext:        np.ndarray,
                 n_trips:         int,
                 delta_max:       float = DELTA_MAX_SEC):

        n_var  = n_trips
        n_obj  = 2
        n_ieq  = len(headway_arrays["prev_idx"])
        xl     = np.full(n_var, -delta_max)
        xu     = np.full(n_var,  delta_max)

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq,
                         xl=xl, xu=xu)

        self.feeder_idx = transfer_arrays["feeder_idx"]
        self.conn_idx   = transfer_arrays["conn_idx"]
        self.wait_sec   = transfer_arrays["scheduled_wait_sec"]
        self.arr_sec    = transfer_arrays["feeder_arrival_sec"]
        self.shade      = transfer_arrays["shade"]
        self.freq       = transfer_arrays["freq"]

        self.prev_idx   = headway_arrays["prev_idx"]
        self.next_idx   = headway_arrays["next_idx"]
        self.hw_sec     = headway_arrays["headway_sec"]

        self.hours_ext  = hours_ext
        self.wbgt_ext   = wbgt_ext

        self.delta_max  = delta_max
        self.n_transfers= len(self.feeder_idx)
        self.n_headways = n_ieq

        log.info("TransferSyncProblem initialised:")
        log.info("  n_var=%d  n_obj=2  n_ieq=%d (headway)  "
                 "missed-connections counted post-hoc", n_var, n_ieq)
        log.info("  δ ∈ [%.0f, %.0f] s  (±%.1f min)",
                 -delta_max, delta_max, delta_max / 60)
        log.info("  n_transfers=%d", self.n_transfers)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        pop  = X.shape[0]
        dF   = X[:, self.feeder_idx]
        dC   = X[:, self.conn_idx]

        new_wait_sec = self.wait_sec + dC - dF
        new_wait_min = np.maximum(0.0, new_wait_sec) / 60.0

        f1 = new_wait_min.sum(axis=1)

        shifted_arr = (self.arr_sec + dF) % 86400.0
        wbgt_flat   = np.interp(shifted_arr.ravel() / 3600.0,
                                self.hours_ext, self.wbgt_ext)
        wbgt        = wbgt_flat.reshape(pop, self.n_transfers)
        f2          = (wbgt * new_wait_min * self.shade * self.freq).sum(axis=1)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = (X[:, self.prev_idx] - X[:, self.next_idx]) \
                   - HEADWAY_FACTOR * self.hw_sec



    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: evaluate a single solution and return scalar objectives
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_single(self, delta: np.ndarray) -> dict:
        """Evaluate one solution and return full diagnostics."""
        delta = np.asarray(delta, dtype=float).reshape(1, -1)
        out   = {}
        self._evaluate(delta, out)
        F  = out["F"][0]
        G  = out["G"][0]

        dF           = delta[0, self.feeder_idx]
        dC           = delta[0, self.conn_idx]
        new_wait_sec = self.wait_sec + dC - dF

        return {
            "f1"                       : float(F[0]),
            "f2"                       : float(F[1]),
            "is_feasible"              : bool((G <= 0).all()),
            "max_constraint_violation" : float(np.maximum(0, G).max()),
            "n_violated_constraints"   : int((G > 0).sum()),
            "n_missed_connections"     : int((new_wait_sec < self.MIN_WAIT_SEC).sum()),
            "new_wait_sec"             : new_wait_sec,
            "constraint_violations"    : G,
        }

    def headway_violation_summary(self, delta: np.ndarray,
                                   hw_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of violated headway constraints for a solution."""
        delta = np.asarray(delta, dtype=float)
        dP = delta[self.prev_idx]
        dN = delta[self.next_idx]
        G  = (dP - dN) - HEADWAY_FACTOR * self.hw_sec

        result = hw_df.copy()
        result["delta_prev"] = dP
        result["delta_next"] = dN
        result["g_value"]    = G
        result["violated"]   = G > 0
        result["new_hw_sec"] = result["headway_sec"] + (dN - dP)
        return result[result["violated"]].sort_values("g_value", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Baseline verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_baseline(problem: TransferSyncProblem,
                    hei_df: pd.DataFrame) -> None:
    log.info("Verifying baseline (δ = 0) …")
    delta_zero = np.zeros(problem.n_var)
    res = problem.evaluate_single(delta_zero)

    expected_f1 = float(hei_df["wait_min"].sum())
    expected_f2 = float(hei_df["hei_weighted_mean"].sum())
    f1_err = abs(res["f1"] - expected_f1)
    f2_err = abs(res["f2"] - expected_f2)

    log.info("  f1 (total wait min)  : computed=%.3f  expected=%.3f  Δ=%.6f",
             res["f1"], expected_f1, f1_err)
    log.info("  f2 (weighted HEI)    : computed=%.3f  expected=%.3f  Δ=%.6f",
             res["f2"], expected_f2, f2_err)
    log.info("  is_feasible          : %s", res["is_feasible"])
    log.info("  n_missed_connections : %d", res["n_missed_connections"])

    tol = 1e-4
    assert f1_err < tol, f"f1 mismatch at δ=0: Δ={f1_err:.6f}"
    assert f2_err < tol, f"f2 mismatch at δ=0: Δ={f2_err:.6f}"
    assert res["is_feasible"], "Baseline δ=0 is infeasible"
    assert res["n_missed_connections"] == 0, "Missed connections at δ=0"
    log.info("  Baseline verification passed ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Problem diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_problem_diagnostics(problem: TransferSyncProblem,
                               hw_df: pd.DataFrame,
                               trip_index: pd.DataFrame) -> None:
    """Log a structured summary of the optimisation problem."""

    log.info("=" * 65)
    log.info("OPTIMISATION PROBLEM DIAGNOSTICS")
    log.info("=" * 65)

    # ── Dimensions ────────────────────────────────────────────────────────
    log.info("Dimensions:")
    log.info("  Decision variables   : %d  (δᵢ ∈ [%.0f, %.0f] s)",
             problem.n_var, -problem.delta_max, problem.delta_max)
    log.info("  Objectives           : 2  (f₁ wait-time, f₂ heat-exposure)")
    log.info("  Inequality constraints: %d  (intra-block headway stability)",
             problem.n_ieq_constr)
    log.info("  Transfer pairs       : %d", problem.n_transfers)

    # ── Constraint analysis ───────────────────────────────────────────────
    log.info("")
    log.info("Headway constraint analysis:")
    hw = problem.hw_sec
    min_delta_allowed = -HEADWAY_FACTOR * hw   # lower bound on (δN - δP)
    log.info("  Min allowed (δ_next−δ_prev) across all pairs: %.1f s",
             min_delta_allowed.min())
    log.info("  Pairs where Δ=300s cannot violate constraint: %d / %d",
             (min_delta_allowed < -2 * problem.delta_max).sum(),
             len(hw))
    log.info("  Binding pairs (constraint can be violated): %d",
             (min_delta_allowed >= -2 * problem.delta_max).sum())
    log.info("")
    log.info("  Per-route headway stats (block-level, minutes):")
    stats = (
        hw_df.groupby("route_id")["headway_sec"]
        .agg(n="count", mean="mean", min_hw="min", max_hw="max")
        .assign(mean_min=lambda d: d["mean"] / 60,
                min_min =lambda d: d["min_hw"] / 60,
                binding =lambda d: d["min_hw"] < 3 * problem.delta_max)
        .reset_index()
    )
    log.info("  %-8s  %5s  %9s  %7s  %8s", "route", "n", "mean(min)", "min(min)", "binding?")
    for _, row in stats.sort_values("min_min").iterrows():
        log.info("  %-8s  %5d  %9.1f  %7.1f  %8s",
                 row["route_id"], int(row["n"]),
                 row["mean_min"], row["min_min"],
                 "YES" if row["binding"] else "no")

    # ── Objective landscape at δ=0 ────────────────────────────────────────
    log.info("")
    log.info("Baseline objective values (δ = 0):")
    log.info("  f₁ (total wait time) : %.2f min  = %.1f h",
             problem.wait_sec.sum() / 60,
             problem.wait_sec.sum() / 3600)
    wbgt_baseline = np.interp(
        (problem.arr_sec % 86400) / 3600,
        problem.hours_ext, problem.wbgt_ext)
    hei_baseline = (wbgt_baseline * problem.wait_sec / 60
                    * problem.shade * problem.freq)
    log.info("  f₂ (weighted HEI)    : %.2f °C·min", hei_baseline.sum())
    log.info("  WBGT range at transfers: %.2f – %.2f °C",
             wbgt_baseline.min(), wbgt_baseline.max())

    # ── Sensitivity: what is the maximum achievable f₁ reduction? ────────
    # Lower bound on f₁: all waits reduced to MIN_TRANSFER_SEC (60s)
    min_possible_f1 = len(problem.feeder_idx) * 60.0 / 60.0
    log.info("")
    log.info("Objective bounds (approximate):")
    log.info("  f₁ lower bound (all waits = 60 s) : %.1f min", min_possible_f1)
    log.info("  f₁ upper bound (δ=0)              : %.1f min",
             problem.wait_sec.sum() / 60)
    log.info("  Max possible f₁ reduction          : %.1f%%",
             100 * (1 - min_possible_f1 / (problem.wait_sec.sum() / 60)))

    # ── Decision space volume ─────────────────────────────────────────────
    log.info("")
    log.info("Search space: hypercube [%.0f, %.0f]^%d",
             -problem.delta_max, problem.delta_max, problem.n_var)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    log.info("Step 5 — Formulating bi-objective optimisation problem")

    # ── Load inputs ───────────────────────────────────────────────────────
    stop_events   = pd.read_parquet(GTFS_DIR  / "stop_events.parquet")
    hei_df        = pd.read_parquet(GTFS_DIR  / "transfer_hei.parquet")
    wbgt_mean_arr = np.load(WEATHER_DIR / "wbgt_lookup.npy")

    log.info("Loaded %d stop events, %d transfer pairs, %d WBGT hours",
             len(stop_events), len(hei_df), len(wbgt_mean_arr))

    # ── Build all precomputed arrays ──────────────────────────────────────
    trip_index      = build_trip_index(stop_events)
    transfer_arrays = build_transfer_arrays(hei_df, trip_index)
    headway_arrays  = build_headway_arrays(stop_events, trip_index)
    hours_ext, wbgt_ext = build_wbgt_interp_arrays(wbgt_mean_arr)

    # ── Instantiate problem ───────────────────────────────────────────────
    problem = TransferSyncProblem(
        transfer_arrays = transfer_arrays,
        headway_arrays  = headway_arrays,
        hours_ext       = hours_ext,
        wbgt_ext        = wbgt_ext,
        n_trips         = len(trip_index),
        delta_max       = DELTA_MAX_SEC,
    )

    # ── Verify baseline ───────────────────────────────────────────────────
    verify_baseline(problem, hei_df)

    # ── Print diagnostics ─────────────────────────────────────────────────
    print_problem_diagnostics(problem, headway_arrays["hw_df"], trip_index)

    # ── Benchmark evaluation speed ────────────────────────────────────────
    import time
    rng   = np.random.default_rng(42)
    X_test = rng.uniform(-DELTA_MAX_SEC, DELTA_MAX_SEC,
                          size=(200, problem.n_var))
    t0 = time.perf_counter()
    out = {}
    problem._evaluate(X_test, out)
    elapsed = time.perf_counter() - t0
    log.info("")
    log.info("Evaluation benchmark:")
    log.info("  200 solutions in %.3f s  (%.1f ms / solution)",
             elapsed, 1000 * elapsed / 200)
    log.info("  Projected time for 500 generations × pop=200: %.1f s",
             500 * elapsed)

    # ── Save headway DataFrame for Step 6/7 diagnostics ──────────────────
    log.info("Saving problem arrays …")
    headway_arrays["hw_df"].to_parquet(PROB_DIR / "headway_pairs.parquet",
                                       index=False)

    log.info("Step 5 complete.")

    return {
        "problem"    : problem,
        "trip_index" : trip_index,
        "hei_df"     : hei_df,
        "hw_df"      : headway_arrays["hw_df"],
    }


if __name__ == "__main__":
    result = main()
