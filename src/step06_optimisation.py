"""
Step 6 — Greedy Heuristic, NSGA-II, and Pareto Analysis
=========================================================
Implements and compares three optimisation approaches:

  Approach A — Greedy-f1
    Sorts all transfers by descending scheduled wait time and greedily
    shifts trips to eliminate the largest waits first, respecting
    headway constraints.

  Approach B — Greedy-f2
    Sorts all transfers by descending HEI (= WBGT × wait × shade × freq)
    and greedily shifts trips to eliminate the highest heat-exposure
    transfers first, respecting headway constraints.

  Approach C — NSGA-II
    Bi-objective evolutionary optimisation that returns a full Pareto
    front approximation.  Uses pymoo's NSGA-II with SBX crossover,
    polynomial mutation, and CV-based constraint dominance.

Greedy Algorithm
----------------
For each transfer pair k (sorted by priority descending):

  1. Desired shift: Δ = scheduled_wait_k / 2 (split the wait equally
     between feeder → later and connector → earlier).
  2. Maximum feasible positive shift for the feeder trip i:
       cap_pos_i = min over all headway pairs (i→j): 0.2 × hw_{ij} + δ_j − δ_i
                   (also bounded by DELTA_MAX)
  3. Maximum feasible negative shift for the connector trip j:
       cap_neg_j = max over all headway pairs (l→j): −(0.2 × hw_{lj} + δ_l − δ_j)
                   (also bounded by DELTA_MAX)
  4. Apply: δ_i += min(Δ, cap_pos_i),  δ_j −= min(Δ, cap_neg_j)
  5. Accumulate into the running δ vector.

Because the greedy modifies the δ vector incrementally, feasibility is
maintained by construction at each step.

NSGA-II Parameters
------------------
  population     : 200
  generations    : 300
  crossover      : SBX  (η_c = 15, prob = 0.9)
  mutation       : polynomial  (η_m = 20, prob = 1/n_var)
  constraint     : CV-based dominance (pymoo default)
  seed           : 42 (reproducibility)

Outputs
-------
results/solutions/greedy_f1_delta.npy
results/solutions/greedy_f2_delta.npy
results/solutions/nsga2_pareto_X.npy      — decision vectors on Pareto front
results/solutions/nsga2_pareto_F.npy      — objective values  (f1, f2)
results/solutions/nsga2_all_X.npy         — full final population X
results/solutions/nsga2_all_F.npy         — full final population F
results/tables/tab02_solution_comparison.csv
results/tables/tab03_pareto_front.csv
results/tables/tab08_convergence.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step05_formulation import (
    build_trip_index, build_transfer_arrays, build_headway_arrays,
    build_wbgt_interp_arrays, TransferSyncProblem, DELTA_MAX_SEC, HEADWAY_FACTOR
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
GTFS_DIR = ROOT / "data" / "gtfs"
WX_DIR   = ROOT / "data" / "weather"
SOL_DIR  = ROOT / "results" / "solutions"
TAB_DIR  = ROOT / "results" / "tables"
SOL_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── NSGA-II parameters ────────────────────────────────────────────────────────
POP_SIZE  = 200
N_GEN     = 500
SEED      = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint-feasibility checker
# ═══════════════════════════════════════════════════════════════════════════════

class HeadwayFeasibilityChecker:
    """
    Per-trip feasibility checker for the greedy — enforces both
    headway stability and minimum-wait (≥ 60 s) across ALL transfers
    that share a trip.

    Indexing pre-built at construction; per-call work is O(k) where k
    is the number of constraints touching that trip.
    """

    MIN_WAIT_SEC = 60.0

    def __init__(self, prev_idx, next_idx, hw_sec,
                 feeder_idx, conn_idx, wait_sec,
                 n_trips, delta_max):
        self.n          = n_trips
        self.delta_max  = delta_max
        self.hw_sec     = hw_sec.copy()
        self.prev_idx   = prev_idx.copy()
        self.next_idx   = next_idx.copy()
        self.feeder_idx = feeder_idx.copy()
        self.conn_idx   = conn_idx.copy()
        self.wait_sec   = wait_sec.copy()

        # Headway adjacency lists
        self.as_prev = [[] for _ in range(n_trips)]
        self.as_next = [[] for _ in range(n_trips)]
        for k, (p, q) in enumerate(zip(prev_idx, next_idx)):
            self.as_prev[p].append(k)
            self.as_next[q].append(k)

        # Transfer adjacency lists
        self.as_feeder    = [[] for _ in range(n_trips)]
        self.as_connector = [[] for _ in range(n_trips)]
        for k, (f, c) in enumerate(zip(feeder_idx, conn_idx)):
            self.as_feeder[f].append(k)
            self.as_connector[c].append(k)

    def max_pos_shift(self, i: int, delta: np.ndarray) -> float:
        """
        Max additional positive shift for trip i.

        Capped by:
          (A) Headway: for pairs where i=prev, δ[next]−(δ[i]+d) ≥ −0.2×hw
          (B) Min-wait feeder: shifting i later reduces wait for every
              transfer k where i is feeder:
              new_wait_k = w_k + δ[conn_k] − (δ[i]+d) ≥ 60
              → d ≤ w_k + δ[conn_k] − δ[i] − 60  = cur_wait_k − 60
          (C) Box: δ[i]+d ≤ delta_max
        """
        cap = self.delta_max - delta[i]

        # (A) Headway
        for k in self.as_prev[i]:
            q   = self.next_idx[k]
            cap = min(cap, delta[q] - delta[i] + HEADWAY_FACTOR * self.hw_sec[k])

        # (B) Min-wait: i is feeder — shifting i later reduces wait
        for k in self.as_feeder[i]:
            c        = self.conn_idx[k]
            cur_wait = self.wait_sec[k] + delta[c] - delta[i]
            cap      = min(cap, cur_wait - self.MIN_WAIT_SEC)

        return max(0.0, cap)

    def max_neg_shift(self, i: int, delta: np.ndarray) -> float:
        """
        Max additional negative shift for trip i.

        Capped by:
          (A) Headway: for pairs where i=next, (δ[i]−d)−δ[prev] ≥ −0.2×hw
          (B) Min-wait connector: shifting i earlier reduces wait for every
              transfer k where i is connector:
              new_wait_k = w_k + (δ[i]−d) − δ[feeder_k] ≥ 60
              → d ≤ w_k + δ[i] − δ[feeder_k] − 60 = cur_wait_k − 60
          (C) Box: δ[i]−d ≥ −delta_max
        """
        cap = self.delta_max + delta[i]

        # (A) Headway
        for k in self.as_next[i]:
            p   = self.prev_idx[k]
            cap = min(cap, delta[i] - delta[p] + HEADWAY_FACTOR * self.hw_sec[k])

        # (B) Min-wait: i is connector — shifting i earlier reduces wait
        for k in self.as_connector[i]:
            f        = self.feeder_idx[k]
            cur_wait = self.wait_sec[k] + delta[i] - delta[f]
            cap      = min(cap, cur_wait - self.MIN_WAIT_SEC)

        return max(0.0, cap)


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_greedy(problem: TransferSyncProblem,
                 hei_df: pd.DataFrame,
                 sort_key: str,
                 label: str) -> np.ndarray:
    """
    Greedy slip-shifting heuristic.

    Parameters
    ----------
    sort_key : 'wait_min'   → Greedy-f1 (sort by scheduled wait, descending)
               'hei_mean'   → Greedy-f2 (sort by HEI, descending)
    label    : human-readable name for logging

    Returns
    -------
    delta : (n_var,) float — solution shift vector in seconds
    """
    log.info("Running %s greedy …", label)
    t0 = time.perf_counter()

    delta = np.zeros(problem.n_var, dtype=float)
    chk   = HeadwayFeasibilityChecker(
        problem.prev_idx, problem.next_idx, problem.hw_sec,
        problem.feeder_idx, problem.conn_idx, problem.wait_sec,
        problem.n_var, DELTA_MAX_SEC
    )

    # Sort transfer pairs by priority (descending)
    order = hei_df[sort_key].to_numpy().argsort()[::-1]

    feeder_idx = problem.feeder_idx
    conn_idx   = problem.conn_idx
    wait_sec   = problem.wait_sec

    n_improved = 0
    total_wait_saved = 0.0

    for rank, k in enumerate(order):
        fi = feeder_idx[k]
        ci = conn_idx[k]

        # Current wait for this transfer
        cur_wait = wait_sec[k] + delta[ci] - delta[fi]
        if cur_wait <= 60.0:          # already at minimum — skip
            continue

        # Target: reduce wait to exactly 60 s (min boarding time).
        # Never push below 60 s — this would create a missed connection.
        target_reduction = cur_wait - 60.0

        # Max shifts respecting headway constraints
        cap_f = chk.max_pos_shift(fi, delta)
        cap_c = chk.max_neg_shift(ci, delta)

        # Split reduction: feeder gets half, connector gets remainder,
        # each capped by feasibility. This guarantees new_wait ≥ 60 s.
        half    = target_reduction / 2.0
        shift_f = min(half, cap_f)
        shift_c = min(target_reduction - shift_f, cap_c)

        delta[fi] += shift_f
        delta[ci] -= shift_c

        wait_saved = shift_f + shift_c
        if wait_saved > 0:
            n_improved   += 1
            total_wait_saved += wait_saved

    elapsed = time.perf_counter() - t0
    log.info("  %s greedy: %.3f s  |  %d transfers improved  |  %.1f s wait saved",
             label, elapsed, n_improved, total_wait_saved)

    # Verify feasibility
    res = problem.evaluate_single(delta)
    log.info("  %s: f1=%.2f min  f2=%.2f  feasible=%s  n_viol=%d  missed=%d",
             label, res["f1"], res["f2"],
             res["is_feasible"], res["n_violated_constraints"],
             res["n_missed_connections"])

    assert res["is_feasible"], \
        f"Greedy-{label} produced an infeasible solution " \
        f"({res['n_violated_constraints']} headway violations)"

    return delta


# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II progress callback
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressCallback(Callback):
    """Log Pareto front progress every 50 generations."""

    def __init__(self):
        super().__init__()
        self.t0 = time.perf_counter()
        self.gen_data = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 25 == 0 or gen == 1:
            F    = algorithm.pop.get("F")
            feas = (algorithm.pop.get("CV") <= 0).flatten()
            n_feas   = feas.sum()
            f1_feas  = F[feas, 0].min() if n_feas > 0 else float("nan")
            f2_feas  = F[feas, 1].min() if n_feas > 0 else float("nan")
            elapsed  = time.perf_counter() - self.t0
            log.info("  Gen %4d | %3d feasible | best f1=%.1f | best f2=%.1f | %.1fs",
                     gen, n_feas, f1_feas, f2_feas, elapsed)
            self.gen_data.append({
                "gen": gen, "n_feas": int(n_feas),
                "f1_best": f1_feas, "f2_best": f2_feas, "elapsed": elapsed,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_nsga2(problem: TransferSyncProblem) -> dict:
    """
    Run NSGA-II and return the Pareto front and full final population.

    Returns
    -------
    dict with keys:
      pareto_X   : (n_pareto, n_var)  decision vectors
      pareto_F   : (n_pareto, 2)      objective values [f1, f2]
      all_X      : (pop_size, n_var)  full final population
      all_F      : (pop_size, 2)      full final population objectives
      result     : pymoo Result object
      callback   : ProgressCallback with per-generation data
    """
    log.info("Running NSGA-II  (pop=%d, gen=%d, seed=%d) …",
             POP_SIZE, N_GEN, SEED)

    callback    = ProgressCallback()
    algorithm   = NSGA2(
        pop_size    = POP_SIZE,
        sampling    = FloatRandomSampling(),
        crossover   = SBX(eta=15, prob=0.9),
        mutation    = PM(eta=20, prob=1.0 / problem.n_var),
        eliminate_duplicates = True,
    )
    termination = get_termination("n_gen", N_GEN)

    t0 = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed     = SEED,
        callback = callback,
        verbose  = False,
    )
    elapsed = time.perf_counter() - t0
    log.info("NSGA-II completed in %.1f s", elapsed)

    # ── Extract Pareto front (feasible solutions only) ────────────────────
    all_X = result.pop.get("X")
    all_F = result.pop.get("F")
    all_G = result.pop.get("G")
    cv    = np.maximum(all_G, 0).sum(axis=1)   # constraint violation sum

    # Pareto front = result.opt (pymoo returns the non-dominated set)
    opt_X = result.opt.get("X")
    opt_F = result.opt.get("F")
    opt_G = result.opt.get("G")
    opt_cv = np.maximum(opt_G, 0).sum(axis=1)

    # Keep only feasible Pareto solutions
    feas_mask = (opt_cv <= 0)
    pareto_X  = opt_X[feas_mask]
    pareto_F  = opt_F[feas_mask]

    log.info("Pareto front: %d solutions  (%d feasible / %d total in opt set)",
             len(pareto_F), feas_mask.sum(), len(opt_F))
    log.info("  f1 range: %.2f – %.2f min", pareto_F[:,0].min(), pareto_F[:,0].max())
    log.info("  f2 range: %.2f – %.2f", pareto_F[:,1].min(), pareto_F[:,1].max())

    return {
        "pareto_X" : pareto_X,
        "pareto_F" : pareto_F,
        "all_X"    : all_X,
        "all_F"    : all_F,
        "result"   : result,
        "callback" : callback,
        "elapsed"  : elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Hypervolume indicator
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hypervolume(pareto_F: np.ndarray,
                         baseline_f1: float,
                         baseline_f2: float) -> float:
    """
    Compute the hypervolume indicator of the Pareto front relative to
    the baseline (δ=0) as the reference point.

    Uses the WFG algorithm via pymoo.

    The reference point is set to (baseline_f1 × 1.01, baseline_f2 × 1.01)
    to give a small margin above the baseline — standard practice ensures
    all Pareto solutions are dominated by the reference point.
    """
    from pymoo.indicators.hv import HV

    ref_point = np.array([baseline_f1 * 1.01, baseline_f2 * 1.01])

    # Normalise to [0,1] range for numerical stability
    f1_range = baseline_f1 * 1.01
    f2_range = baseline_f2 * 1.01
    F_norm   = pareto_F / np.array([f1_range, f2_range])
    ref_norm = np.array([1.01, 1.01])

    ind = HV(ref_point=ref_norm)
    hv  = ind(F_norm)
    log.info("Hypervolume indicator (normalised): %.6f", hv)
    return float(hv)


# ═══════════════════════════════════════════════════════════════════════════════
# Solution comparison and analytics
# ═══════════════════════════════════════════════════════════════════════════════

def _solution_metrics(name: str,
                       delta: np.ndarray,
                       problem: TransferSyncProblem,
                       hei_df: pd.DataFrame) -> dict:
    """Compute rich metrics for a single solution."""
    res = problem.evaluate_single(delta)

    new_wait_sec = res["new_wait_sec"]
    new_wait_min = np.maximum(0, new_wait_sec) / 60.0

    shifted_arr = (problem.arr_sec + delta[problem.feeder_idx]) % 86400.0
    wbgt_new    = np.interp(shifted_arr / 3600., problem.hours_ext, problem.wbgt_ext)
    new_hei     = wbgt_new * new_wait_min * problem.shade * problem.freq

    base_wait_min = problem.wait_sec / 60.0
    base_hei      = np.interp((problem.arr_sec % 86400)/3600,
                               problem.hours_ext, problem.wbgt_ext) \
                    * base_wait_min * problem.shade * problem.freq

    return {
        "name"              : name,
        "f1"                : res["f1"],
        "f2"                : res["f2"],
        "f1_reduction_pct"  : 100 * (1 - res["f1"] / base_wait_min.sum()),
        "f2_reduction_pct"  : 100 * (1 - res["f2"] / base_hei.sum()),
        "is_feasible"       : res["is_feasible"],
        "n_viol_constraints": res["n_violated_constraints"],
        "n_missed_conn"     : res["n_missed_connections"],
        "mean_wait_min"     : new_wait_min.mean(),
        "max_wait_min"      : new_wait_min.max(),
        "hei_mean_per_xfer" : new_hei.mean(),
        "delta_min"         : delta.min(),
        "delta_max_val"     : delta.max(),
        "delta_mean"        : delta.mean(),
        "n_trips_shifted"   : (np.abs(delta) > 0.5).sum(),
    }


def analyse_pareto_front(pareto_F: np.ndarray,
                          pareto_X: np.ndarray,
                          baseline_f1: float,
                          baseline_f2: float) -> pd.DataFrame:
    """
    Annotate the Pareto front with reduction percentages and identify
    key solutions: minimum-f1, minimum-f2, and the balanced knee point
    (minimum distance from the normalised ideal point).
    """
    df = pd.DataFrame({
        "f1_min"        : pareto_F[:, 0],
        "f2_hei"        : pareto_F[:, 1],
        "f1_red_pct"    : 100 * (1 - pareto_F[:, 0] / baseline_f1),
        "f2_red_pct"    : 100 * (1 - pareto_F[:, 1] / baseline_f2),
    })

    # Knee point: minimum Euclidean distance to ideal in normalised space
    f1_n = pareto_F[:, 0] / baseline_f1
    f2_n = pareto_F[:, 1] / baseline_f2
    dist = np.sqrt(f1_n**2 + f2_n**2)
    df["knee_score"]   = dist
    df["is_knee"]      = (dist == dist.min())
    df["is_min_f1"]    = (pareto_F[:, 0] == pareto_F[:, 0].min())
    df["is_min_f2"]    = (pareto_F[:, 1] == pareto_F[:, 1].min())
    df["pareto_rank"]  = np.arange(len(df))

    # Sort by f1 for readable output
    df = df.sort_values("f1_min").reset_index(drop=True)
    return df


def compare_solutions(problem: TransferSyncProblem,
                       hei_df: pd.DataFrame,
                       delta_greedy_f1: np.ndarray,
                       delta_greedy_f2: np.ndarray,
                       pareto_F: np.ndarray,
                       pareto_X: np.ndarray) -> pd.DataFrame:
    """
    Build a comparison table: baseline vs greedy-f1 vs greedy-f2 vs
    Pareto min-f1 vs Pareto min-f2 vs Pareto knee.
    """
    baseline_f1 = problem.wait_sec.sum() / 60.0
    baseline_hei = (np.interp((problem.arr_sec % 86400)/3600,
                              problem.hours_ext, problem.wbgt_ext)
                    * (problem.wait_sec/60) * problem.shade * problem.freq).sum()

    rows = []
    rows.append({
        "name": "Baseline (δ=0)", "f1": baseline_f1,
        "f2": baseline_hei, "f1_red_pct": 0.0, "f2_red_pct": 0.0,
        "is_feasible": True, "n_missed_conn": 0, "n_trips_shifted": 0,
    })

    for name, delta in [("Greedy-f1", delta_greedy_f1),
                         ("Greedy-f2", delta_greedy_f2)]:
        m = _solution_metrics(name, delta, problem, hei_df)
        m["f1_red_pct"] = 100 * (1 - m["f1"] / baseline_f1)
        m["f2_red_pct"] = 100 * (1 - m["f2"] / baseline_hei)
        rows.append(m)

    # Pareto extremes and knee
    pf_sorted = pareto_F.copy()
    px_sorted = pareto_X.copy()
    order = pf_sorted[:, 0].argsort()
    pf_sorted = pf_sorted[order]
    px_sorted = px_sorted[order]

    f1_n = pf_sorted[:, 0] / baseline_f1
    f2_n = pf_sorted[:, 1] / baseline_hei
    dist = np.sqrt(f1_n**2 + f2_n**2)

    specials = {
        "NSGA-II  min-f1"  : np.argmin(pf_sorted[:, 0]),
        "NSGA-II  min-f2"  : np.argmin(pf_sorted[:, 1]),
        "NSGA-II  knee"    : np.argmin(dist),
    }
    for name, idx in specials.items():
        m = _solution_metrics(name, px_sorted[idx], problem, hei_df)
        m["f1_red_pct"] = 100 * (1 - m["f1"] / baseline_f1)
        m["f2_red_pct"] = 100 * (1 - m["f2"] / baseline_hei)
        rows.append(m)

    comp = pd.DataFrame(rows)
    return comp, baseline_f1, baseline_hei


def log_comparison(comp: pd.DataFrame) -> None:
    log.info("=" * 80)
    log.info("SOLUTION COMPARISON")
    log.info("=" * 80)
    log.info("%-22s  %10s  %8s  %10s  %8s  %8s  %8s",
             "Solution", "f1(min)", "Δf1(%)", "f2(°C·min)", "Δf2(%)",
             "feasible", "missed")
    log.info("-" * 80)
    for _, row in comp.iterrows():
        log.info("%-22s  %10.1f  %8.2f  %10.1f  %8.2f  %8s  %8d",
                 row["name"], row["f1"], row.get("f1_red_pct", 0),
                 row["f2"], row.get("f2_red_pct", 0),
                 str(row.get("is_feasible", True)),
                 int(row.get("n_missed_conn", 0)))
    log.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-transfer breakdown helper
# ═══════════════════════════════════════════════════════════════════════════════

def build_transfer_breakdown(delta: np.ndarray,
                              problem: TransferSyncProblem,
                              hei_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-transfer table showing before/after wait and HEI."""
    res = problem.evaluate_single(delta)
    new_wait_sec = res["new_wait_sec"]
    new_wait_min = np.maximum(0, new_wait_sec) / 60.0

    shifted_arr = (problem.arr_sec + delta[problem.feeder_idx]) % 86400.0
    wbgt_new    = np.interp(shifted_arr/3600, problem.hours_ext, problem.wbgt_ext)
    hei_new     = wbgt_new * new_wait_min * problem.shade * problem.freq

    df = hei_df[["transfer_id", "stop_id", "stop_name",
                  "feeder_route_short", "connector_route_short",
                  "transfer_clock_hour", "wait_min", "hei_weighted_mean",
                  "shade_factor", "freq_proxy"]].copy()

    df["new_wait_min"] = new_wait_min
    df["new_hei"]      = hei_new
    df["wait_saved_min"] = df["wait_min"] - df["new_wait_min"]
    df["hei_saved"]      = df["hei_weighted_mean"] - df["new_hei"]
    df["delta_feeder"]   = delta[problem.feeder_idx]
    df["delta_connector"]= delta[problem.conn_idx]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    log.info("Step 6 — Greedy Heuristic, NSGA-II, and Pareto Analysis")

    # ── Load and reconstruct problem ──────────────────────────────────────
    log.info("Loading data and reconstructing problem …")
    se     = pd.read_parquet(GTFS_DIR / "stop_events.parquet")
    hei_df = pd.read_parquet(GTFS_DIR / "transfer_hei.parquet")
    wbgt   = np.load(WX_DIR / "wbgt_lookup.npy")

    ti  = build_trip_index(se)
    ta  = build_transfer_arrays(hei_df, ti)
    ha  = build_headway_arrays(se, ti)
    he, we = build_wbgt_interp_arrays(wbgt)
    prob   = TransferSyncProblem(ta, ha, he, we, len(ti), DELTA_MAX_SEC)

    baseline_f1 = prob.wait_sec.sum() / 60.0
    baseline_f2 = (np.interp((prob.arr_sec % 86400)/3600, he, we)
                   * (prob.wait_sec/60) * prob.shade * prob.freq).sum()
    log.info("Baseline: f1=%.2f min  f2=%.2f °C·min", baseline_f1, baseline_f2)

    # ── Approach A: Greedy-f1 ─────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Approach A — Greedy-f1")
    delta_gf1 = solve_greedy(prob, hei_df, sort_key="wait_min", label="Greedy-f1")
    np.save(SOL_DIR / "greedy_f1_delta.npy", delta_gf1)

    # ── Approach B: Greedy-f2 ─────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Approach B — Greedy-f2")
    delta_gf2 = solve_greedy(prob, hei_df, sort_key="hei_mean", label="Greedy-f2")
    np.save(SOL_DIR / "greedy_f2_delta.npy", delta_gf2)

    # ── Approach C: NSGA-II ───────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Approach C — NSGA-II")
    nsga2_result = solve_nsga2(prob)
    pareto_X = nsga2_result["pareto_X"]
    pareto_F = nsga2_result["pareto_F"]

    np.save(SOL_DIR / "nsga2_pareto_X.npy", pareto_X)
    np.save(SOL_DIR / "nsga2_pareto_F.npy", pareto_F)
    np.save(SOL_DIR / "nsga2_all_X.npy",    nsga2_result["all_X"])
    np.save(SOL_DIR / "nsga2_all_F.npy",    nsga2_result["all_F"])

    # ── Hypervolume ───────────────────────────────────────────────────────
    log.info("─" * 60)
    hv = compute_hypervolume(pareto_F, baseline_f1, baseline_f2)

    # ── Solution comparison ───────────────────────────────────────────────
    log.info("─" * 60)
    comp, bf1, bf2 = compare_solutions(
        prob, hei_df, delta_gf1, delta_gf2, pareto_F, pareto_X)
    log_comparison(comp)

    # ── Pareto front analysis ─────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Pareto front detail (sorted by f1):")
    pareto_ann = analyse_pareto_front(pareto_F, pareto_X, bf1, bf2)
    log.info("  %-6s  %10s  %8s  %12s  %8s  %6s  %6s  %6s",
             "rank", "f1(min)", "Δf1(%)", "f2(°C·min)", "Δf2(%)",
             "knee", "min_f1", "min_f2")
    for _, row in pareto_ann.iterrows():
        flags = ("K" if row["is_knee"]   else " ") + \
                ("1" if row["is_min_f1"] else " ") + \
                ("2" if row["is_min_f2"] else " ")
        log.info("  %-6d  %10.1f  %8.2f  %12.1f  %8.2f  %6s",
                 row["pareto_rank"], row["f1_min"], row["f1_red_pct"],
                 row["f2_hei"], row["f2_red_pct"], flags.strip() or "-")

    # ── Greedy-vs-Pareto gap analysis ─────────────────────────────────────
    log.info("─" * 60)
    log.info("Greedy proximity to Pareto front:")
    for gname, gf1, gf2 in [
        ("Greedy-f1", comp.loc[comp.name=="Greedy-f1","f1"].iloc[0],
                      comp.loc[comp.name=="Greedy-f1","f2"].iloc[0]),
        ("Greedy-f2", comp.loc[comp.name=="Greedy-f2","f1"].iloc[0],
                      comp.loc[comp.name=="Greedy-f2","f2"].iloc[0]),
    ]:
        # Distance to nearest Pareto point (normalised)
        f1_n = (pareto_F[:,0] - gf1) / bf1
        f2_n = (pareto_F[:,1] - gf2) / bf2
        dist = np.sqrt(f1_n**2 + f2_n**2)
        nearest_idx = dist.argmin()
        log.info("  %s  (f1=%.1f, f2=%.1f)", gname, gf1, gf2)
        log.info("    Nearest Pareto point: (%.1f, %.1f)  dist=%.4f (normalised)",
                 pareto_F[nearest_idx,0], pareto_F[nearest_idx,1], dist.min())
        log.info("    Δf1 to nearest: %+.1f min   Δf2 to nearest: %+.1f",
                 gf1 - pareto_F[nearest_idx,0], gf2 - pareto_F[nearest_idx,1])

    # ── Save tables ───────────────────────────────────────────────────────
    # Write to both canonical (tab0N) and legacy names so downstream steps work
    comp.to_csv(TAB_DIR / "tab02_solution_comparison.csv", index=False)
    pareto_ann.to_csv(TAB_DIR / "tab03_pareto_front.csv", index=False)

    # Save convergence data under canonical name
    pd.DataFrame(nsga2_result["callback"].gen_data).to_csv(
        TAB_DIR / "tab08_convergence.csv", index=False)

    log.info("All outputs saved.")
    log.info("Step 6 complete.")

    return {
        "problem"       : prob,
        "delta_gf1"     : delta_gf1,
        "delta_gf2"     : delta_gf2,
        "pareto_X"      : pareto_X,
        "pareto_F"      : pareto_F,
        "comp"          : comp,
        "pareto_ann"    : pareto_ann,
        "hypervolume"   : hv,
        "baseline_f1"   : bf1,
        "baseline_f2"   : bf2,
    }


if __name__ == "__main__":
    result = main()
