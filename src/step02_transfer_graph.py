"""
Step 2 — Build the Transfer Graph
===================================
Enumerates every realistic transfer opportunity in the weekday summer
schedule and constructs two complementary representations:

  1. transfer_pairs   — a tidy DataFrame, one row per (feeder trip,
                        connector trip, stop) transfer event. This is
                        the primary data structure for Steps 4–6.

  2. G               — a directed NetworkX MultiDiGraph whose nodes are
                        (stop_id, trip_id) tuples and whose edges are
                        the same transfer opportunities. Useful for
                        graph-level analysis and visualisation.

Transfer definition
-------------------
A transfer is valid when:
  • feeder and connector serve the **same stop**
  • they belong to **different routes** (cross-route transfer)
    OR the same route with **opposite directions** (bidirectional loop)
  • MIN_TRANSFER_SEC ≤ connector_departure_sec − feeder_arrival_sec
                      ≤ MAX_TRANSFER_SEC

We use feeder ARRIVAL and connector DEPARTURE because that is the
passenger experience: the feeder drops them off, then they wait until
the connector leaves.

Parameters
----------
MIN_TRANSFER_SEC : 60   s  (1 min  — minimum physical boarding time)
MAX_TRANSFER_SEC : 900  s  (15 min — maximum practical wait)

Outputs
-------
data/gtfs/transfer_pairs.parquet
data/gtfs/transfer_graph.gpickle  (NetworkX graph)
results/tables/transfer_summary.csv
"""

import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
GTFS_DIR = ROOT / "data" / "gtfs"
RES_DIR  = ROOT / "results" / "tables"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── transfer parameters ───────────────────────────────────────────────────────
MIN_TRANSFER_SEC = 60    # 1 minute
MAX_TRANSFER_SEC = 900   # 15 minutes


# ═══════════════════════════════════════════════════════════════════════════════
# Core enumeration
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_transfers(stop_events: pd.DataFrame,
                        filter_same_block: bool = True) -> pd.DataFrame:
    """
    For every stop served by ≥2 routes, find all (feeder, connector) trip
    pairs whose scheduled wait time falls within [MIN, MAX] seconds.

    Algorithm
    ---------
    For each qualifying stop:
      1. Extract the sub-table of events, sorted by departure_sec.
      2. For every ordered pair (i, j) with i ≠ j:
            wait = events_j.departure_sec − events_i.arrival_sec
         Keep the pair if:
            • MIN_TRANSFER_SEC ≤ wait ≤ MAX_TRANSFER_SEC
            • events_i.route_id ≠ events_j.route_id
              OR (same route but different direction_id)
      3. Record full metadata for both legs.

    This is O(k²) per stop where k = events at that stop.
    With k ≤ 100 and 206 qualifying stops the total is fast (<1 s).

    Returns
    -------
    DataFrame with one row per valid transfer pair.
    """
    log.info("Enumerating transfer opportunities …")
    log.info("  Window: [%d, %d] seconds  ([%.0f, %.0f] min)",
             MIN_TRANSFER_SEC, MAX_TRANSFER_SEC,
             MIN_TRANSFER_SEC / 60, MAX_TRANSFER_SEC / 60)

    # Identify stops served by ≥2 distinct routes
    route_counts = stop_events.groupby("stop_id")["route_id"].nunique()
    candidate_stops = route_counts[route_counts >= 2].index
    log.info("  Candidate stops (≥2 routes): %d", len(candidate_stops))

    sub = stop_events[stop_events["stop_id"].isin(candidate_stops)].copy()

    # Columns we need on both legs of every transfer
    keep_cols = [
        "trip_id", "route_id", "route_short_name", "route_long_name",
        "direction_id", "trip_headsign", "block_id",
        "stop_id", "stop_name", "stop_lat", "stop_lon",
        "is_sheltered", "shade_factor",
        "arrival_sec", "departure_sec", "clock_hour", "minute_of_day",
    ]
    sub = sub[keep_cols].copy()

    if filter_same_block:
        log.info("  Same-block filtering: enabled (removes intra-block "
                 "continuations that are not real passenger transfers)")

    records = []

    for stop_id, group in sub.groupby("stop_id"):
        group = group.sort_values("departure_sec").reset_index(drop=True)
        n = len(group)

        # Pre-extract arrays for speed
        arr_arr  = group["arrival_sec"].values
        dep_arr  = group["departure_sec"].values
        rte_arr  = group["route_id"].values
        dir_arr  = group["direction_id"].values
        trip_arr = group["trip_id"].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Passenger arrives on trip i, boards trip j
                wait = dep_arr[j] - arr_arr[i]

                if wait < MIN_TRANSFER_SEC or wait > MAX_TRANSFER_SEC:
                    continue

                # Cross-route transfer OR same-route opposite direction
                same_route = (rte_arr[i] == rte_arr[j])
                same_dir   = (dir_arr[i] == dir_arr[j])

                if same_route and same_dir:
                    continue

                # Exclude same-block continuations: a bus that departs stop X
                # on block B and later returns to stop X on block B is the same
                # vehicle looping — no passenger transfer is needed.
                if filter_same_block:
                    block_i = group.iloc[i]["block_id"]
                    block_j = group.iloc[j]["block_id"]
                    if pd.notna(block_i) and pd.notna(block_j) and block_i == block_j:
                        continue

                r = group.iloc[i]
                c = group.iloc[j]

                records.append({
                    # ── Identifiers ─────────────────────────────────────────
                    "stop_id"                : int(stop_id),
                    "stop_name"              : r["stop_name"],
                    "stop_lat"               : r["stop_lat"],
                    "stop_lon"               : r["stop_lon"],
                    "is_sheltered"           : bool(r["is_sheltered"]),
                    "shade_factor"           : float(r["shade_factor"]),
                    # ── Feeder leg ──────────────────────────────────────────
                    "feeder_trip_id"         : int(r["trip_id"]),
                    "feeder_route_id"        : r["route_id"],
                    "feeder_route_short"     : r["route_short_name"],
                    "feeder_route_long"      : r["route_long_name"],
                    "feeder_direction"       : int(r["direction_id"]),
                    "feeder_headsign"        : r["trip_headsign"],
                    "feeder_arrival_sec"     : float(r["arrival_sec"]),
                    # ── Connector leg ───────────────────────────────────────
                    "connector_trip_id"      : int(c["trip_id"]),
                    "connector_route_id"     : c["route_id"],
                    "connector_route_short"  : c["route_short_name"],
                    "connector_route_long"   : c["route_long_name"],
                    "connector_direction"    : int(c["direction_id"]),
                    "connector_headsign"     : c["trip_headsign"],
                    "connector_departure_sec": float(c["departure_sec"]),
                    # ── Transfer metrics ────────────────────────────────────
                    "wait_sec"               : float(wait),
                    "wait_min"               : float(wait) / 60.0,
                    # Clock hour of transfer (feeder arrival, mod 24 for display)
                    "transfer_clock_hour"    : int(r["clock_hour"]),
                    "transfer_minute_of_day" : int(r["minute_of_day"]),
                    # Transfer type flag
                    "is_cross_route"         : not same_route,
                    "is_direction_reversal"  : (same_route and not same_dir),
                })

    pairs = pd.DataFrame(records)

    assert not pairs.empty, \
        "No transfer pairs found — check TRANSFER_WINDOW_SEC or input data"

    # Add a stable unique transfer ID for downstream joins
    pairs = pairs.reset_index(drop=True)
    pairs.insert(0, "transfer_id", pairs.index.astype(int))

    log.info("  Transfer pairs found: %d", len(pairs))
    log.info("    Cross-route:         %d  (%.1f%%)",
             pairs["is_cross_route"].sum(),
             100 * pairs["is_cross_route"].mean())
    log.info("    Direction reversal:  %d  (%.1f%%)",
             pairs["is_direction_reversal"].sum(),
             100 * pairs["is_direction_reversal"].mean())

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# NetworkX graph construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_networkx_graph(pairs: pd.DataFrame,
                         stop_events: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Construct a directed multigraph from the transfer pairs.

    Nodes  : (stop_id, trip_id)  — a bus visit to a stop
    Edges  : feeder_node → connector_node, weighted by wait_sec
    """
    log.info("Building NetworkX MultiDiGraph …")

    G = nx.MultiDiGraph()

    # ── Node attribute lookup: one row per (stop_id, trip_id) ─────────────
    node_meta = (
        stop_events[["stop_id", "trip_id", "route_id", "route_short_name",
                     "stop_name", "stop_lat", "stop_lon",
                     "is_sheltered", "shade_factor", "clock_hour"]]
        .drop_duplicates(subset=["stop_id", "trip_id"])
        .set_index(["stop_id", "trip_id"])
    )

    # ── Add nodes ─────────────────────────────────────────────────────────
    feeder_nodes    = pairs[["stop_id", "feeder_trip_id",
                              "feeder_route_short", "stop_name",
                              "stop_lat", "stop_lon",
                              "is_sheltered", "shade_factor"]].copy()
    feeder_nodes.columns = ["stop_id", "trip_id", "route_short_name",
                             "stop_name", "stop_lat", "stop_lon",
                             "is_sheltered", "shade_factor"]

    connector_nodes = pairs[["stop_id", "connector_trip_id",
                               "connector_route_short", "stop_name",
                               "stop_lat", "stop_lon",
                               "is_sheltered", "shade_factor"]].copy()
    connector_nodes.columns = feeder_nodes.columns

    all_nodes = pd.concat([feeder_nodes, connector_nodes]).drop_duplicates(
        subset=["stop_id", "trip_id"])

    for _, row in all_nodes.iterrows():
        nid = (int(row["stop_id"]), int(row["trip_id"]))
        G.add_node(nid,
                   stop_id      = int(row["stop_id"]),
                   trip_id      = int(row["trip_id"]),
                   route        = row["route_short_name"],
                   stop_name    = row["stop_name"],
                   lat          = float(row["stop_lat"]),
                   lon          = float(row["stop_lon"]),
                   is_sheltered = bool(row["is_sheltered"]),
                   shade_factor = float(row["shade_factor"]))

    # ── Add edges ─────────────────────────────────────────────────────────
    for _, row in pairs.iterrows():
        u = (int(row["stop_id"]), int(row["feeder_trip_id"]))
        v = (int(row["stop_id"]), int(row["connector_trip_id"]))
        G.add_edge(u, v,
                   transfer_id     = int(row["transfer_id"]),
                   stop_id         = int(row["stop_id"]),
                   feeder_route    = row["feeder_route_short"],
                   connector_route = row["connector_route_short"],
                   wait_sec        = float(row["wait_sec"]),
                   wait_min        = float(row["wait_min"]),
                   is_cross_route  = bool(row["is_cross_route"]),
                   transfer_hour   = int(row["transfer_clock_hour"]))

    log.info("  Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a human-readable summary table of the transfer landscape.
    Returns a multi-section DataFrame written to CSV for the paper.
    """
    log.info("Computing summary statistics …")

    # ── Wait-time distribution ────────────────────────────────────────────
    log.info("── Wait-time distribution (minutes) ──")
    for label, val in [
        ("min",   pairs["wait_min"].min()),
        ("p25",   pairs["wait_min"].quantile(0.25)),
        ("p50",   pairs["wait_min"].median()),
        ("p75",   pairs["wait_min"].quantile(0.75)),
        ("p90",   pairs["wait_min"].quantile(0.90)),
        ("max",   pairs["wait_min"].max()),
        ("mean",  pairs["wait_min"].mean()),
        ("stdev", pairs["wait_min"].std()),
    ]:
        log.info("  %-6s  %.2f min", label, val)

    # ── Busiest transfer stops ────────────────────────────────────────────
    log.info("── Top 15 busiest transfer stops ──")
    stop_counts = (
        pairs.groupby(["stop_id", "stop_name", "is_sheltered"])
        .size()
        .reset_index(name="n_transfers")
        .sort_values("n_transfers", ascending=False)
        .head(15)
    )
    for _, row in stop_counts.iterrows():
        shade = "S" if row["is_sheltered"] else " "
        log.info("  [%s] stop %5d  %-35s  %4d transfers",
                 shade, row["stop_id"], row["stop_name"], row["n_transfers"])

    # ── Most common route-pair combinations ──────────────────────────────
    log.info("── Top 15 route-pair combinations ──")
    pair_counts = (
        pairs.groupby(["feeder_route_short", "connector_route_short"])
        .size()
        .reset_index(name="n_transfers")
        .sort_values("n_transfers", ascending=False)
        .head(15)
    )
    for _, row in pair_counts.iterrows():
        log.info("  %-5s → %-5s  %4d transfers",
                 row["feeder_route_short"], row["connector_route_short"],
                 row["n_transfers"])

    # ── Hourly distribution ───────────────────────────────────────────────
    log.info("── Transfer count by clock hour ──")
    hour_counts = pairs["transfer_clock_hour"].value_counts().sort_index()
    for hour, count in hour_counts.items():
        bar = "█" * (count // 20)
        log.info("  %02dh  %4d  %s", hour, count, bar)

    # ── Shade breakdown ───────────────────────────────────────────────────
    n_sheltered = pairs["is_sheltered"].sum()
    n_exposed   = (~pairs["is_sheltered"]).sum()
    log.info("── Shade breakdown ──")
    log.info("  Sheltered: %5d  (%.1f%%)", n_sheltered, 100 * n_sheltered / len(pairs))
    log.info("  Exposed:   %5d  (%.1f%%)", n_exposed,   100 * n_exposed   / len(pairs))
    log.info("  Mean wait (sheltered): %.2f min",
             pairs[pairs["is_sheltered"]]["wait_min"].mean() if n_sheltered > 0 else float("nan"))
    log.info("  Mean wait (exposed):   %.2f min",
             pairs[~pairs["is_sheltered"]]["wait_min"].mean())

    return stop_counts, pair_counts


def validate_transfers(pairs: pd.DataFrame) -> None:
    """Defensive checks on the transfer pair table."""
    log.info("Validating transfer pairs …")

    assert (pairs["wait_sec"] >= MIN_TRANSFER_SEC).all(), \
        "Found transfers below minimum transfer time"
    assert (pairs["wait_sec"] <= MAX_TRANSFER_SEC).all(), \
        "Found transfers exceeding maximum transfer time"
    assert (pairs["feeder_trip_id"] != pairs["connector_trip_id"]).all(), \
        "Found self-transfers (same trip on both legs)"
    assert not pairs["transfer_id"].duplicated().any(), \
        "Duplicate transfer_ids found"

    # No same-route + same-direction transfers
    bad = pairs[
        (pairs["feeder_route_id"] == pairs["connector_route_id"]) &
        (pairs["feeder_direction"] == pairs["connector_direction"])
    ]
    assert len(bad) == 0, f"Found {len(bad)} same-route same-direction transfers"

    log.info("  All validation checks passed ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    log.info("Step 2 — Building transfer graph")

    # ── Load Step 1 output ────────────────────────────────────────────────
    stop_events = pd.read_parquet(GTFS_DIR / "stop_events.parquet")
    log.info("Loaded stop_events: %d rows", len(stop_events))

    # ── Enumerate transfers ───────────────────────────────────────────────
    pairs = enumerate_transfers(stop_events)
    validate_transfers(pairs)

    # ── Build graph ───────────────────────────────────────────────────────
    G = build_networkx_graph(pairs, stop_events)

    # ── Summary stats ─────────────────────────────────────────────────────
    stop_counts, pair_counts = compute_summary(pairs)

    # ── Save outputs ──────────────────────────────────────────────────────
    log.info("Saving outputs …")

    pairs.to_parquet(GTFS_DIR / "transfer_pairs.parquet", index=False)
    log.info("  Saved transfer_pairs.parquet  (%d rows)", len(pairs))

    with open(GTFS_DIR / "transfer_graph.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("  Saved transfer_graph.gpickle  (%d nodes, %d edges)",
             G.number_of_nodes(), G.number_of_edges())

    stop_counts.to_csv(RES_DIR / "top_transfer_stops.csv", index=False)
    pair_counts.to_csv(RES_DIR / "top_route_pairs.csv",    index=False)
    log.info("  Saved summary tables to results/tables/")

    log.info("Step 2 complete.")

    return {"transfer_pairs": pairs, "graph": G}


if __name__ == "__main__":
    result = main()
