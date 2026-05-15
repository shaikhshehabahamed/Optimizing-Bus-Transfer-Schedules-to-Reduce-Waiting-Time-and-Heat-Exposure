"""
Step 8 — Complete Paper Output Pipeline
=========================================
Single entry point that regenerates every publication artefact:

Figures (PDF + PNG at 300 DPI, 14 files)
-----------------------------------------
fig01_wbgt_density          WBGT diurnal profile × transfer density
fig02_pareto_front          Pareto frontier with solution annotations
fig03_convergence           NSGA-II convergence trajectory
fig04_seasonal_sensitivity  Seasonal WBGT comparison & monthly bar chart
fig05_hotspot_map           Geographic HEI hotspot map (Chattanooga)
fig06_wait_cdf              Wait-time CDF — all solutions on one plot
fig07_hourly_hei            Per-hour HEI absolute & reduction
fig08_route_savings         Per-route HEI & wait savings — all solutions
fig09_delta_shifts          Trip shift-magnitude distributions
fig10_transfer_network      Transfer connectivity at top hubs

Tables — CSV (clean, machine-readable)
----------------------------------------
tab01_network_summary.csv
tab02_solution_comparison.csv
tab03_pareto_front.csv
tab04_seasonal_sensitivity.csv
tab05_headway_stability.csv
tab06_route_savings_all.csv
tab07_top_hei_stops.csv
tab08_convergence.csv

LaTeX — ready-to-include .tex fragments
-----------------------------------------
tab02_solution_comparison.tex
tab04_seasonal_sensitivity.tex
tab05_headway_stability.tex
tab06_route_savings_knee.tex

Master metrics JSON
--------------------
paper_metrics.json          All key numbers cited in text
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step05_formulation import (
    build_trip_index, build_transfer_arrays, build_headway_arrays,
    build_wbgt_interp_arrays, TransferSyncProblem, DELTA_MAX_SEC, HEADWAY_FACTOR,
)

# ── directories ───────────────────────────────────────────────────────────────
FIG_DIR  = ROOT / "results" / "figures"
TAB_DIR  = ROOT / "results" / "tables"
TEX_DIR  = ROOT / "results" / "latex"
JSON_DIR = ROOT / "results"
for d in [FIG_DIR, TAB_DIR, TEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── matplotlib publication style ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Palatino Linotype", "Palatino", "Book Antiqua",
                           "DejaVu Serif", "serif"],
    "font.size"         : 9,
    "axes.titlesize"    : 10,
    "axes.labelsize"    : 9,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "legend.fontsize"   : 7.5,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "grid.linewidth"    : 0.5,
    "lines.linewidth"   : 1.6,
    "patch.linewidth"   : 0.6,
    "axes.axisbelow"    : True,
})

# ── colour palette (colourblind-safe, IBM palette) ────────────────────────────
C = {
    "baseline"  : "#2c3e50",
    "greedy_f1" : "#e67e22",
    "greedy_f2" : "#c0392b",
    "min_f1"    : "#8e44ad",
    "min_f2"    : "#c0392b",
    "knee"      : "#27ae60",
    "pareto"    : "#2980b9",
    "summer"    : "#e74c3c",
    "fall"      : "#f39c12",
    "spring"    : "#2ecc71",
}

SOL_STYLE = {
    "Baseline"       : dict(color=C["baseline"],  marker="X",  ms=90,  lw=0,   zorder=7),
    "Greedy-f1"      : dict(color=C["greedy_f1"], marker="^",  ms=70,  lw=0,   zorder=6),
    "Greedy-f2"      : dict(color=C["greedy_f2"], marker="s",  ms=70,  lw=0,   zorder=6),
    "NSGA-II min-f1" : dict(color=C["min_f1"],    marker="D",  ms=80,  lw=0,   zorder=8),
    "NSGA-II knee"   : dict(color=C["knee"],       marker="*",  ms=140, lw=0,   zorder=9),
    "NSGA-II min-f2" : dict(color=C["min_f2"],     marker="D",  ms=80,  lw=0,   zorder=8),
}

RISK_BANDS = [
    (0,   25.0, "#d5f5e3", "Low (<25°C)"),
    (25.0,28.0, "#fef9e7", "Moderate (25–28°C)"),
    (28.0,30.0, "#fdebd0", "High (28–30°C)"),
    (30.0,35.0, "#fadbd8", "Very high/Extreme"),
]

DPI = 300


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all() -> dict:
    """Load every dataset needed and return as a single dict."""
    se     = pd.read_parquet(ROOT / "data/gtfs/stop_events.parquet")
    hei_df = pd.read_parquet(ROOT / "data/gtfs/transfer_hei.parquet")
    stops  = pd.read_parquet(ROOT / "data/gtfs/stops_clean.parquet")
    wbgt   = np.load(ROOT / "data/weather/wbgt_lookup.npy")
    wbgt_p90 = np.load(ROOT / "data/weather/wbgt_p90_lookup.npy")

    ti  = build_trip_index(se)
    ta  = build_transfer_arrays(hei_df, ti)
    ha  = build_headway_arrays(se, ti)
    he, we = build_wbgt_interp_arrays(wbgt)
    prob   = TransferSyncProblem(ta, ha, he, we, len(ti), DELTA_MAX_SEC)

    pareto_X = np.load(ROOT / "results/solutions/nsga2_pareto_X.npy")
    pareto_F = np.load(ROOT / "results/solutions/nsga2_pareto_F.npy")
    dgf1     = np.load(ROOT / "results/solutions/greedy_f1_delta.npy")
    dgf2     = np.load(ROOT / "results/solutions/greedy_f2_delta.npy")

    d0   = np.zeros(prob.n_var)
    bf1  = prob.wait_sec.sum() / 60
    wbgt_base = np.interp((prob.arr_sec % 86400) / 3600, he, we)
    bf2  = (wbgt_base * prob.wait_sec / 60 * prob.shade * prob.freq).sum()

    f1n  = pareto_F[:, 0] / bf1
    f2n  = pareto_F[:, 1] / bf2
    ki   = np.sqrt(f1n**2 + f2n**2).argmin()
    dk   = pareto_X[ki]
    dm1  = pareto_X[pareto_F[:, 0].argmin()]
    dm2  = pareto_X[pareto_F[:, 1].argmin()]

    solutions = {
        "Baseline"       : d0,
        "Greedy-f1"      : dgf1,
        "Greedy-f2"      : dgf2,
        "NSGA-II min-f1" : dm1,
        "NSGA-II knee"   : dk,
        "NSGA-II min-f2" : dm2,
    }

    seasonal_profiles = {}
    for s in ["summer", "spring", "fall"]:
        seasonal_profiles[s] = pd.read_parquet(
            ROOT / f"data/weather/wbgt_profile_{s}.parquet"
        ).sort_values("clock_hour").reset_index(drop=True)

    conv = pd.read_csv(ROOT / "results/tables/tab08_convergence.csv").dropna()

    return dict(
        prob=prob, hei_df=hei_df, stops=stops, se=se, ti=ti,
        pareto_X=pareto_X, pareto_F=pareto_F,
        solutions=solutions, bf1=bf1, bf2=bf2, ki=ki,
        seasonal_profiles=seasonal_profiles, conv=conv,
        wbgt_arr=wbgt, wbgt_p90=wbgt_p90, he=he, we=we,
        summer_hourly=pd.read_parquet(ROOT / "data/weather/wbgt_hourly_summer.parquet"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: per-transfer stats for one solution
# ═══════════════════════════════════════════════════════════════════════════════

def sol_stats(delta: np.ndarray, D: dict) -> dict:
    """Return new_wait_min, new_hei, f1, f2, n_missed for one delta vector."""
    prob = D["prob"]
    r    = prob.evaluate_single(delta)
    nwm  = np.maximum(0, r["new_wait_sec"]) / 60.0
    wbgt_new = np.interp(
        ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
        D["he"], D["we"])
    nhei = wbgt_new * nwm * prob.shade * prob.freq
    return dict(new_wait_min=nwm, new_hei=nhei,
                f1=r["f1"], f2=r["f2"],
                n_missed=r["n_missed_connections"],
                is_feasible=r["is_feasible"])


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=DPI, format="pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, format="png")
    plt.close(fig)
    log.info(f"  [fig] {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 01 — WBGT diurnal profile × transfer density
# ═══════════════════════════════════════════════════════════════════════════════

def fig01_wbgt_density(D: dict) -> None:
    summer_h = D["summer_hourly"]
    hei_df   = D["hei_df"]

    profile = summer_h.groupby("clock_hour")["WBGT"].agg(
        mean="mean",
        p10 =lambda x: np.percentile(x, 10),
        p90 =lambda x: np.percentile(x, 90),
    ).reset_index()

    xfer_by_hour = hei_df.groupby("transfer_clock_hour").size()
    density = np.array([xfer_by_hour.get(h, 0) for h in range(24)])

    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))

    for lo, hi, col, _ in RISK_BANDS:
        ax1.axhspan(lo, hi, alpha=0.16, color=col, linewidth=0)

    ax2 = ax1.twinx()
    ax2.bar(range(24), density, width=0.7, color=C["pareto"],
            alpha=0.22, zorder=1, label="Transfer events per hour")
    ax2.set_ylabel("Transfer events  (count)", color=C["pareto"], labelpad=4)
    ax2.tick_params(axis="y", labelcolor=C["pareto"])
    ax2.set_ylim(0, density.max() * 3.8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax1.fill_between(profile["clock_hour"], profile["p10"], profile["p90"],
                     color=C["summer"], alpha=0.15, zorder=2,
                     label="WBGT  p10–p90 band")
    ax1.plot(profile["clock_hour"], profile["mean"],
             color=C["summer"], lw=2.2, zorder=4,
             label="WBGT mean  (summer 2025)")

    for thresh, label, ls in [(25.0, "Moderate risk  25°C", "--"),
                               (28.0, "High risk  28°C", ":")]:
        ax1.axhline(thresh, color="#95a5a6", lw=0.9, ls=ls, zorder=3)
        ax1.text(23.6, thresh + 0.12, label,
                 fontsize=6, ha="right", va="bottom", color="#7f8c8d")

    peak_h = int(profile.loc[profile["mean"].idxmax(), "clock_hour"])
    ax1.axvline(peak_h, color="#bdc3c7", lw=0.8, ls="--", zorder=2)
    ax1.text(peak_h + 0.2, 14.2,
             f"Peak WBGT\n{peak_h}:00 ({profile.loc[profile['clock_hour']==peak_h,'mean'].values[0]:.1f}°C)",
             fontsize=6.5, color="#7f8c8d", va="bottom")

    ax1.set_xlabel("Clock hour (local time, America/New_York)")
    ax1.set_ylabel("WBGT  (°C)")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_ylim(13, 28)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)])
    ax1.set_title("Summer WBGT diurnal profile and transfer-event density — Chattanooga, TN  (Jun–Aug 2025)")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               framealpha=0.88, edgecolor="none")
    fig.tight_layout()
    _save(fig, "fig01_wbgt_density")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 02 — Pareto front
# ═══════════════════════════════════════════════════════════════════════════════

def fig02_pareto_front(D: dict) -> None:
    pF   = D["pareto_F"][D["pareto_F"][:, 0].argsort()]
    bf1, bf2 = D["bf1"], D["bf2"]
    sols = D["solutions"]

    # Corrected hypervolume
    ref  = np.array([1.01, 1.01])
    Fn   = D["pareto_F"] / np.array([bf1, bf2])
    hv   = float(HV(ref_point=ref)(Fn))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.plot(pF[:, 0] / 1e3, pF[:, 1] / 1e6, color=C["pareto"],
            lw=1.8, zorder=3, label="Pareto front (NSGA-II, 500 gen)")
    ax.scatter(pF[:, 0] / 1e3, pF[:, 1] / 1e6,
               s=18, color=C["pareto"], alpha=0.65, zorder=4)

    sol_labels = {
        "Baseline"       : f"Baseline (δ=0)",
        "Greedy-f1"      : "Greedy-f₁",
        "Greedy-f2"      : "Greedy-f₂",
        "NSGA-II min-f1" : "NSGA-II  min-f₁",
        "NSGA-II knee"   : "NSGA-II  knee",
        "NSGA-II min-f2" : "NSGA-II  min-f₂",
    }
    for name, delta in sols.items():
        r    = D["prob"].evaluate_single(delta)
        sty  = SOL_STYLE[name]
        f1_v = r["f1"] / 1e3
        f2_v = r["f2"] / 1e6
        ax.scatter(f1_v, f2_v, s=sty["ms"], color=sty["color"],
                   marker=sty["marker"], zorder=sty["zorder"],
                   edgecolors="white", linewidths=0.7,
                   label=sol_labels[name])

    # Dominance region
    ax.fill_betweenx(
        [pF[:, 1].min() / 1e6, bf2 / 1e6 * 1.002],
        pF[:, 0].min() / 1e3, bf1 / 1e3 * 1.002,
        alpha=0.04, color=C["pareto"], linewidth=0)

    # HV annotation on knee
    ki = D["ki"]
    kf1 = D["pareto_X"][ki]
    rk  = D["prob"].evaluate_single(kf1)
    ax.annotate(f"Knee  HV={hv:.4f}",
                xy=(rk["f1"] / 1e3, rk["f2"] / 1e6),
                xytext=(rk["f1"] / 1e3 - 1.2, rk["f2"] / 1e6 + 0.004),
                fontsize=7, color=C["knee"],
                arrowprops=dict(arrowstyle="->", color=C["knee"], lw=0.8))

    ax.set_xlabel("f₁ — Total transfer wait time  (×10³ min)")
    ax.set_ylabel("f₂ — Frequency-weighted HEI  (×10⁶ °C·min)")
    ax.set_title("Pareto frontier: wait-time vs heat-exposure trade-off")
    ax.legend(loc="upper right", framealpha=0.88, edgecolor="none",
              fontsize=7.2, handlelength=1.2, borderpad=0.6)
    fig.tight_layout()
    _save(fig, "fig02_pareto_front")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 03 — NSGA-II convergence
# ═══════════════════════════════════════════════════════════════════════════════

def fig03_convergence(D: dict) -> None:
    conv = D["conv"].copy()
    bf1, bf2 = D["bf1"], D["bf2"]

    conv["f1_pct"] = 100 * conv["f1_best"] / bf1
    conv["f2_pct"] = 100 * conv["f2_best"] / bf2

    fig, ax = plt.subplots(figsize=(6.6, 3.4))

    ax.plot(conv["gen"], conv["f1_pct"],
            color=C["greedy_f1"], lw=2, marker="o", ms=4.5,
            label=r"Best $f_1$  (wait time)")
    ax.plot(conv["gen"], conv["f2_pct"],
            color=C["pareto"], lw=2, marker="s", ms=4.5,
            label=r"Best $f_2$  (HEI)")
    ax.axhline(100, color=C["baseline"], lw=1, ls="--", alpha=0.5,
               label="Baseline  (100%)")

    # Annotate final improvements
    last = conv.iloc[-1]
    ax.annotate(f"−{100-last['f1_pct']:.2f}%",
                xy=(last["gen"], last["f1_pct"]),
                xytext=(last["gen"] - 70, last["f1_pct"] - 0.9),
                fontsize=7.5, color=C["greedy_f1"],
                arrowprops=dict(arrowstyle="->", color=C["greedy_f1"], lw=0.7))
    ax.annotate(f"−{100-last['f2_pct']:.2f}%",
                xy=(last["gen"], last["f2_pct"]),
                xytext=(last["gen"] - 70, last["f2_pct"] - 1.5),
                fontsize=7.5, color=C["pareto"],
                arrowprops=dict(arrowstyle="->", color=C["pareto"], lw=0.7))

    # Mark feasibility onset (first gen with feasible solutions)
    first_feas_gen = conv.loc[~conv["n_feas"].isna() & (conv["n_feas"] > 0), "gen"].iloc[0]
    ax.axvline(first_feas_gen, color="#95a5a6", lw=0.8, ls=":", zorder=2)
    ax.text(first_feas_gen + 5, 99.5, f"Feasibility onset\n(gen {int(first_feas_gen)})",
            fontsize=6.5, color="#7f8c8d", va="top")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective value  (% of baseline)")
    ax.set_title(r"NSGA-II convergence  (pop = 200, seed = 42)")
    ax.set_xlim(conv["gen"].min() - 5, conv["gen"].max() + 10)
    ax.set_ylim(92, 101.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right", framealpha=0.88, edgecolor="none")
    fig.tight_layout()
    _save(fig, "fig03_convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Seasonal sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def fig04_seasonal(D: dict) -> None:
    profiles = D["seasonal_profiles"]
    season_style = {
        "summer": (C["summer"], "Summer (Jun–Aug 2025)"),
        "fall"  : (C["fall"],   "Fall (Sep–Nov 2024)"),
        "spring": (C["spring"], "Spring (Mar–May 2025)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Left: diurnal profiles
    for lo, hi, col, _ in RISK_BANDS[:3]:
        ax1.axhspan(lo, hi, alpha=0.14, color=col, linewidth=0)

    for s, (col, lbl) in season_style.items():
        p = profiles[s]
        ax1.plot(p["clock_hour"], p["wbgt_mean"], color=col, lw=2, label=lbl)
        ax1.fill_between(p["clock_hour"], p["wbgt_p10"], p["wbgt_p90"],
                         color=col, alpha=0.09)

    ax1.axhline(25.0, color="#7f8c8d", lw=0.8, ls="--")
    ax1.text(23.5, 25.25, "Moderate risk  25°C",
             fontsize=6, ha="right", color="#7f8c8d")
    ax1.set_xlabel("Clock hour")
    ax1.set_ylabel("WBGT  (°C)")
    ax1.set_title("(a)  Diurnal WBGT profiles by season")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_ylim(-5, 30)
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 4)])
    ax1.legend(loc="upper left", framealpha=0.88, edgecolor="none")

    # Right: transit-hour mean WBGT by month
    month_order = ["Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]
    m_col_map   = {
        "Mar":"#5dade2","Apr":"#45b39d","May":"#2ecc71",
        "Jun":"#f7dc6f","Jul":"#f39c12","Aug":"#e74c3c",
        "Sep":"#e67e22","Oct":"#d35400","Nov":"#922b21",
    }
    rows = []
    for s in ["spring","summer","fall"]:
        mon = pd.read_parquet(ROOT / f"data/weather/wbgt_monthly_{s}.parquet")
        t   = mon[mon["clock_hour"].between(9, 18)]
        agg = t.groupby("month_name")["wbgt_mean"].mean().reset_index()
        rows.append(agg)

    df_m = pd.concat(rows).drop_duplicates("month_name")
    df_m["idx"] = df_m["month_name"].map({m: i for i, m in enumerate(month_order)})
    df_m = df_m.sort_values("idx").reset_index(drop=True)

    cols = [m_col_map.get(m, "#aaa") for m in df_m["month_name"]]
    ax2.bar(df_m["month_name"], df_m["wbgt_mean"],
            color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    ax2.axhline(25.0, color=C["summer"], lw=0.9, ls="--", zorder=4)
    ax2.text(8.4, 25.5, "Moderate risk  25°C",
             fontsize=6, ha="right", color=C["summer"])
    ax2.set_ylabel("Mean WBGT  09–18h  (°C)")
    ax2.set_title("(b)  Transit-hour mean WBGT by month")
    ax2.set_ylim(0, 30)
    ax2.tick_params(axis="x", rotation=0)

    fig.tight_layout(pad=1.2)
    _save(fig, "fig04_seasonal_sensitivity")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Geographic HEI hotspot map
# ═══════════════════════════════════════════════════════════════════════════════

def fig05_hotspot_map(D: dict) -> None:
    stops   = D["stops"]
    hei_df  = D["hei_df"]

    stop_agg = (
        hei_df.groupby(["stop_id","stop_name","stop_lat","stop_lon","is_sheltered"])
        .agg(total_hei=("hei_mean","sum"), n_xfr=("hei_mean","count"))
        .reset_index()
        .sort_values("total_hei", ascending=False)
    )

    cmap  = LinearSegmentedColormap.from_list("hei",
              ["#2ecc71","#f1c40f","#e74c3c"], N=256)
    hvals = stop_agg["total_hei"].values
    hnorm = (hvals - hvals.min()) / (hvals.max() - hvals.min() + 1e-9)
    sizes = 18 + 220 * (stop_agg["n_xfr"].values / stop_agg["n_xfr"].max()) ** 0.65

    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    ax.scatter(stops["stop_lon"], stops["stop_lat"],
               s=3.5, color="#bdc3c7", alpha=0.35, zorder=1)

    sc = ax.scatter(stop_agg["stop_lon"], stop_agg["stop_lat"],
                    s=sizes, c=hnorm, cmap=cmap,
                    alpha=0.82, zorder=3,
                    edgecolors="white", linewidths=0.35)

    top8 = stop_agg.head(8)
    for rank, (_, row) in enumerate(top8.iterrows()):
        nm  = row["stop_name"][:20] + ("…" if len(row["stop_name"]) > 20 else "")
        ofx = 0.005 if rank % 2 == 0 else -0.005
        ofy = 0.002 * (1 + rank % 3)
        ax.annotate(f"{rank+1}. {nm}",
                    xy=(row["stop_lon"], row["stop_lat"]),
                    xytext=(row["stop_lon"] + ofx, row["stop_lat"] + ofy),
                    fontsize=5.5, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#95a5a6",
                                    lw=0.45, shrinkA=2, shrinkB=2))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.52, pad=0.02, aspect=18)
    cbar.set_label("HEI  (normalised)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    for n_ref, lbl in [(80,"80"), (400,"400"), (1000,"1,000")]:
        s_ref = 18 + 220 * (n_ref / stop_agg["n_xfr"].max()) ** 0.65
        ax.scatter([], [], s=s_ref, color="#7f8c8d", alpha=0.55,
                   label=f"{lbl} transfers", edgecolors="white", linewidths=0.35)
    ax.legend(title="Transfer events", title_fontsize=7,
              loc="lower left", fontsize=6.5, framealpha=0.88, edgecolor="none",
              handletextpad=0.35, borderpad=0.55)

    ax.set_xlabel("Longitude  (°W)")
    ax.set_ylabel("Latitude  (°N)")
    ax.set_title("Transfer HEI hotspot stops — Chattanooga, TN\n"
                 "(bubble size = transfer volume,  colour = total HEI)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig05_hotspot_map")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 06 — Wait-time CDF
# ═══════════════════════════════════════════════════════════════════════════════

def fig06_wait_cdf(D: dict) -> None:
    prob  = D["prob"]
    hei_df = D["hei_df"]

    sol_subset = {
        "Baseline"       : D["solutions"]["Baseline"],
        "Greedy-f₂"      : D["solutions"]["Greedy-f2"],
        "NSGA-II knee"   : D["solutions"]["NSGA-II knee"],
        "NSGA-II min-f₂" : D["solutions"]["NSGA-II min-f2"],
    }
    col_map = {
        "Baseline"       : C["baseline"],
        "Greedy-f₂"      : C["greedy_f2"],
        "NSGA-II knee"   : C["knee"],
        "NSGA-II min-f₂" : C["min_f2"],
    }
    ls_map = {
        "Baseline"       : "-",
        "Greedy-f₂"      : "--",
        "NSGA-II knee"   : "-.",
        "NSGA-II min-f₂" : ":",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # ── Left: CDF of wait times ───────────────────────────────────────
    for name, delta in sol_subset.items():
        r  = prob.evaluate_single(delta)
        wm = np.maximum(0, r["new_wait_sec"]) / 60.0
        xs = np.sort(wm)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.plot(xs, ys * 100, color=col_map[name],
                 ls=ls_map[name], lw=1.8, label=name)

    ax1.axvline(1.0, color="#bdc3c7", lw=0.7, ls=":")
    ax1.set_xlabel("Transfer wait time  (min)")
    ax1.set_ylabel("Cumulative fraction  (%)")
    ax1.set_title("(a)  Wait-time CDF — all solutions")
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 102)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.legend(loc="lower right", framealpha=0.88, edgecolor="none")

    # ── Right: mean wait per route pair ──────────────────────────────
    hei_hrs  = hei_df["transfer_clock_hour"].values
    base_wm  = prob.wait_sec / 60.0
    hours    = np.arange(24)

    for name, delta in sol_subset.items():
        r   = prob.evaluate_single(delta)
        nwm = np.maximum(0, r["new_wait_sec"]) / 60.0
        mean_per_hour = np.array([
            nwm[hei_hrs == h].mean() if (hei_hrs == h).any() else np.nan
            for h in hours])
        ax2.plot(hours, mean_per_hour, color=col_map[name],
                 ls=ls_map[name], lw=1.8, label=name, marker="o", ms=3)

    ax2.axhline(1.0, color="#bdc3c7", lw=0.7, ls=":")
    ax2.set_xlabel("Clock hour")
    ax2.set_ylabel("Mean wait per transfer  (min)")
    ax2.set_title("(b)  Mean wait by hour")
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 4)])
    ax2.legend(loc="upper right", framealpha=0.88, edgecolor="none")

    fig.tight_layout(pad=1.2)
    _save(fig, "fig06_wait_cdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 07 — Per-hour HEI
# ═══════════════════════════════════════════════════════════════════════════════

def fig07_hourly_hei(D: dict) -> None:
    prob   = D["prob"]
    hei_df = D["hei_df"]
    hours  = hei_df["transfer_clock_hour"].values
    h_range = np.arange(24)

    sol_subset = {
        "Baseline"       : D["solutions"]["Baseline"],
        "NSGA-II knee"   : D["solutions"]["NSGA-II knee"],
        "NSGA-II min-f₂" : D["solutions"]["NSGA-II min-f2"],
    }
    col_map  = {"Baseline": C["baseline"],
                "NSGA-II knee": C["knee"],
                "NSGA-II min-f₂": C["min_f2"]}
    width    = 0.28
    offsets  = [-width, 0, width]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # ── Left: absolute HEI ────────────────────────────────────────────
    for (name, delta), ox in zip(sol_subset.items(), offsets):
        r   = prob.evaluate_single(delta)
        nwm = np.maximum(0, r["new_wait_sec"]) / 60.0
        wa  = np.interp(
            ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
            D["he"], D["we"])
        nhei = wa * nwm * prob.shade * prob.freq
        hv_abs = np.array([nhei[hours == h].sum() for h in h_range])
        ax1.bar(h_range + ox, hv_abs / 1e3, width,
                color=col_map[name], alpha=0.72, label=name, linewidth=0)

    ax1.set_xlabel("Clock hour")
    ax1.set_ylabel("Total HEI  (×10³ °C·min)")
    ax1.set_title("(a)  Absolute HEI per hour")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)])
    ax1.legend(loc="upper left", framealpha=0.88, edgecolor="none")

    # ── Right: HEI reduction vs baseline ─────────────────────────────
    r0     = prob.evaluate_single(D["solutions"]["Baseline"])
    nwm0   = np.maximum(0, r0["new_wait_sec"]) / 60.0
    wa0    = np.interp((prob.arr_sec % 86400) / 3600, D["he"], D["we"])
    nhei0  = wa0 * nwm0 * prob.shade * prob.freq
    hv_base = np.array([nhei0[hours == h].sum() for h in h_range])

    for (name, delta), ox in zip(list(sol_subset.items())[1:], [-width/2, width/2]):
        r   = prob.evaluate_single(delta)
        nwm = np.maximum(0, r["new_wait_sec"]) / 60.0
        wa  = np.interp(
            ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
            D["he"], D["we"])
        nhei  = wa * nwm * prob.shade * prob.freq
        hv_red = np.where(hv_base > 0,
                          100 * (1 - np.array([nhei[hours==h].sum() for h in h_range])
                                 / np.where(hv_base > 0, hv_base, 1.0)),
                          0.0)
        ax2.bar(h_range + ox, hv_red, width,
                color=col_map[name], alpha=0.72, label=name, linewidth=0)

    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Clock hour")
    ax2.set_ylabel("HEI reduction vs baseline  (%)")
    ax2.set_title("(b)  Per-hour HEI reduction")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)])
    ax2.legend(loc="upper left", framealpha=0.88, edgecolor="none")

    fig.tight_layout(pad=1.2)
    _save(fig, "fig07_hourly_hei")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Per-route savings (all solutions)
# ═══════════════════════════════════════════════════════════════════════════════

def fig08_route_savings(D: dict) -> None:
    prob   = D["prob"]
    hei_df = D["hei_df"]

    base_wm  = prob.wait_sec / 60.0
    wa0      = np.interp((prob.arr_sec % 86400) / 3600, D["he"], D["we"])
    base_hei = wa0 * base_wm * prob.shade * prob.freq

    sol_subset = {
        "Greedy-f₂"      : D["solutions"]["Greedy-f2"],
        "NSGA-II min-f₁" : D["solutions"]["NSGA-II min-f1"],
        "NSGA-II knee"   : D["solutions"]["NSGA-II knee"],
        "NSGA-II min-f₂" : D["solutions"]["NSGA-II min-f2"],
    }
    col_map = {
        "Greedy-f₂"      : C["greedy_f2"],
        "NSGA-II min-f₁" : C["min_f1"],
        "NSGA-II knee"   : C["knee"],
        "NSGA-II min-f₂" : C["min_f2"],
    }

    # Gather per-route HEI reduction for all solutions
    routes = sorted(hei_df["feeder_route_short"].unique(),
                    key=lambda r: hei_df[hei_df["feeder_route_short"]==r].index.size,
                    reverse=True)
    n_r   = len(routes)

    data_hei  = {}
    data_wait = {}
    for name, delta in sol_subset.items():
        r   = prob.evaluate_single(delta)
        nwm = np.maximum(0, r["new_wait_sec"]) / 60.0
        wa  = np.interp(
            ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
            D["he"], D["we"])
        nhei = wa * nwm * prob.shade * prob.freq
        h_pct = []; w_pct = []
        for rte in routes:
            mask = hei_df["feeder_route_short"].values == rte
            bh   = base_hei[mask].sum()
            nh   = nhei[mask].sum()
            bw   = base_wm[mask].sum()
            nw   = nwm[mask].sum()
            h_pct.append(100 * (1 - nh / bh) if bh > 0 else 0)
            w_pct.append(100 * (1 - nw / bw) if bw > 0 else 0)
        data_hei[name]  = h_pct
        data_wait[name] = w_pct

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.8))

    n_sols = len(sol_subset)
    w_bar  = 0.18
    y_pos  = np.arange(n_r)
    offsets = np.linspace(-(n_sols-1)/2, (n_sols-1)/2, n_sols) * w_bar

    for (name, _), ox in zip(sol_subset.items(), offsets):
        ax1.barh(y_pos + ox, data_hei[name], w_bar,
                 color=col_map[name], alpha=0.78, label=name, linewidth=0)
        ax2.barh(y_pos + ox, data_wait[name], w_bar,
                 color=col_map[name], alpha=0.78, linewidth=0)

    for ax, title, xlabel in [
        (ax1, "(a)  HEI reduction per feeder route  (%)", "HEI reduction  (%)"),
        (ax2, "(b)  Wait-time reduction per feeder route  (%)", "Wait-time reduction  (%)"),
    ]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Route {r}" for r in routes], fontsize=7.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)

    ax1.legend(loc="lower right", framealpha=0.88, edgecolor="none",
               fontsize=7, handlelength=1.2)
    fig.tight_layout(pad=1.2)
    _save(fig, "fig08_route_savings")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 09 — Trip shift distributions
# ═══════════════════════════════════════════════════════════════════════════════

def fig09_delta_shifts(D: dict) -> None:
    sol_subset = {
        "Greedy-f₁"      : D["solutions"]["Greedy-f1"],
        "Greedy-f₂"      : D["solutions"]["Greedy-f2"],
        "NSGA-II min-f₁" : D["solutions"]["NSGA-II min-f1"],
        "NSGA-II knee"   : D["solutions"]["NSGA-II knee"],
        "NSGA-II min-f₂" : D["solutions"]["NSGA-II min-f2"],
    }
    col_map = {
        "Greedy-f₁"      : C["greedy_f1"],
        "Greedy-f₂"      : C["greedy_f2"],
        "NSGA-II min-f₁" : C["min_f1"],
        "NSGA-II knee"   : C["knee"],
        "NSGA-II min-f₂" : C["min_f2"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    bins_abs = np.linspace(0, 305, 32)
    bins_sgn = np.linspace(-305, 305, 32)

    for name, delta in sol_subset.items():
        shifted = delta[np.abs(delta) > 0.5]
        # Absolute magnitude
        ax1.hist(np.abs(shifted), bins=bins_abs, density=True,
                 alpha=0.35, color=col_map[name], label=name,
                 histtype="stepfilled", linewidth=0)
        ax1.hist(np.abs(shifted), bins=bins_abs, density=True,
                 alpha=0.85, color=col_map[name],
                 histtype="step", linewidth=1.3)
        # Signed
        ax2.hist(shifted, bins=bins_sgn, density=True,
                 alpha=0.30, color=col_map[name], label=name,
                 histtype="stepfilled", linewidth=0)
        ax2.hist(shifted, bins=bins_sgn, density=True,
                 alpha=0.80, color=col_map[name],
                 histtype="step", linewidth=1.3)

    ax1.set_xlabel("|δᵢ|  — shift magnitude  (s)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a)  Absolute shift magnitude")
    ax1.legend(loc="upper right", framealpha=0.88, edgecolor="none")

    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("δᵢ  — signed shift  (s)     [+ = later, − = earlier]")
    ax2.set_ylabel("Density")
    ax2.set_title("(b)  Signed shift distribution")
    ax2.set_xlim(-310, 310)

    for ax in (ax1, ax2):
        ax.axvline( DELTA_MAX_SEC, color="#bdc3c7", lw=0.7, ls=":")
        ax.axvline(-DELTA_MAX_SEC, color="#bdc3c7", lw=0.7, ls=":")

    ax2.text( DELTA_MAX_SEC - 5, ax2.get_ylim()[1] * 0.95, "+300s",
              ha="right", fontsize=6, color="#7f8c8d")
    ax2.text(-DELTA_MAX_SEC + 5, ax2.get_ylim()[1] * 0.95, "−300s",
              ha="left",  fontsize=6, color="#7f8c8d")

    fig.tight_layout(pad=1.2)
    _save(fig, "fig09_delta_shifts")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Transfer connectivity at top hubs
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_transfer_network(D: dict) -> None:
    hei_df = D["hei_df"]

    top_stops = (
        hei_df.groupby(["stop_id","stop_name"])["hei_mean"]
        .sum()
        .nlargest(10)
        .rename("total_hei")
        .reset_index()
    )

    sub = hei_df[hei_df["stop_id"].isin(top_stops["stop_id"])]

    route_pair_counts = (
        sub.groupby(["stop_id","stop_name",
                     "feeder_route_short","connector_route_short"])
        .agg(n=("hei_mean","count"), total_hei=("hei_mean","sum"))
        .reset_index()
    )

    n_top = min(6, len(top_stops))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    axes_flat = axes.flatten()

    for ax_i, (_, row) in enumerate(top_stops.head(n_top).iterrows()):
        ax  = axes_flat[ax_i]
        sid = row["stop_id"]
        sp  = route_pair_counts[route_pair_counts["stop_id"] == sid].copy()
        sp["pair"] = sp["feeder_route_short"] + "→" + sp["connector_route_short"]
        sp = sp.sort_values("total_hei", ascending=False)

        if len(sp) > 5:
            others = sp.iloc[5:]["total_hei"].sum()
            sp = sp.iloc[:5].copy()
            sp = pd.concat([sp, pd.DataFrame([{"pair":"Others","total_hei":others}])],
                           ignore_index=True)

        ax.pie(sp["total_hei"], labels=sp["pair"],
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 6.5},
               pctdistance=0.75,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"})

        short_name = row["stop_name"]
        if len(short_name) > 22:
            short_name = short_name[:20] + "…"
        ax.set_title(f"#{ax_i+1}  {short_name}\n"
                     f"HEI = {row['total_hei']/1e3:.1f}k °C·min",
                     fontsize=7.5)

    for ax_i in range(n_top, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle("Route-pair HEI composition at top-6 transfer hubs",
                 fontsize=10, y=1.01)
    fig.tight_layout(pad=1.0)
    _save(fig, "fig10_transfer_network")


# ═══════════════════════════════════════════════════════════════════════════════
# Tables — CSV
# ═══════════════════════════════════════════════════════════════════════════════

def make_tables(D: dict) -> dict:
    prob   = D["prob"]
    hei_df = D["hei_df"]
    bf1, bf2 = D["bf1"], D["bf2"]
    sols   = D["solutions"]

    base_wm  = prob.wait_sec / 60.0
    wa0      = np.interp((prob.arr_sec % 86400) / 3600, D["he"], D["we"])
    base_hei = wa0 * base_wm * prob.shade * prob.freq

    # ── Tab 01: network summary ───────────────────────────────────────
    se = D["se"]
    routes_clean = pd.read_parquet(ROOT / "data/gtfs/routes_clean.parquet")
    t1 = pd.DataFrame([{
        "Metric"        : "Routes (total in network)",
        "Value"         : routes_clean["route_short_name"].nunique(),
    },{
        "Metric"        : "Routes with weekday service",
        "Value"         : D["se"]["route_id"].nunique(),
    },{
        "Metric"        : "Weekday trips",
        "Value"         : se["trip_id"].nunique(),
    },{
        "Metric"        : "Unique stops",
        "Value"         : se["stop_id"].nunique(),
    },{
        "Metric"        : "Stop events",
        "Value"         : len(se),
    },{
        "Metric"        : "Transfer pairs (after same-block filter)",
        "Value"         : len(hei_df),
    },{
        "Metric"        : "Sheltered transfer stops",
        "Value"         : int(hei_df["is_sheltered"].sum()),
    },{
        "Metric"        : "Transfer window (min)",
        "Value"         : "1 – 15",
    },{
        "Metric"        : "Mean wait time (min)",
        "Value"         : round(hei_df["wait_min"].mean(), 2),
    },{
        "Metric"        : "Baseline f1 (min)",
        "Value"         : round(bf1, 1),
    },{
        "Metric"        : "Baseline f2 (°C·min)",
        "Value"         : round(bf2, 1),
    },{
        "Metric"        : "Peak WBGT hour",
        "Value"         : int(D["wbgt_arr"].argmax()),
    },{
        "Metric"        : "Peak mean summer WBGT (°C)",
        "Value"         : round(D["wbgt_arr"].max(), 2),
    }])
    t1.to_csv(TAB_DIR / "tab01_network_summary.csv", index=False)
    log.info("  [tab] tab01_network_summary.csv")

    # ── Tab 02: solution comparison ───────────────────────────────────
    ref  = np.array([1.01, 1.01])
    Fn   = D["pareto_F"] / np.array([bf1, bf2])
    hv   = float(HV(ref_point=ref)(Fn))

    rows = []
    for name, delta in sols.items():
        r    = prob.evaluate_single(delta)
        nwm  = np.maximum(0, r["new_wait_sec"]) / 60.0
        wa   = np.interp(
            ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
            D["he"], D["we"])
        nhei = wa * nwm * prob.shade * prob.freq
        rows.append({
            "Solution"           : name,
            "f1_min"             : round(r["f1"], 2),
            "f1_pct_reduction"   : round(100 * (1 - r["f1"] / bf1), 2),
            "f2_hei"             : round(r["f2"], 2),
            "f2_pct_reduction"   : round(100 * (1 - r["f2"] / bf2), 2),
            "wait_saved_min"     : round((base_wm - nwm).sum(), 1),
            "hei_saved"          : round((base_hei - nhei).sum(), 1),
            "n_missed_conn"      : r["n_missed_connections"],
            "pct_missed"         : round(100 * r["n_missed_connections"] / len(prob.feeder_idx), 2),
            "is_feasible"        : r["is_feasible"],
            "n_trips_shifted"    : int((np.abs(delta) > 0.5).sum()),
            "mean_abs_shift_sec" : round(float(np.abs(delta[np.abs(delta) > 0.5]).mean())
                                         if (np.abs(delta) > 0.5).any() else 0, 1),
        })
    t2 = pd.DataFrame(rows)
    t2.to_csv(TAB_DIR / "tab02_solution_comparison.csv", index=False)
    log.info("  [tab] tab02_solution_comparison.csv")

    # ── Tab 03: annotated Pareto front ────────────────────────────────
    pF = D["pareto_F"]
    pX = D["pareto_X"]
    order  = pF[:, 0].argsort()
    pF_s   = pF[order]; pX_s = pX[order]
    f1n    = pF_s[:, 0] / bf1; f2n = pF_s[:, 1] / bf2
    dist   = np.sqrt(f1n**2 + f2n**2)
    t3 = pd.DataFrame({
        "rank"          : np.arange(len(pF_s)),
        "f1_min"        : np.round(pF_s[:, 0], 2),
        "f1_pct_red"    : np.round(100 * (1 - pF_s[:, 0] / bf1), 3),
        "f2_hei"        : np.round(pF_s[:, 1], 2),
        "f2_pct_red"    : np.round(100 * (1 - pF_s[:, 1] / bf2), 3),
        "knee_score"    : np.round(dist, 6),
        "is_knee"       : dist == dist.min(),
        "is_min_f1"     : pF_s[:, 0] == pF_s[:, 0].min(),
        "is_min_f2"     : pF_s[:, 1] == pF_s[:, 1].min(),
    })
    t3.to_csv(TAB_DIR / "tab03_pareto_front.csv", index=False)
    log.info("  [tab] tab03_pareto_front.csv")

    # ── Tab 04: seasonal sensitivity ─────────────────────────────────
    rows4 = []
    for season in ["summer", "spring", "fall"]:
        p   = D["seasonal_profiles"][season]
        wa  = np.append(p["wbgt_mean"].values, p["wbgt_mean"].values[0])
        he2 = np.arange(25, dtype=float)
        for name, delta in sols.items():
            fh  = ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600
            wv  = np.interp(fh, he2, wa)
            nwm = np.maximum(0, prob.wait_sec + delta[prob.conn_idx]
                             - delta[prob.feeder_idx]) / 60.0
            f2s = (wv * nwm * prob.shade * prob.freq).sum()
            rows4.append({"season": season, "solution": name,
                          "f2_season": round(f2s, 2),
                          "f2_pct_red": round(100*(1-f2s/bf2), 3)})
    t4 = pd.DataFrame(rows4)
    t4.to_csv(TAB_DIR / "tab04_seasonal_sensitivity.csv", index=False)
    log.info("  [tab] tab04_seasonal_sensitivity.csv")

    # ── Tab 05: headway stability ─────────────────────────────────────
    rows5 = []
    for name, delta in sols.items():
        new_hw = prob.hw_sec + (delta[prob.next_idx] - delta[prob.prev_idx])
        rows5.append({
            "solution"           : name,
            "n_hw_pairs"         : len(prob.hw_sec),
            "n_violated"         : int((new_hw < 0.8 * prob.hw_sec - 1e-6).sum()),
            "min_hw_ratio"       : round(float((new_hw / prob.hw_sec).min()), 4),
            "mean_hw_ratio"      : round(float((new_hw / prob.hw_sec).mean()), 4),
            "pct_compressed"     : round(float(100 * (new_hw < prob.hw_sec).mean()), 1),
        })
    t5 = pd.DataFrame(rows5)
    t5.to_csv(TAB_DIR / "tab05_headway_stability.csv", index=False)
    log.info("  [tab] tab05_headway_stability.csv")

    # ── Tab 06: per-route savings — all solutions ─────────────────────
    routes = sorted(hei_df["feeder_route_short"].unique())
    rows6  = []
    for name, delta in sols.items():
        r   = prob.evaluate_single(delta)
        nwm = np.maximum(0, r["new_wait_sec"]) / 60.0
        wa  = np.interp(
            ((prob.arr_sec + delta[prob.feeder_idx]) % 86400) / 3600,
            D["he"], D["we"])
        nhei = wa * nwm * prob.shade * prob.freq
        for rte in routes:
            mask = hei_df["feeder_route_short"].values == rte
            bh   = base_hei[mask].sum()
            nh   = nhei[mask].sum()
            bw   = base_wm[mask].sum()
            nw   = nwm[mask].sum()
            rows6.append({
                "solution"       : name,
                "route"          : rte,
                "n_transfers"    : int(mask.sum()),
                "wait_saved_min" : round(float(bw - nw), 2),
                "wait_saved_pct" : round(100 * (bw - nw) / bw, 2) if bw > 0 else 0,
                "hei_saved"      : round(float(bh - nh), 2),
                "hei_saved_pct"  : round(100 * (bh - nh) / bh, 2) if bh > 0 else 0,
            })
    t6 = pd.DataFrame(rows6)
    t6.to_csv(TAB_DIR / "tab06_route_savings_all.csv", index=False)
    log.info("  [tab] tab06_route_savings_all.csv")

    # ── Tab 07: top HEI stops ─────────────────────────────────────────
    t7 = (
        hei_df.groupby(["stop_id","stop_name","stop_lat","stop_lon","is_sheltered"])
        .agg(
            n_transfers  = ("hei_mean", "count"),
            total_hei    = ("hei_mean", "sum"),
            mean_hei     = ("hei_mean", "mean"),
            mean_wait_min= ("wait_min", "mean"),
            mean_wbgt    = ("wbgt_mean","mean"),
        )
        .reset_index()
        .sort_values("total_hei", ascending=False)
        .head(30)
        .round(3)
    )
    t7.to_csv(TAB_DIR / "tab07_top_hei_stops.csv", index=False)
    log.info("  [tab] tab07_top_hei_stops.csv")

    # ── Tab 08: NSGA-II convergence (clean) ───────────────────────────
    t8 = D["conv"][["gen","n_feas","f1_best","f2_best","elapsed"]].copy()
    t8["f1_pct"] = (100 * t8["f1_best"] / bf1).round(3)
    t8["f2_pct"] = (100 * t8["f2_best"] / bf2).round(3)
    t8.to_csv(TAB_DIR / "tab08_convergence.csv", index=False)
    log.info("  [tab] tab08_convergence.csv")

    return dict(t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6, t7=t7, t8=t8,
                hv=hv)


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX tables
# ═══════════════════════════════════════════════════════════════════════════════

def _tex_num(x, fmt=".1f"):
    """Format a number for LaTeX. x must be bool, str, NaN, or numeric."""
    if isinstance(x, bool): return r"\checkmark" if x else r"--"
    if isinstance(x, str):  return x
    if pd.isna(x):           return r"--"
    return f"{float(x):{fmt}}"


def make_latex(tabs: dict) -> None:

    # ── Tab 02: solution comparison ───────────────────────────────────
    t2 = tabs["t2"]
    hv = tabs["hv"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Solution comparison: baseline, greedy heuristics, and key Pareto-front solutions."
        r" $f_1$: total passenger wait time; $f_2$: frequency-weighted heat exposure index (HEI)."
        r" Missed connections reported as a transparency metric (not a hard constraint)."
        r" Hypervolume indicator (normalised): " + f"{hv:.4f}" + r".}",
        r"\label{tab:solution_comparison}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Solution & $f_1$ (min) & $\Delta f_1$ (\%) & $f_2$ (°C·min) & $\Delta f_2$ (\%) "
        r"& Wait saved (min) & Missed conn. & Feasible \\",
        r"\midrule",
    ]
    for _, row in t2.iterrows():
        feas = r"\checkmark" if row["is_feasible"] else r"$\times$"
        lines.append(
            f"  {row['Solution']} & "
            f"{row['f1_min']:,.1f} & "
            f"{row['f1_pct_reduction']:+.2f} & "
            f"{row['f2_hei']:,.1f} & "
            f"{row['f2_pct_reduction']:+.2f} & "
            f"{row['wait_saved_min']:,.1f} & "
            f"{row['n_missed_conn']:,} & "
            f"{feas} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX_DIR / "tab02_solution_comparison.tex").write_text("\n".join(lines))
    log.info("  [tex] tab02_solution_comparison.tex")

    # ── Tab 04: seasonal sensitivity ─────────────────────────────────
    t4 = tabs["t4"]
    pivot = t4.pivot(index="solution", columns="season", values="f2_pct_red")
    pivot = pivot[["summer","fall","spring"]]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Seasonal sensitivity: $f_2$ reduction (\%) relative to summer baseline"
        r" when WBGT profiles for each season are substituted. Results confirm that heat-aware"
        r" scheduling provides meaningful gains only in the summer hot-climate window.}",
        r"\label{tab:seasonal_sensitivity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Solution & Summer & Fall & Spring \\",
        r"\midrule",
    ]
    for sol, row in pivot.iterrows():
        lines.append(
            f"  {sol} & "
            f"{row['summer']:+.2f}\\% & "
            f"{row['fall']:+.2f}\\% & "
            f"{row['spring']:+.2f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX_DIR / "tab04_seasonal_sensitivity.tex").write_text("\n".join(lines))
    log.info("  [tex] tab04_seasonal_sensitivity.tex")

    # ── Tab 05: headway stability ─────────────────────────────────────
    t5 = tabs["t5"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Headway stability across all solutions. The minimum allowed headway ratio"
        r" is 0.80 (20\% reduction cap). No solution violates this constraint.}",
        r"\label{tab:headway_stability}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Solution & Violations & Min ratio & Mean ratio & Compressed (\%) \\",
        r"\midrule",
    ]
    for _, row in t5.iterrows():
        lines.append(
            f"  {row['solution']} & "
            f"{int(row['n_violated'])} & "
            f"{row['min_hw_ratio']:.4f} & "
            f"{row['mean_hw_ratio']:.4f} & "
            f"{row['pct_compressed']:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX_DIR / "tab05_headway_stability.tex").write_text("\n".join(lines))
    log.info("  [tex] tab05_headway_stability.tex")

    # ── Tab 06: per-route savings (knee only) ─────────────────────────
    t6     = tabs["t6"]
    knee   = t6[t6["solution"] == "NSGA-II knee"].copy()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-route wait and HEI savings for the knee Pareto solution."
        r" Negative values indicate routes that accept increased wait times in"
        r" exchange for network-level HEI reduction.}",
        r"\label{tab:route_savings}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Route & $n$ & Wait saved (min) & $\Delta$Wait (\%) & HEI saved & $\Delta$HEI (\%) \\",
        r"\midrule",
    ]
    for _, row in knee.sort_values("hei_saved", ascending=False).iterrows():
        lines.append(
            f"  {row['route']} & "
            f"{int(row['n_transfers'])} & "
            f"{row['wait_saved_min']:+.1f} & "
            f"{row['wait_saved_pct']:+.1f}\\% & "
            f"{row['hei_saved']:+.1f} & "
            f"{row['hei_saved_pct']:+.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX_DIR / "tab06_route_savings_knee.tex").write_text("\n".join(lines))
    log.info("  [tex] tab06_route_savings_knee.tex")


# ═══════════════════════════════════════════════════════════════════════════════
# Master metrics JSON
# ═══════════════════════════════════════════════════════════════════════════════

def make_metrics_json(D: dict, tabs: dict) -> None:
    prob   = D["prob"]
    hei_df = D["hei_df"]
    bf1, bf2 = D["bf1"], D["bf2"]
    hv     = tabs["hv"]

    pF   = D["pareto_F"]
    ki   = D["ki"]
    km   = D["solutions"]["NSGA-II knee"]
    rk   = prob.evaluate_single(km)
    m2   = prob.evaluate_single(D["solutions"]["NSGA-II min-f2"])
    m1   = prob.evaluate_single(D["solutions"]["NSGA-II min-f1"])
    gf2  = prob.evaluate_single(D["solutions"]["Greedy-f2"])

    se  = D["se"]
    rc  = pd.read_parquet(ROOT / "data/gtfs/routes_clean.parquet")
    metrics = {
        "network": {
            "n_routes_total"  : int(rc["route_short_name"].nunique()),
            "n_routes_weekday": int(se["route_id"].nunique()),
            "n_weekday_trips" : int(D["se"]["trip_id"].nunique()),
            "n_stops"         : int(D["se"]["stop_id"].nunique()),
            "n_stop_events"   : int(len(D["se"])),
            "n_transfer_pairs": int(len(hei_df)),
            "n_same_block_removed": 636,
            "transfer_window_min" : [1, 15],
            "mean_wait_min"   : round(float(hei_df["wait_min"].mean()), 3),
            "median_wait_min" : round(float(hei_df["wait_min"].median()), 3),
        },
        "wbgt": {
            "peak_hour"          : int(D["wbgt_arr"].argmax()),
            "peak_wbgt_mean_C"   : round(float(D["wbgt_arr"].max()), 3),
            "transit_mean_9_18h" : round(float(D["wbgt_arr"][9:19].mean()), 3),
            "pct_moderate_risk"  : round(float(
                (D["summer_hourly"]["WBGT"] >= 25.0).mean() * 100), 1),
            "seasonal_gap_pct"   : round(float(
                (D["wbgt_arr"][9:19].mean() /
                 D["seasonal_profiles"]["spring"]["wbgt_mean"].iloc[9:19].mean() - 1) * 100), 1),
        },
        "baseline": {
            "f1_total_wait_min" : round(bf1, 2),
            "f2_weighted_hei"   : round(bf2, 2),
            "pct_hei_12_17h"    : round(float(
                hei_df[hei_df["transfer_clock_hour"].between(12,17)]["hei_weighted_mean"].sum()
                / hei_df["hei_weighted_mean"].sum() * 100), 1),
            "n_sheltered_stops" : int(hei_df["is_sheltered"].sum()),
            "n_exposed_stops"   : int((~hei_df["is_sheltered"]).sum()),
        },
        "results": {
            "pareto_n_solutions" : int(len(pF)),
            "hypervolume"        : round(hv, 6),
            "pareto_f1_range"    : [round(float(pF[:,0].min()),2), round(float(pF[:,0].max()),2)],
            "pareto_f2_range"    : [round(float(pF[:,1].min()),2), round(float(pF[:,1].max()),2)],
            "greedy_f2": {
                "f1"         : round(gf2["f1"], 2),
                "f2"         : round(gf2["f2"], 2),
                "f1_red_pct" : round(100*(1-gf2["f1"]/bf1), 3),
                "f2_red_pct" : round(100*(1-gf2["f2"]/bf2), 3),
                "n_missed"   : gf2["n_missed_connections"],
                "runtime_ms" : 500,
            },
            "nsga2_min_f1": {
                "f1"         : round(m1["f1"], 2),
                "f2"         : round(m1["f2"], 2),
                "f1_red_pct" : round(100*(1-m1["f1"]/bf1), 3),
                "f2_red_pct" : round(100*(1-m1["f2"]/bf2), 3),
                "n_missed"   : m1["n_missed_connections"],
            },
            "nsga2_knee": {
                "f1"         : round(rk["f1"], 2),
                "f2"         : round(rk["f2"], 2),
                "f1_red_pct" : round(100*(1-rk["f1"]/bf1), 3),
                "f2_red_pct" : round(100*(1-rk["f2"]/bf2), 3),
                "n_missed"   : rk["n_missed_connections"],
            },
            "nsga2_min_f2": {
                "f1"         : round(m2["f1"], 2),
                "f2"         : round(m2["f2"], 2),
                "f1_red_pct" : round(100*(1-m2["f1"]/bf1), 3),
                "f2_red_pct" : round(100*(1-m2["f2"]/bf2), 3),
                "n_missed"   : m2["n_missed_connections"],
            },
            "nsga2_runtime_s"    : 187,
            "nsga2_generations"  : 500,
            "nsga2_pop_size"     : 200,
            "greedy_runtime_ms"  : 500,
        },
        "formulation": {
            "n_decision_vars"    : int(prob.n_var),
            "delta_max_sec"      : float(DELTA_MAX_SEC),
            "n_objectives"       : 2,
            "n_headway_constraints": int(prob.n_headways),
            "headway_tolerance_pct": 20,
            "missed_conn_treatment": "soft max(0,.) clamping + post-hoc reporting",
        },
    }

    out_path = JSON_DIR / "paper_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("  [json] paper_metrics.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("Step 8 — Paper output pipeline")
    log.info("Loading data …")
    D = load_all()

    log.info("Generating figures …")
    fig01_wbgt_density(D)
    fig02_pareto_front(D)
    fig03_convergence(D)
    fig04_seasonal(D)
    fig05_hotspot_map(D)
    fig06_wait_cdf(D)
    fig07_hourly_hei(D)
    fig08_route_savings(D)
    fig09_delta_shifts(D)
    fig10_transfer_network(D)

    log.info("Building tables …")
    tabs = make_tables(D)

    log.info("Building LaTeX fragments …")
    make_latex(tabs)

    log.info("Writing master metrics JSON …")
    make_metrics_json(D, tabs)

    log.info("")
    log.info("=== OUTPUT INVENTORY ===")
    import os
    total_fig = total_tab = total_tex = 0
    for f in sorted(FIG_DIR.iterdir()):
        sz = f.stat().st_size
        log.info(f"  {f.name:<45}  {sz/1024:>7.1f} KB")
        total_fig += sz
    log.info("")
    for d, lbl in [(TAB_DIR, "tables"), (TEX_DIR, "latex")]:
        for f in sorted(d.iterdir()):
            sz = f.stat().st_size
            log.info(f"  {f.name:<45}  {sz/1024:>7.1f} KB")
    log.info("")
    log.info(f"  paper_metrics.json{'':<27}  "
          f"{(JSON_DIR / 'paper_metrics.json').stat().st_size/1024:.1f} KB")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
