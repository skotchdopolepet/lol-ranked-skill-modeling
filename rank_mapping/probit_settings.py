from __future__ import annotations

from lol_rank_sim import DEFAULT_TARGET_PERCENTAGES

# Default global fallback cutoffs.
GM_CUTOFF_LP = 1000.0
CHALLENGER_CUTOFF_LP = 1250.0
RANK1_LP = 2400.0

# Robust server-level apex cutoffs derived from out_prod aggregated ladder snapshots.
# Interpretation:
# - gm_cutoff_lp = Grandmaster p05
# - challenger_cutoff_lp = Challenger p05
# - rank1_lp = Challenger p99
SERVER_APEX_LP_CUTOFFS = {
    "BR1": {"gm_cutoff_lp": 685.0, "challenger_cutoff_lp": 1033.4, "rank1_lp": 2014.4},
    "EUN1": {"gm_cutoff_lp": 664.0, "challenger_cutoff_lp": 948.0, "rank1_lp": 2200.2},
    "EUW1": {"gm_cutoff_lp": 1029.0, "challenger_cutoff_lp": 1366.0, "rank1_lp": 2303.0},
    "JP1": {"gm_cutoff_lp": 705.5, "challenger_cutoff_lp": 1007.4, "rank1_lp": 1581.7},
    "KR": {"gm_cutoff_lp": 1030.0, "challenger_cutoff_lp": 1291.0, "rank1_lp": 2006.0},
    "LA1": {"gm_cutoff_lp": 598.2, "challenger_cutoff_lp": 943.0, "rank1_lp": 2300.6},
    "LA2": {"gm_cutoff_lp": 577.4, "challenger_cutoff_lp": 907.4, "rank1_lp": 1849.5},
    "NA1": {"gm_cutoff_lp": 626.0, "challenger_cutoff_lp": 969.6, "rank1_lp": 2351.0},
    "OC1": {"gm_cutoff_lp": 478.0, "challenger_cutoff_lp": 808.0, "rank1_lp": 1851.7},
    "SG2": {"gm_cutoff_lp": 227.0, "challenger_cutoff_lp": 575.0, "rank1_lp": 1721.9},
    "ME1": {"gm_cutoff_lp": 478.0, "challenger_cutoff_lp": 808.0, "rank1_lp": 1851.7},
    "TR1": {"gm_cutoff_lp": 478.0, "challenger_cutoff_lp": 808.0, "rank1_lp": 1851.7},
    "TW2": {"gm_cutoff_lp": 245.0, "challenger_cutoff_lp": 643.0, "rank1_lp": 1600.3},
    "VN2": {"gm_cutoff_lp": 702.0, "challenger_cutoff_lp": 1026.0, "rank1_lp": 2085.0},
}

# Per-server Soloqueue rank shares from League of Graphs, captured on 2026-04-06.
# Values are per-division percentages in lol_rank_sim.RANK_NAMES order:
# Iron IV .. Challenger. They are intentionally left unnormalized because the
# site rounds displayed percentages; RankLpProbitMapper normalizes them.
SERVER_TARGET_PERCENTAGES = {
    "BR1": (
        0.14, 0.21, 0.55, 1.3, 4.3, 3.8, 4.1, 4.0, 6.9, 6.0, 6.0, 4.8, 8.8,
        6.3, 5.7, 4.0, 7.2, 4.6, 3.7, 2.5, 4.4, 2.5, 1.8, 1.7, 1.7, 0.75,
        0.57, 0.56, 1.1, 0.063, 0.025,
    ),
    "EUN1": (
        0.18, 0.27, 0.65, 1.4, 4.5, 3.8, 4.2, 4.0, 6.7, 5.8, 5.6, 4.5, 8.5,
        6.2, 5.6, 4.0, 7.3, 4.6, 3.7, 2.5, 4.8, 2.6, 1.9, 1.7, 1.8, 0.76,
        0.62, 0.63, 1.1, 0.049, 0.019,
    ),
    "EUW1": (
        0.24, 0.28, 0.67, 1.3, 4.3, 3.8, 4.1, 3.9, 6.8, 5.9, 5.8, 4.7, 8.9,
        6.4, 5.6, 4.0, 7.3, 4.5, 3.6, 2.4, 4.5, 2.5, 1.8, 1.7, 1.7, 0.75,
        0.61, 0.65, 1.2, 0.03, 0.013,
    ),
    "JP1": (
        0.41, 0.55, 1.1, 2.1, 5.0, 3.9, 4.2, 3.9, 6.8, 5.5, 5.5, 4.3, 8.3,
        6.0, 5.5, 4.1, 7.5, 4.5, 3.7, 2.3, 4.5, 2.3, 1.7, 1.6, 1.7, 0.72,
        0.61, 0.55, 1.1, 0.053, 0.026,
    ),
    "KR": (
        0.19, 0.26, 0.66, 1.5, 4.6, 3.9, 4.2, 4.0, 6.9, 5.9, 5.8, 4.7, 8.6,
        6.1, 5.5, 3.9, 7.2, 4.5, 3.6, 2.4, 4.6, 2.4, 1.8, 1.8, 1.7, 0.75,
        0.61, 0.88, 0.96, 0.032, 0.014,
    ),
    "LA1": (
        0.14, 0.22, 0.58, 1.3, 4.3, 3.7, 4.1, 3.9, 6.7, 5.8, 5.7, 4.6, 8.7,
        6.3, 5.7, 4.2, 7.4, 4.6, 3.8, 2.6, 4.6, 2.6, 1.9, 1.6, 1.9, 0.83,
        0.57, 0.53, 1.1, 0.095, 0.038,
    ),
    "LA2": (
        0.14, 0.23, 0.61, 1.4, 4.4, 3.9, 4.2, 4.1, 6.9, 6.0, 5.9, 4.7, 8.7,
        6.3, 5.6, 4.1, 7.1, 4.5, 3.7, 2.5, 4.4, 2.4, 1.8, 1.6, 1.7, 0.76,
        0.58, 0.61, 1.0, 0.091, 0.036,
    ),
    "NA1": (
        0.23, 0.28, 0.65, 1.5, 4.5, 3.9, 4.3, 4.2, 7.0, 6.1, 6.0, 4.8, 9.0,
        6.3, 5.6, 4.0, 7.2, 4.4, 3.5, 2.3, 4.4, 2.3, 1.7, 1.5, 1.6, 0.7,
        0.55, 0.57, 1.0, 0.071, 0.03,
    ),
    "OC1": (
        0.22, 0.28, 0.69, 1.4, 4.4, 3.9, 4.3, 4.1, 6.9, 6.2, 6.0, 4.8, 8.9,
        6.4, 5.6, 4.0, 7.2, 4.3, 3.4, 2.4, 4.5, 2.3, 1.7, 1.7, 1.6, 0.71,
        0.52, 0.57, 0.96, 0.084, 0.042,
    ),
    "SG2": (
        0.12, 0.19, 0.52, 1.4, 4.7, 4.0, 4.4, 4.4, 6.8, 6.0, 6.0, 4.9, 8.1,
        6.2, 5.5, 4.2, 6.8, 4.6, 3.8, 2.6, 4.3, 2.5, 1.8, 1.7, 1.6, 0.76,
        0.55, 0.5, 0.74, 0.35, 0.15,
    ),
    "ME1": (
        0.17, 0.27, 0.6, 1.4, 4.7, 3.7, 4.0, 3.9, 6.2, 5.5, 5.5, 4.2, 8.0,
        6.5, 5.6, 4.3, 7.2, 4.9, 3.8, 2.9, 4.8, 2.7, 1.9, 1.9, 1.9, 0.73,
        0.65, 0.64, 0.85, 0.32, 0.16,
    ),
    "TR1": (
        0.16, 0.26, 0.69, 1.5, 4.4, 3.8, 4.2, 4.1, 6.5, 5.8, 5.8, 4.6, 8.0,
        6.3, 5.8, 4.2, 7.0, 4.6, 3.9, 2.6, 4.7, 2.6, 2.0, 1.8, 1.8, 0.75,
        0.6, 0.64, 0.95, 0.1, 0.038,
    ),
    "TW2": (
        0.16, 0.15, 0.44, 1.2, 4.5, 4.1, 4.4, 4.2, 6.8, 5.9, 5.9, 4.6, 8.8,
        6.2, 5.6, 3.9, 7.5, 4.4, 3.6, 2.5, 4.5, 2.3, 1.7, 1.6, 1.8, 0.78,
        0.56, 0.52, 0.81, 0.31, 0.12,
    ),
    "VN2": (
        0.091, 0.14, 0.48, 1.4, 4.2, 3.6, 3.9, 4.0, 6.0, 5.5, 5.7, 4.7, 7.6,
        6.0, 5.7, 4.3, 7.2, 4.9, 4.1, 2.9, 4.7, 2.8, 2.1, 1.8, 1.9, 0.92,
        0.7, 0.67, 1.6, 0.12, 0.051,
    ),
}

# Default location for latest generated artifacts.
LATEST_OUT_DIR = "runtime/out_latest/analysis/simulations"


def target_percentages() -> tuple[float, ...]:
    return tuple(float(v) for v in DEFAULT_TARGET_PERCENTAGES)


def target_percentages_for_server(server: str | None) -> tuple[float, ...]:
    if server:
        key = str(server).upper()
        if key in SERVER_TARGET_PERCENTAGES:
            return tuple(float(v) for v in SERVER_TARGET_PERCENTAGES[key])
    return target_percentages()


def apex_lp_cutoffs() -> dict[str, float]:
    return {
        "gm_cutoff_lp": GM_CUTOFF_LP,
        "challenger_cutoff_lp": CHALLENGER_CUTOFF_LP,
        "rank1_lp": RANK1_LP,
    }


def apex_lp_cutoffs_for_server(server: str | None) -> dict[str, float]:
    if server:
        key = str(server).upper()
        if key in SERVER_APEX_LP_CUTOFFS:
            return dict(SERVER_APEX_LP_CUTOFFS[key])
    return apex_lp_cutoffs()
