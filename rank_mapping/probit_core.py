from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence

from lol_rank_sim import DEFAULT_TARGET_PERCENTAGES, RANK_NAMES
from probit_settings import CHALLENGER_CUTOFF_LP, GM_CUTOFF_LP, RANK1_LP


STD_NORMAL = NormalDist()
APEX_RANKS = {"Master", "GrandMaster", "Challenger"}
DEFAULT_APEX_LP_CUTOFFS = {
    "gm_cutoff_lp": GM_CUTOFF_LP,
    "challenger_cutoff_lp": CHALLENGER_CUTOFF_LP,
    "rank1_lp": RANK1_LP,
}


def safe_inv_cdf(p: float) -> float:
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return STD_NORMAL.inv_cdf(p)


def normal_pdf(z: float) -> float:
    return STD_NORMAL.pdf(z)


def truncated_normal_mean(a: float, b: float) -> float:
    pa = 0.0 if a == float("-inf") else STD_NORMAL.cdf(a)
    pb = 1.0 if b == float("inf") else STD_NORMAL.cdf(b)
    mass = pb - pa
    if mass <= 0.0:
        return float("nan")
    phi_a = 0.0 if a == float("-inf") else normal_pdf(a)
    phi_b = 0.0 if b == float("inf") else normal_pdf(b)
    return (phi_a - phi_b) / mass


def normalize_percentages(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        raise ValueError("target percentages must sum to > 0")
    return [100.0 * (v / total) for v in values]


@dataclass(frozen=True)
class RankBin:
    rank: str
    lower_pct: float
    upper_pct: float
    target_pct: float


class RankLpProbitMapper:
    def __init__(
        self,
        target_percentages: Sequence[float] = DEFAULT_TARGET_PERCENTAGES,
        rank_names: Sequence[str] = RANK_NAMES,
        floor_epsilon_pct: float = 0.01,
        ceil_epsilon_pct: float = 0.01,
        apex_lp_cutoffs: Mapping[str, float] | None = None,
    ) -> None:
        if len(target_percentages) != len(rank_names):
            raise ValueError("target_percentages and rank_names must have same length.")
        if floor_epsilon_pct < 0.0 or ceil_epsilon_pct < 0.0:
            raise ValueError("epsilon percentages must be >= 0.")
        if floor_epsilon_pct + ceil_epsilon_pct >= 100.0:
            raise ValueError("epsilon percentages are too large.")

        self.floor_epsilon_pct = float(floor_epsilon_pct)
        self.ceil_epsilon_pct = float(ceil_epsilon_pct)

        merged_cutoffs = dict(DEFAULT_APEX_LP_CUTOFFS)
        if apex_lp_cutoffs:
            for key, value in apex_lp_cutoffs.items():
                v = float(value)
                if v <= 0.0:
                    raise ValueError(f"apex cutoff must be > 0 for key {key}")
                merged_cutoffs[key] = v
        self.gm_cutoff_lp = merged_cutoffs["gm_cutoff_lp"]
        self.challenger_cutoff_lp = merged_cutoffs["challenger_cutoff_lp"]
        self.rank1_lp = merged_cutoffs["rank1_lp"]
        if not (self.gm_cutoff_lp < self.challenger_cutoff_lp < self.rank1_lp):
            raise ValueError("Apex LP cutoffs must satisfy gm < challenger < rank1.")

        target = normalize_percentages(target_percentages)
        bins: list[RankBin] = []
        by_rank: dict[str, RankBin] = {}
        cum = 0.0
        for rank, pct in zip(rank_names, target):
            lower = cum
            upper = cum + pct
            cum = upper
            rb = RankBin(rank=rank, lower_pct=lower, upper_pct=upper, target_pct=pct)
            bins.append(rb)
            by_rank[rank] = rb
        self.rank_bins = bins
        self.rank_to_bin = by_rank

    def _clip_pct(self, pct: float) -> float:
        lo = self.floor_epsilon_pct
        hi = 100.0 - self.ceil_epsilon_pct
        if pct < lo:
            return lo
        if pct > hi:
            return hi
        return pct

    @staticmethod
    def _linear_lp_fraction(lp: float) -> float:
        if lp <= 0.0:
            return 0.0
        if lp >= 100.0:
            return 1.0
        return lp / 100.0

    def _apex_lp_fraction(self, rank: str, lp: float) -> float:
        if rank == "Master":
            lo, hi = 0.0, self.gm_cutoff_lp
        elif rank == "GrandMaster":
            lo, hi = self.gm_cutoff_lp, self.challenger_cutoff_lp
        elif rank == "Challenger":
            lo, hi = self.challenger_cutoff_lp, self.rank1_lp
        else:
            raise ValueError(f"Unknown apex rank '{rank}'.")
        if lp <= lo:
            return 0.0
        if lp >= hi:
            return 1.0
        return (lp - lo) / (hi - lo)

    @staticmethod
    def _sanitize_lp(lp: float) -> float:
        # API safety: negative LP is treated as 0 LP.
        return max(0.0, float(lp))

    def _effective_rank(self, rank: str, lp: float) -> str:
        lp_value = self._sanitize_lp(lp)
        if rank == "Master" and lp_value >= self.gm_cutoff_lp:
            return "GrandMaster"
        if rank == "GrandMaster" and lp_value >= self.challenger_cutoff_lp:
            return "Challenger"
        return rank

    def rank_lp_to_percentile(self, rank: str, lp: float) -> float:
        effective_rank = self._effective_rank(rank, lp)
        if effective_rank not in self.rank_to_bin:
            raise ValueError(f"Unknown rank '{rank}'.")
        rb = self.rank_to_bin[effective_rank]
        width = rb.upper_pct - rb.lower_pct
        if width <= 0.0:
            raise ValueError(f"Non-positive rank width for rank '{effective_rank}'.")
        lp_value = self._sanitize_lp(lp)

        if effective_rank in APEX_RANKS:
            frac = self._apex_lp_fraction(effective_rank, lp_value)
        else:
            frac = self._linear_lp_fraction(lp_value)

        pct = rb.lower_pct + frac * width
        return self._clip_pct(pct)

    def rank_lp_to_probit(self, rank: str, lp: float) -> float:
        pct = self.rank_lp_to_percentile(rank, lp)
        return safe_inv_cdf(pct / 100.0)

    def rank_table(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for rb in self.rank_bins:
            a = safe_inv_cdf(rb.lower_pct / 100.0)
            b = safe_inv_cdf(rb.upper_pct / 100.0)
            mid = safe_inv_cdf(((rb.lower_pct + rb.upper_pct) / 2.0) / 100.0)
            rows.append(
                {
                    "rank": rb.rank,
                    "target_pct": rb.target_pct,
                    "lower_pct": rb.lower_pct,
                    "upper_pct": rb.upper_pct,
                    "z_lower": a,
                    "z_upper": b,
                    "z_midpoint": mid,
                    "z_bin_mean": truncated_normal_mean(a, b),
                }
            )
        return rows
