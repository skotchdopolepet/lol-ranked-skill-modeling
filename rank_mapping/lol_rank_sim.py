from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TIERS = ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond"]
DIVISIONS = ["IV", "III", "II", "I"]
RANK_NAMES = [f"{tier} {division}" for tier in TIERS for division in DIVISIONS]
RANK_NAMES.extend(["Master", "GrandMaster", "Challenger"])
N_RANKS = len(RANK_NAMES)

MASTER_IDX = N_RANKS - 3
GRANDMASTER_IDX = N_RANKS - 2
CHALLENGER_IDX = N_RANKS - 1

TIER_FLOOR_MASK = np.zeros(N_RANKS, dtype=bool)
for i, name in enumerate(RANK_NAMES):
    if name.endswith("IV"):
        TIER_FLOOR_MASK[i] = True

DEFAULT_TARGET_PERCENTAGES = (
    0.074,
    0.14,
    0.45,
    1.3,
    4.9,
    4.2,
    4.5,
    4.5,
    7.3,
    6.3,
    6.2,
    5.0,
    8.5,
    6.3,
    5.6,
    4.1,
    6.7,
    4.4,
    3.5,
    2.4,
    3.9,
    2.2,
    1.7,
    1.1,
    1.7,
    0.81,
    0.55,
    0.44,
    0.96,
    0.094,
    0.044,
)

# Division-level activity (queue probability each round).
DEFAULT_RANK_ACTIVITY_PROBS = (
    0.22,
    0.24,
    0.26,
    0.28,
    0.27,
    0.29,
    0.31,
    0.33,
    0.33,
    0.35,
    0.37,
    0.39,
    0.38,
    0.40,
    0.42,
    0.44,
    0.43,
    0.45,
    0.47,
    0.49,
    0.48,
    0.50,
    0.52,
    0.54,
    0.53,
    0.55,
    0.57,
    0.59,
    0.62,
    0.62,
    0.62,
)


@dataclass
class SimulationConfig:
    n_players: int = 100_000
    n_rounds: int = 1200
    team_size: int = 5
    seed: int = 42
    start_rank_idx: int = 7  # Bronze I
    lp_win: float = 25.0
    lp_loss: float = 25.0
    demotion_lp: float = 80.0
    # Smaller buffer => easier demotion from IV divisions.
    tier_floor_buffer_lp: float = 25.0
    # Extra protection losses allowed when losing from exactly 0 LP (non-Iron).
    zero_lp_protection_losses: int = 1
    outcome_model: str = "logistic"  # logistic | gaussian_sum
    # Used in logistic model: smaller => stronger skill impact on outcomes.
    skill_outcome_scale: float = 0.3
    # Used in gaussian_sum model.
    noise_std: float = 1.2
    # Skill prior model.
    skill_distribution: str = "normal"  # normal | normal_mixture | flat_right_skew
    # For normal_mixture: (1-w)*N(0, base_std) + w*N(0, tail_std)
    mixture_tail_weight: float = 0.10
    mixture_base_std: float = 0.85
    mixture_tail_std: float = 2.0
    target_percentages: Sequence[float] = DEFAULT_TARGET_PERCENTAGES
    rank_activity_probs: Sequence[float] = DEFAULT_RANK_ACTIVITY_PROBS
    challenger_cap: int = 44
    gm_cap: int = 94

    def __post_init__(self) -> None:
        if self.n_players <= 0:
            raise ValueError("n_players must be positive.")
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive.")
        if self.team_size <= 0:
            raise ValueError("team_size must be positive.")
        if not (0 <= self.start_rank_idx < N_RANKS):
            raise ValueError("start_rank_idx must be within rank bounds.")
        if len(self.target_percentages) != N_RANKS:
            raise ValueError(f"target_percentages must have {N_RANKS} values.")
        if len(self.rank_activity_probs) != N_RANKS:
            raise ValueError(f"rank_activity_probs must have {N_RANKS} values.")
        if any(p < 0 or p > 1 for p in self.rank_activity_probs):
            raise ValueError("rank_activity_probs values must be in [0, 1].")
        if self.lp_win <= 0 or self.lp_loss <= 0:
            raise ValueError("lp_win and lp_loss must be positive.")
        if self.demotion_lp < 0:
            raise ValueError("demotion_lp must be >= 0.")
        if self.tier_floor_buffer_lp < 0:
            raise ValueError("tier_floor_buffer_lp must be >= 0.")
        if self.zero_lp_protection_losses < 0:
            raise ValueError("zero_lp_protection_losses must be >= 0.")
        if self.outcome_model not in {"logistic", "gaussian_sum"}:
            raise ValueError("outcome_model must be 'logistic' or 'gaussian_sum'.")
        if self.skill_outcome_scale <= 0:
            raise ValueError("skill_outcome_scale must be > 0.")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be > 0.")
        if self.skill_distribution not in {"normal", "normal_mixture", "flat_right_skew"}:
            raise ValueError(
                "skill_distribution must be one of: normal, normal_mixture, flat_right_skew."
            )
        if not (0.0 <= self.mixture_tail_weight <= 1.0):
            raise ValueError("mixture_tail_weight must be in [0, 1].")
        if self.mixture_base_std <= 0 or self.mixture_tail_std <= 0:
            raise ValueError("mixture_base_std and mixture_tail_std must be > 0.")


@dataclass
class SimulationState:
    skill: np.ndarray
    rank_idx: np.ndarray
    lp: np.ndarray
    games_played: np.ndarray
    zero_lp_loss_streak: np.ndarray


@dataclass
class SimulationResult:
    rank_counts: np.ndarray
    rank_percentages: np.ndarray
    skill_by_rank_stats: dict[str, dict[str, float]]
    division_skill_numbers: dict[str, dict[str, float]]
    fit_metrics: dict[str, float]
    state_snapshot: dict[str, np.ndarray]


@dataclass
class RankMatchBatch:
    rank_idx: int
    team_a: np.ndarray
    team_b: np.ndarray


def normalize_percentages(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        raise ValueError("Percentage vector must have a positive sum.")
    return arr * (100.0 / total)


def compute_fit_metrics(sim_pct: np.ndarray, target_pct: np.ndarray) -> dict[str, float]:
    eps = 1e-3
    abs_error = np.abs(sim_pct - target_pct)
    weights = 1.0 / np.sqrt(target_pct + eps)
    weights /= weights.sum()
    weighted_rmse = float(np.sqrt(np.sum(weights * np.square(sim_pct - target_pct))))
    chi_squared = float(np.sum(np.square(sim_pct - target_pct) / (target_pct + eps)))
    return {
        "weighted_rmse": weighted_rmse,
        "chi_squared": chi_squared,
        "mae": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
    }


def initialize_state(config: SimulationConfig, rng: np.random.Generator) -> SimulationState:
    if config.skill_distribution == "normal":
        skill = rng.normal(0.0, 1.0, size=config.n_players).astype(np.float32)
    elif config.skill_distribution == "normal_mixture":
        use_tail = rng.random(config.n_players) < config.mixture_tail_weight
        skill = rng.normal(0.0, config.mixture_base_std, size=config.n_players)
        if np.any(use_tail):
            skill[use_tail] = rng.normal(0.0, config.mixture_tail_std, size=int(np.count_nonzero(use_tail)))
        skill = skill.astype(np.float32)
    else:
        # Flatter-than-normal core with slight right skew, then standardize.
        z = rng.normal(0.0, 1.0, size=config.n_players)
        u = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=config.n_players)
        core = 0.65 * z + 0.35 * u
        skew = 0.12 * (z**2 - 1.0)
        skill = core + skew
        skill = (skill - skill.mean()) / (skill.std() + 1e-9)
        skill = skill.astype(np.float32)
    rank_idx = np.full(config.n_players, config.start_rank_idx, dtype=np.int16)
    lp = np.zeros(config.n_players, dtype=np.float32)
    games_played = np.zeros(config.n_players, dtype=np.int32)
    zero_lp_loss_streak = np.zeros(config.n_players, dtype=np.int8)
    return SimulationState(
        skill=skill,
        rank_idx=rank_idx,
        lp=lp,
        games_played=games_played,
        zero_lp_loss_streak=zero_lp_loss_streak,
    )


def build_matches_by_rank(
    rank_idx: np.ndarray,
    queue_mask: np.ndarray,
    team_size: int,
    rng: np.random.Generator,
) -> list[RankMatchBatch]:
    match_size = 2 * team_size
    batches: list[RankMatchBatch] = []
    for current_rank in range(N_RANKS):
        candidates = np.flatnonzero(queue_mask & (rank_idx == current_rank))
        if candidates.size < match_size:
            continue
        rng.shuffle(candidates)
        usable = (candidates.size // match_size) * match_size
        grouped = candidates[:usable].reshape(-1, match_size)
        batches.append(
            RankMatchBatch(
                rank_idx=current_rank,
                team_a=grouped[:, :team_size],
                team_b=grouped[:, team_size:],
            )
        )
    return batches


def resolve_match_winners(
    skill: np.ndarray,
    team_a: np.ndarray,
    team_b: np.ndarray,
    outcome_model: str,
    skill_outcome_scale: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_matches = team_a.shape[0]
    if outcome_model == "gaussian_sum":
        scale = noise_std * np.sqrt(team_a.shape[1])
        a_total = skill[team_a].sum(axis=1) + rng.normal(0.0, scale, size=n_matches)
        b_total = skill[team_b].sum(axis=1) + rng.normal(0.0, scale, size=n_matches)
        return a_total > b_total

    a_mean = skill[team_a].mean(axis=1)
    b_mean = skill[team_b].mean(axis=1)
    diff = (a_mean - b_mean) / skill_outcome_scale
    p_a = 1.0 / (1.0 + np.exp(-diff))
    return rng.random(n_matches) < p_a


def process_match_outcomes(
    state: SimulationState,
    winner_ids: np.ndarray,
    loser_ids: np.ndarray,
    config: SimulationConfig,
) -> None:
    if winner_ids.size:
        state.games_played[winner_ids] += 1
        state.lp[winner_ids] += config.lp_win
        state.zero_lp_loss_streak[winner_ids] = 0
        can_promote = (state.lp[winner_ids] >= 100.0) & (state.rank_idx[winner_ids] < CHALLENGER_IDX)
        if np.any(can_promote):
            promoted_ids = winner_ids[can_promote]
            state.rank_idx[promoted_ids] += 1
            state.lp[promoted_ids] -= 100.0

    if not loser_ids.size:
        return

    state.games_played[loser_ids] += 1
    prev_rank = state.rank_idx[loser_ids].copy()
    prev_lp = state.lp[loser_ids].copy()
    state.lp[loser_ids] -= config.lp_loss
    new_lp = state.lp[loser_ids]
    needs_demotion = new_lp < 0.0

    no_demotion_ids = loser_ids[~needs_demotion]
    if no_demotion_ids.size:
        state.zero_lp_loss_streak[no_demotion_ids] = 0

    if not np.any(needs_demotion):
        return

    ids = loser_ids[needs_demotion]
    ranks = prev_rank[needs_demotion]
    prev_lp2 = prev_lp[needs_demotion]
    lp_after_loss = new_lp[needs_demotion]

    at_iron = ranks == 0
    if np.any(at_iron):
        iron_ids = ids[at_iron]
        state.rank_idx[iron_ids] = 0
        state.lp[iron_ids] = 0.0

    non_iron = ~at_iron
    if not np.any(non_iron):
        return

    ids2 = ids[non_iron]
    ranks2 = ranks[non_iron]
    prev_lp_non_iron = prev_lp2[non_iron]
    lp2 = lp_after_loss[non_iron]

    # Tier-floor shield: hold only if underflow is smaller than the buffer.
    # Strict comparison avoids infinite holds at exactly -buffer.
    floor_hold = TIER_FLOOR_MASK[ranks2] & (lp2 > -config.tier_floor_buffer_lp)

    # Universal one-step 0 LP protection for all non-Iron players.
    was_zero_lp = np.isclose(prev_lp_non_iron, 0.0)
    if np.any(was_zero_lp):
        ids_zero = ids2[was_zero_lp]
        state.zero_lp_loss_streak[ids_zero] += 1
    if np.any(~was_zero_lp):
        ids_non_zero = ids2[~was_zero_lp]
        state.zero_lp_loss_streak[ids_non_zero] = 0
    zero_hold = was_zero_lp & (
        state.zero_lp_loss_streak[ids2] <= config.zero_lp_protection_losses
    )
    hold_mask = floor_hold | zero_hold

    if np.any(hold_mask):
        hold_ids = ids2[hold_mask]
        state.lp[hold_ids] = 0.0

    if np.any(~hold_mask):
        demote_ids = ids2[~hold_mask]
        state.rank_idx[demote_ids] -= 1
        state.lp[demote_ids] = config.demotion_lp
        state.zero_lp_loss_streak[demote_ids] = 0


def apply_apex_caps(state: SimulationState, config: SimulationConfig) -> None:
    # Master is intentionally uncapped. Only GM/Challenger are hard-capped.
    master_plus = np.flatnonzero(state.rank_idx >= MASTER_IDX)
    if master_plus.size == 0:
        return
    order = master_plus[np.lexsort((master_plus, -state.games_played[master_plus], -state.lp[master_plus]))]
    state.rank_idx[master_plus] = MASTER_IDX

    n_challenger = min(config.challenger_cap, order.size)
    n_gm = min(config.gm_cap, max(0, order.size - n_challenger))
    if n_gm > 0:
        gm_ids = order[n_challenger : n_challenger + n_gm]
        state.rank_idx[gm_ids] = GRANDMASTER_IDX
    if n_challenger > 0:
        chall_ids = order[:n_challenger]
        state.rank_idx[chall_ids] = CHALLENGER_IDX


def run_simulation(config: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    state = initialize_state(config, rng)
    activity = np.asarray(config.rank_activity_probs, dtype=np.float32)

    for _ in range(config.n_rounds):
        queue_mask = rng.random(config.n_players) < activity[state.rank_idx]
        batches = build_matches_by_rank(state.rank_idx, queue_mask, config.team_size, rng)
        for batch in batches:
            a_wins = resolve_match_winners(
                skill=state.skill,
                team_a=batch.team_a,
                team_b=batch.team_b,
                outcome_model=config.outcome_model,
                skill_outcome_scale=config.skill_outcome_scale,
                noise_std=config.noise_std,
                rng=rng,
            )
            winners = np.where(a_wins[:, None], batch.team_a, batch.team_b).reshape(-1)
            losers = np.where(a_wins[:, None], batch.team_b, batch.team_a).reshape(-1)
            process_match_outcomes(state, winners, losers, config)
        apply_apex_caps(state, config)

    rank_counts = np.bincount(state.rank_idx, minlength=N_RANKS).astype(np.int64)
    rank_percentages = rank_counts / float(config.n_players) * 100.0
    target = normalize_percentages(config.target_percentages)
    fit = compute_fit_metrics(rank_percentages, target)

    return SimulationResult(
        rank_counts=rank_counts,
        rank_percentages=rank_percentages,
        skill_by_rank_stats=compute_skill_map(state.skill, state.rank_idx, RANK_NAMES),
        division_skill_numbers=compute_division_skill_numbers(state.skill, state.rank_idx, RANK_NAMES),
        fit_metrics=fit,
        state_snapshot={
            "skill": state.skill.copy(),
            "rank_idx": state.rank_idx.copy(),
            "lp": state.lp.copy(),
            "games_played": state.games_played.copy(),
            "zero_lp_loss_streak": state.zero_lp_loss_streak.copy(),
        },
    )


def compute_skill_map(
    skill: np.ndarray,
    rank_idx: np.ndarray,
    rank_names: list[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for idx, rank_name in enumerate(rank_names):
        mask = rank_idx == idx
        count = int(np.count_nonzero(mask))
        if count == 0:
            stats[rank_name] = {
                "count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "p10": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "p90": float("nan"),
            }
            continue
        values = skill[mask]
        stats[rank_name] = {
            "count": float(count),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }
    return stats


def compute_division_skill_numbers(
    skill: np.ndarray,
    rank_idx: np.ndarray,
    rank_names: list[str],
) -> dict[str, dict[str, float]]:
    global_sorted_skill = np.sort(skill)
    n = global_sorted_skill.size
    out: dict[str, dict[str, float]] = {}
    for idx, rank_name in enumerate(rank_names):
        mask = rank_idx == idx
        count = int(np.count_nonzero(mask))
        if count == 0:
            out[rank_name] = {
                "count": 0.0,
                "median_skill": float("nan"),
                "mean_skill": float("nan"),
                "skill_percentile": float("nan"),
                "skill_score_3000": float("nan"),
            }
            continue
        values = skill[mask]
        median_skill = float(np.median(values))
        mean_skill = float(np.mean(values))
        pos = int(np.searchsorted(global_sorted_skill, median_skill, side="right"))
        percentile = 100.0 * (pos / n)
        score_3000 = 3000.0 * (percentile / 100.0)
        out[rank_name] = {
            "count": float(count),
            "median_skill": median_skill,
            "mean_skill": mean_skill,
            "skill_percentile": float(percentile),
            "skill_score_3000": float(score_3000),
        }
    return out


def plot_distribution(rank_names: list[str], target_pct: np.ndarray, sim_pct: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(rank_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, target_pct, width, label="Target", alpha=0.8)
    ax.bar(x + width / 2, sim_pct, width, label="Simulated", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_names, rotation=90, fontsize=7)
    ax.set_ylabel("% of players")
    ax.set_title("Target vs Simulated Rank Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_skill_map_csv(skill_map: dict[str, dict[str, float]], out_path: Path) -> None:
    fieldnames = ["rank", "count", "mean", "std", "p10", "p25", "p50", "p75", "p90"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank_name in RANK_NAMES:
            row: dict[str, Any] = {"rank": rank_name}
            row.update(skill_map[rank_name])
            w.writerow(row)


def write_division_skill_numbers_csv(
    division_skill_numbers: dict[str, dict[str, float]], out_path: Path
) -> None:
    fieldnames = [
        "rank",
        "count",
        "median_skill",
        "mean_skill",
        "skill_percentile",
        "skill_score_3000",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank_name in RANK_NAMES:
            row: dict[str, Any] = {"rank": rank_name}
            row.update(division_skill_numbers[rank_name])
            w.writerow(row)


def print_distribution_table(
    sim_pct: np.ndarray,
    target_pct: np.ndarray,
    skill_by_rank_stats: dict[str, dict[str, float]],
) -> None:
    print("\n--- Rank Distribution ---")
    print(f"{'Rank':>20s} {'Target%':>10s} {'Sim%':>10s} {'AbsDelta':>10s} {'AvgSkill':>10s}")
    for i, name in enumerate(RANK_NAMES):
        target = target_pct[i]
        sim = sim_pct[i]
        avg_skill = skill_by_rank_stats[name]["mean"]
        avg_skill_s = "nan" if np.isnan(avg_skill) else f"{avg_skill:.3f}"
        print(f"{name:>20s} {target:10.3f} {sim:10.3f} {abs(sim-target):10.3f} {avg_skill_s:>10s}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoL ranked ladder simulator")
    p.add_argument("--players", type=int, default=100_000)
    p.add_argument("--rounds", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-rank-idx", type=int, default=7)
    p.add_argument("--lp-win", type=float, default=25.0)
    p.add_argument("--lp-loss", type=float, default=25.0)
    p.add_argument("--demotion-lp", type=float, default=80.0)
    p.add_argument("--tier-floor-buffer-lp", type=float, default=25.0)
    p.add_argument("--zero-lp-protection-losses", type=int, default=1)
    p.add_argument("--outcome-model", choices=["logistic", "gaussian_sum"], default="logistic")
    p.add_argument("--skill-outcome-scale", type=float, default=0.3)
    p.add_argument("--noise-std", type=float, default=1.2)
    p.add_argument(
        "--skill-distribution",
        choices=["normal", "normal_mixture", "flat_right_skew"],
        default="normal",
    )
    p.add_argument("--mixture-tail-weight", type=float, default=0.10)
    p.add_argument("--mixture-base-std", type=float, default=0.85)
    p.add_argument("--mixture-tail-std", type=float, default=2.0)
    p.add_argument("--out-dir", type=str, default="runtime/out_latest")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        n_players=args.players,
        n_rounds=args.rounds,
        seed=args.seed,
        start_rank_idx=args.start_rank_idx,
        lp_win=args.lp_win,
        lp_loss=args.lp_loss,
        demotion_lp=args.demotion_lp,
        tier_floor_buffer_lp=args.tier_floor_buffer_lp,
        zero_lp_protection_losses=args.zero_lp_protection_losses,
        outcome_model=args.outcome_model,
        skill_outcome_scale=args.skill_outcome_scale,
        noise_std=args.noise_std,
        skill_distribution=args.skill_distribution,
        mixture_tail_weight=args.mixture_tail_weight,
        mixture_base_std=args.mixture_base_std,
        mixture_tail_std=args.mixture_tail_std,
    )
    result = run_simulation(config)
    target = normalize_percentages(config.target_percentages)

    print_distribution_table(result.rank_percentages, target, result.skill_by_rank_stats)
    print("\n--- Fit Metrics ---")
    for k, v in result.fit_metrics.items():
        print(f"{k}: {v:.6f}")

    plot_path = out_dir / "rank_distribution_comparison.png"
    skill_path = out_dir / "skill_map.csv"
    division_path = out_dir / "division_skill_numbers.csv"
    npz_path = out_dir / "final_state.npz"

    plot_distribution(RANK_NAMES, target, result.rank_percentages, str(plot_path))
    write_skill_map_csv(result.skill_by_rank_stats, skill_path)
    write_division_skill_numbers_csv(result.division_skill_numbers, division_path)
    np.savez_compressed(npz_path, **result.state_snapshot)

    print("\n--- Outputs ---")
    print(f"Plot: {plot_path}")
    print(f"Skill map: {skill_path}")
    print(f"Division skill numbers: {division_path}")
    print(f"State dump: {npz_path}")


if __name__ == "__main__":
    main()
