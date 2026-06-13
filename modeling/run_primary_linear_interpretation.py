from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "modeling", REPO_ROOT / "rank_mapping"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_primary_linear_analysis as linear
from lol_rank_sim import RANK_NAMES
from probit_core import RankLpProbitMapper
from probit_settings import apex_lp_cutoffs_for_server, target_percentages_for_server


DEFAULT_MODELING_ROOT = REPO_ROOT / "out_prod" / "primary_dataset" / "modeling"
DEFAULT_INTERPRETATION_ROOT = DEFAULT_MODELING_ROOT / "interpretation"
DEFAULT_SOURCE_RUN = DEFAULT_MODELING_ROOT / "runs" / "20260411_192856_linear_primary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret the primary linear benchmark in more human-readable units.")
    parser.add_argument("--db-path", type=Path, default=linear.DEFAULT_DB_PATH)
    parser.add_argument("--table-name", type=str, default=linear.DEFAULT_TABLE_NAME)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_INTERPRETATION_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--pair-sample-size", type=int, default=500000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--top-coefficients", type=int, default=20)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_source_results(source_run_dir: Path) -> dict[str, Any]:
    results_path = source_run_dir / "results.json"
    split_path = source_run_dir / "split_arrays.npz"
    split_info_path = source_run_dir / "split_info.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing linear results.json: {results_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing linear split_arrays.npz: {split_path}")
    if not split_info_path.exists():
        raise FileNotFoundError(f"Missing linear split_info.json: {split_info_path}")
    return {
        "results": json.loads(results_path.read_text(encoding="utf-8")),
        "split_arrays": np.load(split_path),
        "split_info": json.loads(split_info_path.read_text(encoding="utf-8")),
    }


def phi(z: np.ndarray) -> np.ndarray:
    z64 = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + erf(z64 / math.sqrt(2.0)))


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.asarray([], dtype=np.float64)
    return np.quantile(np.asarray(values, dtype=np.float64), q=np.linspace(1.0 / n_bins, 1.0 - (1.0 / n_bins), n_bins - 1))


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(values.shape[0], dtype=np.int16)
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def predict_on_indices(
    *,
    X: np.memmap,
    indices: np.ndarray,
    beta: np.ndarray,
    intercept: float,
    chunk_size: int,
) -> np.ndarray:
    beta64 = np.asarray(beta, dtype=np.float64)
    out = np.empty(indices.shape[0], dtype=np.float64)
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])
        batch_idx = indices[start:end]
        X_batch = np.asarray(X[batch_idx], dtype=np.float64)
        out[start:end] = X_batch @ beta64 + float(intercept)
    return out


def fit_final_models(
    *,
    dev_summary: linear.SummaryStats,
    source_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fit_bundle: dict[str, dict[str, Any]] = {}
    zeros = np.zeros_like(dev_summary.sum_x, dtype=np.float64)
    fit_bundle["naive_mean"] = {
        "fit": linear.LinearFit(beta=zeros, intercept=float(dev_summary.mean_y()), variant="constant", model_name="naive"),
    }
    fit_bundle["ols_raw"] = {"fit": linear.fit_ols(dev_summary, standardized=False)}
    fit_bundle["ols_standardized"] = {"fit": linear.fit_ols(dev_summary, standardized=True)}
    fit_bundle["ridge_raw"] = {
        "fit": linear.fit_ridge(
            dev_summary,
            standardized=False,
            alpha=float(source_results["results"]["results"]["ridge_raw"]["best_alpha"]),
        )
    }
    fit_bundle["ridge_standardized"] = {
        "fit": linear.fit_ridge(
            dev_summary,
            standardized=True,
            alpha=float(source_results["results"]["results"]["ridge_standardized"]["best_alpha"]),
        )
    }
    fit_bundle["lasso_raw"] = {
        "fit": linear.fit_lasso(
            dev_summary,
            standardized=False,
            alpha=float(source_results["results"]["results"]["lasso_raw"]["best_alpha"]),
            warm_start=None,
            max_iter=5000,
            tol=1e-7,
        )
    }
    fit_bundle["lasso_standardized"] = {
        "fit": linear.fit_lasso(
            dev_summary,
            standardized=True,
            alpha=float(source_results["results"]["results"]["lasso_standardized"]["best_alpha"]),
            warm_start=None,
            max_iter=5000,
            tol=1e-7,
        )
    }
    return fit_bundle


def residual_profile(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = np.asarray(y_pred - y_true, dtype=np.float64)
    abs_err = np.abs(err)
    pct_true = 100.0 * phi(y_true)
    pct_pred = 100.0 * phi(y_pred)
    pct_abs_err = np.abs(pct_pred - pct_true)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    pred_mean = float(np.mean(y_pred))
    actual_mean = float(np.mean(y_true))
    var_pred = float(np.var(y_pred))
    if var_pred <= 0.0:
        calibration_slope = float("nan")
        calibration_intercept = float("nan")
    else:
        cov = float(np.mean((y_pred - pred_mean) * (y_true - actual_mean)))
        calibration_slope = cov / var_pred
        calibration_intercept = actual_mean - calibration_slope * pred_mean
    return {
        "correlation": corr,
        "calibration_slope_y_on_pred": float(calibration_slope),
        "calibration_intercept_y_on_pred": float(calibration_intercept),
        "z_error": {
            "mean_signed": float(np.mean(err)),
            "mae": float(np.mean(abs_err)),
            "median_abs": float(np.median(abs_err)),
            "p90_abs": float(np.quantile(abs_err, 0.90)),
            "p95_abs": float(np.quantile(abs_err, 0.95)),
            "within_0p25": float(np.mean(abs_err <= 0.25)),
            "within_0p50": float(np.mean(abs_err <= 0.50)),
            "within_0p75": float(np.mean(abs_err <= 0.75)),
            "within_1p00": float(np.mean(abs_err <= 1.00)),
        },
        "equivalent_percentile_error_pp": {
            "mae": float(np.mean(pct_abs_err)),
            "median_abs": float(np.median(pct_abs_err)),
            "p90_abs": float(np.quantile(pct_abs_err, 0.90)),
            "p95_abs": float(np.quantile(pct_abs_err, 0.95)),
            "within_5pp": float(np.mean(pct_abs_err <= 5.0)),
            "within_10pp": float(np.mean(pct_abs_err <= 10.0)),
            "within_15pp": float(np.mean(pct_abs_err <= 15.0)),
            "within_20pp": float(np.mean(pct_abs_err <= 20.0)),
        },
    }


def tier_index_from_rank_idx(rank_idx: np.ndarray) -> np.ndarray:
    out = np.empty(rank_idx.shape[0], dtype=np.int16)
    mask_regular = rank_idx < 28
    out[mask_regular] = (rank_idx[mask_regular] // 4).astype(np.int16)
    out[~mask_regular] = (rank_idx[~mask_regular] - 21).astype(np.int16)
    return out


def build_server_rank_edges(server_values: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for server in server_values:
        mapper = RankLpProbitMapper(
            target_percentages=target_percentages_for_server(server),
            apex_lp_cutoffs=apex_lp_cutoffs_for_server(server),
            floor_epsilon_pct=0.01,
            ceil_epsilon_pct=0.01,
        )
        upper = np.asarray([float(row["upper_pct"]) for row in mapper.rank_table()], dtype=np.float64)
        upper[-1] = 100.0
        out[str(server)] = upper
    return out


def equivalent_rank_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    server_names: np.ndarray,
    server_edges: dict[str, np.ndarray],
) -> dict[str, Any]:
    pct_true = np.clip(100.0 * phi(y_true), 0.0, 100.0)
    pct_pred = np.clip(100.0 * phi(y_pred), 0.0, 100.0)
    rank_true = np.empty(y_true.shape[0], dtype=np.int16)
    rank_pred = np.empty(y_true.shape[0], dtype=np.int16)
    for server in np.unique(server_names):
        mask = server_names == server
        edges = server_edges[str(server)]
        rank_true[mask] = np.minimum(np.searchsorted(edges, pct_true[mask], side="left"), len(edges) - 1).astype(np.int16)
        rank_pred[mask] = np.minimum(np.searchsorted(edges, pct_pred[mask], side="left"), len(edges) - 1).astype(np.int16)
    tier_true = tier_index_from_rank_idx(rank_true)
    tier_pred = tier_index_from_rank_idx(rank_pred)
    abs_div_gap = np.abs(rank_pred.astype(np.int32) - rank_true.astype(np.int32))
    abs_tier_gap = np.abs(tier_pred.astype(np.int32) - tier_true.astype(np.int32))
    return {
        "mean_abs_division_gap": float(np.mean(abs_div_gap)),
        "median_abs_division_gap": float(np.median(abs_div_gap)),
        "p90_abs_division_gap": float(np.quantile(abs_div_gap, 0.90)),
        "exact_division_accuracy": float(np.mean(abs_div_gap == 0)),
        "within_1_division_accuracy": float(np.mean(abs_div_gap <= 1)),
        "within_4_divisions_accuracy": float(np.mean(abs_div_gap <= 4)),
        "exact_tier_accuracy": float(np.mean(abs_tier_gap == 0)),
        "within_1_tier_accuracy": float(np.mean(abs_tier_gap <= 1)),
    }


def pairwise_ordering_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    n = int(y_true.shape[0])
    rng = np.random.default_rng(random_state)
    left = rng.integers(0, n, size=sample_size, endpoint=False)
    right = rng.integers(0, n, size=sample_size, endpoint=False)
    neq = left != right
    left = left[neq]
    right = right[neq]
    actual_gap = y_true[left] - y_true[right]
    pred_gap = y_pred[left] - y_pred[right]
    valid = actual_gap != 0.0
    actual_gap = actual_gap[valid]
    pred_gap = pred_gap[valid]

    def score(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        prod = actual_gap[mask] * pred_gap[mask]
        tied = pred_gap[mask] == 0.0
        correct = np.mean((prod > 0.0).astype(np.float64) + 0.5 * tied.astype(np.float64))
        return float(correct)

    all_mask = np.ones(actual_gap.shape[0], dtype=bool)
    gap_025 = np.abs(actual_gap) >= 0.25
    gap_050 = np.abs(actual_gap) >= 0.50
    gap_100 = np.abs(actual_gap) >= 1.00
    return {
        "sample_pairs_used": int(actual_gap.shape[0]),
        "all_pairs_accuracy": score(all_mask),
        "gap_ge_0p25_accuracy": score(gap_025),
        "gap_ge_0p50_accuracy": score(gap_050),
        "gap_ge_1p00_accuracy": score(gap_100),
        "pair_fraction_gap_ge_0p25": float(np.mean(gap_025)),
        "pair_fraction_gap_ge_0p50": float(np.mean(gap_050)),
        "pair_fraction_gap_ge_1p00": float(np.mean(gap_100)),
    }


def grouped_mean_profile(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, Any]]:
    err = y_pred - y_true
    pct_abs_err = np.abs(100.0 * phi(y_pred) - 100.0 * phi(y_true))
    rows: list[dict[str, Any]] = []
    for bin_id in range(int(np.max(bins)) + 1):
        mask = bins == bin_id
        if not np.any(mask):
            continue
        lower = float("-inf") if bin_id == 0 else float(edges[bin_id - 1])
        upper = float("inf") if bin_id >= edges.shape[0] else float(edges[bin_id])
        rows.append(
            {
                "bin": int(bin_id),
                "lower": lower,
                "upper": upper,
                "count": int(np.sum(mask)),
                "actual_mean": float(np.mean(y_true[mask])),
                "pred_mean": float(np.mean(y_pred[mask])),
                "rmse": float(math.sqrt(np.mean(np.square(err[mask])))),
                "mae": float(np.mean(np.abs(err[mask]))),
                "equivalent_percentile_mae_pp": float(np.mean(pct_abs_err[mask])),
            }
        )
    return rows


def coefficient_family(name: str) -> str:
    if name.startswith(("top_", "jungle_", "middle_", "bottom_", "support_")):
        stem = name.split("_", 1)[1]
    else:
        stem = name
    if stem.endswith("_pm_avg"):
        return "per_minute_average"
    if stem.endswith("_pm_delta"):
        return "per_minute_delta"
    if stem.endswith("_share_avg"):
        return "share_average"
    if stem.endswith("_delta"):
        return "delta_other"
    if stem.endswith("_avg"):
        return "average_other"
    return "context_or_info"


def coefficient_role(name: str) -> str:
    for prefix in ("top_", "jungle_", "middle_", "bottom_", "support_"):
        if name.startswith(prefix):
            return prefix[:-1]
    return "context"


def coefficient_tables(
    *,
    predictor_columns: list[str],
    std_x: np.ndarray,
    fit: linear.LinearFit,
    top_n: int,
) -> dict[str, Any]:
    standardized_coef = np.asarray(fit.beta, dtype=np.float64) * np.asarray(std_x, dtype=np.float64)
    abs_coef = np.abs(standardized_coef)
    order = np.argsort(-abs_coef)
    top_rows: list[dict[str, Any]] = []
    for idx in order[:top_n]:
        top_rows.append(
            {
                "feature": predictor_columns[int(idx)],
                "standardized_coef": float(standardized_coef[int(idx)]),
                "abs_standardized_coef": float(abs_coef[int(idx)]),
                "role_group": coefficient_role(predictor_columns[int(idx)]),
                "family_group": coefficient_family(predictor_columns[int(idx)]),
            }
        )
    role_groups: dict[str, float] = {}
    family_groups: dict[str, float] = {}
    non_zero_count = int(np.sum(abs_coef > 1e-12))
    for idx, name in enumerate(predictor_columns):
        role_key = coefficient_role(name)
        family_key = coefficient_family(name)
        role_groups[role_key] = role_groups.get(role_key, 0.0) + float(abs_coef[idx])
        family_groups[family_key] = family_groups.get(family_key, 0.0) + float(abs_coef[idx])
    return {
        "non_zero_coefficients": non_zero_count,
        "top_features": top_rows,
        "abs_standardized_coef_sum_by_role": dict(sorted(role_groups.items())),
        "abs_standardized_coef_sum_by_family": dict(sorted(family_groups.items())),
    }


def render_summary(payload: dict[str, Any]) -> str:
    metrics = payload["model_metrics"]
    representative = payload["representative_model"]
    calibration_rows = payload["representative_calibration_by_pred_bin"]
    actual_band_rows = payload["representative_profile_by_actual_bin"]
    coef_ols = payload["coefficient_interpretation"]["ols_standardized"]["top_features"]
    coef_lasso = payload["coefficient_interpretation"]["lasso_standardized"]["top_features"]

    lines = [
        "# Primary Linear Interpretation",
        "",
        f"- Source linear run: `{payload['meta']['source_run_dir']}`",
        f"- Interpretation run: `{payload['meta']['run_dir']}`",
        f"- Test rows: `{payload['meta']['test_rows']:,}`",
        f"- Pairwise sample size: `{payload['meta']['pair_sample_size']:,}`",
        "",
        "## Equivalent-Scale Note",
        "",
        "- `average_skill_level` is the mean of participant probit skill values, so `Phi(y)` is the percentile of a hypothetical equally skilled player with the same latent score",
        "- That percentile is not the arithmetic mean of participant percentiles, but it is the cleanest monotone way to translate the target back into ladder space",
        "",
        "## Model Interpretation Table",
        "",
        "| Model | Test RMSE | Corr(y,pred) | Percentile MAE (pp) | Within +/- 0.50 z | Within +/- 10 pp | Mean abs division gap | Exact tier acc | Pairwise acc | Pairwise acc (gap>=1.0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered_keys = [
        "naive_mean",
        "ols_raw",
        "ols_standardized",
        "ridge_raw",
        "ridge_standardized",
        "lasso_raw",
        "lasso_standardized",
    ]
    for key in ordered_keys:
        row = metrics[key]
        lines.append(
            f"| {key} | {row['test_rmse']:.6f} | {row['residual_profile']['correlation']:.6f} | "
            f"{row['residual_profile']['equivalent_percentile_error_pp']['mae']:.3f} | "
            f"{100.0 * row['residual_profile']['z_error']['within_0p50']:.2f}% | "
            f"{100.0 * row['residual_profile']['equivalent_percentile_error_pp']['within_10pp']:.2f}% | "
            f"{row['equivalent_rank_metrics']['mean_abs_division_gap']:.3f} | "
            f"{100.0 * row['equivalent_rank_metrics']['exact_tier_accuracy']:.2f}% | "
            f"{100.0 * row['pairwise_ordering']['all_pairs_accuracy']:.2f}% | "
            f"{100.0 * row['pairwise_ordering']['gap_ge_1p00_accuracy']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Representative Model",
            "",
            f"- Deep-dive model: `{representative}`",
            f"- Calibration slope `y ~ pred`: `{metrics[representative]['residual_profile']['calibration_slope_y_on_pred']:.4f}`",
            f"- Calibration intercept `y ~ pred`: `{metrics[representative]['residual_profile']['calibration_intercept_y_on_pred']:.4f}`",
            f"- Median absolute z-error: `{metrics[representative]['residual_profile']['z_error']['median_abs']:.4f}`",
            f"- 90th percentile absolute z-error: `{metrics[representative]['residual_profile']['z_error']['p90_abs']:.4f}`",
            f"- Median equivalent-percentile error: `{metrics[representative]['residual_profile']['equivalent_percentile_error_pp']['median_abs']:.3f}` pp",
            f"- 90th percentile equivalent-percentile error: `{metrics[representative]['residual_profile']['equivalent_percentile_error_pp']['p90_abs']:.3f}` pp",
            "",
            "Calibration by predicted decile:",
            "",
            "| Pred bin | Count | Mean pred y | Mean actual y | RMSE | MAE | Percentile MAE (pp) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in calibration_rows:
        lines.append(
            f"| {row['bin']} | {row['count']} | {row['pred_mean']:.4f} | {row['actual_mean']:.4f} | "
            f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['equivalent_percentile_mae_pp']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Error profile by actual decile:",
            "",
            "| Actual bin | Count | Mean actual y | Mean pred y | RMSE | MAE | Percentile MAE (pp) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in actual_band_rows:
        lines.append(
            f"| {row['bin']} | {row['count']} | {row['actual_mean']:.4f} | {row['pred_mean']:.4f} | "
            f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['equivalent_percentile_mae_pp']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Top standardized OLS features by absolute coefficient:",
            "",
            "| Feature | Std coef | Role | Family |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in coef_ols:
        lines.append(
            f"| {row['feature']} | {row['standardized_coef']:.5f} | {row['role_group']} | {row['family_group']} |"
        )
    lines.extend(
        [
            "",
            "Top standardized Lasso features by absolute coefficient:",
            "",
            "| Feature | Std coef | Role | Family |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in coef_lasso:
        lines.append(
            f"| {row['feature']} | {row['standardized_coef']:.5f} | {row['role_group']} | {row['family_group']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_bundle = load_source_results(args.source_run_dir)
    run_name = args.run_name.strip() or f"{utc_stamp()}_linear_interpretation_v1"
    run_dir = args.output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=20000,
    )
    arrays = linear.load_cached_arrays(meta)
    X = arrays["X"]
    y = arrays["y"]
    server_codes = arrays["server_codes"]

    split_arrays = source_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    test_idx = np.asarray(split_arrays["test_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)

    summary_bundle = linear.accumulate_summaries(
        X=X,
        y=y,
        test_idx=test_idx,
        fold_ids=fold_ids,
        n_folds=int(source_bundle["split_info"]["n_folds"]),
        chunk_size=20000,
    )
    dev_summary: linear.SummaryStats = summary_bundle["dev_summary"]
    std_x = np.sqrt(np.maximum(np.diag(dev_summary.centered()[0]) / float(dev_summary.n), 1e-12))

    fit_bundle = fit_final_models(dev_summary=dev_summary, source_results=source_bundle)

    server_values = [str(v) for v in meta["server_values"]]
    server_names = np.asarray(server_values, dtype=object)[server_codes[test_idx]]
    server_edges = build_server_rank_edges(server_values)

    y_test = np.asarray(y[test_idx], dtype=np.float64)
    predictor_columns = [str(v) for v in meta["predictor_columns"]]

    model_metrics: dict[str, Any] = {}
    representative = "ols_standardized"
    predictions_root = run_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    for model_key, bundle in fit_bundle.items():
        fit: linear.LinearFit = bundle["fit"]
        preds = predict_on_indices(X=X, indices=test_idx, beta=fit.beta, intercept=fit.intercept, chunk_size=args.chunk_size)
        np.save(predictions_root / f"{model_key}_test_predictions.npy", preds)
        residuals = residual_profile(y_test, preds)
        eq_rank = equivalent_rank_metrics(
            y_true=y_test,
            y_pred=preds,
            server_names=server_names,
            server_edges=server_edges,
        )
        pairwise = pairwise_ordering_metrics(
            y_true=y_test,
            y_pred=preds,
            sample_size=args.pair_sample_size,
            random_state=args.random_state,
        )
        model_metrics[model_key] = {
            "test_rmse": float(math.sqrt(np.mean(np.square(preds - y_test)))),
            "test_mae": float(np.mean(np.abs(preds - y_test))),
            "test_r2": float(1.0 - np.sum(np.square(preds - y_test)) / np.sum(np.square(y_test - np.mean(y_test)))),
            "residual_profile": residuals,
            "equivalent_rank_metrics": eq_rank,
            "pairwise_ordering": pairwise,
        }

    rep_preds = np.load(predictions_root / f"{representative}_test_predictions.npy")
    pred_edges = quantile_edges(rep_preds, args.calibration_bins)
    pred_bins = assign_bins(rep_preds, pred_edges)
    actual_edges = quantile_edges(y_test, args.calibration_bins)
    actual_bins = assign_bins(y_test, actual_edges)

    coefficient_interpretation = {
        "ols_standardized": coefficient_tables(
            predictor_columns=predictor_columns,
            std_x=std_x,
            fit=fit_bundle["ols_standardized"]["fit"],
            top_n=args.top_coefficients,
        ),
        "lasso_standardized": coefficient_tables(
            predictor_columns=predictor_columns,
            std_x=std_x,
            fit=fit_bundle["lasso_standardized"]["fit"],
            top_n=args.top_coefficients,
        ),
    }

    payload = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_run_dir": str(args.source_run_dir),
            "run_dir": str(run_dir),
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
            "test_rows": int(test_idx.shape[0]),
            "pair_sample_size": int(args.pair_sample_size),
            "random_state": int(args.random_state),
            "elapsed_seconds": float(time.time() - t0),
        },
        "model_metrics": model_metrics,
        "representative_model": representative,
        "representative_calibration_by_pred_bin": grouped_mean_profile(
            y_true=y_test,
            y_pred=rep_preds,
            edges=pred_edges,
            bins=pred_bins,
        ),
        "representative_profile_by_actual_bin": grouped_mean_profile(
            y_true=y_test,
            y_pred=rep_preds,
            edges=actual_edges,
            bins=actual_bins,
        ),
        "coefficient_interpretation": coefficient_interpretation,
    }
    json_dump(run_dir / "interpretation_results.json", payload)
    (run_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
