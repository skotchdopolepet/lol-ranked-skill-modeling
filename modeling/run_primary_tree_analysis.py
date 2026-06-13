from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "modeling",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_primary_linear_analysis as linear


DEFAULT_RUNTIME_ROOT = REPO_ROOT / "runtime" / "out_prod" / "primary_dataset"
DEFAULT_DB_PATH = DEFAULT_RUNTIME_ROOT / "primary_dataset.sqlite3"
DEFAULT_TABLE_NAME = "primary_dataset_v1"
DEFAULT_MODELING_ROOT = DEFAULT_RUNTIME_ROOT / "modeling"
DEFAULT_SOURCE_SPLIT_RUN = DEFAULT_MODELING_ROOT / "runs" / "20260411_192856_linear_primary_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_MODELING_ROOT / "tree_bakeoff"
DEFAULT_LOG_PATH = DEFAULT_OUTPUT_ROOT / "TREE_MODELING_EXECUTION_LOG_20260417.md"
DEFAULT_RANDOM_STATE = 42
DEFAULT_LIGHTGBM_PHASE1A_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260418_001631_lightgbm_phase1a_v1"
DEFAULT_LIGHTGBM_PHASE1B_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260420_000000_lightgbm_phase1b_v1"
DEFAULT_XGBOOST_PHASE1A_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260419_ew2_vm_xgboost_phase1a_v1"
DEFAULT_XGBOOST_PHASE1B_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260422_013600_xgboost_phase1b_v1"
DEFAULT_CATBOOST_PHASE1A_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260420_en1_vm_catboost_phase1a_v1"
DEFAULT_CATBOOST_PHASE1B_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "runs" / "20260422_064000_catboost_phase1b_v1"
PHASE1B_GLOBAL_SEED_OFFSET = 1_000_000
PHASE1C_TRIAL_SEED_OFFSET = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primary tree-model analysis on the primary dataset.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--table-name", type=str, default=DEFAULT_TABLE_NAME)
    parser.add_argument("--source-split-run-dir", type=Path, default=DEFAULT_SOURCE_SPLIT_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--mode",
        choices=[
            "smoke_histgb",
            "smoke_lightgbm",
            "smoke_xgboost",
            "smoke_catboost",
            "phase1a_lightgbm",
            "phase1b_lightgbm",
            "phase1c_lightgbm",
            "phase1a_xgboost",
            "phase1b_xgboost",
            "phase1a_catboost",
            "phase1b_catboost",
            "phase1c_catboost",
            "phase2_lightgbm",
            "phase2_xgboost",
            "phase2_catboost",
            "phase3_lightgbm",
            "phase3_xgboost",
            "phase3_catboost",
        ],
        default="smoke_histgb",
    )
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--smoke-fold", type=int, default=0)
    parser.add_argument("--smoke-train-cap", type=int, default=150000)
    parser.add_argument("--smoke-val-cap", type=int, default=100000)
    parser.add_argument("--smoke-learning-rate", type=float, default=0.05)
    parser.add_argument("--smoke-max-iter", type=int, default=300)
    parser.add_argument("--smoke-max-leaf-nodes", type=int, default=63)
    parser.add_argument("--smoke-max-depth", type=int, default=8)
    parser.add_argument("--smoke-min-samples-leaf", type=int, default=200)
    parser.add_argument("--smoke-l2-regularization", type=float, default=1.0)
    parser.add_argument("--smoke-validation-fraction", type=float, default=0.1)
    parser.add_argument("--smoke-n-iter-no-change", type=int, default=30)
    parser.add_argument("--smoke-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--phase1a-fold", type=int, default=0)
    parser.add_argument("--phase1a-trials", type=int, default=40)
    parser.add_argument("--phase1a-max-iter", type=int, default=5000)
    parser.add_argument("--phase1a-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--phase1a-n-jobs", type=int, default=4)
    parser.add_argument("--phase1b-source-run-dir", type=Path, default=DEFAULT_LIGHTGBM_PHASE1A_SOURCE_RUN)
    parser.add_argument("--phase1b-fold", type=int, default=0)
    parser.add_argument("--phase1b-trials", type=int, default=50)
    parser.add_argument("--phase1b-focused-trials", type=int, default=40)
    parser.add_argument("--phase1b-global-trials", type=int, default=10)
    parser.add_argument("--phase1b-topk", type=int, default=8)
    parser.add_argument("--phase1b-max-iter", type=int, default=5000)
    parser.add_argument("--phase1b-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--phase1b-n-jobs", type=int, default=4)
    parser.add_argument("--phase1c-source-run-dir", type=Path, default=DEFAULT_LIGHTGBM_PHASE1B_SOURCE_RUN)
    parser.add_argument("--phase1c-aux-source-run-dir", type=Path, default=DEFAULT_LIGHTGBM_PHASE1A_SOURCE_RUN)
    parser.add_argument("--phase1c-fold", type=int, default=0)
    parser.add_argument("--phase1c-trials", type=int, default=30)
    parser.add_argument("--phase1c-topk", type=int, default=30)
    parser.add_argument("--phase1c-max-iter", type=int, default=5000)
    parser.add_argument("--phase1c-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--phase1c-n-jobs", type=int, default=4)
    parser.add_argument("--phase2-source-run-dir", action="append", type=Path, default=[])
    parser.add_argument("--phase2-topk", type=int, default=5)
    parser.add_argument("--phase2-max-iter", type=int, default=5000)
    parser.add_argument("--phase2-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--phase2-n-jobs", type=int, default=4)
    parser.add_argument("--phase2-config-index-offset", type=int, default=0)
    parser.add_argument(
        "--phase3-source-run-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "runs" / "20260422_064000_lightgbm_phase2_top10_v1",
    )
    parser.add_argument("--phase3-config-index", type=int, default=1)
    parser.add_argument("--phase3-max-iter", type=int, default=5000)
    parser.add_argument("--phase3-n-jobs", type=int, default=4)
    parser.add_argument("--phase3-save-test-predictions", action="store_true")
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def log(msg: str) -> None:
    print(msg, flush=True)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_source_split(source_run_dir: Path) -> dict[str, Any]:
    split_arrays_path = source_run_dir / "split_arrays.npz"
    split_info_path = source_run_dir / "split_info.json"
    if not split_arrays_path.exists():
        raise FileNotFoundError(f"Missing split arrays: {split_arrays_path}")
    if not split_info_path.exists():
        raise FileNotFoundError(f"Missing split info: {split_info_path}")
    return {
        "split_arrays": np.load(split_arrays_path),
        "split_info": json.loads(split_info_path.read_text(encoding="utf-8")),
    }


def detect_optional_dependency(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}


def dependency_status() -> dict[str, Any]:
    return {
        "xgboost": detect_optional_dependency("xgboost"),
        "lightgbm": detect_optional_dependency("lightgbm"),
        "catboost": detect_optional_dependency("catboost"),
        "optuna": detect_optional_dependency("optuna"),
        "sklearn": detect_optional_dependency("sklearn"),
        "numpy": detect_optional_dependency("numpy"),
    }


def sample_indices(indices: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if cap <= 0 or idx.shape[0] <= cap:
        return np.sort(idx)
    chosen = rng.choice(idx, size=int(cap), replace=False)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    rmse = math.sqrt(float(np.mean(np.square(err))))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def append_markdown_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def render_smoke_summary(payload: dict[str, Any]) -> str:
    metrics_train = payload["metrics"]["train"]
    metrics_val = payload["metrics"]["validation"]
    deps = payload["dependency_status"]
    lines = [
        f"# {payload['model']['name']} Smoke Run",
        "",
        f"- Run dir: `{payload['meta']['run_dir']}`",
        f"- Mode: `{payload['meta']['mode']}`",
        f"- Source split run: `{payload['meta']['source_split_run_dir']}`",
        f"- Sampled train rows: `{payload['sampling']['train_rows_used']:,}` / `{payload['sampling']['train_rows_available']:,}`",
        f"- Sampled validation rows: `{payload['sampling']['validation_rows_used']:,}` / `{payload['sampling']['validation_rows_available']:,}`",
        "",
        "## Dependency Status",
        "",
        "| Package | Available | Details |",
        "| --- | --- | --- |",
    ]
    for name in ("sklearn", "numpy", "lightgbm", "xgboost", "catboost"):
        entry = deps[name]
        detail = entry.get("version", entry.get("error", ""))
        lines.append(f"| {name} | {entry['available']} | {detail} |")
    lines.extend(
        [
            "",
            f"## {payload['model']['name']} Smoke Metrics",
            "",
            "| Split | RMSE | MAE | R^2 |",
            "| --- | ---: | ---: | ---: |",
            f"| train | {metrics_train['rmse']:.6f} | {metrics_train['mae']:.6f} | {metrics_train['r2']:.6f} |",
            f"| validation | {metrics_val['rmse']:.6f} | {metrics_val['mae']:.6f} | {metrics_val['r2']:.6f} |",
            "",
            "## Runtime",
            "",
            "| Stage | Seconds |",
            "| --- | ---: |",
        ]
    )
    for key, value in payload["timings_sec"].items():
        lines.append(f"| {key} | {value:.1f} |")
    return "\n".join(lines) + "\n"


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sample_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(float(low), float(high)))


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    lo = math.log(float(low))
    hi = math.log(float(high))
    return float(math.exp(rng.uniform(lo, hi)))


def sample_choice(rng: np.random.Generator, choices: list[Any]) -> Any:
    return choices[int(rng.integers(0, len(choices)))]


def sample_int(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(int(low), int(high) + 1))


def sample_lightgbm_phase1a_params(rng: np.random.Generator) -> dict[str, Any]:
    params = {
        "learning_rate": sample_log_uniform(rng, 0.02, 0.10),
        "num_leaves": sample_int(rng, 31, 255),
        "max_depth": int(sample_choice(rng, [-1, 6, 8, 10, 12])),
        "min_child_samples": sample_int(rng, 50, 1000),
        "feature_fraction": sample_uniform(rng, 0.60, 1.00),
        "bagging_fraction": sample_uniform(rng, 0.60, 1.00),
        "bagging_freq": int(sample_choice(rng, [1, 5])),
        "lambda_l1": sample_log_uniform(rng, 1.0e-4, 10.0),
        "lambda_l2": sample_log_uniform(rng, 1.0e-3, 100.0),
        "min_split_gain": sample_uniform(rng, 0.0, 1.0),
        "max_bin": int(sample_choice(rng, [255, 511])),
    }
    max_depth = int(params["max_depth"])
    if max_depth > 0:
        params["num_leaves"] = int(min(int(params["num_leaves"]), 2 ** max_depth))
    return params


def lightgbm_param_signature(params: dict[str, Any]) -> str:
    canonical = {
        "bagging_fraction": float(params["bagging_fraction"]),
        "bagging_freq": int(params["bagging_freq"]),
        "feature_fraction": float(params["feature_fraction"]),
        "lambda_l1": float(params["lambda_l1"]),
        "lambda_l2": float(params["lambda_l2"]),
        "learning_rate": float(params["learning_rate"]),
        "max_bin": int(params["max_bin"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "min_split_gain": float(params["min_split_gain"]),
        "num_leaves": int(params["num_leaves"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


LIGHTGBM_PHASE1A_MAX_DEPTH_CHOICES = [-1, 6, 8, 10, 12]
LIGHTGBM_PHASE1A_BAGGING_FREQ_CHOICES = [1, 5]
LIGHTGBM_PHASE1A_MAX_BIN_CHOICES = [255, 511]


def sample_lightgbm_from_space(
    rng: np.random.Generator,
    *,
    learning_rate: tuple[float, float],
    num_leaves: tuple[int, int],
    max_depth_choices: list[int],
    min_child_samples: tuple[int, int],
    feature_fraction: tuple[float, float],
    bagging_fraction: tuple[float, float],
    bagging_freq_choices: list[int],
    lambda_l1: tuple[float, float],
    lambda_l2: tuple[float, float],
    min_split_gain: tuple[float, float],
    max_bin_choices: list[int],
) -> dict[str, Any]:
    params = {
        "learning_rate": sample_log_uniform(rng, learning_rate[0], learning_rate[1]),
        "num_leaves": sample_int(rng, num_leaves[0], num_leaves[1]),
        "max_depth": int(sample_choice(rng, list(max_depth_choices))),
        "min_child_samples": sample_int(rng, min_child_samples[0], min_child_samples[1]),
        "feature_fraction": sample_uniform(rng, feature_fraction[0], feature_fraction[1]),
        "bagging_fraction": sample_uniform(rng, bagging_fraction[0], bagging_fraction[1]),
        "bagging_freq": int(sample_choice(rng, list(bagging_freq_choices))),
        "lambda_l1": sample_log_uniform(rng, lambda_l1[0], lambda_l1[1]),
        "lambda_l2": sample_log_uniform(rng, lambda_l2[0], lambda_l2[1]),
        "min_split_gain": sample_uniform(rng, min_split_gain[0], min_split_gain[1]),
        "max_bin": int(sample_choice(rng, list(max_bin_choices))),
    }
    max_depth = int(params["max_depth"])
    if max_depth > 0:
        params["num_leaves"] = int(min(int(params["num_leaves"]), 2 ** max_depth))
    return params


def sample_xgboost_phase1a_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "learning_rate": sample_log_uniform(rng, 0.02, 0.10),
        "max_depth": sample_int(rng, 4, 10),
        "min_child_weight": sample_log_uniform(rng, 1.0, 64.0),
        "subsample": sample_uniform(rng, 0.60, 1.00),
        "colsample_bytree": sample_uniform(rng, 0.60, 1.00),
        "reg_alpha": sample_log_uniform(rng, 1.0e-4, 10.0),
        "reg_lambda": sample_log_uniform(rng, 1.0e-3, 100.0),
        "gamma": sample_uniform(rng, 0.0, 5.0),
        "max_bin": int(sample_choice(rng, [256, 512])),
    }


def sample_xgboost_from_space(
    rng: np.random.Generator,
    *,
    learning_rate: tuple[float, float],
    max_depth: tuple[int, int],
    min_child_weight: tuple[float, float],
    subsample: tuple[float, float],
    colsample_bytree: tuple[float, float],
    reg_alpha: tuple[float, float],
    reg_lambda: tuple[float, float],
    gamma: tuple[float, float],
    max_bin_choices: list[int],
) -> dict[str, Any]:
    return {
        "learning_rate": sample_log_uniform(rng, learning_rate[0], learning_rate[1]),
        "max_depth": sample_int(rng, max_depth[0], max_depth[1]),
        "min_child_weight": sample_log_uniform(rng, min_child_weight[0], min_child_weight[1]),
        "subsample": sample_uniform(rng, subsample[0], subsample[1]),
        "colsample_bytree": sample_uniform(rng, colsample_bytree[0], colsample_bytree[1]),
        "reg_alpha": sample_log_uniform(rng, reg_alpha[0], reg_alpha[1]),
        "reg_lambda": sample_log_uniform(rng, reg_lambda[0], reg_lambda[1]),
        "gamma": sample_uniform(rng, gamma[0], gamma[1]),
        "max_bin": int(sample_choice(rng, list(max_bin_choices))),
    }


def xgboost_param_signature(params: dict[str, Any]) -> str:
    canonical = {
        "colsample_bytree": float(params["colsample_bytree"]),
        "gamma": float(params["gamma"]),
        "learning_rate": float(params["learning_rate"]),
        "max_bin": int(params["max_bin"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "subsample": float(params["subsample"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def sample_catboost_phase1a_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "learning_rate": sample_log_uniform(rng, 0.02, 0.10),
        "depth": sample_int(rng, 5, 10),
        "l2_leaf_reg": sample_log_uniform(rng, 1.0, 30.0),
        "rsm": sample_uniform(rng, 0.60, 1.00),
        "bagging_temperature": sample_uniform(rng, 0.0, 5.0),
        "border_count": int(sample_choice(rng, [128, 254])),
    }


def sample_catboost_from_space(
    rng: np.random.Generator,
    *,
    learning_rate: tuple[float, float],
    depth: tuple[int, int],
    l2_leaf_reg: tuple[float, float],
    rsm: tuple[float, float],
    bagging_temperature: tuple[float, float],
    border_count_choices: list[int],
) -> dict[str, Any]:
    return {
        "learning_rate": sample_log_uniform(rng, learning_rate[0], learning_rate[1]),
        "depth": sample_int(rng, depth[0], depth[1]),
        "l2_leaf_reg": sample_log_uniform(rng, l2_leaf_reg[0], l2_leaf_reg[1]),
        "rsm": sample_uniform(rng, rsm[0], rsm[1]),
        "bagging_temperature": sample_uniform(rng, bagging_temperature[0], bagging_temperature[1]),
        "border_count": int(sample_choice(rng, list(border_count_choices))),
    }


def catboost_param_signature(params: dict[str, Any]) -> str:
    canonical = {
        "bagging_temperature": float(params["bagging_temperature"]),
        "border_count": int(params["border_count"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "learning_rate": float(params["learning_rate"]),
        "rsm": float(params["rsm"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def trial_sort_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["metrics"]["validation"]["rmse"]),
        float(row["metrics"]["rmse_gap"]),
        float(row["timings_sec"]["fit"]),
        int(row["trial_id"]),
    )


def rank_trial_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(trials, key=trial_sort_key)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = int(rank)
    return ordered


def write_trial_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_id",
        "trial_seed",
        "best_iteration",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
        "train_rmse_eval",
        "rmse_gap",
        "fit_seconds",
        "total_trial_seconds",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "min_split_gain",
        "max_bin",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            params = row["params"]
            writer.writerow(
                {
                    "rank": int(row.get("rank", 0)),
                    "trial_id": int(row["trial_id"]),
                    "trial_seed": int(row["trial_seed"]),
                    "best_iteration": int(row["best_iteration"]),
                    "validation_rmse": f"{row['metrics']['validation']['rmse']:.6f}",
                    "validation_mae": f"{row['metrics']['validation']['mae']:.6f}",
                    "validation_r2": f"{row['metrics']['validation']['r2']:.6f}",
                    "train_rmse_eval": f"{row['metrics']['train']['rmse']:.6f}",
                    "rmse_gap": f"{row['metrics']['rmse_gap']:.6f}",
                    "fit_seconds": f"{row['timings_sec']['fit']:.1f}",
                    "total_trial_seconds": f"{row['timings_sec']['trial_total']:.1f}",
                    "learning_rate": f"{params['learning_rate']:.8f}",
                    "num_leaves": int(params["num_leaves"]),
                    "max_depth": int(params["max_depth"]),
                    "min_child_samples": int(params["min_child_samples"]),
                    "feature_fraction": f"{params['feature_fraction']:.6f}",
                    "bagging_fraction": f"{params['bagging_fraction']:.6f}",
                    "bagging_freq": int(params["bagging_freq"]),
                    "lambda_l1": f"{params['lambda_l1']:.8f}",
                    "lambda_l2": f"{params['lambda_l2']:.8f}",
                    "min_split_gain": f"{params['min_split_gain']:.8f}",
                    "max_bin": int(params["max_bin"]),
                }
            )


def write_trial_csv_xgboost(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_id",
        "trial_seed",
        "best_iteration",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
        "train_rmse_eval",
        "rmse_gap",
        "fit_seconds",
        "total_trial_seconds",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "max_bin",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            params = row["params"]
            writer.writerow(
                {
                    "rank": int(row.get("rank", 0)),
                    "trial_id": int(row["trial_id"]),
                    "trial_seed": int(row["trial_seed"]),
                    "best_iteration": int(row["best_iteration"]),
                    "validation_rmse": f"{row['metrics']['validation']['rmse']:.6f}",
                    "validation_mae": f"{row['metrics']['validation']['mae']:.6f}",
                    "validation_r2": f"{row['metrics']['validation']['r2']:.6f}",
                    "train_rmse_eval": f"{row['metrics']['train']['rmse']:.6f}",
                    "rmse_gap": f"{row['metrics']['rmse_gap']:.6f}",
                    "fit_seconds": f"{row['timings_sec']['fit']:.1f}",
                    "total_trial_seconds": f"{row['timings_sec']['trial_total']:.1f}",
                    "learning_rate": f"{params['learning_rate']:.8f}",
                    "max_depth": int(params["max_depth"]),
                    "min_child_weight": f"{params['min_child_weight']:.8f}",
                    "subsample": f"{params['subsample']:.6f}",
                    "colsample_bytree": f"{params['colsample_bytree']:.6f}",
                    "reg_alpha": f"{params['reg_alpha']:.8f}",
                    "reg_lambda": f"{params['reg_lambda']:.8f}",
                    "gamma": f"{params['gamma']:.8f}",
                    "max_bin": int(params["max_bin"]),
                }
            )


def write_trial_csv_catboost(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_id",
        "trial_seed",
        "best_iteration",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
        "train_rmse_eval",
        "rmse_gap",
        "fit_seconds",
        "total_trial_seconds",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "rsm",
        "bagging_temperature",
        "border_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            params = row["params"]
            writer.writerow(
                {
                    "rank": int(row.get("rank", 0)),
                    "trial_id": int(row["trial_id"]),
                    "trial_seed": int(row["trial_seed"]),
                    "best_iteration": int(row["best_iteration"]),
                    "validation_rmse": f"{row['metrics']['validation']['rmse']:.6f}",
                    "validation_mae": f"{row['metrics']['validation']['mae']:.6f}",
                    "validation_r2": f"{row['metrics']['validation']['r2']:.6f}",
                    "train_rmse_eval": f"{row['metrics']['train']['rmse']:.6f}",
                    "rmse_gap": f"{row['metrics']['rmse_gap']:.6f}",
                    "fit_seconds": f"{row['timings_sec']['fit']:.1f}",
                    "total_trial_seconds": f"{row['timings_sec']['trial_total']:.1f}",
                    "learning_rate": f"{params['learning_rate']:.8f}",
                    "depth": int(params["depth"]),
                    "l2_leaf_reg": f"{params['l2_leaf_reg']:.8f}",
                    "rsm": f"{params['rsm']:.6f}",
                    "bagging_temperature": f"{params['bagging_temperature']:.8f}",
                    "border_count": int(params["border_count"]),
                }
            )


def write_trial_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_trial_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    unique_by_trial: dict[int, dict[str, Any]] = {}
    for row in rows:
        unique_by_trial[int(row["trial_id"])] = row
    return [unique_by_trial[key] for key in sorted(unique_by_trial)]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_created_utc(results_json_path: Path, fallback: str) -> str:
    if not results_json_path.exists():
        return fallback
    try:
        payload = json.loads(results_json_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    return str(payload.get("meta", {}).get("created_utc", fallback))


def split_lightgbm_attempts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    signature_to_trial_id: dict[str, int] = {}
    ordered = sorted(rows, key=lambda row: int(row["trial_id"]))
    for raw_row in ordered:
        row = dict(raw_row)
        params = row.get("params") or {}
        signature = lightgbm_param_signature(params) if params else ""
        row["param_signature"] = signature
        outcome = str(row.get("outcome", "completed"))
        if outcome != "completed":
            duplicate_rows.append(row)
            continue
        if signature in signature_to_trial_id:
            row["outcome"] = "duplicate_completed"
            row["duplicate_of_trial_id"] = int(signature_to_trial_id[signature])
            duplicate_rows.append(row)
            continue
        row["outcome"] = "completed"
        signature_to_trial_id[signature] = int(row["trial_id"])
        unique_rows.append(row)
    return unique_rows, duplicate_rows, signature_to_trial_id


def split_xgboost_attempts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    signature_to_trial_id: dict[str, int] = {}
    ordered = sorted(rows, key=lambda row: int(row["trial_id"]))
    for raw_row in ordered:
        row = dict(raw_row)
        params = row.get("params") or {}
        signature = xgboost_param_signature(params) if params else ""
        row["param_signature"] = signature
        outcome = str(row.get("outcome", "completed"))
        if outcome != "completed":
            duplicate_rows.append(row)
            continue
        if signature in signature_to_trial_id:
            row["outcome"] = "duplicate_completed"
            row["duplicate_of_trial_id"] = int(signature_to_trial_id[signature])
            duplicate_rows.append(row)
            continue
        row["outcome"] = "completed"
        signature_to_trial_id[signature] = int(row["trial_id"])
        unique_rows.append(row)
    return unique_rows, duplicate_rows, signature_to_trial_id


def split_catboost_attempts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    signature_to_trial_id: dict[str, int] = {}
    ordered = sorted(rows, key=lambda row: int(row["trial_id"]))
    for raw_row in ordered:
        row = dict(raw_row)
        params = row.get("params") or {}
        signature = catboost_param_signature(params) if params else ""
        row["param_signature"] = signature
        outcome = str(row.get("outcome", "completed"))
        if outcome != "completed":
            duplicate_rows.append(row)
            continue
        if signature in signature_to_trial_id:
            row["outcome"] = "duplicate_completed"
            row["duplicate_of_trial_id"] = int(signature_to_trial_id[signature])
            duplicate_rows.append(row)
            continue
        row["outcome"] = "completed"
        signature_to_trial_id[signature] = int(row["trial_id"])
        unique_rows.append(row)
    return unique_rows, duplicate_rows, signature_to_trial_id


def expand_linear_range(values: list[float], *, low: float, high: float, min_width: float, pad_fraction: float) -> tuple[float, float]:
    observed_low = float(min(values))
    observed_high = float(max(values))
    span = observed_high - observed_low
    pad = max(min_width, span * pad_fraction)
    return (max(float(low), observed_low - pad), min(float(high), observed_high + pad))


def expand_log_range(values: list[float], *, low: float, high: float, min_log_width: float, pad_fraction: float) -> tuple[float, float]:
    logs = [math.log(float(v)) for v in values]
    lo = min(logs)
    hi = max(logs)
    span = hi - lo
    pad = max(min_log_width, span * pad_fraction)
    out_low = max(math.log(float(low)), lo - pad)
    out_high = min(math.log(float(high)), hi + pad)
    return (float(math.exp(out_low)), float(math.exp(out_high)))


def expand_int_range(values: list[int], *, low: int, high: int, min_margin: int, pad_fraction: float) -> tuple[int, int]:
    observed_low = int(min(values))
    observed_high = int(max(values))
    span = observed_high - observed_low
    margin = max(int(math.ceil(span * pad_fraction)), int(min_margin))
    return (max(int(low), observed_low - margin), min(int(high), observed_high + margin))


def build_lightgbm_phase1b_space(source_trials: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    if not source_trials:
        raise ValueError("Phase 1B requires at least one unique LightGBM Phase 1A trial.")
    ranked = rank_trial_rows(list(source_trials))
    seed_trials = ranked[: max(1, int(topk))]
    params_list = [row["params"] for row in seed_trials]
    max_depth_choices = sorted({int(p["max_depth"]) for p in params_list})
    if not max_depth_choices:
        max_depth_choices = list(LIGHTGBM_PHASE1A_MAX_DEPTH_CHOICES)
    bagging_freq_choices = sorted({int(p["bagging_freq"]) for p in params_list})
    if not bagging_freq_choices:
        bagging_freq_choices = list(LIGHTGBM_PHASE1A_BAGGING_FREQ_CHOICES)
    max_bin_choices = sorted({int(p["max_bin"]) for p in params_list})
    if not max_bin_choices:
        max_bin_choices = list(LIGHTGBM_PHASE1A_MAX_BIN_CHOICES)
    return {
        "seed_trial_ids": [int(row["trial_id"]) for row in seed_trials],
        "topk": int(len(seed_trials)),
        "learning_rate": expand_log_range(
            [float(p["learning_rate"]) for p in params_list],
            low=0.02,
            high=0.10,
            min_log_width=0.20,
            pad_fraction=0.20,
        ),
        "num_leaves": expand_int_range(
            [int(p["num_leaves"]) for p in params_list],
            low=31,
            high=255,
            min_margin=8,
            pad_fraction=0.20,
        ),
        "max_depth_choices": [int(x) for x in max_depth_choices],
        "min_child_samples": expand_int_range(
            [int(p["min_child_samples"]) for p in params_list],
            low=50,
            high=1000,
            min_margin=40,
            pad_fraction=0.20,
        ),
        "feature_fraction": expand_linear_range(
            [float(p["feature_fraction"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.03,
            pad_fraction=0.20,
        ),
        "bagging_fraction": expand_linear_range(
            [float(p["bagging_fraction"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.03,
            pad_fraction=0.20,
        ),
        "bagging_freq_choices": [int(x) for x in bagging_freq_choices],
        "lambda_l1": expand_log_range(
            [float(p["lambda_l1"]) for p in params_list],
            low=1.0e-4,
            high=10.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "lambda_l2": expand_log_range(
            [float(p["lambda_l2"]) for p in params_list],
            low=1.0e-3,
            high=100.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "min_split_gain": expand_linear_range(
            [float(p["min_split_gain"]) for p in params_list],
            low=0.0,
            high=1.0,
            min_width=0.05,
            pad_fraction=0.20,
        ),
        "max_bin_choices": [int(x) for x in max_bin_choices],
    }


def sample_lightgbm_phase1b_focused_params(rng: np.random.Generator, focused_space: dict[str, Any]) -> dict[str, Any]:
    return sample_lightgbm_from_space(
        rng,
        learning_rate=tuple(focused_space["learning_rate"]),
        num_leaves=tuple(focused_space["num_leaves"]),
        max_depth_choices=list(focused_space["max_depth_choices"]),
        min_child_samples=tuple(focused_space["min_child_samples"]),
        feature_fraction=tuple(focused_space["feature_fraction"]),
        bagging_fraction=tuple(focused_space["bagging_fraction"]),
        bagging_freq_choices=list(focused_space["bagging_freq_choices"]),
        lambda_l1=tuple(focused_space["lambda_l1"]),
        lambda_l2=tuple(focused_space["lambda_l2"]),
        min_split_gain=tuple(focused_space["min_split_gain"]),
        max_bin_choices=list(focused_space["max_bin_choices"]),
    )


def build_xgboost_phase1b_space(source_trials: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    if not source_trials:
        raise ValueError("Phase 1B requires at least one unique XGBoost Phase 1A trial.")
    ranked = rank_trial_rows(list(source_trials))
    seed_trials = ranked[: max(1, int(topk))]
    params_list = [row["params"] for row in seed_trials]
    max_bin_choices = sorted({int(p["max_bin"]) for p in params_list})
    if not max_bin_choices:
        max_bin_choices = [256, 512]
    return {
        "seed_trial_ids": [int(row["trial_id"]) for row in seed_trials],
        "topk": int(len(seed_trials)),
        "learning_rate": expand_log_range(
            [float(p["learning_rate"]) for p in params_list],
            low=0.02,
            high=0.10,
            min_log_width=0.20,
            pad_fraction=0.20,
        ),
        "max_depth": expand_int_range(
            [int(p["max_depth"]) for p in params_list],
            low=4,
            high=10,
            min_margin=1,
            pad_fraction=0.20,
        ),
        "min_child_weight": expand_log_range(
            [float(p["min_child_weight"]) for p in params_list],
            low=1.0,
            high=64.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "subsample": expand_linear_range(
            [float(p["subsample"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.03,
            pad_fraction=0.20,
        ),
        "colsample_bytree": expand_linear_range(
            [float(p["colsample_bytree"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.03,
            pad_fraction=0.20,
        ),
        "reg_alpha": expand_log_range(
            [float(p["reg_alpha"]) for p in params_list],
            low=1.0e-4,
            high=10.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "reg_lambda": expand_log_range(
            [float(p["reg_lambda"]) for p in params_list],
            low=1.0e-3,
            high=100.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "gamma": expand_linear_range(
            [float(p["gamma"]) for p in params_list],
            low=0.0,
            high=5.0,
            min_width=0.05,
            pad_fraction=0.20,
        ),
        "max_bin_choices": [int(x) for x in max_bin_choices],
    }


def sample_xgboost_phase1b_focused_params(rng: np.random.Generator, focused_space: dict[str, Any]) -> dict[str, Any]:
    return sample_xgboost_from_space(
        rng,
        learning_rate=tuple(focused_space["learning_rate"]),
        max_depth=tuple(focused_space["max_depth"]),
        min_child_weight=tuple(focused_space["min_child_weight"]),
        subsample=tuple(focused_space["subsample"]),
        colsample_bytree=tuple(focused_space["colsample_bytree"]),
        reg_alpha=tuple(focused_space["reg_alpha"]),
        reg_lambda=tuple(focused_space["reg_lambda"]),
        gamma=tuple(focused_space["gamma"]),
        max_bin_choices=list(focused_space["max_bin_choices"]),
    )


def build_catboost_phase1b_space(source_trials: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    if not source_trials:
        raise ValueError("Phase 1B requires at least one unique CatBoost Phase 1A trial.")
    ranked = rank_trial_rows(list(source_trials))
    seed_trials = ranked[: max(1, int(topk))]
    params_list = [row["params"] for row in seed_trials]
    border_count_choices = sorted({int(p["border_count"]) for p in params_list})
    if not border_count_choices:
        border_count_choices = [128, 254]
    return {
        "seed_trial_ids": [int(row["trial_id"]) for row in seed_trials],
        "topk": int(len(seed_trials)),
        "learning_rate": expand_log_range(
            [float(p["learning_rate"]) for p in params_list],
            low=0.02,
            high=0.10,
            min_log_width=0.20,
            pad_fraction=0.20,
        ),
        "depth": expand_int_range(
            [int(p["depth"]) for p in params_list],
            low=5,
            high=10,
            min_margin=1,
            pad_fraction=0.20,
        ),
        "l2_leaf_reg": expand_log_range(
            [float(p["l2_leaf_reg"]) for p in params_list],
            low=1.0,
            high=30.0,
            min_log_width=0.50,
            pad_fraction=0.20,
        ),
        "rsm": expand_linear_range(
            [float(p["rsm"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.03,
            pad_fraction=0.20,
        ),
        "bagging_temperature": expand_linear_range(
            [float(p["bagging_temperature"]) for p in params_list],
            low=0.0,
            high=5.0,
            min_width=0.10,
            pad_fraction=0.20,
        ),
        "border_count_choices": [int(x) for x in border_count_choices],
    }


def sample_catboost_phase1b_focused_params(rng: np.random.Generator, focused_space: dict[str, Any]) -> dict[str, Any]:
    return sample_catboost_from_space(
        rng,
        learning_rate=tuple(focused_space["learning_rate"]),
        depth=tuple(focused_space["depth"]),
        l2_leaf_reg=tuple(focused_space["l2_leaf_reg"]),
        rsm=tuple(focused_space["rsm"]),
        bagging_temperature=tuple(focused_space["bagging_temperature"]),
        border_count_choices=list(focused_space["border_count_choices"]),
    )


def build_catboost_phase1c_space(source_trials: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    if not source_trials:
        raise ValueError("Phase 1C requires at least one unique CatBoost source trial.")
    ranked = rank_trial_rows(list(source_trials))
    seed_trials = ranked[: max(1, int(topk))]
    params_list = [row["params"] for row in seed_trials]
    border_count_choices = sorted({int(p["border_count"]) for p in params_list})
    if not border_count_choices:
        border_count_choices = [128, 254]
    return {
        "seed_trial_ids": [int(row["trial_id"]) for row in seed_trials],
        "seed_trial_sources": [str(row.get("source_phase", "unknown")) for row in seed_trials],
        "topk": int(len(seed_trials)),
        "learning_rate": expand_log_range(
            [float(p["learning_rate"]) for p in params_list],
            low=0.02,
            high=0.10,
            min_log_width=0.12,
            pad_fraction=0.12,
        ),
        "depth": expand_int_range(
            [int(p["depth"]) for p in params_list],
            low=5,
            high=10,
            min_margin=1,
            pad_fraction=0.12,
        ),
        "l2_leaf_reg": expand_log_range(
            [float(p["l2_leaf_reg"]) for p in params_list],
            low=1.0,
            high=30.0,
            min_log_width=0.35,
            pad_fraction=0.12,
        ),
        "rsm": expand_linear_range(
            [float(p["rsm"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.02,
            pad_fraction=0.12,
        ),
        "bagging_temperature": expand_linear_range(
            [float(p["bagging_temperature"]) for p in params_list],
            low=0.0,
            high=5.0,
            min_width=0.10,
            pad_fraction=0.12,
        ),
        "border_count_choices": [int(x) for x in border_count_choices],
    }


def optuna_catboost_distributions(params: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    import optuna

    return {
        "learning_rate": optuna.distributions.FloatDistribution(
            float(space["learning_rate"][0]), float(space["learning_rate"][1]), log=True
        ),
        "depth": optuna.distributions.IntDistribution(int(space["depth"][0]), int(space["depth"][1])),
        "l2_leaf_reg": optuna.distributions.FloatDistribution(float(space["l2_leaf_reg"][0]), float(space["l2_leaf_reg"][1]), log=True),
        "rsm": optuna.distributions.FloatDistribution(float(space["rsm"][0]), float(space["rsm"][1])),
        "bagging_temperature": optuna.distributions.FloatDistribution(
            float(space["bagging_temperature"][0]), float(space["bagging_temperature"][1])
        ),
        "border_count": optuna.distributions.CategoricalDistribution(list(space["border_count_choices"])),
    }


def catboost_params_within_space(params: dict[str, Any], space: dict[str, Any]) -> bool:
    try:
        if int(params["border_count"]) not in {int(x) for x in space["border_count_choices"]}:
            return False
        checks = [
            ("learning_rate", float(space["learning_rate"][0]), float(space["learning_rate"][1])),
            ("depth", int(space["depth"][0]), int(space["depth"][1])),
            ("l2_leaf_reg", float(space["l2_leaf_reg"][0]), float(space["l2_leaf_reg"][1])),
            ("rsm", float(space["rsm"][0]), float(space["rsm"][1])),
            ("bagging_temperature", float(space["bagging_temperature"][0]), float(space["bagging_temperature"][1])),
        ]
        for key, low, high in checks:
            value = float(params[key])
            if value < float(low) or value > float(high):
                return False
        return True
    except Exception:
        return False


def suggest_catboost_phase1c_params(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(trial.suggest_float("learning_rate", float(space["learning_rate"][0]), float(space["learning_rate"][1]), log=True)),
        "depth": int(trial.suggest_int("depth", int(space["depth"][0]), int(space["depth"][1]))),
        "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", float(space["l2_leaf_reg"][0]), float(space["l2_leaf_reg"][1]), log=True)),
        "rsm": float(trial.suggest_float("rsm", float(space["rsm"][0]), float(space["rsm"][1]))),
        "bagging_temperature": float(
            trial.suggest_float("bagging_temperature", float(space["bagging_temperature"][0]), float(space["bagging_temperature"][1]))
        ),
        "border_count": int(trial.suggest_categorical("border_count", list(space["border_count_choices"]))),
    }


def render_phase1a_lightgbm_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    status = payload["status"]
    best_trial = payload.get("best_trial")
    lines = [
        "# LightGBM Phase 1A Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{status}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_split_gain",
            "max_bin",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1a_lightgbm_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_lightgbm_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    best_trial = leaderboard[0] if leaderboard else None
    payload = {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1a_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": int(args.phase1a_max_iter),
            "early_stopping_rounds": int(args.phase1a_early_stopping_rounds),
            "force_col_wise": True,
        },
        "progress": {
            "planned_trials": int(args.phase1a_trials),
            "completed_trials": int(len(unique_trials)),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1a_n_jobs),
        },
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }
    return payload


def write_phase1a_lightgbm_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1a_lightgbm_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1a_lightgbm_summary(payload), encoding="utf-8")
    return payload


def render_phase1b_lightgbm_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    status = payload["status"]
    best_trial = payload.get("best_trial")
    lines = [
        "# LightGBM Phase 1B Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{status}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Phase 1A source run: `{meta['phase1a_source_run_dir']}`",
        f"- Source signatures blocked: `{meta['blocked_source_signatures']}`",
        f"- Focused seed trials from Phase 1A: `{', '.join(str(x) for x in payload['focused_space']['seed_trial_ids'])}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Focused unique trials completed: `{progress['focused_completed_trials']}` / `{progress['focused_target_trials']}`",
        f"- Global unique trials completed: `{progress['global_completed_trials']}` / `{progress['global_target_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Search stage: `{best_trial.get('search_stage', 'unknown')}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_split_gain",
            "max_bin",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Stage | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row.get('search_stage', 'unknown')} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1b_lightgbm_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_lightgbm_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    best_trial = leaderboard[0] if leaderboard else None
    focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
    global_completed = sum(1 for row in unique_trials if row.get("search_stage") == "global")
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase1a_source_run_dir": str(args.phase1b_source_run_dir),
            "blocked_source_signatures": int(len(source_unique_trials)),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1b_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": int(args.phase1b_max_iter),
            "early_stopping_rounds": int(args.phase1b_early_stopping_rounds),
            "force_col_wise": True,
        },
        "focused_space": focused_space,
        "progress": {
            "planned_trials": int(args.phase1b_trials),
            "focused_target_trials": int(args.phase1b_focused_trials),
            "global_target_trials": int(args.phase1b_global_trials),
            "completed_trials": int(len(unique_trials)),
            "focused_completed_trials": int(focused_completed),
            "global_completed_trials": int(global_completed),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1b_n_jobs),
        },
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1b_lightgbm_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1b_lightgbm_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        source_unique_trials=source_unique_trials,
        focused_space=focused_space,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "focused_completed_trials": payload["progress"]["focused_completed_trials"],
            "global_completed_trials": payload["progress"]["global_completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1b_lightgbm_summary(payload), encoding="utf-8")
    return payload


def build_lightgbm_phase1c_space(source_trials: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    if not source_trials:
        raise ValueError("Phase 1C requires at least one unique LightGBM source trial.")
    ranked = rank_trial_rows(list(source_trials))
    seed_trials = ranked[: max(1, int(topk))]
    params_list = [row["params"] for row in seed_trials]
    max_depth_choices = sorted({int(p["max_depth"]) for p in params_list})
    if not max_depth_choices:
        max_depth_choices = list(LIGHTGBM_PHASE1A_MAX_DEPTH_CHOICES)
    bagging_freq_choices = sorted({int(p["bagging_freq"]) for p in params_list})
    if not bagging_freq_choices:
        bagging_freq_choices = list(LIGHTGBM_PHASE1A_BAGGING_FREQ_CHOICES)
    max_bin_choices = sorted({int(p["max_bin"]) for p in params_list})
    if not max_bin_choices:
        max_bin_choices = list(LIGHTGBM_PHASE1A_MAX_BIN_CHOICES)
    return {
        "seed_trial_ids": [int(row["trial_id"]) for row in seed_trials],
        "seed_trial_sources": [str(row.get("source_phase", "unknown")) for row in seed_trials],
        "topk": int(len(seed_trials)),
        "learning_rate": expand_log_range(
            [float(p["learning_rate"]) for p in params_list],
            low=0.02,
            high=0.10,
            min_log_width=0.12,
            pad_fraction=0.12,
        ),
        "num_leaves": expand_int_range(
            [int(p["num_leaves"]) for p in params_list],
            low=31,
            high=255,
            min_margin=6,
            pad_fraction=0.12,
        ),
        "max_depth_choices": [int(x) for x in max_depth_choices],
        "min_child_samples": expand_int_range(
            [int(p["min_child_samples"]) for p in params_list],
            low=50,
            high=1000,
            min_margin=30,
            pad_fraction=0.12,
        ),
        "feature_fraction": expand_linear_range(
            [float(p["feature_fraction"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.02,
            pad_fraction=0.12,
        ),
        "bagging_fraction": expand_linear_range(
            [float(p["bagging_fraction"]) for p in params_list],
            low=0.60,
            high=1.00,
            min_width=0.02,
            pad_fraction=0.12,
        ),
        "bagging_freq_choices": [int(x) for x in bagging_freq_choices],
        "lambda_l1": expand_log_range(
            [float(p["lambda_l1"]) for p in params_list],
            low=1.0e-4,
            high=10.0,
            min_log_width=0.35,
            pad_fraction=0.12,
        ),
        "lambda_l2": expand_log_range(
            [float(p["lambda_l2"]) for p in params_list],
            low=1.0e-3,
            high=100.0,
            min_log_width=0.35,
            pad_fraction=0.12,
        ),
        "min_split_gain": expand_linear_range(
            [float(p["min_split_gain"]) for p in params_list],
            low=0.0,
            high=1.0,
            min_width=0.03,
            pad_fraction=0.12,
        ),
        "max_bin_choices": [int(x) for x in max_bin_choices],
    }


def optuna_lightgbm_distributions(params: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    import optuna

    max_depth = int(params["max_depth"])
    num_leaves_low = int(space["num_leaves"][0])
    num_leaves_high = int(space["num_leaves"][1])
    if max_depth > 0:
        num_leaves_high = min(num_leaves_high, 2 ** max_depth)
    num_leaves_low = min(num_leaves_low, num_leaves_high)
    return {
        "learning_rate": optuna.distributions.FloatDistribution(
            float(space["learning_rate"][0]), float(space["learning_rate"][1]), log=True
        ),
        "max_depth": optuna.distributions.CategoricalDistribution(list(space["max_depth_choices"])),
        "num_leaves": optuna.distributions.IntDistribution(num_leaves_low, num_leaves_high),
        "min_child_samples": optuna.distributions.IntDistribution(
            int(space["min_child_samples"][0]), int(space["min_child_samples"][1])
        ),
        "feature_fraction": optuna.distributions.FloatDistribution(
            float(space["feature_fraction"][0]), float(space["feature_fraction"][1])
        ),
        "bagging_fraction": optuna.distributions.FloatDistribution(
            float(space["bagging_fraction"][0]), float(space["bagging_fraction"][1])
        ),
        "bagging_freq": optuna.distributions.CategoricalDistribution(list(space["bagging_freq_choices"])),
        "lambda_l1": optuna.distributions.FloatDistribution(float(space["lambda_l1"][0]), float(space["lambda_l1"][1]), log=True),
        "lambda_l2": optuna.distributions.FloatDistribution(float(space["lambda_l2"][0]), float(space["lambda_l2"][1]), log=True),
        "min_split_gain": optuna.distributions.FloatDistribution(
            float(space["min_split_gain"][0]), float(space["min_split_gain"][1])
        ),
        "max_bin": optuna.distributions.CategoricalDistribution(list(space["max_bin_choices"])),
    }


def lightgbm_params_within_space(params: dict[str, Any], space: dict[str, Any]) -> bool:
    try:
        max_depth = int(params["max_depth"])
        num_leaves = int(params["num_leaves"])
        if max_depth not in {int(x) for x in space["max_depth_choices"]}:
            return False
        if int(params["bagging_freq"]) not in {int(x) for x in space["bagging_freq_choices"]}:
            return False
        if int(params["max_bin"]) not in {int(x) for x in space["max_bin_choices"]}:
            return False
        checks = [
            ("learning_rate", float(space["learning_rate"][0]), float(space["learning_rate"][1])),
            ("num_leaves", int(space["num_leaves"][0]), int(space["num_leaves"][1])),
            ("min_child_samples", int(space["min_child_samples"][0]), int(space["min_child_samples"][1])),
            ("feature_fraction", float(space["feature_fraction"][0]), float(space["feature_fraction"][1])),
            ("bagging_fraction", float(space["bagging_fraction"][0]), float(space["bagging_fraction"][1])),
            ("lambda_l1", float(space["lambda_l1"][0]), float(space["lambda_l1"][1])),
            ("lambda_l2", float(space["lambda_l2"][0]), float(space["lambda_l2"][1])),
            ("min_split_gain", float(space["min_split_gain"][0]), float(space["min_split_gain"][1])),
        ]
        for key, low, high in checks:
            value = float(params[key])
            if value < float(low) or value > float(high):
                return False
        if max_depth > 0 and num_leaves > 2 ** max_depth:
            return False
        return True
    except Exception:
        return False


def suggest_lightgbm_phase1c_params(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    max_depth = int(trial.suggest_categorical("max_depth", list(space["max_depth_choices"])))
    num_leaves_low = int(space["num_leaves"][0])
    num_leaves_high = int(space["num_leaves"][1])
    if max_depth > 0:
        num_leaves_high = min(num_leaves_high, 2 ** max_depth)
    num_leaves_low = min(num_leaves_low, num_leaves_high)
    return {
        "learning_rate": float(trial.suggest_float("learning_rate", float(space["learning_rate"][0]), float(space["learning_rate"][1]), log=True)),
        "num_leaves": int(trial.suggest_int("num_leaves", num_leaves_low, num_leaves_high)),
        "max_depth": int(max_depth),
        "min_child_samples": int(trial.suggest_int("min_child_samples", int(space["min_child_samples"][0]), int(space["min_child_samples"][1]))),
        "feature_fraction": float(
            trial.suggest_float("feature_fraction", float(space["feature_fraction"][0]), float(space["feature_fraction"][1]))
        ),
        "bagging_fraction": float(
            trial.suggest_float("bagging_fraction", float(space["bagging_fraction"][0]), float(space["bagging_fraction"][1]))
        ),
        "bagging_freq": int(trial.suggest_categorical("bagging_freq", list(space["bagging_freq_choices"]))),
        "lambda_l1": float(trial.suggest_float("lambda_l1", float(space["lambda_l1"][0]), float(space["lambda_l1"][1]), log=True)),
        "lambda_l2": float(trial.suggest_float("lambda_l2", float(space["lambda_l2"][0]), float(space["lambda_l2"][1]), log=True)),
        "min_split_gain": float(trial.suggest_float("min_split_gain", float(space["min_split_gain"][0]), float(space["min_split_gain"][1]))),
        "max_bin": int(trial.suggest_categorical("max_bin", list(space["max_bin_choices"]))),
    }


def render_phase1c_lightgbm_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    best_trial = payload.get("best_trial")
    lines = [
        "# LightGBM Phase 1C Optuna Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Phase 1B source run: `{meta['phase1b_source_run_dir']}`",
        f"- Phase 1A auxiliary source run: `{meta['phase1a_aux_source_run_dir']}`",
        f"- Optuna prior observations loaded: `{payload['optuna']['prior_observations']}`",
        f"- Optuna prior source counts: `{payload['optuna']['prior_observations_by_phase']}`",
        f"- Unique new 1C trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best 1C Trial",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Optuna trial number: `{best_trial.get('optuna_trial_number')}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_split_gain",
            "max_bin",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top New 1C Trials",
            "",
            "| Rank | Trial | Optuna | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row.get('optuna_trial_number', '')} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    source_best = payload.get("source_best_trial")
    if source_best is not None:
        lines.extend(
            [
                "",
                "## Source Benchmark",
                "",
                f"- Best source phase: `{source_best.get('source_phase', 'unknown')}`",
                f"- Best source trial: `{source_best['trial_id']}`",
                f"- Best source validation RMSE: `{source_best['metrics']['validation']['rmse']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1c_lightgbm_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    optuna_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
    prior_observations: int,
    prior_observation_trials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prior_observation_trials = list(prior_observation_trials or [])
    prior_counts_by_phase: dict[str, int] = {}
    for row in prior_observation_trials:
        phase = str(row.get("source_phase", "unknown"))
        prior_counts_by_phase[phase] = int(prior_counts_by_phase.get(phase, 0) + 1)
    unique_trials, duplicate_trials, _ = split_lightgbm_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    source_leaderboard = rank_trial_rows(list(source_unique_trials))
    source_best = source_leaderboard[0] if source_leaderboard else None
    best_trial = leaderboard[0] if leaderboard else None
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase1b_source_run_dir": str(args.phase1c_source_run_dir),
            "phase1a_aux_source_run_dir": str(args.phase1c_aux_source_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1c_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": int(args.phase1c_max_iter),
            "early_stopping_rounds": int(args.phase1c_early_stopping_rounds),
            "force_col_wise": True,
        },
        "optuna": {
            "sampler": "TPESampler",
            "direction": "minimize",
            "multivariate": True,
            "group": False,
            "prior_observations": int(prior_observations),
            "prior_observations_by_phase": prior_counts_by_phase,
            "prior_observation_trials": prior_observation_trials,
            "space_topk": int(args.phase1c_topk),
        },
        "optuna_space": optuna_space,
        "progress": {
            "planned_trials": int(args.phase1c_trials),
            "completed_trials": int(len(unique_trials)),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1c_n_jobs),
        },
        "source_best_trial": source_best,
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1c_lightgbm_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    optuna_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
    prior_observations: int,
    prior_observation_trials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_phase1c_lightgbm_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        source_unique_trials=source_unique_trials,
        optuna_space=optuna_space,
        trials=trials,
        status=status,
        started_utc=started_utc,
        prior_observations=prior_observations,
        prior_observation_trials=prior_observation_trials,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
            "source_best_validation_rmse": None
            if payload["source_best_trial"] is None
            else float(payload["source_best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1c_lightgbm_summary(payload), encoding="utf-8")
    return payload


def run_phase1c_lightgbm(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb
    import optuna

    source_paths = [
        ("phase1b", args.phase1c_source_run_dir / "trial_results.jsonl"),
        ("phase1a", args.phase1c_aux_source_run_dir / "trial_results.jsonl"),
    ]
    source_unique_trials: list[dict[str, Any]] = []
    source_signature_to_trial_id: dict[str, int] = {}
    source_signature_to_origin: dict[str, dict[str, Any]] = {}
    source_metric_by_signature: dict[str, float] = {}
    for source_phase, source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing LightGBM Phase 1C source JSONL: {source_path}")
        source_rows = load_trial_jsonl(source_path)
        source_unique, _, source_signatures = split_lightgbm_attempts(source_rows)
        for row in source_unique:
            enriched = dict(row)
            enriched["source_phase"] = str(source_phase)
            source_unique_trials.append(enriched)
            signature = lightgbm_param_signature(enriched["params"])
            source_signature_to_trial_id[signature] = int(enriched["trial_id"])
            source_signature_to_origin[signature] = {
                "source_phase": str(source_phase),
                "source_trial_id": int(enriched["trial_id"]),
            }
            source_metric_by_signature[signature] = float(enriched["metrics"]["validation"]["rmse"])

    optuna_space = build_lightgbm_phase1c_space(source_unique_trials, int(args.phase1c_topk))
    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    cache_meta_path = cache_root / "meta.json"
    if cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1c_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1c_fold)], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 1C Run: LightGBM Optuna",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 1B source run: `{args.phase1c_source_run_dir}`",
            f"- Phase 1A auxiliary source run: `{args.phase1c_aux_source_run_dir}`",
            f"- Optuna trial budget: `{args.phase1c_trials}`",
            f"- Search-space source top-k: `{args.phase1c_topk}`",
            f"- Fold: `{args.phase1c_fold}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1c_max_iter}`",
            f"- Early stopping rounds: `{args.phase1c_early_stopping_rounds}`",
            f"- LightGBM threads: `{args.phase1c_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for LightGBM Phase 1C")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float64)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, current_signature_to_trial_id = split_lightgbm_attempts(trials)
    blocked_signatures: dict[str, int] = dict(source_signature_to_trial_id)
    for signature, trial_id in current_signature_to_trial_id.items():
        blocked_signatures[signature] = int(trial_id)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1

    sampler = optuna.samplers.TPESampler(seed=int(args.random_state), n_startup_trials=0, multivariate=True, group=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    prior_observations = 0
    prior_observation_trials: list[dict[str, Any]] = []
    prior_rows = [
        row
        for row in rank_trial_rows(list(source_unique_trials))
        if lightgbm_params_within_space(dict(row["params"]), optuna_space)
    ]
    for row in prior_rows:
        try:
            params = dict(row["params"])
            distributions = optuna_lightgbm_distributions(params, optuna_space)
            source_phase = str(row.get("source_phase", "unknown"))
            source_trial_id = int(row["trial_id"])
            validation_rmse = float(row["metrics"]["validation"]["rmse"])
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=validation_rmse,
                    user_attrs={
                        "source_phase": source_phase,
                        "source_trial_id": source_trial_id,
                    },
                )
            )
            prior_observations += 1
            prior_observation_trials.append(
                {
                    "source_phase": source_phase,
                    "source_trial_id": source_trial_id,
                    "source_rank_inside_all_lightgbm": int(row.get("rank", 0)),
                    "validation_rmse": validation_rmse,
                    "param_signature": lightgbm_param_signature(params),
                }
            )
        except Exception:
            continue
    json_dump(run_dir / "optuna_prior_observations.json", {"prior_observations": prior_observation_trials})
    for row in unique_trials:
        try:
            params = dict(row["params"])
            distributions = optuna_lightgbm_distributions(params, optuna_space)
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(row["metrics"]["validation"]["rmse"]),
                    user_attrs={"source_phase": "phase1c_resume", "source_trial_id": int(row["trial_id"])},
                )
            )
        except Exception:
            continue

    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1c_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: LightGBM Optuna",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed 1C trials already on disk: `{payload['progress']['completed_trials']}`",
                f"- Remaining unique 1C trials to run: `{max(int(args.phase1c_trials) - payload['progress']['completed_trials'], 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                f"- Prior Optuna observations loaded: `{prior_observations}`",
                "",
            ],
        )
    else:
        payload = write_phase1c_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Prior Optuna observations loaded: `{prior_observations}`",
                f"- Prior observations by source: `{build_phase1c_lightgbm_payload(args=args, run_dir=run_dir, meta=meta, train_idx=train_idx, val_idx=val_idx, source_unique_trials=source_unique_trials, optuna_space=optuna_space, trials=trials, status='running', started_utc=started_utc, prior_observations=prior_observations, prior_observation_trials=prior_observation_trials)['optuna']['prior_observations_by_phase']}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1c_trials):
        return write_phase1c_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )

    try:
        while len(unique_trials) < int(args.phase1c_trials):
            trial_id = int(next_trial_id)
            next_trial_id += 1
            trial_seed = int(args.random_state) + PHASE1C_TRIAL_SEED_OFFSET + trial_id
            optuna_trial = study.ask()
            params = suggest_lightgbm_phase1c_params(optuna_trial, optuna_space)
            param_signature = lightgbm_param_signature(params)
            if param_signature in blocked_signatures:
                duplicate_of_trial_id = int(blocked_signatures[param_signature])
                duplicate_source = "phase1c" if param_signature in current_signature_to_trial_id else "source"
                duplicate_origin = source_signature_to_origin.get(param_signature, {})
                known_rmse = source_metric_by_signature.get(param_signature)
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "optuna_trial_number": int(optuna_trial.number),
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "duplicate_source": str(duplicate_source),
                    "duplicate_source_phase": duplicate_origin.get("source_phase"),
                    "duplicate_source_trial_id": duplicate_origin.get("source_trial_id"),
                    "known_validation_rmse": None if known_rmse is None else float(known_rmse),
                    "timings_sec": {"fit": 0.0, "trial_total": 0.0},
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                if known_rmse is not None:
                    study.tell(optuna_trial, float(known_rmse))
                else:
                    study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
                payload = write_phase1c_lightgbm_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    source_unique_trials=source_unique_trials,
                    optuna_space=optuna_space,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                    prior_observations=prior_observations,
                    prior_observation_trials=prior_observation_trials,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### LightGBM Phase 1C Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                    f"- Duplicate source: `{duplicate_source}`",
                    f"- Duplicate source phase/trial: `{duplicate_origin.get('source_phase', '')}` / `{duplicate_origin.get('source_trial_id', '')}`",
                    f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique 1C trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue

            fit_started = time.time()
            model = lgb.LGBMRegressor(
                objective="regression",
                boosting_type="gbdt",
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(args.phase1c_max_iter),
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                min_child_samples=int(params["min_child_samples"]),
                subsample=float(params["bagging_fraction"]),
                subsample_freq=int(params["bagging_freq"]),
                colsample_bytree=float(params["feature_fraction"]),
                reg_alpha=float(params["lambda_l1"]),
                reg_lambda=float(params["lambda_l2"]),
                min_split_gain=float(params["min_split_gain"]),
                max_bin=int(params["max_bin"]),
                force_col_wise=True,
                random_state=trial_seed,
                bagging_seed=trial_seed,
                feature_fraction_seed=trial_seed,
                data_random_seed=trial_seed,
                deterministic=True,
                n_jobs=max(1, int(args.phase1c_n_jobs)),
                verbosity=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "validation"],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(int(args.phase1c_early_stopping_rounds), verbose=False)],
            )
            fit_seconds = time.time() - fit_started
            best_iteration = int(getattr(model, "best_iteration_", args.phase1c_max_iter) or args.phase1c_max_iter)
            if best_iteration <= 0:
                best_iteration = int(args.phase1c_max_iter)
            evals_result = getattr(model, "evals_result_", {})
            train_rmse_curve = evals_result.get("train", {}).get("rmse", [])
            train_rmse = float(train_rmse_curve[best_iteration - 1]) if best_iteration - 1 < len(train_rmse_curve) else float("nan")
            val_pred = model.predict(X_val, num_iteration=best_iteration)
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            study.tell(optuna_trial, float(val_metrics["rmse"]))
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "optuna_trial_number": int(optuna_trial.number),
                "best_iteration": int(best_iteration),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            blocked_signatures[param_signature] = int(trial_id)
            current_signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1c_lightgbm_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                source_unique_trials=source_unique_trials,
                optuna_space=optuna_space,
                trials=trials,
                status="running",
                started_utc=started_utc,
                prior_observations=prior_observations,
                prior_observation_trials=prior_observation_trials,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### LightGBM Phase 1C Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Optuna trial number: `{optuna_trial.number}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best 1C trial: `{best_row['trial_id']}`",
                    f"- Current best 1C validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del model
            gc.collect()
        payload = write_phase1c_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: LightGBM Optuna",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best 1C trial: `{payload['best_trial']['trial_id']}`",
                f"- Best 1C validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Source best validation RMSE: `{payload['source_best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1c_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status="failed",
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: LightGBM Optuna",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def render_phase1a_xgboost_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    status = payload["status"]
    best_trial = payload.get("best_trial")
    lines = [
        "# XGBoost Phase 1A Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{status}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        f"- Tree method: `{payload['fixed_settings']['tree_method']}`",
        f"- Matrix type: `{payload['fixed_settings']['matrix_type']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "max_bin",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1a_xgboost_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_xgboost_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    best_trial = leaderboard[0] if leaderboard else None
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1a_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "reg:squarederror",
            "metric": "rmse",
            "tree_method": "hist",
            "n_estimators": int(args.phase1a_max_iter),
            "early_stopping_rounds": int(args.phase1a_early_stopping_rounds),
            "matrix_type": "QuantileDMatrix",
        },
        "progress": {
            "planned_trials": int(args.phase1a_trials),
            "completed_trials": int(len(unique_trials)),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1a_n_jobs),
        },
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1a_xgboost_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1a_xgboost_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv_xgboost(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1a_xgboost_summary(payload), encoding="utf-8")
    return payload


def render_phase1b_xgboost_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    status = payload["status"]
    best_trial = payload.get("best_trial")
    lines = [
        "# XGBoost Phase 1B Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{status}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Phase 1A source run: `{meta['phase1a_source_run_dir']}`",
        f"- Source signatures blocked: `{meta['blocked_source_signatures']}`",
        f"- Focused seed trials from Phase 1A: `{', '.join(str(x) for x in payload['focused_space']['seed_trial_ids'])}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Focused unique trials completed: `{progress['focused_completed_trials']}` / `{progress['focused_target_trials']}`",
        f"- Global unique trials completed: `{progress['global_completed_trials']}` / `{progress['global_target_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        f"- Tree method: `{payload['fixed_settings']['tree_method']}`",
        f"- Matrix type: `{payload['fixed_settings']['matrix_type']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Search stage: `{best_trial.get('search_stage', 'unknown')}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "max_bin",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Stage | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row.get('search_stage', 'unknown')} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1b_xgboost_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_xgboost_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    best_trial = leaderboard[0] if leaderboard else None
    focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
    global_completed = sum(1 for row in unique_trials if row.get("search_stage") == "global")
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase1a_source_run_dir": str(args.phase1b_source_run_dir),
            "blocked_source_signatures": int(len(source_unique_trials)),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1b_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "reg:squarederror",
            "metric": "rmse",
            "tree_method": "hist",
            "n_estimators": int(args.phase1b_max_iter),
            "early_stopping_rounds": int(args.phase1b_early_stopping_rounds),
            "matrix_type": "QuantileDMatrix",
        },
        "focused_space": focused_space,
        "progress": {
            "planned_trials": int(args.phase1b_trials),
            "focused_target_trials": int(args.phase1b_focused_trials),
            "global_target_trials": int(args.phase1b_global_trials),
            "completed_trials": int(len(unique_trials)),
            "focused_completed_trials": int(focused_completed),
            "global_completed_trials": int(global_completed),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1b_n_jobs),
        },
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1b_xgboost_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1b_xgboost_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        source_unique_trials=source_unique_trials,
        focused_space=focused_space,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "focused_completed_trials": payload["progress"]["focused_completed_trials"],
            "global_completed_trials": payload["progress"]["global_completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv_xgboost(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1b_xgboost_summary(payload), encoding="utf-8")
    return payload


def run_phase1a_lightgbm(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    cache_meta_path = cache_root / "meta.json"
    if cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1a_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1a_fold)], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 1A Run: LightGBM",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Fold: `{args.phase1a_fold}`",
            f"- Planned trials: `{args.phase1a_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1a_max_iter}`",
            f"- Early stopping rounds: `{args.phase1a_early_stopping_rounds}`",
            f"- LightGBM threads: `{args.phase1a_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for LightGBM Phase 1A")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float64)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, signature_to_trial_id = split_lightgbm_attempts(trials)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1
    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1a_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: LightGBM",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials already on disk: `{len(unique_trials)}`",
                f"- Duplicate attempts already on disk: `{len(duplicate_trials)}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1a_trials) - len(unique_trials), 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_phase1a_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1a_trials):
        payload = write_phase1a_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: LightGBM",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- No additional work required because `{len(unique_trials)}` unique completed trials were already present.",
                "",
            ],
        )
        return payload

    try:
        while len(unique_trials) < int(args.phase1a_trials):
            trial_id = int(next_trial_id)
            next_trial_id += 1
            trial_seed = int(args.random_state) + trial_id
            trial_rng = np.random.default_rng(trial_seed)
            params = sample_lightgbm_phase1a_params(trial_rng)
            param_signature = lightgbm_param_signature(params)
            if param_signature in signature_to_trial_id:
                duplicate_of_trial_id = int(signature_to_trial_id[param_signature])
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "timings_sec": {
                        "fit": 0.0,
                        "trial_total": 0.0,
                    },
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_phase1a_lightgbm_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### LightGBM Phase 1A Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue
            fit_started = time.time()
            model = lgb.LGBMRegressor(
                objective="regression",
                boosting_type="gbdt",
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(args.phase1a_max_iter),
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                min_child_samples=int(params["min_child_samples"]),
                subsample=float(params["bagging_fraction"]),
                subsample_freq=int(params["bagging_freq"]),
                colsample_bytree=float(params["feature_fraction"]),
                reg_alpha=float(params["lambda_l1"]),
                reg_lambda=float(params["lambda_l2"]),
                min_split_gain=float(params["min_split_gain"]),
                max_bin=int(params["max_bin"]),
                force_col_wise=True,
                random_state=trial_seed,
                bagging_seed=trial_seed,
                feature_fraction_seed=trial_seed,
                data_random_seed=trial_seed,
                deterministic=True,
                n_jobs=max(1, int(args.phase1a_n_jobs)),
                verbosity=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "validation"],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(int(args.phase1a_early_stopping_rounds), verbose=False)],
            )
            fit_seconds = time.time() - fit_started
            best_iteration = int(getattr(model, "best_iteration_", args.phase1a_max_iter) or args.phase1a_max_iter)
            if best_iteration <= 0:
                best_iteration = int(args.phase1a_max_iter)
            evals_result = getattr(model, "evals_result_", {})
            train_rmse_curve = evals_result.get("train", {}).get("rmse", [])
            train_rmse = float(train_rmse_curve[best_iteration - 1]) if best_iteration - 1 < len(train_rmse_curve) else float("nan")
            val_pred = model.predict(X_val, num_iteration=best_iteration)
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "best_iteration": int(best_iteration),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1a_lightgbm_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                trials=trials,
                status="running",
                started_utc=started_utc,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### LightGBM Phase 1A Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            gc.collect()
        payload = write_phase1a_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: LightGBM",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1a_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: LightGBM",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def run_phase1b_lightgbm(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    if int(args.phase1b_focused_trials) + int(args.phase1b_global_trials) != int(args.phase1b_trials):
        raise ValueError("Phase 1B LightGBM trial counts must satisfy focused + global == total trials.")

    source_trial_jsonl_path = args.phase1b_source_run_dir / "trial_results.jsonl"
    if not source_trial_jsonl_path.exists():
        raise FileNotFoundError(f"Missing Phase 1A LightGBM trial JSONL: {source_trial_jsonl_path}")

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1b_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1b_fold)], dtype=np.int64))

    source_trials_all = load_trial_jsonl(source_trial_jsonl_path)
    source_unique_trials, _, source_signature_to_trial_id = split_lightgbm_attempts(source_trials_all)
    focused_space = build_lightgbm_phase1b_space(source_unique_trials, int(args.phase1b_topk))

    append_markdown_log(
        log_path,
        [
            "### Phase 1B Run: LightGBM",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 1A source run: `{args.phase1b_source_run_dir}`",
            f"- Blocked source signatures: `{len(source_unique_trials)}`",
            f"- Focused seed trials: `{', '.join(str(x) for x in focused_space['seed_trial_ids'])}`",
            f"- Fold: `{args.phase1b_fold}`",
            f"- Planned trials: `{args.phase1b_trials}`",
            f"- Focused / global plan: `{args.phase1b_focused_trials}` / `{args.phase1b_global_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1b_max_iter}`",
            f"- Early stopping rounds: `{args.phase1b_early_stopping_rounds}`",
            f"- LightGBM threads: `{args.phase1b_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for LightGBM Phase 1B")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float64)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, current_signature_to_trial_id = split_lightgbm_attempts(trials)
    blocked_signatures: dict[str, int] = dict(source_signature_to_trial_id)
    for signature, trial_id in current_signature_to_trial_id.items():
        blocked_signatures[signature] = int(trial_id)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1

    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1b_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: LightGBM",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials already on disk: `{payload['progress']['completed_trials']}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1b_trials) - payload['progress']['completed_trials'], 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_phase1b_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1b_trials):
        payload = write_phase1b_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        return payload

    try:
        while len(unique_trials) < int(args.phase1b_trials):
            focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
            search_stage = "focused" if focused_completed < int(args.phase1b_focused_trials) else "global"
            trial_id = int(next_trial_id)
            next_trial_id += 1
            seed_offset = PHASE1B_GLOBAL_SEED_OFFSET if search_stage == "global" else 0
            trial_seed = int(args.random_state) + seed_offset + trial_id
            trial_rng = np.random.default_rng(trial_seed)
            if search_stage == "focused":
                params = sample_lightgbm_phase1b_focused_params(trial_rng, focused_space)
            else:
                params = sample_lightgbm_phase1a_params(trial_rng)
            param_signature = lightgbm_param_signature(params)
            duplicate_source = "phase1a" if param_signature in source_signature_to_trial_id else "phase1b"
            if param_signature in blocked_signatures:
                duplicate_of_trial_id = int(blocked_signatures[param_signature])
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "params": params,
                    "search_stage": str(search_stage),
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "duplicate_source": str(duplicate_source),
                    "timings_sec": {
                        "fit": 0.0,
                        "trial_total": 0.0,
                    },
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_phase1b_lightgbm_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    source_unique_trials=source_unique_trials,
                    focused_space=focused_space,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### LightGBM Phase 1B Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Search stage: `{search_stage}`",
                        f"- Duplicate source: `{duplicate_source}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue
            fit_started = time.time()
            model = lgb.LGBMRegressor(
                objective="regression",
                boosting_type="gbdt",
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(args.phase1b_max_iter),
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                min_child_samples=int(params["min_child_samples"]),
                subsample=float(params["bagging_fraction"]),
                subsample_freq=int(params["bagging_freq"]),
                colsample_bytree=float(params["feature_fraction"]),
                reg_alpha=float(params["lambda_l1"]),
                reg_lambda=float(params["lambda_l2"]),
                min_split_gain=float(params["min_split_gain"]),
                max_bin=int(params["max_bin"]),
                force_col_wise=True,
                random_state=trial_seed,
                bagging_seed=trial_seed,
                feature_fraction_seed=trial_seed,
                data_random_seed=trial_seed,
                deterministic=True,
                n_jobs=max(1, int(args.phase1b_n_jobs)),
                verbosity=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "validation"],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(int(args.phase1b_early_stopping_rounds), verbose=False)],
            )
            fit_seconds = time.time() - fit_started
            best_iteration = int(getattr(model, "best_iteration_", args.phase1b_max_iter) or args.phase1b_max_iter)
            if best_iteration <= 0:
                best_iteration = int(args.phase1b_max_iter)
            evals_result = getattr(model, "evals_result_", {})
            train_rmse_curve = evals_result.get("train", {}).get("rmse", [])
            train_rmse = float(train_rmse_curve[best_iteration - 1]) if best_iteration - 1 < len(train_rmse_curve) else float("nan")
            val_pred = model.predict(X_val, num_iteration=best_iteration)
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "search_stage": str(search_stage),
                "best_iteration": int(best_iteration),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            blocked_signatures[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1b_lightgbm_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                source_unique_trials=source_unique_trials,
                focused_space=focused_space,
                trials=trials,
                status="running",
                started_utc=started_utc,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### LightGBM Phase 1B Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Search stage: `{search_stage}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del model
            gc.collect()
        payload = write_phase1b_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: LightGBM",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1b_lightgbm_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: LightGBM",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def run_phase1a_xgboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import xgboost as xgb

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1a_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1a_fold)], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 1A Run: XGBoost",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Fold: `{args.phase1a_fold}`",
            f"- Planned trials: `{args.phase1a_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1a_max_iter}`",
            f"- Early stopping rounds: `{args.phase1a_early_stopping_rounds}`",
            f"- XGBoost threads: `{args.phase1a_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for XGBoost Phase 1A")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0

    dmatrix_cache: dict[int, tuple[Any, Any]] = {}
    dmatrix_build_seconds_by_max_bin: dict[int, float] = {}

    def get_quantile_matrices(max_bin: int) -> tuple[Any, Any]:
        cache_key = int(max_bin)
        if cache_key in dmatrix_cache:
            return dmatrix_cache[cache_key]
        log(f"Building QuantileDMatrix objects for XGBoost Phase 1A with max_bin={cache_key}")
        t0_local = time.time()
        dtrain_local = xgb.QuantileDMatrix(
            X_train,
            label=y_train,
            max_bin=cache_key,
            nthread=max(1, int(args.phase1a_n_jobs)),
        )
        dval_local = xgb.QuantileDMatrix(
            X_val,
            label=y_val,
            ref=dtrain_local,
            max_bin=cache_key,
            nthread=max(1, int(args.phase1a_n_jobs)),
        )
        dmatrix_cache[cache_key] = (dtrain_local, dval_local)
        dmatrix_build_seconds_by_max_bin[cache_key] = float(time.time() - t0_local)
        return dtrain_local, dval_local

    dmatrix_build_seconds = 0.0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, signature_to_trial_id = split_xgboost_attempts(trials)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1
    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1a_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: XGBoost",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials already on disk: `{len(unique_trials)}`",
                f"- Duplicate attempts already on disk: `{len(duplicate_trials)}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1a_trials) - len(unique_trials), 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                f"- QuantileDMatrix build seconds this launch: `{dmatrix_build_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_phase1a_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- QuantileDMatrix build seconds: `{dmatrix_build_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1a_trials):
        payload = write_phase1a_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: XGBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- No additional work required because `{len(unique_trials)}` unique completed trials were already present.",
                "",
            ],
        )
        return payload

    try:
        while len(unique_trials) < int(args.phase1a_trials):
            trial_id = int(next_trial_id)
            next_trial_id += 1
            trial_seed = int(args.random_state) + trial_id
            trial_rng = np.random.default_rng(trial_seed)
            params = sample_xgboost_phase1a_params(trial_rng)
            param_signature = xgboost_param_signature(params)
            if param_signature in signature_to_trial_id:
                duplicate_of_trial_id = int(signature_to_trial_id[param_signature])
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "timings_sec": {
                        "fit": 0.0,
                        "trial_total": 0.0,
                    },
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_phase1a_xgboost_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### XGBoost Phase 1A Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue
            xgb_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "learning_rate": float(params["learning_rate"]),
                "max_depth": int(params["max_depth"]),
                "min_child_weight": float(params["min_child_weight"]),
                "subsample": float(params["subsample"]),
                "colsample_bytree": float(params["colsample_bytree"]),
                "reg_alpha": float(params["reg_alpha"]),
                "reg_lambda": float(params["reg_lambda"]),
                "gamma": float(params["gamma"]),
                "max_bin": int(params["max_bin"]),
                "seed": int(trial_seed),
                "nthread": max(1, int(args.phase1a_n_jobs)),
                "verbosity": 0,
            }
            dtrain, dval = get_quantile_matrices(int(params["max_bin"]))
            fit_started = time.time()
            evals_result: dict[str, Any] = {}
            booster = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=int(args.phase1a_max_iter),
                evals=[(dtrain, "train"), (dval, "validation")],
                evals_result=evals_result,
                early_stopping_rounds=int(args.phase1a_early_stopping_rounds),
                verbose_eval=False,
            )
            fit_seconds = time.time() - fit_started
            best_iteration = int(getattr(booster, "best_iteration", args.phase1a_max_iter - 1))
            if best_iteration < 0:
                best_iteration = int(args.phase1a_max_iter - 1)
            best_iteration_one_based = int(best_iteration + 1)
            train_rmse_curve = evals_result.get("train", {}).get("rmse", [])
            train_rmse = float(train_rmse_curve[best_iteration]) if best_iteration < len(train_rmse_curve) else float("nan")
            val_pred = booster.predict(dval, iteration_range=(0, best_iteration_one_based))
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "best_iteration": int(best_iteration_one_based),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1a_xgboost_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                trials=trials,
                status="running",
                started_utc=started_utc,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### XGBoost Phase 1A Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration_one_based}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del booster
            del evals_result
            gc.collect()
        payload = write_phase1a_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: XGBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1a_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: XGBoost",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def run_phase1b_xgboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import xgboost as xgb

    if int(args.phase1b_focused_trials) + int(args.phase1b_global_trials) != int(args.phase1b_trials):
        raise ValueError("Phase 1B XGBoost trial counts must satisfy focused + global == total trials.")

    source_trial_jsonl_path = args.phase1b_source_run_dir / "trial_results.jsonl"
    if not source_trial_jsonl_path.exists():
        raise FileNotFoundError(f"Missing Phase 1A XGBoost trial JSONL: {source_trial_jsonl_path}")

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1b_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1b_fold)], dtype=np.int64))

    source_trials_all = load_trial_jsonl(source_trial_jsonl_path)
    source_unique_trials, _, source_signature_to_trial_id = split_xgboost_attempts(source_trials_all)
    focused_space = build_xgboost_phase1b_space(source_unique_trials, int(args.phase1b_topk))

    append_markdown_log(
        log_path,
        [
            "### Phase 1B Run: XGBoost",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 1A source run: `{args.phase1b_source_run_dir}`",
            f"- Blocked source signatures: `{len(source_unique_trials)}`",
            f"- Focused seed trials: `{', '.join(str(x) for x in focused_space['seed_trial_ids'])}`",
            f"- Fold: `{args.phase1b_fold}`",
            f"- Planned trials: `{args.phase1b_trials}`",
            f"- Focused / global plan: `{args.phase1b_focused_trials}` / `{args.phase1b_global_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1b_max_iter}`",
            f"- Early stopping rounds: `{args.phase1b_early_stopping_rounds}`",
            f"- XGBoost threads: `{args.phase1b_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for XGBoost Phase 1B")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0

    dmatrix_cache: dict[int, tuple[Any, Any]] = {}
    dmatrix_build_seconds_by_max_bin: dict[int, float] = {}

    def get_quantile_matrices(max_bin: int) -> tuple[Any, Any]:
        cache_key = int(max_bin)
        if cache_key in dmatrix_cache:
            return dmatrix_cache[cache_key]
        log(f"Building QuantileDMatrix objects for XGBoost Phase 1B with max_bin={cache_key}")
        t0_local = time.time()
        dtrain_local = xgb.QuantileDMatrix(
            X_train,
            label=y_train,
            max_bin=cache_key,
            nthread=max(1, int(args.phase1b_n_jobs)),
        )
        dval_local = xgb.QuantileDMatrix(
            X_val,
            label=y_val,
            ref=dtrain_local,
            max_bin=cache_key,
            nthread=max(1, int(args.phase1b_n_jobs)),
        )
        dmatrix_cache[cache_key] = (dtrain_local, dval_local)
        dmatrix_build_seconds_by_max_bin[cache_key] = float(time.time() - t0_local)
        return dtrain_local, dval_local

    dmatrix_build_seconds = 0.0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, current_signature_to_trial_id = split_xgboost_attempts(trials)
    blocked_signatures: dict[str, int] = dict(source_signature_to_trial_id)
    for signature, trial_id in current_signature_to_trial_id.items():
        blocked_signatures[signature] = int(trial_id)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1

    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1b_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: XGBoost",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed Phase 1B trials already on disk: `{payload['progress']['completed_trials']}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1b_trials) - payload['progress']['completed_trials'], 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                f"- QuantileDMatrix build seconds this launch: `{dmatrix_build_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_phase1b_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- QuantileDMatrix build seconds: `{dmatrix_build_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1b_trials):
        payload = write_phase1b_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: XGBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- No additional work required because `{len(unique_trials)}` unique completed Phase 1B trials were already present.",
                "",
            ],
        )
        return payload

    try:
        while len(unique_trials) < int(args.phase1b_trials):
            focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
            search_stage = "focused" if focused_completed < int(args.phase1b_focused_trials) else "global"
            trial_id = int(next_trial_id)
            next_trial_id += 1
            if search_stage == "focused":
                trial_seed = int(args.random_state) + trial_id
                trial_rng = np.random.default_rng(trial_seed)
                params = sample_xgboost_phase1b_focused_params(trial_rng, focused_space)
            else:
                trial_seed = int(args.random_state) + PHASE1B_GLOBAL_SEED_OFFSET + trial_id
                trial_rng = np.random.default_rng(trial_seed)
                params = sample_xgboost_phase1a_params(trial_rng)
            param_signature = xgboost_param_signature(params)
            if param_signature in blocked_signatures:
                duplicate_of_trial_id = int(blocked_signatures[param_signature])
                duplicate_source = "phase1a" if param_signature in source_signature_to_trial_id else "phase1b"
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "search_stage": search_stage,
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_source": duplicate_source,
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "timings_sec": {
                        "fit": 0.0,
                        "trial_total": 0.0,
                    },
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_phase1b_xgboost_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    source_unique_trials=source_unique_trials,
                    focused_space=focused_space,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### XGBoost Phase 1B Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Search stage: `{search_stage}`",
                        f"- Duplicate source: `{duplicate_source}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue

            xgb_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "learning_rate": float(params["learning_rate"]),
                "max_depth": int(params["max_depth"]),
                "min_child_weight": float(params["min_child_weight"]),
                "subsample": float(params["subsample"]),
                "colsample_bytree": float(params["colsample_bytree"]),
                "reg_alpha": float(params["reg_alpha"]),
                "reg_lambda": float(params["reg_lambda"]),
                "gamma": float(params["gamma"]),
                "max_bin": int(params["max_bin"]),
                "seed": int(trial_seed),
                "nthread": max(1, int(args.phase1b_n_jobs)),
                "verbosity": 0,
            }
            dtrain, dval = get_quantile_matrices(int(params["max_bin"]))
            fit_started = time.time()
            evals_result: dict[str, Any] = {}
            booster = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=int(args.phase1b_max_iter),
                evals=[(dtrain, "train"), (dval, "validation")],
                evals_result=evals_result,
                early_stopping_rounds=int(args.phase1b_early_stopping_rounds),
                verbose_eval=False,
            )
            fit_seconds = time.time() - fit_started
            best_iteration = int(getattr(booster, "best_iteration", args.phase1b_max_iter - 1))
            if best_iteration < 0:
                best_iteration = int(args.phase1b_max_iter - 1)
            best_iteration_one_based = int(best_iteration + 1)
            train_rmse_curve = evals_result.get("train", {}).get("rmse", [])
            train_rmse = float(train_rmse_curve[best_iteration]) if best_iteration < len(train_rmse_curve) else float("nan")
            val_pred = booster.predict(dval, iteration_range=(0, best_iteration_one_based))
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "search_stage": search_stage,
                "best_iteration": int(best_iteration_one_based),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            blocked_signatures[param_signature] = int(trial_id)
            current_signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1b_xgboost_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                source_unique_trials=source_unique_trials,
                focused_space=focused_space,
                trials=trials,
                status="running",
                started_utc=started_utc,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### XGBoost Phase 1B Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Search stage: `{search_stage}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration_one_based}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del booster
            del evals_result
            gc.collect()

        payload = write_phase1b_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: XGBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Focused / global completed: `{payload['progress']['focused_completed_trials']}` / `{payload['progress']['global_completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1b_xgboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: XGBoost",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def render_phase1a_catboost_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    status = payload["status"]
    best_trial = payload.get("best_trial")
    lines = [
        "# CatBoost Phase 1A Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{status}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed max iterations: `{payload['fixed_settings']['iterations']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        f"- Bootstrap type: `{payload['fixed_settings']['bootstrap_type']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in (
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "rsm",
            "bagging_temperature",
            "border_count",
        ):
            value = best_trial["params"][key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.8f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['trial_id']} | "
            f"{row['metrics']['validation']['rmse']:.6f} | "
            f"{row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | "
            f"{row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | "
            f"{row['best_iteration']} | "
            f"{row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1a_catboost_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_catboost_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    best_trial = leaderboard[0] if leaderboard else None
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1a_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": int(args.phase1a_max_iter),
            "early_stopping_rounds": int(args.phase1a_early_stopping_rounds),
            "bootstrap_type": "Bayesian",
        },
        "progress": {
            "planned_trials": int(args.phase1a_trials),
            "completed_trials": int(len(unique_trials)),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1a_n_jobs),
        },
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1a_catboost_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1a_catboost_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv_catboost(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1a_catboost_summary(payload), encoding="utf-8")
    return payload


def render_phase1b_catboost_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    best_trial = payload.get("best_trial")
    lines = [
        "# CatBoost Phase 1B Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Phase 1A source run: `{meta['phase1a_source_run_dir']}`",
        f"- Source signatures blocked: `{meta['blocked_source_signatures']}`",
        f"- Focused seed trials from Phase 1A: `{', '.join(str(x) for x in payload['focused_space']['seed_trial_ids'])}`",
        f"- Unique trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Focused unique trials completed: `{progress['focused_completed_trials']}` / `{progress['focused_target_trials']}`",
        f"- Global unique trials completed: `{progress['global_completed_trials']}` / `{progress['global_target_trials']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed iterations: `{payload['fixed_settings']['iterations']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Search stage: `{best_trial.get('search_stage', 'unknown')}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in ("learning_rate", "depth", "l2_leaf_reg", "rsm", "bagging_temperature", "border_count"):
            value = best_trial["params"][key]
            lines.append(f"- `{key}`: `{value:.8f}`" if isinstance(value, float) else f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Trials",
            "",
            "| Rank | Trial | Stage | Val RMSE | Val MAE | Val R^2 | Train RMSE | Gap | Iter | Fit sec |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["leaderboard"][:10]:
        lines.append(
            "| "
            f"{row['rank']} | {row['trial_id']} | {row.get('search_stage', 'unknown')} | "
            f"{row['metrics']['validation']['rmse']:.6f} | {row['metrics']['validation']['mae']:.6f} | "
            f"{row['metrics']['validation']['r2']:.6f} | {row['metrics']['train']['rmse']:.6f} | "
            f"{row['metrics']['rmse_gap']:.6f} | {row['best_iteration']} | {row['timings_sec']['fit']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{meta['run_dir']}\\results.json`",
            f"- Trial JSONL: `{meta['run_dir']}\\trial_results.jsonl`",
            f"- Trial CSV: `{meta['run_dir']}\\trial_results.csv`",
            f"- Status JSON: `{meta['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase1b_catboost_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    unique_trials, duplicate_trials, _ = split_catboost_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
    global_completed = sum(1 for row in unique_trials if row.get("search_stage") == "global")
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase1a_source_run_dir": str(args.phase1b_source_run_dir),
            "blocked_source_signatures": int(len(source_unique_trials)),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1b_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": int(args.phase1b_max_iter),
            "early_stopping_rounds": int(args.phase1b_early_stopping_rounds),
            "bootstrap_type": "Bayesian",
        },
        "focused_space": focused_space,
        "progress": {
            "planned_trials": int(args.phase1b_trials),
            "focused_target_trials": int(args.phase1b_focused_trials),
            "global_target_trials": int(args.phase1b_global_trials),
            "completed_trials": int(len(unique_trials)),
            "focused_completed_trials": int(focused_completed),
            "global_completed_trials": int(global_completed),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1b_n_jobs),
        },
        "best_trial": leaderboard[0] if leaderboard else None,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1b_catboost_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_unique_trials: list[dict[str, Any]],
    focused_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase1b_catboost_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        source_unique_trials=source_unique_trials,
        focused_space=focused_space,
        trials=trials,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "focused_completed_trials": payload["progress"]["focused_completed_trials"],
            "global_completed_trials": payload["progress"]["global_completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv_catboost(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1b_catboost_summary(payload), encoding="utf-8")
    return payload


def render_phase1c_catboost_summary(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = payload["progress"]
    split = payload["split"]
    best_trial = payload.get("best_trial")
    lines = [
        "# CatBoost Phase 1C Optuna Search",
        "",
        f"- Run dir: `{meta['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Source split run: `{meta['source_split_run_dir']}`",
        f"- Phase 1B source run: `{meta['phase1b_source_run_dir']}`",
        f"- Phase 1A auxiliary source run: `{meta['phase1a_aux_source_run_dir']}`",
        f"- Optuna prior observations loaded: `{payload['optuna']['prior_observations']}`",
        f"- Optuna prior source counts: `{payload['optuna']['prior_observations_by_phase']}`",
        f"- Unique new 1C trials completed: `{progress['completed_trials']}` / `{progress['planned_trials']}`",
        f"- Total attempts logged: `{progress['total_attempts']}`",
        f"- Duplicate attempts detected: `{progress['duplicate_attempts']}`",
        f"- Fold-0 training rows: `{split['train_rows']:,}`",
        f"- Fold-0 validation rows: `{split['validation_rows']:,}`",
        f"- Fixed iterations: `{payload['fixed_settings']['iterations']}`",
        f"- Early stopping rounds: `{payload['fixed_settings']['early_stopping_rounds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Current Best 1C Trial",
                "",
                f"- Trial: `{best_trial['trial_id']}`",
                f"- Rank: `{best_trial['rank']}`",
                f"- Optuna trial number: `{best_trial.get('optuna_trial_number')}`",
                f"- Validation RMSE: `{best_trial['metrics']['validation']['rmse']:.6f}`",
                f"- Validation MAE: `{best_trial['metrics']['validation']['mae']:.6f}`",
                f"- Validation R^2: `{best_trial['metrics']['validation']['r2']:.6f}`",
                f"- Train RMSE at best iteration: `{best_trial['metrics']['train']['rmse']:.6f}`",
                f"- RMSE gap: `{best_trial['metrics']['rmse_gap']:.6f}`",
                f"- Best iteration: `{best_trial['best_iteration']}`",
                f"- Fit seconds: `{best_trial['timings_sec']['fit']:.1f}`",
                "",
                "### Best Params",
                "",
            ]
        )
        for key in ("learning_rate", "depth", "l2_leaf_reg", "rsm", "bagging_temperature", "border_count"):
            lines.append(f"- `{key}`: `{best_trial['params'][key]}`")
        lines.append("")
    source_best = payload.get("source_best_trial")
    if source_best is not None:
        lines.extend(
            [
                "## Best Source Trial",
                "",
                f"- Source phase: `{source_best.get('source_phase', 'unknown')}`",
                f"- Source trial: `{source_best['trial_id']}`",
                f"- Validation RMSE: `{source_best['metrics']['validation']['rmse']:.6f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def build_phase1c_catboost_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    phase1b_source_run_dir: Path,
    phase1a_aux_source_run_dir: Path,
    source_unique_trials: list[dict[str, Any]],
    optuna_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
    prior_observations: int,
    prior_observation_trials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prior_observation_trials = list(prior_observation_trials or [])
    prior_counts_by_phase: dict[str, int] = {}
    for row in prior_observation_trials:
        phase = str(row.get("source_phase", "unknown"))
        prior_counts_by_phase[phase] = int(prior_counts_by_phase.get(phase, 0) + 1)
    unique_trials, duplicate_trials, _ = split_catboost_attempts(trials)
    leaderboard = rank_trial_rows(unique_trials)
    source_leaderboard = rank_trial_rows(list(source_unique_trials))
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase1b_source_run_dir": str(phase1b_source_run_dir),
            "phase1a_aux_source_run_dir": str(phase1a_aux_source_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": status,
        "dependency_status": dependency_status(),
        "split": {
            "fold": int(args.phase1c_fold),
            "train_rows": int(train_idx.shape[0]),
            "validation_rows": int(val_idx.shape[0]),
        },
        "fixed_settings": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": int(args.phase1c_max_iter),
            "early_stopping_rounds": int(args.phase1c_early_stopping_rounds),
            "bootstrap_type": "Bayesian",
        },
        "optuna": {
            "sampler": "TPESampler",
            "direction": "minimize",
            "multivariate": True,
            "group": False,
            "prior_observations": int(prior_observations),
            "prior_observations_by_phase": prior_counts_by_phase,
            "prior_observation_trials": prior_observation_trials,
            "space_topk": int(args.phase1c_topk),
        },
        "optuna_space": optuna_space,
        "progress": {
            "planned_trials": int(args.phase1c_trials),
            "completed_trials": int(len(unique_trials)),
            "total_attempts": int(len(trials)),
            "duplicate_attempts": int(len(duplicate_trials)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase1c_n_jobs),
        },
        "source_best_trial": source_leaderboard[0] if source_leaderboard else None,
        "best_trial": leaderboard[0] if leaderboard else None,
        "leaderboard": leaderboard,
        "trials": leaderboard,
        "duplicate_attempts": duplicate_trials,
    }


def write_phase1c_catboost_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    phase1b_source_run_dir: Path,
    phase1a_aux_source_run_dir: Path,
    source_unique_trials: list[dict[str, Any]],
    optuna_space: dict[str, Any],
    trials: list[dict[str, Any]],
    status: str,
    started_utc: str,
    prior_observations: int,
    prior_observation_trials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_phase1c_catboost_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        train_idx=train_idx,
        val_idx=val_idx,
        phase1b_source_run_dir=phase1b_source_run_dir,
        phase1a_aux_source_run_dir=phase1a_aux_source_run_dir,
        source_unique_trials=source_unique_trials,
        optuna_space=optuna_space,
        trials=trials,
        status=status,
        started_utc=started_utc,
        prior_observations=prior_observations,
        prior_observation_trials=prior_observation_trials,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_trials": payload["progress"]["planned_trials"],
            "completed_trials": payload["progress"]["completed_trials"],
            "total_attempts": payload["progress"]["total_attempts"],
            "duplicate_attempts": payload["progress"]["duplicate_attempts"],
            "best_trial_id": None if payload["best_trial"] is None else int(payload["best_trial"]["trial_id"]),
            "best_validation_rmse": None
            if payload["best_trial"] is None
            else float(payload["best_trial"]["metrics"]["validation"]["rmse"]),
            "source_best_validation_rmse": None
            if payload["source_best_trial"] is None
            else float(payload["source_best_trial"]["metrics"]["validation"]["rmse"]),
        },
    )
    write_trial_csv_catboost(run_dir / "trial_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase1c_catboost_summary(payload), encoding="utf-8")
    return payload


def resolve_catboost_phase1c_sources(args: argparse.Namespace) -> tuple[Path, Path]:
    phase1b_source_run_dir = args.phase1c_source_run_dir
    phase1a_aux_source_run_dir = args.phase1c_aux_source_run_dir
    if phase1b_source_run_dir == DEFAULT_LIGHTGBM_PHASE1B_SOURCE_RUN:
        phase1b_source_run_dir = DEFAULT_CATBOOST_PHASE1B_SOURCE_RUN
    if phase1a_aux_source_run_dir == DEFAULT_LIGHTGBM_PHASE1A_SOURCE_RUN:
        phase1a_aux_source_run_dir = DEFAULT_CATBOOST_PHASE1A_SOURCE_RUN
    return phase1b_source_run_dir, phase1a_aux_source_run_dir


def run_phase1c_catboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    import optuna

    phase1b_source_run_dir, phase1a_aux_source_run_dir = resolve_catboost_phase1c_sources(args)
    source_paths = [
        ("phase1b", phase1b_source_run_dir / "trial_results.jsonl"),
        ("phase1a", phase1a_aux_source_run_dir / "trial_results.jsonl"),
    ]
    source_unique_trials: list[dict[str, Any]] = []
    source_signature_to_trial_id: dict[str, int] = {}
    source_signature_to_origin: dict[str, dict[str, Any]] = {}
    source_metric_by_signature: dict[str, float] = {}
    for source_phase, source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing CatBoost Phase 1C source JSONL: {source_path}")
        source_unique, _, _ = split_catboost_attempts(load_trial_jsonl(source_path))
        for row in source_unique:
            enriched = dict(row)
            enriched["source_phase"] = str(source_phase)
            source_unique_trials.append(enriched)
            signature = catboost_param_signature(enriched["params"])
            source_signature_to_trial_id[signature] = int(enriched["trial_id"])
            source_signature_to_origin[signature] = {"source_phase": str(source_phase), "source_trial_id": int(enriched["trial_id"])}
            source_metric_by_signature[signature] = float(enriched["metrics"]["validation"]["rmse"])

    optuna_space = build_catboost_phase1c_space(source_unique_trials, int(args.phase1c_topk))
    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1c_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1c_fold)], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 1C Run: CatBoost Optuna",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 1B source run: `{phase1b_source_run_dir}`",
            f"- Phase 1A auxiliary source run: `{phase1a_aux_source_run_dir}`",
            f"- Optuna trial budget: `{args.phase1c_trials}`",
            f"- Search-space source top-k: `{args.phase1c_topk}`",
            f"- Fold: `{args.phase1c_fold}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1c_max_iter}`",
            f"- Early stopping rounds: `{args.phase1c_early_stopping_rounds}`",
            f"- CatBoost threads: `{args.phase1c_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for CatBoost Phase 1C")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, current_signature_to_trial_id = split_catboost_attempts(trials)
    blocked_signatures: dict[str, int] = dict(source_signature_to_trial_id)
    for signature, trial_id in current_signature_to_trial_id.items():
        blocked_signatures[signature] = int(trial_id)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1

    sampler = optuna.samplers.TPESampler(seed=int(args.random_state), n_startup_trials=0, multivariate=True, group=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    prior_observations = 0
    prior_observation_trials: list[dict[str, Any]] = []
    prior_rows = [
        row
        for row in rank_trial_rows(list(source_unique_trials))
        if catboost_params_within_space(dict(row["params"]), optuna_space)
    ]
    for row in prior_rows:
        try:
            params = dict(row["params"])
            source_phase = str(row.get("source_phase", "unknown"))
            source_trial_id = int(row["trial_id"])
            validation_rmse = float(row["metrics"]["validation"]["rmse"])
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=optuna_catboost_distributions(params, optuna_space),
                    value=validation_rmse,
                    user_attrs={"source_phase": source_phase, "source_trial_id": source_trial_id},
                )
            )
            prior_observations += 1
            prior_observation_trials.append(
                {
                    "source_phase": source_phase,
                    "source_trial_id": source_trial_id,
                    "source_rank_inside_all_catboost": int(row.get("rank", 0)),
                    "validation_rmse": validation_rmse,
                    "param_signature": catboost_param_signature(params),
                }
            )
        except Exception:
            continue
    json_dump(run_dir / "optuna_prior_observations.json", {"prior_observations": prior_observation_trials})
    for row in unique_trials:
        try:
            params = dict(row["params"])
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=optuna_catboost_distributions(params, optuna_space),
                    value=float(row["metrics"]["validation"]["rmse"]),
                    user_attrs={"source_phase": "phase1c_resume", "source_trial_id": int(row["trial_id"])},
                )
            )
        except Exception:
            continue

    def write_state(status: str) -> dict[str, Any]:
        return write_phase1c_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            phase1b_source_run_dir=phase1b_source_run_dir,
            phase1a_aux_source_run_dir=phase1a_aux_source_run_dir,
            source_unique_trials=source_unique_trials,
            optuna_space=optuna_space,
            trials=trials,
            status=status,
            started_utc=started_utc,
            prior_observations=prior_observations,
            prior_observation_trials=prior_observation_trials,
        )

    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_state("running")
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: CatBoost Optuna",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed 1C trials already on disk: `{payload['progress']['completed_trials']}`",
                f"- Remaining unique 1C trials to run: `{max(int(args.phase1c_trials) - payload['progress']['completed_trials'], 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                f"- Prior Optuna observations loaded: `{prior_observations}`",
                "",
            ],
        )
    else:
        payload = write_state("running")
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Prior Optuna observations loaded: `{prior_observations}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1c_trials):
        return write_state("completed")

    try:
        while len(unique_trials) < int(args.phase1c_trials):
            trial_id = int(next_trial_id)
            next_trial_id += 1
            trial_seed = int(args.random_state) + PHASE1C_TRIAL_SEED_OFFSET + trial_id
            optuna_trial = study.ask()
            params = suggest_catboost_phase1c_params(optuna_trial, optuna_space)
            param_signature = catboost_param_signature(params)
            if param_signature in blocked_signatures:
                duplicate_of_trial_id = int(blocked_signatures[param_signature])
                duplicate_source = "phase1c" if param_signature in current_signature_to_trial_id else "source"
                duplicate_origin = source_signature_to_origin.get(param_signature, {})
                known_rmse = source_metric_by_signature.get(param_signature)
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "optuna_trial_number": int(optuna_trial.number),
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "duplicate_source": str(duplicate_source),
                    "duplicate_source_phase": duplicate_origin.get("source_phase"),
                    "duplicate_source_trial_id": duplicate_origin.get("source_trial_id"),
                    "known_validation_rmse": None if known_rmse is None else float(known_rmse),
                    "timings_sec": {"fit": 0.0, "trial_total": 0.0},
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                if known_rmse is not None:
                    study.tell(optuna_trial, float(known_rmse))
                else:
                    study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
                payload = write_state("running")
                append_markdown_log(
                    log_path,
                    [
                        f"### CatBoost Phase 1C Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Duplicate source: `{duplicate_source}`",
                        f"- Duplicate source phase/trial: `{duplicate_origin.get('source_phase', '')}` / `{duplicate_origin.get('source_trial_id', '')}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique 1C trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue

            fit_started = time.time()
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=int(args.phase1c_max_iter),
                learning_rate=float(params["learning_rate"]),
                depth=int(params["depth"]),
                l2_leaf_reg=float(params["l2_leaf_reg"]),
                rsm=float(params["rsm"]),
                bootstrap_type="Bayesian",
                bagging_temperature=float(params["bagging_temperature"]),
                border_count=int(params["border_count"]),
                random_seed=int(trial_seed),
                od_type="Iter",
                od_wait=int(args.phase1c_early_stopping_rounds),
                thread_count=max(1, int(args.phase1c_n_jobs)),
                use_best_model=True,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            fit_seconds = time.time() - fit_started
            best_iteration_zero_based = int(model.get_best_iteration() or 0)
            if best_iteration_zero_based < 0:
                best_iteration_zero_based = 0
            best_iteration_one_based = int(best_iteration_zero_based + 1)
            evals_result = model.get_evals_result()
            train_rmse_curve = evals_result.get("learn", {}).get("RMSE", [])
            train_rmse = (
                float(train_rmse_curve[best_iteration_zero_based])
                if best_iteration_zero_based < len(train_rmse_curve)
                else float("nan")
            )
            val_pred = model.predict(X_val, ntree_end=best_iteration_one_based)
            val_metrics = compute_metrics(y_val, val_pred)
            study.tell(optuna_trial, float(val_metrics["rmse"]))
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "optuna_trial_number": int(optuna_trial.number),
                "best_iteration": int(best_iteration_one_based),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(time.time() - fit_started),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            blocked_signatures[param_signature] = int(trial_id)
            current_signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_state("running")
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### CatBoost Phase 1C Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Optuna trial number: `{optuna_trial.number}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration_one_based}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best 1C trial: `{best_row['trial_id']}`",
                    f"- Current best 1C validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del model
            del evals_result
            gc.collect()
        payload = write_state("completed")
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: CatBoost Optuna",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best 1C trial: `{payload['best_trial']['trial_id']}`",
                f"- Best 1C validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Source best validation RMSE: `{payload['source_best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_state("failed")
        append_markdown_log(
            log_path,
            [
                "### Phase 1C Run: CatBoost Optuna",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def run_phase1a_catboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1a_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1a_fold)], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 1A Run: CatBoost",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Fold: `{args.phase1a_fold}`",
            f"- Planned trials: `{args.phase1a_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1a_max_iter}`",
            f"- Early stopping rounds: `{args.phase1a_early_stopping_rounds}`",
            f"- CatBoost threads: `{args.phase1a_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for CatBoost Phase 1A")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, signature_to_trial_id = split_catboost_attempts(trials)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1
    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_phase1a_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: CatBoost",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials already on disk: `{len(unique_trials)}`",
                f"- Duplicate attempts already on disk: `{len(duplicate_trials)}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1a_trials) - len(unique_trials), 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_phase1a_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="running",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1a_trials):
        payload = write_phase1a_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: CatBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- No additional work required because `{len(unique_trials)}` unique completed trials were already present.",
                "",
            ],
        )
        return payload

    try:
        while len(unique_trials) < int(args.phase1a_trials):
            trial_id = int(next_trial_id)
            next_trial_id += 1
            trial_seed = int(args.random_state) + trial_id
            trial_rng = np.random.default_rng(trial_seed)
            params = sample_catboost_phase1a_params(trial_rng)
            param_signature = catboost_param_signature(params)
            if param_signature in signature_to_trial_id:
                duplicate_of_trial_id = int(signature_to_trial_id[param_signature])
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "timings_sec": {
                        "fit": 0.0,
                        "trial_total": 0.0,
                    },
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_phase1a_catboost_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    trials=trials,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### CatBoost Phase 1A Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                        "",
                    ],
                )
                continue
            fit_started = time.time()
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=int(args.phase1a_max_iter),
                learning_rate=float(params["learning_rate"]),
                depth=int(params["depth"]),
                l2_leaf_reg=float(params["l2_leaf_reg"]),
                rsm=float(params["rsm"]),
                bootstrap_type="Bayesian",
                bagging_temperature=float(params["bagging_temperature"]),
                border_count=int(params["border_count"]),
                random_seed=int(trial_seed),
                od_type="Iter",
                od_wait=int(args.phase1a_early_stopping_rounds),
                thread_count=max(1, int(args.phase1a_n_jobs)),
                use_best_model=True,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            fit_seconds = time.time() - fit_started
            best_iteration_zero_based = int(model.get_best_iteration() or 0)
            if best_iteration_zero_based < 0:
                best_iteration_zero_based = 0
            best_iteration_one_based = int(best_iteration_zero_based + 1)
            evals_result = model.get_evals_result()
            train_rmse_curve = evals_result.get("learn", {}).get("RMSE", [])
            train_rmse = (
                float(train_rmse_curve[best_iteration_zero_based])
                if best_iteration_zero_based < len(train_rmse_curve)
                else float("nan")
            )
            val_pred = model.predict(X_val, ntree_end=best_iteration_one_based)
            val_metrics = compute_metrics(y_val, val_pred)
            trial_seconds = time.time() - fit_started
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "best_iteration": int(best_iteration_one_based),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(trial_seconds),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_phase1a_catboost_state(
                args=args,
                run_dir=run_dir,
                meta=meta,
                train_idx=train_idx,
                val_idx=val_idx,
                trials=trials,
                status="running",
                started_utc=started_utc,
            )
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### CatBoost Phase 1A Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration_one_based}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    f"- Trial CSV: `{run_dir / 'trial_results.csv'}`",
                    "",
                ],
            )
            del model
            del evals_result
            gc.collect()
        payload = write_phase1a_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: CatBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged: `{payload['progress']['total_attempts']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase1a_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            trials=trials,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                "### Phase 1A Run: CatBoost",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Total attempts logged before failure: `{payload['progress']['total_attempts']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                f"- Summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )
        raise


def run_phase1b_catboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    if int(args.phase1b_focused_trials) + int(args.phase1b_global_trials) != int(args.phase1b_trials):
        raise ValueError("Phase 1B CatBoost trial counts must satisfy focused + global == total trials.")
    source_trial_jsonl_path = args.phase1b_source_run_dir / "trial_results.jsonl"
    if not source_trial_jsonl_path.exists():
        raise FileNotFoundError(f"Missing Phase 1A CatBoost trial JSONL: {source_trial_jsonl_path}")

    started_utc = now_utc_iso()
    trial_jsonl_path = run_dir / "trial_results.jsonl"
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(args.phase1b_fold)], dtype=np.int64))
    train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(args.phase1b_fold)], dtype=np.int64))

    source_trials_all = load_trial_jsonl(source_trial_jsonl_path)
    source_unique_trials, _, source_signature_to_trial_id = split_catboost_attempts(source_trials_all)
    focused_space = build_catboost_phase1b_space(source_unique_trials, int(args.phase1b_topk))

    append_markdown_log(
        log_path,
        [
            "### Phase 1B Run: CatBoost",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 1A source run: `{args.phase1b_source_run_dir}`",
            f"- Blocked source signatures: `{len(source_unique_trials)}`",
            f"- Focused seed trials: `{', '.join(str(x) for x in focused_space['seed_trial_ids'])}`",
            f"- Fold: `{args.phase1b_fold}`",
            f"- Planned trials: `{args.phase1b_trials}`",
            f"- Focused / global plan: `{args.phase1b_focused_trials}` / `{args.phase1b_global_trials}`",
            f"- Train rows: `{train_idx.shape[0]:,}`",
            f"- Validation rows: `{val_idx.shape[0]:,}`",
            f"- Max iterations: `{args.phase1b_max_iter}`",
            f"- Early stopping rounds: `{args.phase1b_early_stopping_rounds}`",
            f"- CatBoost threads: `{args.phase1b_n_jobs}`",
            "",
        ],
    )

    log("Loading full fold-0 train and validation matrices into memory for CatBoost Phase 1B")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    trials = load_trial_jsonl(trial_jsonl_path)
    unique_trials, duplicate_trials, current_signature_to_trial_id = split_catboost_attempts(trials)
    blocked_signatures: dict[str, int] = dict(source_signature_to_trial_id)
    for signature, trial_id in current_signature_to_trial_id.items():
        blocked_signatures[signature] = int(trial_id)
    next_trial_id = max((int(row["trial_id"]) for row in trials), default=0) + 1

    def write_state(status: str) -> dict[str, Any]:
        return write_phase1b_catboost_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            train_idx=train_idx,
            val_idx=val_idx,
            source_unique_trials=source_unique_trials,
            focused_space=focused_space,
            trials=trials,
            status=status,
            started_utc=started_utc,
        )

    if trials:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
        payload = write_state("running")
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: CatBoost",
                "",
                "Status:",
                "",
                "- resumed",
                "",
                "Notes:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed Phase 1B trials already on disk: `{payload['progress']['completed_trials']}`",
                f"- Remaining unique trials to run: `{max(int(args.phase1b_trials) - payload['progress']['completed_trials'], 0)}`",
                f"- Matrix materialization seconds this launch: `{matrix_load_seconds:.1f}`",
                "",
            ],
        )
    else:
        payload = write_state("running")
        append_markdown_log(
            log_path,
            [
                "Measured result:",
                "",
                f"- Matrix materialization seconds: `{matrix_load_seconds:.1f}`",
                f"- Initial summary markdown: `{run_dir / 'summary.md'}`",
                "",
            ],
        )

    if len(unique_trials) >= int(args.phase1b_trials):
        return write_state("completed")

    try:
        while len(unique_trials) < int(args.phase1b_trials):
            focused_completed = sum(1 for row in unique_trials if row.get("search_stage") == "focused")
            search_stage = "focused" if focused_completed < int(args.phase1b_focused_trials) else "global"
            trial_id = int(next_trial_id)
            next_trial_id += 1
            if search_stage == "focused":
                trial_seed = int(args.random_state) + trial_id
                trial_rng = np.random.default_rng(trial_seed)
                params = sample_catboost_phase1b_focused_params(trial_rng, focused_space)
            else:
                trial_seed = int(args.random_state) + PHASE1B_GLOBAL_SEED_OFFSET + trial_id
                trial_rng = np.random.default_rng(trial_seed)
                params = sample_catboost_phase1a_params(trial_rng)
            param_signature = catboost_param_signature(params)
            if param_signature in blocked_signatures:
                duplicate_of_trial_id = int(blocked_signatures[param_signature])
                duplicate_source = "phase1a" if param_signature in source_signature_to_trial_id else "phase1b"
                duplicate_row = {
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "search_stage": search_stage,
                    "params": params,
                    "outcome": "duplicate_skipped",
                    "duplicate_source": duplicate_source,
                    "duplicate_of_trial_id": int(duplicate_of_trial_id),
                    "timings_sec": {"fit": 0.0, "trial_total": 0.0},
                }
                trials.append(duplicate_row)
                duplicate_trials.append(dict(duplicate_row))
                write_trial_jsonl(trial_jsonl_path, duplicate_row)
                payload = write_state("running")
                append_markdown_log(
                    log_path,
                    [
                        f"### CatBoost Phase 1B Trial {trial_id:03d}",
                        "",
                        "Status:",
                        "",
                        "- skipped duplicate configuration",
                        "",
                        "Notes:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Search stage: `{search_stage}`",
                        f"- Duplicate source: `{duplicate_source}`",
                        f"- Duplicate of trial: `{duplicate_of_trial_id}`",
                        f"- Unique trials completed remains: `{payload['progress']['completed_trials']}` / `{payload['progress']['planned_trials']}`",
                        "",
                    ],
                )
                continue

            fit_started = time.time()
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=int(args.phase1b_max_iter),
                learning_rate=float(params["learning_rate"]),
                depth=int(params["depth"]),
                l2_leaf_reg=float(params["l2_leaf_reg"]),
                rsm=float(params["rsm"]),
                bootstrap_type="Bayesian",
                bagging_temperature=float(params["bagging_temperature"]),
                border_count=int(params["border_count"]),
                random_seed=int(trial_seed),
                od_type="Iter",
                od_wait=int(args.phase1b_early_stopping_rounds),
                thread_count=max(1, int(args.phase1b_n_jobs)),
                use_best_model=True,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            fit_seconds = time.time() - fit_started
            best_iteration_zero_based = int(model.get_best_iteration() or 0)
            if best_iteration_zero_based < 0:
                best_iteration_zero_based = 0
            best_iteration_one_based = int(best_iteration_zero_based + 1)
            evals_result = model.get_evals_result()
            train_rmse_curve = evals_result.get("learn", {}).get("RMSE", [])
            train_rmse = (
                float(train_rmse_curve[best_iteration_zero_based])
                if best_iteration_zero_based < len(train_rmse_curve)
                else float("nan")
            )
            val_pred = model.predict(X_val, ntree_end=best_iteration_one_based)
            val_metrics = compute_metrics(y_val, val_pred)
            row = {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_seed),
                "search_stage": search_stage,
                "best_iteration": int(best_iteration_one_based),
                "params": params,
                "metrics": {
                    "train": {"rmse": float(train_rmse)},
                    "validation": val_metrics,
                    "rmse_gap": float(val_metrics["rmse"] - train_rmse),
                },
                "timings_sec": {
                    "fit": float(fit_seconds),
                    "trial_total": float(time.time() - fit_started),
                },
            }
            trials.append(row)
            unique_trials.append(dict(row))
            blocked_signatures[param_signature] = int(trial_id)
            current_signature_to_trial_id[param_signature] = int(trial_id)
            write_trial_jsonl(trial_jsonl_path, row)
            payload = write_state("running")
            best_row = payload["best_trial"]
            append_markdown_log(
                log_path,
                [
                    f"### CatBoost Phase 1B Trial {trial_id:03d}",
                    "",
                    "Status:",
                    "",
                    "- completed",
                    "",
                    "Measured result:",
                    "",
                    f"- Run directory: `{run_dir}`",
                    f"- Search stage: `{search_stage}`",
                    f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                    f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                    f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                    f"- Train RMSE at best iteration: `{train_rmse:.6f}`",
                    f"- RMSE gap: `{row['metrics']['rmse_gap']:.6f}`",
                    f"- Best iteration: `{best_iteration_one_based}`",
                    f"- Fit seconds: `{fit_seconds:.1f}`",
                    f"- Current best trial: `{best_row['trial_id']}`",
                    f"- Current best validation RMSE: `{best_row['metrics']['validation']['rmse']:.6f}`",
                    "",
                ],
            )
            del model
            del evals_result
            gc.collect()

        payload = write_state("completed")
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: CatBoost",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials: `{payload['progress']['completed_trials']}`",
                f"- Focused / global completed: `{payload['progress']['focused_completed_trials']}` / `{payload['progress']['global_completed_trials']}`",
                f"- Best trial: `{payload['best_trial']['trial_id']}`",
                f"- Best validation RMSE: `{payload['best_trial']['metrics']['validation']['rmse']:.6f}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_state("failed")
        append_markdown_log(
            log_path,
            [
                "### Phase 1B Run: CatBoost",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Unique completed trials before failure: `{payload['progress']['completed_trials']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                "",
            ],
        )
        raise


def load_ranked_unique_trials_from_sources(
    source_run_dirs: list[Path],
    *,
    split_fn: Any,
    signature_fn: Any,
) -> list[dict[str, Any]]:
    by_signature: dict[str, dict[str, Any]] = {}
    for source_dir in source_run_dirs:
        trial_path = source_dir / "trial_results.jsonl"
        if not trial_path.exists():
            raise FileNotFoundError(f"Missing source trial JSONL: {trial_path}")
        source_trials, _, _ = split_fn(load_trial_jsonl(trial_path))
        for row in source_trials:
            params = row.get("params") or {}
            signature = signature_fn(params)
            current = by_signature.get(signature)
            candidate = dict(row)
            candidate["source_run_dir"] = str(source_dir)
            if current is None or trial_sort_key(candidate) < trial_sort_key(current):
                by_signature[signature] = candidate
    return rank_trial_rows(list(by_signature.values()))


def render_phase2_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['model_name']} Phase 2 Five-Fold Confirmation",
        "",
        f"- Run dir: `{payload['meta']['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Source runs: `{', '.join(payload['meta']['source_run_dirs'])}`",
        f"- Shortlisted configs: `{payload['progress']['completed_configs']}` / `{payload['progress']['planned_configs']}`",
        f"- Completed folds: `{payload['progress']['completed_folds']}` / `{payload['progress']['planned_folds']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
        "## Leaderboard",
        "",
        "| Rank | Config | Source trial | Mean train RMSE | Mean val RMSE | SD val RMSE | Mean gap | Max gap | Mean MAE | Mean R^2 | Mean iter | Mean fit sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            "| "
            f"{row['rank']} | {row['config_index']} | {row['source_trial_id']} | "
            f"{row['mean_train_rmse']:.6f} | "
            f"{row['mean_validation_rmse']:.6f} | {row['sd_validation_rmse']:.6f} | "
            f"{row['mean_rmse_gap']:.6f} | {row['max_rmse_gap']:.6f} | "
            f"{row['mean_validation_mae']:.6f} | {row['mean_validation_r2']:.6f} | "
            f"{row['mean_best_iteration']:.1f} | {row['mean_fit_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{payload['meta']['run_dir']}\\results.json`",
            f"- Fold JSONL: `{payload['meta']['run_dir']}\\fold_results.jsonl`",
            f"- Config CSV: `{payload['meta']['run_dir']}\\config_results.csv`",
            f"- Status JSON: `{payload['meta']['run_dir']}\\status.json`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase2_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "config_index",
        "source_trial_id",
        "source_rank",
        "mean_train_rmse",
        "sd_train_rmse",
        "mean_validation_rmse",
        "sd_validation_rmse",
        "mean_rmse_gap",
        "max_rmse_gap",
        "mean_validation_mae",
        "sd_validation_mae",
        "mean_validation_r2",
        "sd_validation_r2",
        "mean_best_iteration",
        "mean_fit_seconds",
        "source_run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_phase2_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    model_name: str,
    source_run_dirs: list[Path],
    selected_configs: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in fold_rows:
        if row.get("outcome", "completed") == "completed":
            grouped.setdefault(int(row["config_index"]), []).append(row)
    leaderboard: list[dict[str, Any]] = []
    for config in selected_configs:
        idx = int(config["config_index"])
        rows = sorted(grouped.get(idx, []), key=lambda row: int(row["fold"]))
        if len(rows) < 5:
            continue
        train_rmses = [float(row["metrics"]["train"]["rmse"]) for row in rows]
        rmses = [float(row["metrics"]["validation"]["rmse"]) for row in rows]
        gaps = [float(row["metrics"]["rmse_gap"]) for row in rows]
        maes = [float(row["metrics"]["validation"]["mae"]) for row in rows]
        r2s = [float(row["metrics"]["validation"]["r2"]) for row in rows]
        iters = [float(row["best_iteration"]) for row in rows]
        fits = [float(row["timings_sec"]["fit"]) for row in rows]
        leaderboard.append(
            {
                "config_index": idx,
                "source_trial_id": int(config["source_trial_id"]),
                "source_rank": int(config["source_rank"]),
                "source_run_dir": str(config["source_run_dir"]),
                "params": config["params"],
                "folds": rows,
                "mean_train_rmse": float(np.mean(train_rmses)),
                "sd_train_rmse": float(np.std(train_rmses, ddof=1)),
                "mean_validation_rmse": float(np.mean(rmses)),
                "sd_validation_rmse": float(np.std(rmses, ddof=1)),
                "mean_rmse_gap": float(np.mean(gaps)),
                "max_rmse_gap": float(np.max(gaps)),
                "mean_validation_mae": float(np.mean(maes)),
                "sd_validation_mae": float(np.std(maes, ddof=1)),
                "mean_validation_r2": float(np.mean(r2s)),
                "sd_validation_r2": float(np.std(r2s, ddof=1)),
                "mean_best_iteration": float(np.mean(iters)),
                "mean_fit_seconds": float(np.mean(fits)),
            }
        )
    leaderboard.sort(key=lambda row: (row["mean_validation_rmse"], row["sd_validation_rmse"], row["mean_fit_seconds"]))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = int(rank)
    completed_pairs = {(int(row["config_index"]), int(row["fold"])) for row in fold_rows if row.get("outcome", "completed") == "completed"}
    completed_configs = sum(1 for config in selected_configs if all((int(config["config_index"]), fold) in completed_pairs for fold in range(5)))
    return {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "source_run_dirs": [str(path) for path in source_run_dirs],
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "model_name": model_name,
        "status": status,
        "dependency_status": dependency_status(),
        "fixed_settings": {
            "max_iterations": int(args.phase2_max_iter),
            "early_stopping_rounds": int(args.phase2_early_stopping_rounds),
        },
        "progress": {
            "planned_configs": int(len(selected_configs)),
            "completed_configs": int(completed_configs),
            "planned_folds": int(len(selected_configs) * 5),
            "completed_folds": int(len(completed_pairs)),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase2_n_jobs),
        },
        "selected_configs": selected_configs,
        "fold_results": fold_rows,
        "leaderboard": leaderboard,
        "best_config": leaderboard[0] if leaderboard else None,
    }


def write_phase2_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    meta: dict[str, Any],
    model_name: str,
    source_run_dirs: list[Path],
    selected_configs: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    payload = build_phase2_payload(
        args=args,
        run_dir=run_dir,
        meta=meta,
        model_name=model_name,
        source_run_dirs=source_run_dirs,
        selected_configs=selected_configs,
        fold_rows=fold_rows,
        status=status,
        started_utc=started_utc,
    )
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "planned_configs": payload["progress"]["planned_configs"],
            "completed_configs": payload["progress"]["completed_configs"],
            "planned_folds": payload["progress"]["planned_folds"],
            "completed_folds": payload["progress"]["completed_folds"],
            "best_config_index": None if payload["best_config"] is None else int(payload["best_config"]["config_index"]),
            "best_mean_validation_rmse": None
            if payload["best_config"] is None
            else float(payload["best_config"]["mean_validation_rmse"]),
        },
    )
    write_phase2_csv(run_dir / "config_results.csv", payload["leaderboard"])
    (run_dir / "summary.md").write_text(render_phase2_summary(payload), encoding="utf-8")
    return payload


def run_phase2_tree(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
    model_name: str,
) -> dict[str, Any]:
    if model_name == "LightGBM":
        import lightgbm as lgb

        default_sources = [
            DEFAULT_LIGHTGBM_PHASE1A_SOURCE_RUN,
            DEFAULT_LIGHTGBM_PHASE1B_SOURCE_RUN,
            DEFAULT_OUTPUT_ROOT / "runs" / "20260422_004800_lightgbm_phase1c_optuna_v2",
        ]
        source_run_dirs = list(args.phase2_source_run_dir or default_sources)
        ranked = load_ranked_unique_trials_from_sources(
            source_run_dirs,
            split_fn=split_lightgbm_attempts,
            signature_fn=lightgbm_param_signature,
        )
    elif model_name == "XGBoost":
        import xgboost as xgb

        default_sources = [DEFAULT_XGBOOST_PHASE1A_SOURCE_RUN, DEFAULT_XGBOOST_PHASE1B_SOURCE_RUN]
        source_run_dirs = list(args.phase2_source_run_dir or default_sources)
        ranked = load_ranked_unique_trials_from_sources(
            source_run_dirs,
            split_fn=split_xgboost_attempts,
            signature_fn=xgboost_param_signature,
        )
    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor

        default_sources = [
            DEFAULT_CATBOOST_PHASE1A_SOURCE_RUN,
            DEFAULT_CATBOOST_PHASE1B_SOURCE_RUN,
            DEFAULT_OUTPUT_ROOT / "runs" / "20260425_183000_phase1c_focused_catboost_v1",
        ]
        source_run_dirs = list(args.phase2_source_run_dir or default_sources)
        ranked = load_ranked_unique_trials_from_sources(
            source_run_dirs,
            split_fn=split_catboost_attempts,
            signature_fn=catboost_param_signature,
        )
    else:
        raise ValueError(f"Unsupported Phase 2 model: {model_name}")

    selected_configs = []
    for idx, row in enumerate(ranked[: int(args.phase2_topk)], start=1):
        config_index = int(args.phase2_config_index_offset) + int(idx)
        selected_configs.append(
            {
                "config_index": int(config_index),
                "source_trial_id": int(row["trial_id"]),
                "source_rank": int(row.get("rank", config_index)),
                "source_run_dir": str(row.get("source_run_dir", "")),
                "source_validation_rmse": float(row["metrics"]["validation"]["rmse"]),
                "params": row["params"],
            }
        )
    if not selected_configs:
        raise ValueError("Phase 2 requires at least one selected config.")

    started_utc = now_utc_iso()
    fold_jsonl_path = run_dir / "fold_results.jsonl"
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    cache_meta_path = cache_root / "meta.json"
    if cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)

    fold_rows = load_jsonl_rows(fold_jsonl_path)
    if fold_rows:
        started_utc = load_created_utc(run_dir / "results.json", started_utc)
    completed_pairs = {(int(row["config_index"]), int(row["fold"])) for row in fold_rows if row.get("outcome", "completed") == "completed"}

    append_markdown_log(
        log_path,
        [
            f"### Phase 2 Run: {model_name}",
            "",
            "Status:",
            "",
            "- started" if not fold_rows else "- resumed",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Source runs: `{', '.join(str(x) for x in source_run_dirs)}`",
            f"- Top configs: `{len(selected_configs)}`",
            f"- Completed folds already present: `{len(completed_pairs)}` / `{len(selected_configs) * 5}`",
            f"- Max iterations: `{args.phase2_max_iter}`",
            f"- Early stopping rounds: `{args.phase2_early_stopping_rounds}`",
            f"- Threads: `{args.phase2_n_jobs}`",
            "",
        ],
    )

    payload = write_phase2_state(
        args=args,
        run_dir=run_dir,
        meta=meta,
        model_name=model_name,
        source_run_dirs=source_run_dirs,
        selected_configs=selected_configs,
        fold_rows=fold_rows,
        status="running",
        started_utc=started_utc,
    )

    try:
        for config in selected_configs:
            config_index = int(config["config_index"])
            params = config["params"]
            for fold in range(5):
                if (config_index, fold) in completed_pairs:
                    continue
                val_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] == int(fold)], dtype=np.int64))
                train_idx = np.sort(np.asarray(dev_idx[fold_ids[dev_idx] != int(fold)], dtype=np.int64))
                X_train = np.asarray(X[train_idx], dtype=np.float32)
                X_val = np.asarray(X[val_idx], dtype=np.float32)
                if model_name == "LightGBM":
                    y_train = np.asarray(y[train_idx], dtype=np.float64)
                    y_val = np.asarray(y[val_idx], dtype=np.float64)
                    fit_started = time.time()
                    model = lgb.LGBMRegressor(
                        objective="regression",
                        metric="rmse",
                        boosting_type="gbdt",
                        n_estimators=int(args.phase2_max_iter),
                        learning_rate=float(params["learning_rate"]),
                        num_leaves=int(params["num_leaves"]),
                        max_depth=int(params["max_depth"]),
                        min_child_samples=int(params["min_child_samples"]),
                        feature_fraction=float(params["feature_fraction"]),
                        bagging_fraction=float(params["bagging_fraction"]),
                        bagging_freq=int(params["bagging_freq"]),
                        lambda_l1=float(params["lambda_l1"]),
                        lambda_l2=float(params["lambda_l2"]),
                        min_split_gain=float(params["min_split_gain"]),
                        max_bin=int(params["max_bin"]),
                        random_state=int(args.random_state) + config_index * 100 + fold,
                        n_jobs=max(1, int(args.phase2_n_jobs)),
                        verbosity=-1,
                        force_col_wise=True,
                    )
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="rmse",
                        callbacks=[lgb.early_stopping(int(args.phase2_early_stopping_rounds), verbose=False)],
                    )
                    fit_seconds = time.time() - fit_started
                    best_iteration = int(getattr(model, "best_iteration_", args.phase2_max_iter) or args.phase2_max_iter)
                    train_pred = model.predict(X_train, num_iteration=best_iteration)
                    val_pred = model.predict(X_val, num_iteration=best_iteration)
                elif model_name == "XGBoost":
                    y_train = np.asarray(y[train_idx], dtype=np.float32)
                    y_val = np.asarray(y[val_idx], dtype=np.float32)
                    fit_started = time.time()
                    dtrain = xgb.QuantileDMatrix(
                        X_train,
                        label=y_train,
                        max_bin=int(params["max_bin"]),
                        nthread=max(1, int(args.phase2_n_jobs)),
                    )
                    dval = xgb.QuantileDMatrix(
                        X_val,
                        label=y_val,
                        ref=dtrain,
                        max_bin=int(params["max_bin"]),
                        nthread=max(1, int(args.phase2_n_jobs)),
                    )
                    evals_result: dict[str, Any] = {}
                    booster = xgb.train(
                        params={
                            "objective": "reg:squarederror",
                            "eval_metric": "rmse",
                            "tree_method": "hist",
                            "learning_rate": float(params["learning_rate"]),
                            "max_depth": int(params["max_depth"]),
                            "min_child_weight": float(params["min_child_weight"]),
                            "subsample": float(params["subsample"]),
                            "colsample_bytree": float(params["colsample_bytree"]),
                            "reg_alpha": float(params["reg_alpha"]),
                            "reg_lambda": float(params["reg_lambda"]),
                            "gamma": float(params["gamma"]),
                            "max_bin": int(params["max_bin"]),
                            "seed": int(args.random_state) + config_index * 100 + fold,
                            "nthread": max(1, int(args.phase2_n_jobs)),
                            "verbosity": 0,
                        },
                        dtrain=dtrain,
                        num_boost_round=int(args.phase2_max_iter),
                        evals=[(dtrain, "train"), (dval, "validation")],
                        evals_result=evals_result,
                        early_stopping_rounds=int(args.phase2_early_stopping_rounds),
                        verbose_eval=False,
                    )
                    fit_seconds = time.time() - fit_started
                    best_zero = int(getattr(booster, "best_iteration", args.phase2_max_iter - 1))
                    if best_zero < 0:
                        best_zero = int(args.phase2_max_iter - 1)
                    best_iteration = int(best_zero + 1)
                    train_pred = booster.predict(dtrain, iteration_range=(0, best_iteration))
                    val_pred = booster.predict(dval, iteration_range=(0, best_iteration))
                    del booster
                    del evals_result
                else:
                    y_train = np.asarray(y[train_idx], dtype=np.float64)
                    y_val = np.asarray(y[val_idx], dtype=np.float64)
                    fit_started = time.time()
                    model = CatBoostRegressor(
                        loss_function="RMSE",
                        eval_metric="RMSE",
                        iterations=int(args.phase2_max_iter),
                        learning_rate=float(params["learning_rate"]),
                        depth=int(params["depth"]),
                        l2_leaf_reg=float(params["l2_leaf_reg"]),
                        rsm=float(params["rsm"]),
                        bootstrap_type="Bayesian",
                        bagging_temperature=float(params["bagging_temperature"]),
                        border_count=int(params["border_count"]),
                        random_seed=int(args.random_state) + config_index * 100 + fold,
                        od_type="Iter",
                        od_wait=int(args.phase2_early_stopping_rounds),
                        thread_count=max(1, int(args.phase2_n_jobs)),
                        use_best_model=True,
                        allow_writing_files=False,
                        verbose=False,
                    )
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
                    fit_seconds = time.time() - fit_started
                    best_zero = int(model.get_best_iteration() or 0)
                    if best_zero < 0:
                        best_zero = 0
                    best_iteration = int(best_zero + 1)
                    train_pred = model.predict(X_train, ntree_end=best_iteration)
                    val_pred = model.predict(X_val, ntree_end=best_iteration)
                    del model
                train_metrics = compute_metrics(y_train, train_pred)
                val_metrics = compute_metrics(y_val, val_pred)
                row = {
                    "config_index": int(config_index),
                    "fold": int(fold),
                    "source_trial_id": int(config["source_trial_id"]),
                    "source_run_dir": str(config["source_run_dir"]),
                    "best_iteration": int(best_iteration),
                    "params": params,
                    "metrics": {
                        "train": train_metrics,
                        "validation": val_metrics,
                        "rmse_gap": float(val_metrics["rmse"] - train_metrics["rmse"]),
                    },
                    "timings_sec": {"fit": float(fit_seconds)},
                    "outcome": "completed",
                }
                fold_rows.append(row)
                completed_pairs.add((config_index, fold))
                write_trial_jsonl(fold_jsonl_path, row)
                payload = write_phase2_state(
                    args=args,
                    run_dir=run_dir,
                    meta=meta,
                    model_name=model_name,
                    source_run_dirs=source_run_dirs,
                    selected_configs=selected_configs,
                    fold_rows=fold_rows,
                    status="running",
                    started_utc=started_utc,
                )
                append_markdown_log(
                    log_path,
                    [
                        f"### {model_name} Phase 2 Config {config_index:02d} Fold {fold}",
                        "",
                        "Status:",
                        "",
                        "- completed",
                        "",
                        "Measured result:",
                        "",
                        f"- Run directory: `{run_dir}`",
                        f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
                        f"- Validation MAE: `{val_metrics['mae']:.6f}`",
                        f"- Validation R^2: `{val_metrics['r2']:.6f}`",
                        f"- Best iteration: `{best_iteration}`",
                        f"- Fit seconds: `{fit_seconds:.1f}`",
                        "",
                    ],
                )
                del X_train
                del X_val
                del y_train
                del y_val
                gc.collect()
        payload = write_phase2_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            model_name=model_name,
            source_run_dirs=source_run_dirs,
            selected_configs=selected_configs,
            fold_rows=fold_rows,
            status="completed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                f"### Phase 2 Run: {model_name}",
                "",
                "Status:",
                "",
                "- completed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Completed folds: `{payload['progress']['completed_folds']}` / `{payload['progress']['planned_folds']}`",
                f"- Best config: `{payload['best_config']['config_index'] if payload['best_config'] else None}`",
                f"- Best mean validation RMSE: `{payload['best_config']['mean_validation_rmse'] if payload['best_config'] else None}`",
                "",
            ],
        )
        return payload
    except Exception as exc:
        payload = write_phase2_state(
            args=args,
            run_dir=run_dir,
            meta=meta,
            model_name=model_name,
            source_run_dirs=source_run_dirs,
            selected_configs=selected_configs,
            fold_rows=fold_rows,
            status="failed",
            started_utc=started_utc,
        )
        append_markdown_log(
            log_path,
            [
                f"### Phase 2 Run: {model_name}",
                "",
                "Status:",
                "",
                "- failed",
                "",
                "Measured result:",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Completed folds before failure: `{payload['progress']['completed_folds']}` / `{payload['progress']['planned_folds']}`",
                f"- Error: `{exc.__class__.__name__}: {exc}`",
                "",
            ],
        )
        raise


def render_phase3_lightgbm_summary(payload: dict[str, Any]) -> str:
    train = payload["metrics"]["train_dev"]
    test = payload["metrics"]["test"]
    cv = payload["source_phase2_config"]
    lines = [
        "# LightGBM Phase 3 Final Refit/Test",
        "",
        f"- Run dir: `{payload['meta']['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Phase 2 source run: `{payload['meta']['phase2_source_run_dir']}`",
        f"- Selected config index: `{payload['selected_config']['config_index']}`",
        f"- Source trial: `{payload['selected_config']['source_trial_id']}`",
        f"- Fit rows: `{payload['split']['dev_rows']:,}`",
        f"- Test rows: `{payload['split']['test_rows']:,}`",
        f"- Fixed iterations: `{payload['fixed_settings']['n_estimators']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
        "## Metrics",
        "",
        "| Split | RMSE | MAE | R^2 |",
        "| --- | ---: | ---: | ---: |",
        f"| Full 90% dev train | {train['rmse']:.9f} | {train['mae']:.9f} | {train['r2']:.9f} |",
        f"| Frozen 10% test | {test['rmse']:.9f} | {test['mae']:.9f} | {test['r2']:.9f} |",
        "",
        "## Comparison To Phase 2",
        "",
        f"- Phase 2 mean CV validation RMSE: `{cv['mean_validation_rmse']:.9f}`",
        f"- Test RMSE minus Phase 2 mean CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
        f"- Test RMSE minus full-dev train RMSE: `{payload['metrics']['test_minus_train_dev_rmse']:.9f}`",
        "",
        "## Selected Params",
        "",
    ]
    for key, value in payload["selected_config"]["params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model text: `{payload['artifacts']['model_txt']}`",
            f"- Results JSON: `{payload['meta']['run_dir']}\\results.json`",
            f"- Status JSON: `{payload['meta']['run_dir']}\\status.json`",
            "",
        ]
    )
    if payload["artifacts"].get("test_predictions_npz"):
        lines.append(f"- Test predictions NPZ: `{payload['artifacts']['test_predictions_npz']}`")
    return "\n".join(lines) + "\n"


def write_phase3_lightgbm_state(
    *,
    run_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "config_index": int(payload["selected_config"]["config_index"]),
            "source_trial_id": int(payload["selected_config"]["source_trial_id"]),
            "train_dev_rmse": float(payload["metrics"]["train_dev"]["rmse"]),
            "test_rmse": float(payload["metrics"]["test"]["rmse"]),
            "phase2_mean_validation_rmse": float(payload["source_phase2_config"]["mean_validation_rmse"]),
            "test_minus_phase2_cv_rmse": float(payload["metrics"]["test_minus_phase2_cv_rmse"]),
        },
    )
    (run_dir / "summary.md").write_text(render_phase3_lightgbm_summary(payload), encoding="utf-8")
    return payload


def run_phase3_lightgbm(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    phase2_results_path = args.phase3_source_run_dir / "results.json"
    if not phase2_results_path.exists():
        raise FileNotFoundError(f"Missing LightGBM Phase 2 results JSON: {phase2_results_path}")
    phase2_payload = json.loads(phase2_results_path.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in phase2_payload.get("leaderboard", [])
        if int(row.get("config_index", -1)) == int(args.phase3_config_index)
    ]
    if not candidates:
        raise ValueError(f"Phase 2 config index {args.phase3_config_index} was not found in {phase2_results_path}")
    selected = dict(candidates[0])
    params = dict(selected["params"])
    started_utc = now_utc_iso()
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    meta_path = cache_root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Phase 3 workers are often restored from a Windows cache archive, so
        # normalize cache paths instead of forcing a SQLite rebuild.
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.sort(np.asarray(split_arrays["dev_idx"], dtype=np.int64))
    test_idx = np.sort(np.asarray(split_arrays["test_idx"], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: LightGBM Final Refit/Test",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 2 source run: `{args.phase3_source_run_dir}`",
            f"- Selected config index: `{args.phase3_config_index}`",
            f"- Source trial: `{selected['source_trial_id']}`",
            f"- Dev rows: `{dev_idx.shape[0]:,}`",
            f"- Test rows: `{test_idx.shape[0]:,}`",
            f"- Fixed iterations: `{args.phase3_max_iter}`",
            f"- Threads: `{args.phase3_n_jobs}`",
            "",
        ],
    )

    t0 = time.time()
    X_dev = np.asarray(X[dev_idx], dtype=np.float32)
    y_dev = np.asarray(y[dev_idx], dtype=np.float64)
    X_test = np.asarray(X[test_idx], dtype=np.float32)
    y_test = np.asarray(y[test_idx], dtype=np.float64)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=int(args.phase3_max_iter),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=float(params["feature_fraction"]),
        bagging_fraction=float(params["bagging_fraction"]),
        bagging_freq=int(params["bagging_freq"]),
        lambda_l1=float(params["lambda_l1"]),
        lambda_l2=float(params["lambda_l2"]),
        min_split_gain=float(params["min_split_gain"]),
        max_bin=int(params["max_bin"]),
        random_state=int(args.random_state) + 30_000 + int(args.phase3_config_index),
        bagging_seed=int(args.random_state) + 30_000 + int(args.phase3_config_index),
        feature_fraction_seed=int(args.random_state) + 30_000 + int(args.phase3_config_index),
        data_random_seed=int(args.random_state) + 30_000 + int(args.phase3_config_index),
        deterministic=True,
        n_jobs=max(1, int(args.phase3_n_jobs)),
        verbosity=-1,
        force_col_wise=True,
    )
    fit_started = time.time()
    model.fit(X_dev, y_dev)
    fit_seconds = time.time() - fit_started
    best_iteration = int(getattr(model, "best_iteration_", args.phase3_max_iter) or args.phase3_max_iter)
    if best_iteration <= 0:
        best_iteration = int(args.phase3_max_iter)
    train_pred = model.predict(X_dev, num_iteration=best_iteration)
    test_pred = model.predict(X_test, num_iteration=best_iteration)
    train_metrics = compute_metrics(y_dev, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    model_txt = run_dir / "lightgbm_phase3_model.txt"
    model.booster_.save_model(str(model_txt), num_iteration=best_iteration)
    predictions_npz = None
    if bool(args.phase3_save_test_predictions):
        predictions_npz = run_dir / "test_predictions.npz"
        np.savez_compressed(
            predictions_npz,
            test_idx=test_idx,
            y_test=y_test,
            y_pred=np.asarray(test_pred, dtype=np.float64),
        )

    phase2_mean_rmse = float(selected["mean_validation_rmse"])
    payload = {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase2_source_run_dir": str(args.phase3_source_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": "completed",
        "dependency_status": dependency_status(),
        "selected_config": {
            "config_index": int(selected["config_index"]),
            "source_trial_id": int(selected["source_trial_id"]),
            "source_rank": int(selected.get("source_rank", selected.get("rank", 0))),
            "source_run_dir": str(selected["source_run_dir"]),
            "params": params,
        },
        "source_phase2_config": selected,
        "split": {
            "dev_rows": int(dev_idx.shape[0]),
            "test_rows": int(test_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": int(args.phase3_max_iter),
            "best_iteration": int(best_iteration),
        },
        "metrics": {
            "train_dev": train_metrics,
            "test": test_metrics,
            "test_minus_phase2_cv_rmse": float(test_metrics["rmse"] - phase2_mean_rmse),
            "test_minus_train_dev_rmse": float(test_metrics["rmse"] - train_metrics["rmse"]),
        },
        "timings_sec": {
            "matrix_load": float(matrix_load_seconds),
            "fit": float(fit_seconds),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase3_n_jobs),
        },
        "artifacts": {
            "model_txt": str(model_txt),
            "test_predictions_npz": None if predictions_npz is None else str(predictions_npz),
        },
    }
    write_phase3_lightgbm_state(run_dir=run_dir, payload=payload)
    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: LightGBM Final Refit/Test",
            "",
            "Status:",
            "",
            "- completed",
            "",
            "Measured result:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Full-dev train RMSE: `{train_metrics['rmse']:.9f}`",
            f"- Frozen test RMSE: `{test_metrics['rmse']:.9f}`",
            f"- Phase 2 mean CV RMSE: `{phase2_mean_rmse:.9f}`",
            f"- Test minus Phase 2 CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
            f"- Fit seconds: `{fit_seconds:.1f}`",
            f"- Model artifact: `{model_txt}`",
            "",
        ],
    )
    return payload


def render_phase3_xgboost_summary(payload: dict[str, Any]) -> str:
    train = payload["metrics"]["train_dev"]
    test = payload["metrics"]["test"]
    cv = payload["source_phase2_config"]
    lines = [
        "# XGBoost Phase 3 Final Refit/Test",
        "",
        f"- Run dir: `{payload['meta']['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Phase 2 source run: `{payload['meta']['phase2_source_run_dir']}`",
        f"- Selected config index: `{payload['selected_config']['config_index']}`",
        f"- Source trial: `{payload['selected_config']['source_trial_id']}`",
        f"- Fit rows: `{payload['split']['dev_rows']:,}`",
        f"- Test rows: `{payload['split']['test_rows']:,}`",
        f"- Fixed boosting rounds: `{payload['fixed_settings']['num_boost_round']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
        "## Metrics",
        "",
        "| Split | RMSE | MAE | R^2 |",
        "| --- | ---: | ---: | ---: |",
        f"| Full 90% dev train | {train['rmse']:.9f} | {train['mae']:.9f} | {train['r2']:.9f} |",
        f"| Frozen 10% test | {test['rmse']:.9f} | {test['mae']:.9f} | {test['r2']:.9f} |",
        "",
        "## Comparison To Phase 2",
        "",
        f"- Phase 2 mean CV validation RMSE: `{cv['mean_validation_rmse']:.9f}`",
        f"- Test RMSE minus Phase 2 mean CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
        f"- Test RMSE minus full-dev train RMSE: `{payload['metrics']['test_minus_train_dev_rmse']:.9f}`",
        "",
        "## Selected Params",
        "",
    ]
    for key, value in payload["selected_config"]["params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model JSON: `{payload['artifacts']['model_json']}`",
            f"- Results JSON: `{payload['meta']['run_dir']}\\results.json`",
            f"- Status JSON: `{payload['meta']['run_dir']}\\status.json`",
            "",
        ]
    )
    if payload["artifacts"].get("test_predictions_npz"):
        lines.append(f"- Test predictions NPZ: `{payload['artifacts']['test_predictions_npz']}`")
    return "\n".join(lines) + "\n"


def write_phase3_xgboost_state(
    *,
    run_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "config_index": int(payload["selected_config"]["config_index"]),
            "source_trial_id": int(payload["selected_config"]["source_trial_id"]),
            "train_dev_rmse": float(payload["metrics"]["train_dev"]["rmse"]),
            "test_rmse": float(payload["metrics"]["test"]["rmse"]),
            "phase2_mean_validation_rmse": float(payload["source_phase2_config"]["mean_validation_rmse"]),
            "test_minus_phase2_cv_rmse": float(payload["metrics"]["test_minus_phase2_cv_rmse"]),
        },
    )
    (run_dir / "summary.md").write_text(render_phase3_xgboost_summary(payload), encoding="utf-8")
    return payload


def run_phase3_xgboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import xgboost as xgb

    phase2_results_path = args.phase3_source_run_dir / "results.json"
    if not phase2_results_path.exists():
        raise FileNotFoundError(f"Missing XGBoost Phase 2 results JSON: {phase2_results_path}")
    phase2_payload = json.loads(phase2_results_path.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in phase2_payload.get("leaderboard", [])
        if int(row.get("config_index", -1)) == int(args.phase3_config_index)
    ]
    if not candidates:
        raise ValueError(f"Phase 2 config index {args.phase3_config_index} was not found in {phase2_results_path}")
    selected = dict(candidates[0])
    params = dict(selected["params"])
    started_utc = now_utc_iso()
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    meta_path = cache_root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.sort(np.asarray(split_arrays["dev_idx"], dtype=np.int64))
    test_idx = np.sort(np.asarray(split_arrays["test_idx"], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: XGBoost Final Refit/Test",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 2 source run: `{args.phase3_source_run_dir}`",
            f"- Selected config index: `{args.phase3_config_index}`",
            f"- Source trial: `{selected['source_trial_id']}`",
            f"- Dev rows: `{dev_idx.shape[0]:,}`",
            f"- Test rows: `{test_idx.shape[0]:,}`",
            f"- Fixed boosting rounds: `{args.phase3_max_iter}`",
            f"- Threads: `{args.phase3_n_jobs}`",
            "",
        ],
    )

    t0 = time.time()
    X_dev = np.asarray(X[dev_idx], dtype=np.float32)
    y_dev = np.asarray(y[dev_idx], dtype=np.float32)
    X_test = np.asarray(X[test_idx], dtype=np.float32)
    y_test = np.asarray(y[test_idx], dtype=np.float32)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    dtrain = xgb.QuantileDMatrix(
        X_dev,
        label=y_dev,
        max_bin=int(params["max_bin"]),
        nthread=max(1, int(args.phase3_n_jobs)),
    )
    dtest = xgb.QuantileDMatrix(
        X_test,
        label=y_test,
        ref=dtrain,
        max_bin=int(params["max_bin"]),
        nthread=max(1, int(args.phase3_n_jobs)),
    )
    fit_started = time.time()
    booster = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "reg_alpha": float(params["reg_alpha"]),
            "reg_lambda": float(params["reg_lambda"]),
            "gamma": float(params["gamma"]),
            "max_bin": int(params["max_bin"]),
            "seed": int(args.random_state) + 30_000 + int(args.phase3_config_index),
            "nthread": max(1, int(args.phase3_n_jobs)),
            "verbosity": 0,
        },
        dtrain=dtrain,
        num_boost_round=int(args.phase3_max_iter),
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )
    fit_seconds = time.time() - fit_started
    train_pred = booster.predict(dtrain, iteration_range=(0, int(args.phase3_max_iter)))
    test_pred = booster.predict(dtest, iteration_range=(0, int(args.phase3_max_iter)))
    train_metrics = compute_metrics(y_dev, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    model_json = run_dir / "xgboost_phase3_model.json"
    booster.save_model(str(model_json))
    predictions_npz = None
    if bool(args.phase3_save_test_predictions):
        predictions_npz = run_dir / "test_predictions.npz"
        np.savez_compressed(
            predictions_npz,
            test_idx=test_idx,
            y_test=y_test,
            y_pred=np.asarray(test_pred, dtype=np.float64),
        )

    phase2_mean_rmse = float(selected["mean_validation_rmse"])
    payload = {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase2_source_run_dir": str(args.phase3_source_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": "completed",
        "dependency_status": dependency_status(),
        "selected_config": {
            "config_index": int(selected["config_index"]),
            "source_trial_id": int(selected["source_trial_id"]),
            "source_rank": int(selected.get("source_rank", selected.get("rank", 0))),
            "source_run_dir": str(selected["source_run_dir"]),
            "params": params,
        },
        "source_phase2_config": selected,
        "split": {
            "dev_rows": int(dev_idx.shape[0]),
            "test_rows": int(test_idx.shape[0]),
        },
        "fixed_settings": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "num_boost_round": int(args.phase3_max_iter),
        },
        "metrics": {
            "train_dev": train_metrics,
            "test": test_metrics,
            "test_minus_phase2_cv_rmse": float(test_metrics["rmse"] - phase2_mean_rmse),
            "test_minus_train_dev_rmse": float(test_metrics["rmse"] - train_metrics["rmse"]),
        },
        "timings_sec": {
            "matrix_load": float(matrix_load_seconds),
            "fit": float(fit_seconds),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase3_n_jobs),
        },
        "artifacts": {
            "model_json": str(model_json),
            "test_predictions_npz": None if predictions_npz is None else str(predictions_npz),
        },
    }
    write_phase3_xgboost_state(run_dir=run_dir, payload=payload)
    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: XGBoost Final Refit/Test",
            "",
            "Status:",
            "",
            "- completed",
            "",
            "Measured result:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Full-dev train RMSE: `{train_metrics['rmse']:.9f}`",
            f"- Frozen test RMSE: `{test_metrics['rmse']:.9f}`",
            f"- Phase 2 mean CV RMSE: `{phase2_mean_rmse:.9f}`",
            f"- Test minus Phase 2 CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
            f"- Fit seconds: `{fit_seconds:.1f}`",
            f"- Model artifact: `{model_json}`",
            "",
        ],
    )
    return payload


def render_phase3_catboost_summary(payload: dict[str, Any]) -> str:
    train = payload["metrics"]["train_dev"]
    test = payload["metrics"]["test"]
    cv = payload["source_phase2_config"]
    lines = [
        "# CatBoost Phase 3 Final Refit/Test",
        "",
        f"- Run dir: `{payload['meta']['run_dir']}`",
        f"- Status: `{payload['status']}`",
        f"- Phase 2 source run: `{payload['meta']['phase2_source_run_dir']}`",
        f"- Selected config index: `{payload['selected_config']['config_index']}`",
        f"- Source trial: `{payload['selected_config']['source_trial_id']}`",
        f"- Fit rows: `{payload['split']['dev_rows']:,}`",
        f"- Test rows: `{payload['split']['test_rows']:,}`",
        f"- Fixed iterations: `{payload['fixed_settings']['iterations']}`",
        f"- Thread count: `{payload['runtime_context']['n_jobs']}`",
        "",
        "## Metrics",
        "",
        "| Split | RMSE | MAE | R^2 |",
        "| --- | ---: | ---: | ---: |",
        f"| Full 90% dev train | {train['rmse']:.9f} | {train['mae']:.9f} | {train['r2']:.9f} |",
        f"| Frozen 10% test | {test['rmse']:.9f} | {test['mae']:.9f} | {test['r2']:.9f} |",
        "",
        "## Comparison To Phase 2",
        "",
        f"- Phase 2 mean CV validation RMSE: `{cv['mean_validation_rmse']:.9f}`",
        f"- Test RMSE minus Phase 2 mean CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
        f"- Test RMSE minus full-dev train RMSE: `{payload['metrics']['test_minus_train_dev_rmse']:.9f}`",
        "",
        "## Selected Params",
        "",
    ]
    for key, value in payload["selected_config"]["params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model CBM: `{payload['artifacts']['model_cbm']}`",
            f"- Model JSON: `{payload['artifacts']['model_json']}`",
            f"- Results JSON: `{payload['meta']['run_dir']}\\results.json`",
            f"- Status JSON: `{payload['meta']['run_dir']}\\status.json`",
            "",
        ]
    )
    if payload["artifacts"].get("test_predictions_npz"):
        lines.append(f"- Test predictions NPZ: `{payload['artifacts']['test_predictions_npz']}`")
    return "\n".join(lines) + "\n"


def write_phase3_catboost_state(
    *,
    run_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    json_dump(run_dir / "results.json", payload)
    json_dump(
        run_dir / "status.json",
        {
            "status": payload["status"],
            "updated_utc": payload["meta"]["updated_utc"],
            "config_index": int(payload["selected_config"]["config_index"]),
            "source_trial_id": int(payload["selected_config"]["source_trial_id"]),
            "train_dev_rmse": float(payload["metrics"]["train_dev"]["rmse"]),
            "test_rmse": float(payload["metrics"]["test"]["rmse"]),
            "phase2_mean_validation_rmse": float(payload["source_phase2_config"]["mean_validation_rmse"]),
            "test_minus_phase2_cv_rmse": float(payload["metrics"]["test_minus_phase2_cv_rmse"]),
        },
    )
    (run_dir / "summary.md").write_text(render_phase3_catboost_summary(payload), encoding="utf-8")
    return payload


def run_phase3_catboost(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    phase2_results_path = args.phase3_source_run_dir / "results.json"
    if not phase2_results_path.exists():
        raise FileNotFoundError(f"Missing CatBoost Phase 2 results JSON: {phase2_results_path}")
    phase2_payload = json.loads(phase2_results_path.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in phase2_payload.get("leaderboard", [])
        if int(row.get("config_index", -1)) == int(args.phase3_config_index)
    ]
    if not candidates:
        raise ValueError(f"Phase 2 config index {args.phase3_config_index} was not found in {phase2_results_path}")
    selected = dict(candidates[0])
    params = dict(selected["params"])
    started_utc = now_utc_iso()
    cache_root = DEFAULT_MODELING_ROOT / "cache"
    meta_path = cache_root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["X_path"] = str(cache_root / "X_float32.dat")
        meta["y_path"] = str(cache_root / "y_float64.npy")
        meta["patch_codes_path"] = str(cache_root / "patch_codes_int16.npy")
        meta["server_codes_path"] = str(cache_root / "server_codes_int16.npy")
        meta["mapped_participants_path"] = str(cache_root / "mapped_participants_int8.npy")
    else:
        meta = linear.prepare_cache(
            db_path=args.db_path,
            table_name=args.table_name,
            cache_root=cache_root,
            chunk_size=args.chunk_size,
        )
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.sort(np.asarray(split_arrays["dev_idx"], dtype=np.int64))
    test_idx = np.sort(np.asarray(split_arrays["test_idx"], dtype=np.int64))

    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: CatBoost Final Refit/Test",
            "",
            "Status:",
            "",
            "- started",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Phase 2 source run: `{args.phase3_source_run_dir}`",
            f"- Selected config index: `{args.phase3_config_index}`",
            f"- Source trial: `{selected['source_trial_id']}`",
            f"- Dev rows: `{dev_idx.shape[0]:,}`",
            f"- Test rows: `{test_idx.shape[0]:,}`",
            f"- Fixed iterations: `{args.phase3_max_iter}`",
            f"- Threads: `{args.phase3_n_jobs}`",
            "",
        ],
    )

    t0 = time.time()
    X_dev = np.asarray(X[dev_idx], dtype=np.float32)
    y_dev = np.asarray(y[dev_idx], dtype=np.float64)
    X_test = np.asarray(X[test_idx], dtype=np.float32)
    y_test = np.asarray(y[test_idx], dtype=np.float64)
    matrix_load_seconds = time.time() - t0
    del arrays
    del split_bundle
    gc.collect()

    train_dir = run_dir / "catboost_info"
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=int(args.phase3_max_iter),
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        rsm=float(params["rsm"]),
        bagging_temperature=float(params["bagging_temperature"]),
        border_count=int(params["border_count"]),
        random_seed=int(args.random_state) + 30_000 + int(args.phase3_config_index),
        thread_count=max(1, int(args.phase3_n_jobs)),
        train_dir=str(train_dir),
        allow_writing_files=True,
        verbose=False,
    )
    fit_started = time.time()
    model.fit(X_dev, y_dev, verbose=False)
    fit_seconds = time.time() - fit_started
    actual_iterations = int(model.tree_count_ or args.phase3_max_iter)
    train_pred = model.predict(X_dev)
    test_pred = model.predict(X_test)
    train_metrics = compute_metrics(y_dev, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    model_cbm = run_dir / "catboost_phase3_model.cbm"
    model_json = run_dir / "catboost_phase3_model.json"
    model.save_model(str(model_cbm))
    model.save_model(str(model_json), format="json")
    predictions_npz = None
    if bool(args.phase3_save_test_predictions):
        predictions_npz = run_dir / "test_predictions.npz"
        np.savez_compressed(
            predictions_npz,
            test_idx=test_idx,
            y_test=y_test,
            y_pred=np.asarray(test_pred, dtype=np.float64),
        )

    phase2_mean_rmse = float(selected["mean_validation_rmse"])
    payload = {
        "meta": {
            "created_utc": started_utc,
            "updated_utc": now_utc_iso(),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "phase2_source_run_dir": str(args.phase3_source_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "status": "completed",
        "dependency_status": dependency_status(),
        "selected_config": {
            "config_index": int(selected["config_index"]),
            "source_trial_id": int(selected["source_trial_id"]),
            "source_rank": int(selected.get("source_rank", selected.get("rank", 0))),
            "source_run_dir": str(selected["source_run_dir"]),
            "params": params,
        },
        "source_phase2_config": selected,
        "split": {
            "dev_rows": int(dev_idx.shape[0]),
            "test_rows": int(test_idx.shape[0]),
        },
        "fixed_settings": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": int(args.phase3_max_iter),
            "actual_iterations_fit": int(actual_iterations),
        },
        "metrics": {
            "train_dev": train_metrics,
            "test": test_metrics,
            "test_minus_phase2_cv_rmse": float(test_metrics["rmse"] - phase2_mean_rmse),
            "test_minus_train_dev_rmse": float(test_metrics["rmse"] - train_metrics["rmse"]),
        },
        "timings_sec": {
            "matrix_load": float(matrix_load_seconds),
            "fit": float(fit_seconds),
        },
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
            "n_jobs": int(args.phase3_n_jobs),
        },
        "artifacts": {
            "model_cbm": str(model_cbm),
            "model_json": str(model_json),
            "test_predictions_npz": None if predictions_npz is None else str(predictions_npz),
        },
    }
    write_phase3_catboost_state(run_dir=run_dir, payload=payload)
    append_markdown_log(
        log_path,
        [
            "### Phase 3 Run: CatBoost Final Refit/Test",
            "",
            "Status:",
            "",
            "- completed",
            "",
            "Measured result:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Full-dev train RMSE: `{train_metrics['rmse']:.9f}`",
            f"- Frozen test RMSE: `{test_metrics['rmse']:.9f}`",
            f"- Phase 2 mean CV RMSE: `{phase2_mean_rmse:.9f}`",
            f"- Test minus Phase 2 CV RMSE: `{payload['metrics']['test_minus_phase2_cv_rmse']:.9f}`",
            f"- Fit seconds: `{fit_seconds:.1f}`",
            f"- Model artifact: `{model_cbm}`",
            "",
        ],
    )
    return payload


def run_smoke_histgb(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    rng = np.random.default_rng(args.random_state)

    log("Preparing cached modeling inputs")
    t0 = time.time()
    meta = linear.prepare_cache(
        db_path=args.db_path,
        table_name=args.table_name,
        cache_root=DEFAULT_MODELING_ROOT / "cache",
        chunk_size=args.chunk_size,
    )
    timings["cache_prepare"] = time.time() - t0

    log("Loading cached arrays and frozen split")
    t0 = time.time()
    arrays = linear.load_cached_arrays(meta)
    split_bundle = load_source_split(args.source_split_run_dir)
    X = arrays["X"]
    y = arrays["y"]
    split_arrays = split_bundle["split_arrays"]
    dev_idx = np.asarray(split_arrays["dev_idx"], dtype=np.int64)
    fold_ids = np.asarray(split_arrays["fold_ids"], dtype=np.int8)
    val_idx_full = dev_idx[fold_ids[dev_idx] == int(args.smoke_fold)]
    train_idx_full = dev_idx[fold_ids[dev_idx] != int(args.smoke_fold)]
    train_idx = sample_indices(train_idx_full, int(args.smoke_train_cap), rng)
    val_idx = sample_indices(val_idx_full, int(args.smoke_val_cap), rng)
    timings["data_and_split_load"] = time.time() - t0

    log("Materializing sampled train and validation matrices")
    t0 = time.time()
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float64)
    timings["matrix_materialization"] = time.time() - t0

    log("Fitting smoke model")
    t0 = time.time()
    model_name = ""
    model_params: dict[str, Any]
    actual_iterations = int(args.smoke_max_iter)
    if args.mode == "smoke_histgb":
        model_name = "HistGradientBoostingRegressor"
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(args.smoke_learning_rate),
            max_iter=int(args.smoke_max_iter),
            max_leaf_nodes=int(args.smoke_max_leaf_nodes),
            max_depth=int(args.smoke_max_depth),
            min_samples_leaf=int(args.smoke_min_samples_leaf),
            l2_regularization=float(args.smoke_l2_regularization),
            validation_fraction=float(args.smoke_validation_fraction),
            early_stopping=True,
            n_iter_no_change=int(args.smoke_n_iter_no_change),
            random_state=int(args.random_state),
        )
        model.fit(X_train, y_train)
        actual_iterations = int(getattr(model, "n_iter_", args.smoke_max_iter))
        model_params = {
            "learning_rate": float(args.smoke_learning_rate),
            "max_iter": int(args.smoke_max_iter),
            "max_leaf_nodes": int(args.smoke_max_leaf_nodes),
            "max_depth": int(args.smoke_max_depth),
            "min_samples_leaf": int(args.smoke_min_samples_leaf),
            "l2_regularization": float(args.smoke_l2_regularization),
            "validation_fraction": float(args.smoke_validation_fraction),
            "n_iter_no_change": int(args.smoke_n_iter_no_change),
        }
    elif args.mode == "smoke_lightgbm":
        import lightgbm as lgb

        model_name = "LightGBM"
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=float(args.smoke_learning_rate),
            n_estimators=int(args.smoke_max_iter),
            num_leaves=int(args.smoke_max_leaf_nodes),
            max_depth=int(args.smoke_max_depth),
            min_child_samples=int(args.smoke_min_samples_leaf),
            reg_lambda=float(args.smoke_l2_regularization),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=int(args.random_state),
            n_jobs=max(1, int(os.cpu_count() or 1) - 1),
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(int(args.smoke_early_stopping_rounds), verbose=False)],
        )
        actual_iterations = int(getattr(model, "best_iteration_", args.smoke_max_iter) or args.smoke_max_iter)
        model_params = {
            "learning_rate": float(args.smoke_learning_rate),
            "n_estimators": int(args.smoke_max_iter),
            "num_leaves": int(args.smoke_max_leaf_nodes),
            "max_depth": int(args.smoke_max_depth),
            "min_child_samples": int(args.smoke_min_samples_leaf),
            "reg_lambda": float(args.smoke_l2_regularization),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": int(args.smoke_early_stopping_rounds),
        }
    elif args.mode == "smoke_xgboost":
        import xgboost as xgb

        model_name = "XGBoost"
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            learning_rate=float(args.smoke_learning_rate),
            n_estimators=int(args.smoke_max_iter),
            max_depth=int(args.smoke_max_depth),
            min_child_weight=max(1.0, float(args.smoke_min_samples_leaf) / 50.0),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=float(args.smoke_l2_regularization),
            max_bin=256,
            early_stopping_rounds=int(args.smoke_early_stopping_rounds),
            random_state=int(args.random_state),
            n_jobs=max(1, int(os.cpu_count() or 1) - 1),
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        actual_iterations = int(getattr(model, "best_iteration", args.smoke_max_iter) or args.smoke_max_iter)
        if actual_iterations <= 0:
            actual_iterations = int(args.smoke_max_iter)
        model_params = {
            "learning_rate": float(args.smoke_learning_rate),
            "n_estimators": int(args.smoke_max_iter),
            "max_depth": int(args.smoke_max_depth),
            "min_child_weight": max(1.0, float(args.smoke_min_samples_leaf) / 50.0),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": float(args.smoke_l2_regularization),
            "max_bin": 256,
            "early_stopping_rounds": int(args.smoke_early_stopping_rounds),
        }
    elif args.mode == "smoke_catboost":
        from catboost import CatBoostRegressor

        model_name = "CatBoost"
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=int(args.smoke_max_iter),
            learning_rate=float(args.smoke_learning_rate),
            depth=int(args.smoke_max_depth),
            l2_leaf_reg=float(args.smoke_l2_regularization),
            rsm=0.8,
            border_count=128,
            random_seed=int(args.random_state),
            od_type="Iter",
            od_wait=int(args.smoke_early_stopping_rounds),
            train_dir=str(run_dir / "catboost_info"),
            verbose=False,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
        actual_iterations = int(model.get_best_iteration() or args.smoke_max_iter)
        if actual_iterations <= 0:
            actual_iterations = int(args.smoke_max_iter)
        model_params = {
            "iterations": int(args.smoke_max_iter),
            "learning_rate": float(args.smoke_learning_rate),
            "depth": int(args.smoke_max_depth),
            "l2_leaf_reg": float(args.smoke_l2_regularization),
            "rsm": 0.8,
            "border_count": 128,
            "od_wait": int(args.smoke_early_stopping_rounds),
            "train_dir": str(run_dir / "catboost_info"),
        }
    else:
        raise ValueError(f"Unsupported smoke mode: {args.mode}")
    timings["model_fit"] = time.time() - t0

    log("Scoring smoke model")
    t0 = time.time()
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)
    np.save(run_dir / "histgb_smoke_validation_predictions.npy", val_pred)
    timings["scoring_and_export"] = time.time() - t0

    payload = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_dir": str(run_dir),
            "mode": args.mode,
            "source_split_run_dir": str(args.source_split_run_dir),
            "db_path": str(args.db_path),
            "table_name": args.table_name,
            "row_count": int(meta["row_count"]),
            "predictor_count": int(meta["predictor_count"]),
        },
        "dependency_status": dependency_status(),
        "sampling": {
            "smoke_fold": int(args.smoke_fold),
            "train_rows_available": int(train_idx_full.shape[0]),
            "validation_rows_available": int(val_idx_full.shape[0]),
            "train_rows_used": int(train_idx.shape[0]),
            "validation_rows_used": int(val_idx.shape[0]),
        },
        "model": {
            "name": model_name,
            "params": {
                **model_params,
                "actual_iterations_fit": int(actual_iterations),
            },
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
        },
        "timings_sec": timings,
        "runtime_context": {
            "python_version": os.sys.version,
            "logical_cpus": int(os.cpu_count() or 0),
        },
    }

    json_dump(run_dir / "results.json", payload)
    (run_dir / "summary.md").write_text(render_smoke_summary(payload), encoding="utf-8")

    append_markdown_log(
        log_path,
        [
            f"### Smoke Run: {model_name}",
            "",
            "Status:",
            "",
            "- completed",
            "",
            "Notes:",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Mode: `{args.mode}`",
            f"- Smoke fold: `{args.smoke_fold}`",
            f"- Sampled train rows: `{train_idx.shape[0]:,}` from `{train_idx_full.shape[0]:,}` available",
            f"- Sampled validation rows: `{val_idx.shape[0]:,}` from `{val_idx_full.shape[0]:,}` available",
            f"- Actual fitted iterations: `{actual_iterations}`",
            "",
            "Measured result:",
            "",
            f"- Train RMSE: `{train_metrics['rmse']:.6f}`",
            f"- Train MAE: `{train_metrics['mae']:.6f}`",
            f"- Train R^2: `{train_metrics['r2']:.6f}`",
            f"- Validation RMSE: `{val_metrics['rmse']:.6f}`",
            f"- Validation MAE: `{val_metrics['mae']:.6f}`",
            f"- Validation R^2: `{val_metrics['r2']:.6f}`",
            f"- Summary markdown: `{run_dir / 'summary.md'}`",
            f"- Results JSON: `{run_dir / 'results.json'}`",
            "",
        ],
    )
    return payload


def main() -> None:
    args = parse_args()
    run_name = args.run_name.strip() or f"{utc_stamp()}_tree_primary_v1"
    run_dir = args.output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"smoke_histgb", "smoke_lightgbm", "smoke_xgboost", "smoke_catboost"}:
        payload = run_smoke_histgb(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote smoke results to {run_dir}")
        log(
            "Validation metrics: "
            f"RMSE={payload['metrics']['validation']['rmse']:.6f}, "
            f"MAE={payload['metrics']['validation']['mae']:.6f}, "
            f"R2={payload['metrics']['validation']['r2']:.6f}"
        )
        return
    if args.mode == "phase1a_lightgbm":
        payload = run_phase1a_lightgbm(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote LightGBM Phase 1A results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1b_lightgbm":
        payload = run_phase1b_lightgbm(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote LightGBM Phase 1B results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1c_lightgbm":
        payload = run_phase1c_lightgbm(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote LightGBM Phase 1C results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1a_xgboost":
        payload = run_phase1a_xgboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote XGBoost Phase 1A results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1b_xgboost":
        payload = run_phase1b_xgboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote XGBoost Phase 1B results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1a_catboost":
        payload = run_phase1a_catboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote CatBoost Phase 1A results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1b_catboost":
        payload = run_phase1b_catboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote CatBoost Phase 1B results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase1c_catboost":
        payload = run_phase1c_catboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote CatBoost Phase 1C results to {run_dir}")
        if payload.get("best_trial") is not None:
            log(
                "Current best validation metrics: "
                f"RMSE={payload['best_trial']['metrics']['validation']['rmse']:.6f}, "
                f"MAE={payload['best_trial']['metrics']['validation']['mae']:.6f}, "
                f"R2={payload['best_trial']['metrics']['validation']['r2']:.6f}"
            )
        return
    if args.mode == "phase2_lightgbm":
        payload = run_phase2_tree(args=args, run_dir=run_dir, log_path=args.log_path, model_name="LightGBM")
        log(f"Wrote LightGBM Phase 2 results to {run_dir}")
        if payload.get("best_config") is not None:
            log(f"Current best mean validation RMSE={payload['best_config']['mean_validation_rmse']:.6f}")
        return
    if args.mode == "phase2_xgboost":
        payload = run_phase2_tree(args=args, run_dir=run_dir, log_path=args.log_path, model_name="XGBoost")
        log(f"Wrote XGBoost Phase 2 results to {run_dir}")
        if payload.get("best_config") is not None:
            log(f"Current best mean validation RMSE={payload['best_config']['mean_validation_rmse']:.6f}")
        return
    if args.mode == "phase2_catboost":
        payload = run_phase2_tree(args=args, run_dir=run_dir, log_path=args.log_path, model_name="CatBoost")
        log(f"Wrote CatBoost Phase 2 results to {run_dir}")
        if payload.get("best_config") is not None:
            log(f"Current best mean validation RMSE={payload['best_config']['mean_validation_rmse']:.6f}")
        return
    if args.mode == "phase3_lightgbm":
        payload = run_phase3_lightgbm(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote LightGBM Phase 3 results to {run_dir}")
        log(
            "Final frozen-test metrics: "
            f"RMSE={payload['metrics']['test']['rmse']:.6f}, "
            f"MAE={payload['metrics']['test']['mae']:.6f}, "
            f"R2={payload['metrics']['test']['r2']:.6f}"
        )
        return
    if args.mode == "phase3_xgboost":
        payload = run_phase3_xgboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote XGBoost Phase 3 results to {run_dir}")
        log(
            "Final frozen-test metrics: "
            f"RMSE={payload['metrics']['test']['rmse']:.6f}, "
            f"MAE={payload['metrics']['test']['mae']:.6f}, "
            f"R2={payload['metrics']['test']['r2']:.6f}"
        )
        return
    if args.mode == "phase3_catboost":
        payload = run_phase3_catboost(args=args, run_dir=run_dir, log_path=args.log_path)
        log(f"Wrote CatBoost Phase 3 results to {run_dir}")
        log(
            "Final frozen-test metrics: "
            f"RMSE={payload['metrics']['test']['rmse']:.6f}, "
            f"MAE={payload['metrics']['test']['mae']:.6f}, "
            f"R2={payload['metrics']['test']['r2']:.6f}"
        )
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
