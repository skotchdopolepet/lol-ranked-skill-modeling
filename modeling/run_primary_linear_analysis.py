from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "out_prod" / "primary_dataset" / "primary_dataset.sqlite3"
DEFAULT_TABLE_NAME = "primary_dataset_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out_prod" / "primary_dataset" / "modeling"
DEFAULT_RANDOM_STATE = 42

METADATA_COLUMNS = {
    "match_id",
    "source_folder",
    "source_file",
    "game_version",
    "patch",
    "queue_id",
    "map_id",
    "server_timezone_fallback_used",
}
SECONDARY_ANALYSIS_COLUMNS = {
    "server",
    "is_weekend",
    "local_time_sin",
    "local_time_cos",
}
TARGET_SIDE_COLUMNS = {
    "average_skill_level",
    "mapped_participants",
    "unique_participants",
    "skill_server",
}


@dataclass
class SummaryStats:
    n: int
    sum_y: float
    yTy: float
    sum_x: np.ndarray
    XTX: np.ndarray
    XTy: np.ndarray

    @classmethod
    def zeros(cls, p: int) -> "SummaryStats":
        return cls(
            n=0,
            sum_y=0.0,
            yTy=0.0,
            sum_x=np.zeros(p, dtype=np.float64),
            XTX=np.zeros((p, p), dtype=np.float64),
            XTy=np.zeros(p, dtype=np.float64),
        )

    def accumulate(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        X64 = np.asarray(X, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        self.n += int(y64.shape[0])
        self.sum_y += float(np.sum(y64))
        self.yTy += float(y64 @ y64)
        self.sum_x += np.sum(X64, axis=0)
        self.XTX += X64.T @ X64
        self.XTy += X64.T @ y64

    def subtract(self, other: "SummaryStats") -> "SummaryStats":
        return SummaryStats(
            n=int(self.n - other.n),
            sum_y=float(self.sum_y - other.sum_y),
            yTy=float(self.yTy - other.yTy),
            sum_x=self.sum_x - other.sum_x,
            XTX=self.XTX - other.XTX,
            XTy=self.XTy - other.XTy,
        )

    def mean_y(self) -> float:
        return float(self.sum_y / self.n)

    def mean_x(self) -> np.ndarray:
        return self.sum_x / float(self.n)

    def centered(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        mean_x = self.mean_x()
        mean_y = self.mean_y()
        XcTXc = self.XTX - np.outer(self.sum_x, self.sum_x) / float(self.n)
        XcTyc = self.XTy - mean_y * self.sum_x
        return XcTXc, XcTyc, mean_x, mean_y, float(self.n)

    def sst(self) -> float:
        return max(float(self.yTy - (self.sum_y * self.sum_y) / float(self.n)), 0.0)


@dataclass
class LinearFit:
    beta: np.ndarray
    intercept: float
    variant: str
    model_name: str
    alpha: float | None = None
    converged: bool | None = None
    iterations: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primary linear analysis on the primary dataset.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--table-name", type=str, default=DEFAULT_TABLE_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--outer-test-size", type=float, default=0.10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--ridge-grid-size", type=int, default=17)
    parser.add_argument("--lasso-grid-size", type=int, default=25)
    parser.add_argument("--lasso-min-ratio", type=float, default=1e-4)
    parser.add_argument("--lasso-max-iter", type=int, default=5000)
    parser.add_argument("--lasso-tol", type=float, default=1e-7)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def log(msg: str) -> None:
    print(msg, flush=True)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_schema(db_path: Path, table_name: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
        return [str(row[1]) for row in rows]
    finally:
        conn.close()


def select_primary_columns(all_columns: list[str]) -> list[str]:
    out: list[str] = []
    for column in all_columns:
        if column in METADATA_COLUMNS:
            continue
        if column in SECONDARY_ANALYSIS_COLUMNS:
            continue
        if column in TARGET_SIDE_COLUMNS:
            continue
        out.append(column)
    return out


def prepare_cache(
    *,
    db_path: Path,
    table_name: str,
    cache_root: Path,
    chunk_size: int,
) -> dict[str, Any]:
    all_columns = load_schema(db_path, table_name)
    predictor_columns = select_primary_columns(all_columns)
    if not predictor_columns:
        raise RuntimeError("No primary predictor columns were found.")

    cache_dir = cache_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    X_path = cache_dir / "X_float32.dat"
    y_path = cache_dir / "y_float64.npy"
    patch_path = cache_dir / "patch_codes_int16.npy"
    server_path = cache_dir / "server_codes_int16.npy"
    mapped_path = cache_dir / "mapped_participants_int8.npy"

    if meta_path.exists() and X_path.exists() and y_path.exists() and patch_path.exists() and server_path.exists() and mapped_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("db_path") == str(db_path)
            and meta.get("table_name") == table_name
            and meta.get("predictor_columns") == predictor_columns
        ):
            log(f"Reusing cached linear inputs from {cache_dir}")
            return meta

    conn = sqlite3.connect(db_path)
    try:
        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0])
        patches = [str(row[0]) for row in conn.execute(f"SELECT DISTINCT patch FROM {quote_ident(table_name)} ORDER BY patch")]
        servers = [str(row[0]) for row in conn.execute(f"SELECT DISTINCT server FROM {quote_ident(table_name)} ORDER BY server")]
        patch_to_code = {value: idx for idx, value in enumerate(patches)}
        server_to_code = {value: idx for idx, value in enumerate(servers)}

        X_mm = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(row_count, len(predictor_columns)))
        y = np.empty(row_count, dtype=np.float64)
        patch_codes = np.empty(row_count, dtype=np.int16)
        server_codes = np.empty(row_count, dtype=np.int16)
        mapped_participants = np.empty(row_count, dtype=np.int8)

        sql_columns = predictor_columns + ["average_skill_level", "patch", "server", "mapped_participants"]
        sql = (
            "SELECT "
            + ", ".join(quote_ident(name) for name in sql_columns)
            + f" FROM {quote_ident(table_name)}"
        )
        cur = conn.execute(sql)
        offset = 0
        start = time.time()
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            batch = np.asarray(rows, dtype=object)
            batch_n = int(batch.shape[0])
            X_mm[offset : offset + batch_n, :] = np.asarray(batch[:, : len(predictor_columns)], dtype=np.float32)
            y[offset : offset + batch_n] = np.asarray(batch[:, len(predictor_columns)], dtype=np.float64)
            patch_codes[offset : offset + batch_n] = np.asarray(
                [patch_to_code[str(v)] for v in batch[:, len(predictor_columns) + 1]],
                dtype=np.int16,
            )
            server_codes[offset : offset + batch_n] = np.asarray(
                [server_to_code[str(v)] for v in batch[:, len(predictor_columns) + 2]],
                dtype=np.int16,
            )
            mapped_participants[offset : offset + batch_n] = np.asarray(
                batch[:, len(predictor_columns) + 3],
                dtype=np.int8,
            )
            offset += batch_n
            if offset % max(chunk_size * 20, 1) == 0 or offset == row_count:
                elapsed = time.time() - start
                log(f"Extracted {offset:,}/{row_count:,} rows into cache in {elapsed:.1f}s")
        X_mm.flush()
        np.save(y_path, y)
        np.save(patch_path, patch_codes)
        np.save(server_path, server_codes)
        np.save(mapped_path, mapped_participants)
        meta = {
            "db_path": str(db_path),
            "table_name": table_name,
            "row_count": row_count,
            "predictor_columns": predictor_columns,
            "predictor_count": len(predictor_columns),
            "patch_values": patches,
            "server_values": servers,
            "X_path": str(X_path),
            "X_dtype": "float32",
            "y_path": str(y_path),
            "patch_codes_path": str(patch_path),
            "server_codes_path": str(server_path),
            "mapped_participants_path": str(mapped_path),
        }
        json_dump(meta_path, meta)
        return meta
    finally:
        conn.close()


def load_cached_arrays(meta: dict[str, Any]) -> dict[str, Any]:
    n = int(meta["row_count"])
    p = int(meta["predictor_count"])
    return {
        "X": np.memmap(meta["X_path"], dtype=np.float32, mode="r", shape=(n, p)),
        "y": np.load(meta["y_path"]),
        "patch_codes": np.load(meta["patch_codes_path"]),
        "server_codes": np.load(meta["server_codes_path"]),
        "mapped_participants": np.load(meta["mapped_participants_path"]),
    }


def make_decile_bins(values: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values64 = np.asarray(values, dtype=np.float64)
    if edges is None:
        edges = np.quantile(values64, q=np.linspace(0.1, 0.9, 9), method="linear")
    bins = np.searchsorted(edges, values64, side="right").astype(np.int16)
    return bins, np.asarray(edges, dtype=np.float64)


def combine_strata(patch_codes: np.ndarray, server_codes: np.ndarray, bins: np.ndarray) -> np.ndarray:
    max_server = int(np.max(server_codes)) + 1
    return (
        patch_codes.astype(np.int64) * (max_server * 10)
        + server_codes.astype(np.int64) * 10
        + bins.astype(np.int64)
    )


def describe_min_stratum(strata: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(strata, return_counts=True)
    return {
        "num_strata": int(unique.shape[0]),
        "min_count": int(np.min(counts)),
        "max_count": int(np.max(counts)),
        "median_count": float(np.median(counts)),
    }


def build_splits(
    *,
    y: np.ndarray,
    patch_codes: np.ndarray,
    server_codes: np.ndarray,
    test_size: float,
    n_folds: int,
    random_state: int,
) -> dict[str, Any]:
    global_bins, outer_edges = make_decile_bins(y)
    outer_strata = combine_strata(patch_codes, server_codes, global_bins)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(splitter.split(np.zeros_like(y), outer_strata))
    dev_idx = np.asarray(dev_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    dev_bins, inner_edges = make_decile_bins(y[dev_idx])
    dev_strata = combine_strata(patch_codes[dev_idx], server_codes[dev_idx], dev_bins)

    fold_ids = np.full(y.shape[0], fill_value=-1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold_id, (_, val_local_idx) in enumerate(skf.split(np.zeros(dev_idx.shape[0], dtype=np.int8), dev_strata)):
        fold_ids[dev_idx[val_local_idx]] = int(fold_id)

    split_info = {
        "random_state": random_state,
        "outer_test_size": float(test_size),
        "n_folds": int(n_folds),
        "outer_bin_edges": outer_edges.tolist(),
        "inner_bin_edges": inner_edges.tolist(),
        "outer_strata_summary": describe_min_stratum(outer_strata),
        "inner_strata_summary": describe_min_stratum(dev_strata),
        "dev_size": int(dev_idx.shape[0]),
        "test_size_rows": int(test_idx.shape[0]),
    }
    return {
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "fold_ids": fold_ids,
        "split_info": split_info,
    }


def accumulate_summaries(
    *,
    X: np.memmap,
    y: np.ndarray,
    test_idx: np.ndarray,
    fold_ids: np.ndarray,
    n_folds: int,
    chunk_size: int,
) -> dict[str, Any]:
    n_rows, p = X.shape
    is_test = np.zeros(n_rows, dtype=bool)
    is_test[test_idx] = True
    dev_summary = SummaryStats.zeros(p)
    test_summary = SummaryStats.zeros(p)
    fold_val_summaries = [SummaryStats.zeros(p) for _ in range(n_folds)]

    start = time.time()
    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        X_chunk = X[chunk_start:chunk_end]
        y_chunk = y[chunk_start:chunk_end]
        test_mask = is_test[chunk_start:chunk_end]
        if np.any(~test_mask):
            dev_summary.accumulate(X_chunk[~test_mask], y_chunk[~test_mask])
        if np.any(test_mask):
            test_summary.accumulate(X_chunk[test_mask], y_chunk[test_mask])
        fold_chunk = fold_ids[chunk_start:chunk_end]
        for fold_id in range(n_folds):
            fold_mask = fold_chunk == fold_id
            if np.any(fold_mask):
                fold_val_summaries[fold_id].accumulate(X_chunk[fold_mask], y_chunk[fold_mask])
        if chunk_end % max(chunk_size * 20, 1) == 0 or chunk_end == n_rows:
            elapsed = time.time() - start
            log(f"Accumulated summary stats for {chunk_end:,}/{n_rows:,} rows in {elapsed:.1f}s")
        del X_chunk, y_chunk
        gc.collect()

    return {
        "dev_summary": dev_summary,
        "test_summary": test_summary,
        "fold_val_summaries": fold_val_summaries,
        "is_test": is_test,
    }


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1e-12) @ rhs


def fit_ols(summary: SummaryStats, standardized: bool) -> LinearFit:
    XcTXc, XcTyc, mean_x, mean_y, n = summary.centered()
    if standardized:
        var = np.diag(XcTXc) / n
        scale = np.sqrt(np.maximum(var, 1e-12))
        ZTZ = XcTXc / np.outer(scale, scale)
        ZTy = XcTyc / scale
        beta_z = np.linalg.pinv(ZTZ, rcond=1e-12) @ ZTy
        beta = beta_z / scale
        intercept = mean_y - float(mean_x @ beta)
        return LinearFit(beta=beta, intercept=float(intercept), variant="standardized", model_name="ols")
    beta = np.linalg.pinv(XcTXc, rcond=1e-12) @ XcTyc
    intercept = mean_y - float(mean_x @ beta)
    return LinearFit(beta=beta, intercept=float(intercept), variant="raw", model_name="ols")


def ridge_alpha_grid(summary: SummaryStats, standardized: bool, grid_size: int) -> np.ndarray:
    XcTXc, _, _, _, n = summary.centered()
    if standardized:
        scale = np.sqrt(np.maximum(np.diag(XcTXc) / n, 1e-12))
        design = (XcTXc / np.outer(scale, scale)) / n
    else:
        design = XcTXc / n
    median_diag = float(np.median(np.diag(design)))
    base = max(median_diag, 1e-8)
    return base * np.power(10.0, np.linspace(-6.0, 4.0, grid_size))


def fit_ridge(summary: SummaryStats, standardized: bool, alpha: float) -> LinearFit:
    XcTXc, XcTyc, mean_x, mean_y, n = summary.centered()
    if standardized:
        var = np.diag(XcTXc) / n
        scale = np.sqrt(np.maximum(var, 1e-12))
        ZTZ = XcTXc / np.outer(scale, scale)
        ZTy = XcTyc / scale
        beta_z = solve_linear_system(ZTZ / n + alpha * np.eye(ZTZ.shape[0]), ZTy / n)
        beta = beta_z / scale
        intercept = mean_y - float(mean_x @ beta)
        return LinearFit(beta=beta, intercept=float(intercept), variant="standardized", model_name="ridge", alpha=float(alpha))
    beta = solve_linear_system(XcTXc / n + alpha * np.eye(XcTXc.shape[0]), XcTyc / n)
    intercept = mean_y - float(mean_x @ beta)
    return LinearFit(beta=beta, intercept=float(intercept), variant="raw", model_name="ridge", alpha=float(alpha))


def soft_threshold(value: float, alpha: float) -> float:
    if value > alpha:
        return value - alpha
    if value < -alpha:
        return value + alpha
    return 0.0


def fit_lasso(
    summary: SummaryStats,
    standardized: bool,
    alpha: float,
    *,
    warm_start: np.ndarray | None,
    max_iter: int,
    tol: float,
) -> LinearFit:
    XcTXc, XcTyc, mean_x, mean_y, n = summary.centered()
    if standardized:
        var = np.diag(XcTXc) / n
        scale = np.sqrt(np.maximum(var, 1e-12))
        G = (XcTXc / np.outer(scale, scale)) / n
        c = (XcTyc / scale) / n
    else:
        scale = None
        G = XcTXc / n
        c = XcTyc / n

    p = G.shape[0]
    beta_work = np.zeros(p, dtype=np.float64) if warm_start is None else np.asarray(warm_start, dtype=np.float64).copy()
    Gb = G @ beta_work
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        max_delta = 0.0
        for j in range(p):
            g_jj = float(G[j, j])
            if g_jj <= 0.0:
                beta_work[j] = 0.0
                continue
            old = float(beta_work[j])
            rho = float(c[j] - (Gb[j] - g_jj * old))
            new = soft_threshold(rho, alpha) / g_jj
            if new != old:
                delta = new - old
                beta_work[j] = new
                Gb += G[:, j] * delta
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
        if max_delta < tol:
            converged = True
            break

    if standardized:
        beta = beta_work / scale
        variant = "standardized"
    else:
        beta = beta_work
        variant = "raw"
    intercept = mean_y - float(mean_x @ beta)
    return LinearFit(
        beta=beta,
        intercept=float(intercept),
        variant=variant,
        model_name="lasso",
        alpha=float(alpha),
        converged=bool(converged),
        iterations=int(n_iter),
    )


def lasso_alpha_grid(summary: SummaryStats, standardized: bool, grid_size: int, min_ratio: float) -> np.ndarray:
    XcTXc, XcTyc, _, _, n = summary.centered()
    if standardized:
        scale = np.sqrt(np.maximum(np.diag(XcTXc) / n, 1e-12))
        c = (XcTyc / scale) / n
    else:
        c = XcTyc / n
    alpha_max = max(float(np.max(np.abs(c))), 1e-12)
    alpha_min = max(alpha_max * float(min_ratio), 1e-8)
    return np.geomspace(alpha_max, alpha_min, num=grid_size)


def sse_from_summary(summary: SummaryStats, beta: np.ndarray, intercept: float) -> float:
    beta64 = np.asarray(beta, dtype=np.float64)
    term = (
        summary.yTy
        - 2.0 * intercept * summary.sum_y
        - 2.0 * float(beta64 @ summary.XTy)
        + 2.0 * intercept * float(beta64 @ summary.sum_x)
        + float(summary.n) * (intercept ** 2)
        + float(beta64 @ (summary.XTX @ beta64))
    )
    return max(float(term), 0.0)


def rmse_r2_from_summary(summary: SummaryStats, beta: np.ndarray, intercept: float) -> dict[str, float]:
    sse = sse_from_summary(summary, beta, intercept)
    rmse = math.sqrt(sse / float(summary.n))
    sst = summary.sst()
    r2 = float("nan") if sst <= 0.0 else 1.0 - sse / sst
    return {"rmse": float(rmse), "r2": float(r2), "sse": float(sse)}


def mae_from_mask(
    *,
    X: np.memmap,
    y: np.ndarray,
    mask: np.ndarray,
    beta: np.ndarray,
    intercept: float,
    chunk_size: int,
) -> float:
    total_abs = 0.0
    total_n = 0
    beta64 = np.asarray(beta, dtype=np.float64)
    for chunk_start in range(0, X.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, X.shape[0])
        chunk_mask = mask[chunk_start:chunk_end]
        if not np.any(chunk_mask):
            continue
        X_chunk = np.asarray(X[chunk_start:chunk_end][chunk_mask], dtype=np.float64)
        y_chunk = np.asarray(y[chunk_start:chunk_end][chunk_mask], dtype=np.float64)
        preds = X_chunk @ beta64 + intercept
        total_abs += float(np.abs(y_chunk - preds).sum())
        total_n += int(y_chunk.shape[0])
    return float(total_abs / max(total_n, 1))


def summarize_cv_metrics(metric_rows: list[dict[str, float]]) -> dict[str, Any]:
    rmse = np.asarray([row["rmse"] for row in metric_rows], dtype=np.float64)
    mae = np.asarray([row["mae"] for row in metric_rows], dtype=np.float64)
    r2 = np.asarray([row["r2"] for row in metric_rows], dtype=np.float64)
    return {
        "folds": metric_rows,
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse, ddof=1)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae, ddof=1)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2, ddof=1)),
    }


def evaluate_cv_fixed_model(
    *,
    model_builder,
    model_kwargs: dict[str, Any],
    X: np.memmap,
    y: np.ndarray,
    fold_ids: np.ndarray,
    dev_summary: SummaryStats,
    fold_val_summaries: list[SummaryStats],
    chunk_size: int,
) -> tuple[list[dict[str, float]], list[LinearFit]]:
    rows: list[dict[str, float]] = []
    fits: list[LinearFit] = []
    for fold_id, val_summary in enumerate(fold_val_summaries):
        train_summary = dev_summary.subtract(val_summary)
        fit = model_builder(train_summary, **model_kwargs)
        fits.append(fit)
        metric_base = rmse_r2_from_summary(val_summary, fit.beta, fit.intercept)
        mask = fold_ids == fold_id
        mae = mae_from_mask(X=X, y=y, mask=mask, beta=fit.beta, intercept=fit.intercept, chunk_size=chunk_size)
        rows.append({"fold": int(fold_id), "rmse": metric_base["rmse"], "mae": mae, "r2": metric_base["r2"]})
    return rows, fits


def evaluate_test_model(
    *,
    fit: LinearFit,
    X: np.memmap,
    y: np.ndarray,
    test_mask: np.ndarray,
    test_summary: SummaryStats,
    chunk_size: int,
) -> dict[str, float]:
    metric_base = rmse_r2_from_summary(test_summary, fit.beta, fit.intercept)
    mae = mae_from_mask(X=X, y=y, mask=test_mask, beta=fit.beta, intercept=fit.intercept, chunk_size=chunk_size)
    return {"rmse": metric_base["rmse"], "mae": mae, "r2": metric_base["r2"]}


def run_naive_baseline(
    *,
    X: np.memmap,
    y: np.ndarray,
    fold_ids: np.ndarray,
    dev_summary: SummaryStats,
    test_summary: SummaryStats,
    fold_val_summaries: list[SummaryStats],
    test_mask: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    cv_rows: list[dict[str, float]] = []
    zeros = np.zeros(X.shape[1], dtype=np.float64)
    for fold_id, val_summary in enumerate(fold_val_summaries):
        train_summary = dev_summary.subtract(val_summary)
        intercept = train_summary.mean_y()
        metric_base = rmse_r2_from_summary(val_summary, zeros, intercept)
        mae = mae_from_mask(X=X, y=y, mask=(fold_ids == fold_id), beta=zeros, intercept=intercept, chunk_size=chunk_size)
        cv_rows.append({"fold": int(fold_id), "rmse": metric_base["rmse"], "mae": mae, "r2": metric_base["r2"]})
    dev_intercept = dev_summary.mean_y()
    test_metric = rmse_r2_from_summary(test_summary, zeros, dev_intercept)
    test_metric["mae"] = mae_from_mask(X=X, y=y, mask=test_mask, beta=zeros, intercept=dev_intercept, chunk_size=chunk_size)
    return {
        "cv": summarize_cv_metrics(cv_rows),
        "test": {"rmse": test_metric["rmse"], "mae": test_metric["mae"], "r2": test_metric["r2"]},
    }


def select_best_ridge_alpha(
    *,
    alphas: np.ndarray,
    standardized: bool,
    dev_summary: SummaryStats,
    fold_val_summaries: list[SummaryStats],
) -> dict[str, Any]:
    search_rows: list[dict[str, float]] = []
    best_alpha = None
    best_score = None
    for alpha in alphas:
        fold_rmse: list[float] = []
        for val_summary in fold_val_summaries:
            train_summary = dev_summary.subtract(val_summary)
            fit = fit_ridge(train_summary, standardized=standardized, alpha=float(alpha))
            fold_rmse.append(rmse_r2_from_summary(val_summary, fit.beta, fit.intercept)["rmse"])
        mean_rmse = float(np.mean(fold_rmse))
        search_rows.append({"alpha": float(alpha), "rmse_mean": mean_rmse, "rmse_std": float(np.std(fold_rmse, ddof=1))})
        if best_score is None or mean_rmse < best_score:
            best_score = mean_rmse
            best_alpha = float(alpha)
    return {"best_alpha": float(best_alpha), "search": search_rows}


def select_best_lasso_alpha(
    *,
    alphas: np.ndarray,
    standardized: bool,
    dev_summary: SummaryStats,
    fold_val_summaries: list[SummaryStats],
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    train_summaries = [dev_summary.subtract(val_summary) for val_summary in fold_val_summaries]
    warm_starts: list[np.ndarray | None] = [None for _ in train_summaries]
    search_rows: list[dict[str, float]] = []
    best_alpha = None
    best_score = None
    best_converged_alpha = None
    best_converged_score = None
    for alpha in alphas:
        fold_rmse: list[float] = []
        converged_all = True
        max_iterations = 0
        for fold_id, (train_summary, val_summary) in enumerate(zip(train_summaries, fold_val_summaries)):
            fit = fit_lasso(
                train_summary,
                standardized=standardized,
                alpha=float(alpha),
                warm_start=warm_starts[fold_id],
                max_iter=max_iter,
                tol=tol,
            )
            warm_starts[fold_id] = fit.beta.copy()
            fold_rmse.append(rmse_r2_from_summary(val_summary, fit.beta, fit.intercept)["rmse"])
            converged_all = converged_all and bool(fit.converged)
            max_iterations = max(max_iterations, int(fit.iterations or 0))
        mean_rmse = float(np.mean(fold_rmse))
        search_rows.append(
            {
                "alpha": float(alpha),
                "rmse_mean": mean_rmse,
                "rmse_std": float(np.std(fold_rmse, ddof=1)),
                "all_folds_converged": bool(converged_all),
                "max_fold_iterations": int(max_iterations),
            }
        )
        if best_score is None or mean_rmse < best_score:
            best_score = mean_rmse
            best_alpha = float(alpha)
        if converged_all and (best_converged_score is None or mean_rmse < best_converged_score):
            best_converged_score = mean_rmse
            best_converged_alpha = float(alpha)
    selected_alpha = best_converged_alpha if best_converged_alpha is not None else best_alpha
    return {
        "best_alpha": float(selected_alpha),
        "search": search_rows,
        "used_converged_preference": bool(best_converged_alpha is not None),
    }


def render_summary(results: dict[str, Any], meta: dict[str, Any], split_info: dict[str, Any], timings: dict[str, float]) -> str:
    lines = [
        "# Primary Linear Analysis",
        "",
        f"- Rows: `{meta['row_count']:,}`",
        f"- Predictors: `{meta['predictor_count']}`",
        f"- Outer split: `{split_info['dev_size']:,}` development / `{split_info['test_size_rows']:,}` test",
        f"- Outer strata min count: `{split_info['outer_strata_summary']['min_count']}`",
        f"- Inner strata min count: `{split_info['inner_strata_summary']['min_count']}`",
        "",
        "## Test Metrics",
        "",
        "| Model | RMSE | MAE | R^2 | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
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
    for model_key in ordered_keys:
        entry = results[model_key]
        test = entry["test"]
        note = f"alpha={entry['best_alpha']:.6g}" if "best_alpha" in entry else ""
        lines.append(f"| {model_key} | {test['rmse']:.6f} | {test['mae']:.6f} | {test['r2']:.6f} | {note} |")
    lines.extend(["", "## CV RMSE", "", "| Model | Mean | Std | Notes |", "| --- | ---: | ---: | --- |"])
    for model_key in ordered_keys:
        entry = results[model_key]
        cv = entry["cv"]
        note = f"alpha={entry['best_alpha']:.6g}" if "best_alpha" in entry else ""
        lines.append(f"| {model_key} | {cv['rmse_mean']:.6f} | {cv['rmse_std']:.6f} | {note} |")
    lines.extend(["", "## Timings", "", "| Stage | Seconds |", "| --- | ---: |"])
    for key, value in timings.items():
        lines.append(f"| {key} | {value:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_name = args.run_name.strip() or f"{utc_stamp()}_linear_primary_v1"
    cache_root = args.output_root / "cache"
    output_dir = args.output_root / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    log("Preparing cached modeling inputs")
    t0 = time.time()
    meta = prepare_cache(db_path=args.db_path, table_name=args.table_name, cache_root=cache_root, chunk_size=args.chunk_size)
    timings["cache_prepare"] = time.time() - t0

    arrays = load_cached_arrays(meta)
    X = arrays["X"]
    y = arrays["y"]
    patch_codes = arrays["patch_codes"]
    server_codes = arrays["server_codes"]
    mapped_participants = arrays["mapped_participants"]

    log("Building the fixed outer split and inner CV folds")
    t0 = time.time()
    split_bundle = build_splits(
        y=y,
        patch_codes=patch_codes,
        server_codes=server_codes,
        test_size=args.outer_test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    timings["split_build"] = time.time() - t0

    dev_idx = split_bundle["dev_idx"]
    test_idx = split_bundle["test_idx"]
    fold_ids = split_bundle["fold_ids"]
    split_info = split_bundle["split_info"]
    json_dump(output_dir / "split_info.json", split_info)
    np.savez_compressed(output_dir / "split_arrays.npz", dev_idx=dev_idx, test_idx=test_idx, fold_ids=fold_ids)

    split_audit = {
        "mapped_participants_full": {"9": int(np.sum(mapped_participants == 9)), "10": int(np.sum(mapped_participants == 10))},
        "mapped_participants_dev": {"9": int(np.sum(mapped_participants[dev_idx] == 9)), "10": int(np.sum(mapped_participants[dev_idx] == 10))},
        "mapped_participants_test": {"9": int(np.sum(mapped_participants[test_idx] == 9)), "10": int(np.sum(mapped_participants[test_idx] == 10))},
    }
    json_dump(output_dir / "split_audit.json", split_audit)

    log("Accumulating partition summaries")
    t0 = time.time()
    summary_bundle = accumulate_summaries(
        X=X,
        y=y,
        test_idx=test_idx,
        fold_ids=fold_ids,
        n_folds=args.n_folds,
        chunk_size=args.chunk_size,
    )
    timings["summary_accumulation"] = time.time() - t0

    dev_summary: SummaryStats = summary_bundle["dev_summary"]
    test_summary: SummaryStats = summary_bundle["test_summary"]
    fold_val_summaries: list[SummaryStats] = summary_bundle["fold_val_summaries"]
    test_mask: np.ndarray = summary_bundle["is_test"]

    results: dict[str, Any] = {}

    log("Running naive mean baseline")
    t0 = time.time()
    results["naive_mean"] = run_naive_baseline(
        X=X,
        y=y,
        fold_ids=fold_ids,
        dev_summary=dev_summary,
        test_summary=test_summary,
        fold_val_summaries=fold_val_summaries,
        test_mask=test_mask,
        chunk_size=args.chunk_size,
    )
    timings["naive_mean"] = time.time() - t0

    for standardized in (False, True):
        variant_key = "standardized" if standardized else "raw"
        log(f"Running OLS ({variant_key})")
        t0 = time.time()
        cv_rows, _ = evaluate_cv_fixed_model(
            model_builder=fit_ols,
            model_kwargs={"standardized": standardized},
            X=X,
            y=y,
            fold_ids=fold_ids,
            dev_summary=dev_summary,
            fold_val_summaries=fold_val_summaries,
            chunk_size=args.chunk_size,
        )
        full_fit = fit_ols(dev_summary, standardized=standardized)
        test_metrics = evaluate_test_model(fit=full_fit, X=X, y=y, test_mask=test_mask, test_summary=test_summary, chunk_size=args.chunk_size)
        results[f"ols_{variant_key}"] = {
            "cv": summarize_cv_metrics(cv_rows),
            "test": test_metrics,
            "fit": {"intercept": float(full_fit.intercept), "coef_l2_norm": float(np.linalg.norm(full_fit.beta))},
        }
        timings[f"ols_{variant_key}"] = time.time() - t0

    for standardized in (False, True):
        variant_key = "standardized" if standardized else "raw"
        log(f"Searching Ridge alpha ({variant_key})")
        t0 = time.time()
        ridge_grid = ridge_alpha_grid(dev_summary, standardized=standardized, grid_size=args.ridge_grid_size)
        ridge_selection = select_best_ridge_alpha(
            alphas=ridge_grid,
            standardized=standardized,
            dev_summary=dev_summary,
            fold_val_summaries=fold_val_summaries,
        )
        best_alpha = float(ridge_selection["best_alpha"])
        cv_rows, _ = evaluate_cv_fixed_model(
            model_builder=fit_ridge,
            model_kwargs={"standardized": standardized, "alpha": best_alpha},
            X=X,
            y=y,
            fold_ids=fold_ids,
            dev_summary=dev_summary,
            fold_val_summaries=fold_val_summaries,
            chunk_size=args.chunk_size,
        )
        full_fit = fit_ridge(dev_summary, standardized=standardized, alpha=best_alpha)
        test_metrics = evaluate_test_model(fit=full_fit, X=X, y=y, test_mask=test_mask, test_summary=test_summary, chunk_size=args.chunk_size)
        results[f"ridge_{variant_key}"] = {
            "best_alpha": best_alpha,
            "search": ridge_selection["search"],
            "cv": summarize_cv_metrics(cv_rows),
            "test": test_metrics,
            "fit": {"intercept": float(full_fit.intercept), "coef_l2_norm": float(np.linalg.norm(full_fit.beta))},
        }
        timings[f"ridge_{variant_key}"] = time.time() - t0

    for standardized in (False, True):
        variant_key = "standardized" if standardized else "raw"
        log(f"Searching Lasso alpha ({variant_key})")
        t0 = time.time()
        alpha_grid = lasso_alpha_grid(dev_summary, standardized=standardized, grid_size=args.lasso_grid_size, min_ratio=args.lasso_min_ratio)
        selection = select_best_lasso_alpha(
            alphas=alpha_grid,
            standardized=standardized,
            dev_summary=dev_summary,
            fold_val_summaries=fold_val_summaries,
            max_iter=args.lasso_max_iter,
            tol=args.lasso_tol,
        )
        best_alpha = float(selection["best_alpha"])
        cv_rows, cv_fits = evaluate_cv_fixed_model(
            model_builder=fit_lasso,
            model_kwargs={
                "standardized": standardized,
                "alpha": best_alpha,
                "warm_start": None,
                "max_iter": args.lasso_max_iter,
                "tol": args.lasso_tol,
            },
            X=X,
            y=y,
            fold_ids=fold_ids,
            dev_summary=dev_summary,
            fold_val_summaries=fold_val_summaries,
            chunk_size=args.chunk_size,
        )
        full_fit = fit_lasso(
            dev_summary,
            standardized=standardized,
            alpha=best_alpha,
            warm_start=cv_fits[-1].beta if cv_fits else None,
            max_iter=args.lasso_max_iter,
            tol=args.lasso_tol,
        )
        test_metrics = evaluate_test_model(fit=full_fit, X=X, y=y, test_mask=test_mask, test_summary=test_summary, chunk_size=args.chunk_size)
        results[f"lasso_{variant_key}"] = {
            "best_alpha": best_alpha,
            "used_converged_preference": bool(selection.get("used_converged_preference", False)),
            "search": selection["search"],
            "cv": summarize_cv_metrics(cv_rows),
            "test": test_metrics,
            "fit": {
                "intercept": float(full_fit.intercept),
                "coef_l2_norm": float(np.linalg.norm(full_fit.beta)),
                "non_zero_coefficients": int(np.sum(np.abs(full_fit.beta) > 1e-10)),
                "converged": bool(full_fit.converged),
                "iterations": int(full_fit.iterations or 0),
            },
        }
        timings[f"lasso_{variant_key}"] = time.time() - t0

    results_payload = {
        "meta": meta,
        "split_info": split_info,
        "split_audit": split_audit,
        "timings_sec": timings,
        "results": results,
        "runtime_context": {"python_version": os.sys.version, "logical_cpus": int(os.cpu_count() or 0)},
    }
    json_dump(output_dir / "results.json", results_payload)
    (output_dir / "summary.md").write_text(render_summary(results, meta, split_info, timings), encoding="utf-8")
    log(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
