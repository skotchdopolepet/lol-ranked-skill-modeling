from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TARGET = ROOT / "stepwise_interpretation"
DB = PROJECT_ROOT / "runtime" / "out_prod_player" / "secondary_dataset" / "player_secondary_dataset.sqlite3"
VIEW = "player_secondary_dataset_v1"
UPDATED = datetime.now().replace(microsecond=0).isoformat()
NIGHT_HOURS = [1, 2, 3, 4, 5, 6]
DEEP_NIGHT_HOURS = [3, 4, 5]
PALETTE = {
    "blue": "#1f4e79",
    "green": "#0b6e4f",
    "red": "#bc3c4a",
    "gold": "#d28b26",
    "gray": "#7b8794",
    "dark": "#243447",
    "light": "#d8dee6",
}


def assert_inside_root(path: Path) -> None:
    resolved = path.resolve()
    if ROOT not in [resolved, *resolved.parents]:
        raise ValueError(f"Refusing to write outside the modeling folder: {resolved}")


def write_text(path: Path, text: str) -> Path:
    assert_inside_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_df(path: Path, df: pd.DataFrame) -> Path:
    assert_inside_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def f6(value: float | int | None) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return ""
    return f"{v:.6f}"


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return ""
    return f"{100 * v:.{digits}f}%"


def md_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def assign_quartiles(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(s, 4, labels=[1, 2, 3, 4], duplicates="raise").astype("Int64")
    except ValueError:
        return pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype("Int64")


def qcut_edges(values: pd.Series) -> list[float]:
    return [float(x) for x in pd.qcut(pd.to_numeric(values, errors="coerce"), 4, retbins=True, duplicates="raise")[1]]


def cut_with_edges(values: pd.Series, edges: list[float]) -> pd.Series:
    bins = [-np.inf, edges[1], edges[2], edges[3], np.inf]
    return pd.cut(values, bins=bins, labels=[1, 2, 3, 4], include_lowest=True).astype("Int64")


def mean_ci(values: pd.Series) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    return {
        "n": int(n),
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_high": mean + 1.96 * se if np.isfinite(se) else np.nan,
    }


def load_data() -> pd.DataFrame:
    query = f"""
    SELECT
        secondary_row_id,
        match_id,
        source_folder,
        server,
        is_weekend,
        residual,
        local_start_hour,
        night_window,
        underperforming_window,
        night_game_share,
        underperforming_game_share,
        night_share_std,
        underperforming_share_std,
        history_game_count
    FROM {VIEW}
    WHERE residual IS NOT NULL
      AND local_start_hour IS NOT NULL
      AND night_window IS NOT NULL
      AND underperforming_window IS NOT NULL
      AND night_game_share IS NOT NULL
      AND underperforming_game_share IS NOT NULL
      AND server IS NOT NULL
      AND source_folder IS NOT NULL
    """
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(query, con)
    finally:
        con.close()

    numeric = [
        "secondary_row_id",
        "is_weekend",
        "residual",
        "local_start_hour",
        "night_window",
        "underperforming_window",
        "night_game_share",
        "underperforming_game_share",
        "night_share_std",
        "underperforming_share_std",
        "history_game_count",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric + ["server", "source_folder"]).copy()
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["night_window"] = df["night_window"].astype(int)
    df["underperforming_window"] = df["underperforming_window"].astype(int)
    df["hour"] = np.floor(df["local_start_hour"]).astype(int) % 24
    df["hour_str"] = df["hour"].astype(str).str.zfill(2)
    df["weekend_str"] = np.where(df["is_weekend"].eq(1), "weekend", "weekday")
    df["night_window_str"] = np.where(df["night_window"].eq(1), "night 01-06", "other hours")
    df["underperforming_window_str"] = np.where(
        df["underperforming_window"].eq(1),
        "underperf window 02-08",
        "other hours",
    )
    df["source_cloud"] = df["source_folder"].astype(str).str.extract(r"_prod_([^_]+)$", expand=False).fillna("unknown")
    df["server_hour"] = df["server"].astype(str) + "_h" + df["hour_str"]
    df["server_night"] = df["server"].astype(str) + "_" + df["night_window_str"].str.replace(" ", "_", regex=False)
    df["weekend_night"] = df["weekend_str"] + "_" + df["night_window_str"].str.replace(" ", "_", regex=False)
    df["cloud_hour"] = df["source_cloud"].astype(str) + "_h" + df["hour_str"]

    df["night_share_q_all"] = assign_quartiles(df["night_game_share"])
    night_edges = qcut_edges(df.loc[df["night_window"].eq(1), "night_game_share"])
    df["night_share_q_night_threshold"] = cut_with_edges(df["night_game_share"], night_edges)
    df["night_share_q_all_str"] = "Q" + df["night_share_q_all"].astype(str)
    df["night_share_q_night_str"] = "Q" + df["night_share_q_night_threshold"].astype(str)
    df["night_q_hour"] = df["night_share_q_night_str"] + "_h" + df["hour_str"]
    df["all_q_hour"] = df["night_share_q_all_str"] + "_h" + df["hour_str"]
    df["night_share_x_night"] = df["night_game_share"] * df["night_window"]
    df["underperf_share_x_underperf"] = df["underperforming_game_share"] * df["underperforming_window"]
    return df


def gap(values: np.ndarray | pd.Series, left: pd.Series, right: pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    a = v[np.asarray(left, dtype=bool)]
    b = v[np.asarray(right, dtype=bool)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(a) - np.mean(b))


@dataclass(frozen=True)
class ModelSpec:
    stage: str
    name: str
    label: str
    cat_cols: tuple[str, ...] = ()
    num_cols: tuple[str, ...] = ()


SINGLE_SPECS = [
    ModelSpec("single", "mean_only", "Mean only"),
    ModelSpec("single", "server", "Server", cat_cols=("server",)),
    ModelSpec("single", "weekend", "Weekend flag", cat_cols=("weekend_str",)),
    ModelSpec("single", "night_window", "Night window flag", cat_cols=("night_window_str",)),
    ModelSpec("single", "underperforming_window", "02:00-08:30 window flag", cat_cols=("underperforming_window_str",)),
    ModelSpec("single", "hour", "Local hour", cat_cols=("hour_str",)),
    ModelSpec("single", "night_share_linear", "Historical night share", num_cols=("night_game_share",)),
    ModelSpec("single", "night_share_q_all", "Historical night share quartile", cat_cols=("night_share_q_all_str",)),
    ModelSpec(
        "single",
        "night_share_q_night_threshold",
        "Night-derived historical night-share quartile",
        cat_cols=("night_share_q_night_str",),
    ),
    ModelSpec("single", "source_cloud", "Source cloud/batch suffix", cat_cols=("source_cloud",)),
    ModelSpec("single", "source_folder", "Source folder", cat_cols=("source_folder",)),
]

ADDITIVE_SPECS = [
    ModelSpec("additive", "server", "Server", cat_cols=("server",)),
    ModelSpec("additive", "server_weekend", "Server + weekend", cat_cols=("server", "weekend_str")),
    ModelSpec("additive", "server_weekend_night", "Server + weekend + night flag", cat_cols=("server", "weekend_str", "night_window_str")),
    ModelSpec("additive", "server_weekend_hour", "Server + weekend + hour", cat_cols=("server", "weekend_str", "hour_str")),
    ModelSpec(
        "additive",
        "server_weekend_hour_night",
        "Server + weekend + hour + night flag",
        cat_cols=("server", "weekend_str", "hour_str", "night_window_str"),
    ),
    ModelSpec(
        "additive",
        "server_weekend_hour_night_share",
        "Server + weekend + hour + night-share quartile",
        cat_cols=("server", "weekend_str", "hour_str", "night_share_q_night_str"),
    ),
    ModelSpec(
        "additive",
        "server_weekend_hour_night_share_linear",
        "Server + weekend + hour + linear night share",
        cat_cols=("server", "weekend_str", "hour_str"),
        num_cols=("night_game_share", "night_share_x_night"),
    ),
    ModelSpec(
        "additive",
        "server_weekend_hour_source_cloud_night_share",
        "Server + weekend + hour + source cloud + night-share quartile",
        cat_cols=("server", "weekend_str", "hour_str", "source_cloud", "night_share_q_night_str"),
    ),
]

INTERACTION_SPECS = [
    ModelSpec("interaction", "server_hour", "Server x hour + weekend", cat_cols=("server_hour", "weekend_str")),
    ModelSpec("interaction", "server_night", "Server x night + weekend + hour", cat_cols=("server_night", "weekend_str", "hour_str")),
    ModelSpec("interaction", "weekend_night", "Weekend x night + server + hour", cat_cols=("weekend_night", "server", "hour_str")),
    ModelSpec(
        "interaction",
        "night_share_q_hour",
        "Night-share quartile x hour + server + weekend",
        cat_cols=("server", "weekend_str", "night_q_hour"),
    ),
    ModelSpec(
        "interaction",
        "server_hour_night_share_q",
        "Server x hour + night-share quartile",
        cat_cols=("server_hour", "weekend_str", "night_share_q_night_str"),
    ),
    ModelSpec(
        "interaction",
        "server_hour_night_q_hour",
        "Server x hour + night-share quartile x hour",
        cat_cols=("server_hour", "weekend_str", "night_q_hour"),
    ),
    ModelSpec(
        "interaction",
        "source_folder_hour_night_q",
        "Source folder + hour + night-share quartile",
        cat_cols=("source_folder", "hour_str", "night_share_q_night_str"),
    ),
]

VALIDATION_MODEL_NAMES = [
    "mean_only",
    "server",
    "source_cloud",
    "source_folder",
    "server_weekend_hour",
    "server_weekend_hour_night_share_linear",
    "night_share_q_hour",
    "server_hour_night_q_hour",
]


class DesignBuilder:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.ohe: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None

    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return self._build(df, fit=True)

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return self._build(df, fit=False)

    def _build(self, df: pd.DataFrame, *, fit: bool) -> sparse.csr_matrix:
        blocks: list[sparse.spmatrix] = []
        n = len(df)
        if self.spec.cat_cols:
            cats = df.loc[:, list(self.spec.cat_cols)].astype(str).fillna("missing")
            if fit:
                self.ohe = safe_ohe()
                blocks.append(self.ohe.fit_transform(cats))
            else:
                if self.ohe is None:
                    raise RuntimeError("Encoder was not fit")
                blocks.append(self.ohe.transform(cats))
        if self.spec.num_cols:
            arr = df.loc[:, list(self.spec.num_cols)].astype(float).to_numpy()
            if fit:
                self.scaler = StandardScaler()
                arr = self.scaler.fit_transform(arr)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler was not fit")
                arr = self.scaler.transform(arr)
            blocks.append(sparse.csr_matrix(arr))
        if not blocks:
            return sparse.csr_matrix((n, 0))
        return sparse.hstack(blocks, format="csr")


def fit_predict(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, int]:
    y_train = train["residual"].to_numpy(float)
    if spec.name == "mean_only":
        return np.full(len(test), float(np.mean(y_train))), 1
    builder = DesignBuilder(spec)
    x_train = builder.fit_transform(train)
    x_test = builder.transform(test)
    model = Ridge(alpha=10.0, solver="lsqr", random_state=42)
    model.fit(x_train, y_train)
    return model.predict(x_test), int(x_train.shape[1])


def metric_row(spec: ModelSpec, test: pd.DataFrame, yhat: np.ndarray, feature_count: int) -> dict:
    y = test["residual"].to_numpy(float)
    after = y - yhat
    night = test["night_window"].eq(1)
    non_night = test["night_window"].eq(0)
    under = test["underperforming_window"].eq(1)
    not_under = test["underperforming_window"].eq(0)
    deep = test["hour"].isin(DEEP_NIGHT_HOURS)
    night_q1 = night & test["night_share_q_night_threshold"].eq(1)
    night_q4 = night & test["night_share_q_night_threshold"].eq(4)
    deep_q1 = deep & test["night_share_q_night_threshold"].eq(1)
    deep_q4 = deep & test["night_share_q_night_threshold"].eq(4)
    raw_deep = gap(y, deep_q1, deep_q4)
    left_deep = gap(after, deep_q1, deep_q4)
    explained = np.nan
    if np.isfinite(raw_deep) and abs(raw_deep) > 1e-12 and np.isfinite(left_deep):
        explained = 1.0 - abs(left_deep) / abs(raw_deep)
    return {
        "stage": spec.stage,
        "model": spec.name,
        "label": spec.label,
        "feature_count": feature_count,
        "rmse": math.sqrt(mean_squared_error(y, yhat)),
        "r2": r2_score(y, yhat),
        "raw_night_gap": gap(y, night, non_night),
        "predicted_night_gap": gap(yhat, night, non_night),
        "remaining_night_gap": gap(after, night, non_night),
        "raw_underperforming_gap": gap(y, under, not_under),
        "remaining_underperforming_gap": gap(after, under, not_under),
        "raw_night_q1_minus_q4": gap(y, night_q1, night_q4),
        "remaining_night_q1_minus_q4": gap(after, night_q1, night_q4),
        "raw_deep_q1_minus_q4": raw_deep,
        "remaining_deep_q1_minus_q4": left_deep,
        "deep_gap_explained_abs": explained,
        "test_n": int(len(test)),
        "night_n": int(night.sum()),
        "deep_q1_n": int(deep_q1.sum()),
        "deep_q4_n": int(deep_q4.sum()),
    }


def run_models(df: pd.DataFrame) -> pd.DataFrame:
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=df["night_window"])
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    specs = SINGLE_SPECS + ADDITIVE_SPECS + INTERACTION_SPECS
    rows = []
    for spec in specs:
        yhat, feature_count = fit_predict(spec, train, test)
        rows.append(metric_row(spec, test, yhat, feature_count))
    out = pd.DataFrame(rows)
    baseline = float(out.loc[out["model"].eq("mean_only"), "rmse"].iloc[0])
    out["rmse_improvement_vs_mean"] = baseline - out["rmse"]
    out["rmse_improvement_pct"] = out["rmse_improvement_vs_mean"] / baseline
    return out


def build_validation_splits(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame]]:
    splits: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=df["night_window"])
    splits.append(("random_25pct", "random", df.iloc[train_idx].copy(), df.iloc[test_idx].copy()))

    ordered = df.sort_values("secondary_row_id").copy()
    cut = int(len(ordered) * 0.75)
    splits.append(("row_order_last_25pct", "row_order_proxy", ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_i, test_i = next(gss.split(df, groups=df["source_folder"]))
    splits.append(("source_folder_group_25pct", "source_folder_group", df.iloc[train_i].copy(), df.iloc[test_i].copy()))
    return splits


def run_selected_validations(df: pd.DataFrame) -> pd.DataFrame:
    specs_by_name = {spec.name: spec for spec in SINGLE_SPECS + ADDITIVE_SPECS + INTERACTION_SPECS}
    specs = [specs_by_name[name] for name in VALIDATION_MODEL_NAMES]
    rows = []
    for split_name, split_type, train, test in build_validation_splits(df):
        for spec in specs:
            yhat, feature_count = fit_predict(spec, train, test)
            row = metric_row(spec, test, yhat, feature_count)
            row["split"] = split_name
            row["split_type"] = split_type
            row["train_n"] = int(len(train))
            row["test_source_folders"] = int(test["source_folder"].nunique())
            row["test_servers"] = int(test["server"].nunique())
            rows.append(row)
    out = pd.DataFrame(rows)
    for split_name, g in out.groupby("split"):
        baseline = float(g.loc[g["model"].eq("mean_only"), "rmse"].iloc[0])
        idx = out["split"].eq(split_name)
        out.loc[idx, "rmse_improvement_vs_split_mean"] = baseline - out.loc[idx, "rmse"]
        out.loc[idx, "rmse_improvement_pct_vs_split_mean"] = out.loc[idx, "rmse_improvement_vs_split_mean"] / baseline
    return out


def summarize_cloud_increment(score: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("single_source_cloud_vs_server", "single", "server", "single", "source_cloud"),
        ("single_source_folder_vs_server", "single", "server", "single", "source_folder"),
        (
            "cloud_added_to_additive_context",
            "additive",
            "server_weekend_hour_night_share",
            "additive",
            "server_weekend_hour_source_cloud_night_share",
        ),
    ]
    rows = []
    for name, base_stage, base_model, cloud_stage, cloud_model in pairs:
        base = score[score["stage"].eq(base_stage) & score["model"].eq(base_model)].iloc[0]
        cloud = score[score["stage"].eq(cloud_stage) & score["model"].eq(cloud_model)].iloc[0]
        rows.append(
            {
                "comparison": name,
                "base_model": base["label"],
                "test_model": cloud["label"],
                "base_r2": float(base["r2"]),
                "test_r2": float(cloud["r2"]),
                "delta_r2": float(cloud["r2"] - base["r2"]),
                "base_rmse_lift_pct": float(base["rmse_improvement_pct"]),
                "test_rmse_lift_pct": float(cloud["rmse_improvement_pct"]),
                "delta_rmse_lift_pct": float(cloud["rmse_improvement_pct"] - base["rmse_improvement_pct"]),
                "base_remaining_deep_q1_q4": float(base["remaining_deep_q1_minus_q4"]),
                "test_remaining_deep_q1_q4": float(cloud["remaining_deep_q1_minus_q4"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_variable_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "variable": "server",
            "type": "categorical context",
            "levels": int(df["server"].nunique()),
            "why_test": "Servers have different residual baselines and local-hour shapes.",
            "interpretation_caution": "Useful for calibration and localization; not a causal gameplay mechanism.",
        },
        {
            "variable": "is_weekend",
            "type": "binary context",
            "levels": 2,
            "why_test": "Weekend can change player mix and play schedule.",
            "interpretation_caution": "Expected to be a modifier or composition check, not a large standalone effect.",
        },
        {
            "variable": "night_window",
            "type": "binary timing flag",
            "levels": 2,
            "why_test": "Primary question: 01:00-07:00 local-time match starts.",
            "interpretation_caution": "Raw gap mixes server, hour, source, and player-history composition.",
        },
        {
            "variable": "underperforming_window",
            "type": "binary timing flag",
            "levels": 2,
            "why_test": "Alternative 02:00-08:30 window from earlier diagnostics.",
            "interpretation_caution": "Should not be used as the headline if it only re-labels the same hour pattern.",
        },
        {
            "variable": "hour",
            "type": "24-level categorical timing",
            "levels": int(df["hour"].nunique()),
            "why_test": "Checks whether the clock shape is broader than a single night flag.",
            "interpretation_caution": "Hour effects are descriptive local-time profiles.",
        },
        {
            "variable": "night_game_share",
            "type": "continuous player-history mix",
            "levels": "continuous",
            "why_test": "Measures how historically night-heavy the eligible players in a match are.",
            "interpretation_caution": "This is roster composition, not a randomized player trait.",
        },
        {
            "variable": "source_cloud",
            "type": "8-level operational/source batch",
            "levels": int(df["source_cloud"].nunique()),
            "why_test": "Tests the user's 'cloud/source' concern without jumping to full source-folder fixed effects.",
            "interpretation_caution": "Operational artifact; if strong, treat as QA/calibration evidence, not theory.",
        },
        {
            "variable": "source_folder",
            "type": "32-level operational/source folder",
            "levels": int(df["source_folder"].nunique()),
            "why_test": "Hard diagnostic for source-specific calibration or crawl-batch structure.",
            "interpretation_caution": "Highly confounded with server and collection process; useful as sensitivity only.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_group_means(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    for key, g in df.groupby(group_col, sort=True, dropna=False):
        stats = mean_ci(g["residual"])
        rows.append(
            {
                group_col: key,
                "n": int(len(g)),
                "share": len(g) / total,
                "mean_residual": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "night_row_share": float(g["night_window"].mean()),
                "weekend_share": float(g["is_weekend"].mean()),
                "mean_night_game_share": float(g["night_game_share"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_descriptives(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for col in ["server", "weekend_str", "night_window_str", "underperforming_window_str", "source_cloud", "source_folder"]:
        tables[col] = summarize_group_means(df, col)
    hour_rows = []
    for hour, g in df.groupby("hour", sort=True):
        stats = mean_ci(g["residual"])
        hour_rows.append(
            {
                "hour": int(hour),
                "n": int(len(g)),
                "mean_residual": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "night_game_share": float(g["night_game_share"].mean()),
                "weekend_share": float(g["is_weekend"].mean()),
            }
        )
    tables["hour"] = pd.DataFrame(hour_rows)

    q_rows = []
    for (window, q), g in df.groupby(["night_window_str", "night_share_q_night_threshold"], observed=True, sort=True):
        stats = mean_ci(g["residual"])
        q_rows.append(
            {
                "window": window,
                "night_share_q_night_threshold": int(q),
                "n": int(len(g)),
                "mean_residual": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "mean_night_game_share": float(g["night_game_share"].mean()),
            }
        )
    tables["night_share_by_window"] = pd.DataFrame(q_rows)
    return tables


def plot_single_variable_scoreboard(score: pd.DataFrame) -> Path:
    path = TARGET / "plot_01_single_variable_scoreboard.png"
    plot = score[score["stage"].eq("single")].copy()
    plot = plot.sort_values("rmse_improvement_pct", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = [PALETTE["gray"] if m == "mean_only" else PALETTE["blue"] for m in plot["model"]]
    ax.barh(y, plot["rmse_improvement_pct"] * 100, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Random-holdout RMSE improvement vs mean-only (%)")
    ax.set_title("Step zero: test each variable by itself")
    for i, r in enumerate(plot.itertuples()):
        ax.text(r.rmse_improvement_pct * 100 + 0.02, i, f"R2={r.r2:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_additive_ladder(score: pd.DataFrame) -> Path:
    path = TARGET / "plot_02_additive_ladder.png"
    plot = score[score["stage"].isin(["single", "additive"])].copy()
    keep = [
        "mean_only",
        "server",
        "weekend",
        "night_window",
        "hour",
        "night_share_q_night_threshold",
        "source_cloud",
        "server_weekend",
        "server_weekend_night",
        "server_weekend_hour",
        "server_weekend_hour_night_share_linear",
        "server_weekend_hour_night_share",
        "server_weekend_hour_source_cloud_night_share",
    ]
    plot = plot[plot["model"].isin(keep)].copy()
    order = {name: i for i, name in enumerate(keep)}
    plot["order"] = plot["model"].map(order)
    plot = plot.sort_values("order")
    x = np.arange(len(plot))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(x, plot["remaining_night_gap"], marker="o", color=PALETTE["blue"], label="Remaining night gap")
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_ylabel("Night - non-night residual")
    axes[0].set_title("Additive ladder: what remains after simple blocks")
    axes[0].legend(loc="best")
    axes[1].plot(x, plot["remaining_deep_q1_minus_q4"], marker="o", color=PALETTE["red"], label="Remaining deep Q1-Q4")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_ylabel("Deep-night Q1 - Q4")
    axes[1].legend(loc="best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot["label"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_interaction_ladder(score: pd.DataFrame) -> Path:
    path = TARGET / "plot_03_interaction_ladder.png"
    plot = score[score["stage"].isin(["additive", "interaction"])].copy()
    keep = [
        "server_weekend_hour_night_share",
        "server_hour",
        "server_night",
        "weekend_night",
        "night_share_q_hour",
        "server_hour_night_share_q",
        "server_hour_night_q_hour",
        "source_folder_hour_night_q",
    ]
    plot = plot[plot["model"].isin(keep)].copy()
    order = {name: i for i, name in enumerate(keep)}
    plot["order"] = plot["model"].map(order)
    plot = plot.sort_values("order")
    x = np.arange(len(plot))
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(x, plot["rmse_improvement_pct"] * 100, color=PALETTE["green"], alpha=0.75, label="RMSE lift")
    ax1.set_ylabel("RMSE lift vs mean-only (%)")
    ax2 = ax1.twinx()
    ax2.plot(x, plot["remaining_deep_q1_minus_q4"].abs(), color=PALETTE["red"], marker="o", label="Abs remaining deep Q1-Q4")
    ax2.set_ylabel("Abs remaining deep-night Q1-Q4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot["label"], rotation=35, ha="right")
    ax1.set_title("Only after additive checks: small named interactions")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_basic_context_means(tables: dict[str, pd.DataFrame]) -> Path:
    path = TARGET / "plot_04_basic_context_means.png"
    server = tables["server"].sort_values("mean_residual")
    cloud = tables["source_cloud"].sort_values("mean_residual")
    binary = pd.concat(
        [
            tables["weekend_str"].rename(columns={"weekend_str": "group"}).assign(variable="weekend"),
            tables["night_window_str"].rename(columns={"night_window_str": "group"}).assign(variable="night flag"),
            tables["underperforming_window_str"].rename(columns={"underperforming_window_str": "group"}).assign(variable="02-08:30 flag"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].barh(server["server"], server["mean_residual"], color=PALETTE["blue"])
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Server means")
    axes[0].set_xlabel("Mean residual")
    axes[1].barh(cloud["source_cloud"], cloud["mean_residual"], color=PALETTE["gold"])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Source cloud/batch means")
    axes[1].set_xlabel("Mean residual")
    labels = binary["variable"] + ": " + binary["group"]
    axes[2].barh(labels, binary["mean_residual"], color=PALETTE["green"])
    axes[2].axvline(0, color="#333333", linewidth=1)
    axes[2].set_title("Binary timing/context means")
    axes[2].set_xlabel("Mean residual")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hour_q_profile(df: pd.DataFrame) -> Path:
    path = TARGET / "plot_05_hour_by_night_share_quartile.png"
    prof = (
        df.groupby(["hour", "night_share_q_night_threshold"], observed=True)["residual"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {1: PALETTE["red"], 2: PALETTE["gold"], 3: PALETTE["blue"], 4: PALETTE["green"]}
    for q, g in prof.groupby("night_share_q_night_threshold", sort=True):
        ax.plot(g["hour"], g["mean"], marker="o", linewidth=1.8, color=colors[int(q)], label=f"Q{int(q)}")
    ax.axhline(0, color="#333333", linewidth=1)
    for h in NIGHT_HOURS:
        ax.axvspan(h - 0.5, h + 0.5, color="#efefef", alpha=0.35, linewidth=0)
    ax.set_xticks(range(24))
    ax.set_xlabel("Local start hour")
    ax.set_ylabel("Mean residual")
    ax.set_title("Simple profile: historical night-share groups by local hour")
    ax.legend(title="Night-derived share quartile")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_validation_summary(validation: pd.DataFrame) -> Path:
    path = TARGET / "plot_06_selected_validation.png"
    keep = [
        "mean_only",
        "server",
        "server_weekend_hour_night_share_linear",
        "night_share_q_hour",
        "server_hour_night_q_hour",
        "source_folder",
    ]
    splits = ["random_25pct", "row_order_last_25pct", "source_folder_group_25pct"]
    plot = validation[validation["model"].isin(keep)].copy()
    order = {m: i for i, m in enumerate(keep)}
    plot["model_order"] = plot["model"].map(order)
    plot = plot.sort_values(["model_order", "split"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=False)
    for _, g in plot.groupby("model_order", sort=True):
        g = g.set_index("split").reindex(splits).reset_index()
        label = g["label"].dropna().iloc[0]
        axes[0].plot(range(len(splits)), g["r2"], marker="o", label=label)
        axes[1].plot(range(len(splits)), g["remaining_deep_q1_minus_q4"], marker="o", label=label)
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_ylabel("R2")
    axes[0].set_title("Selected validation: fit stability")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_ylabel("Remaining deep-night Q1-Q4")
    axes[1].set_title("Selected validation: remaining key gap")
    for ax in axes:
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(["random", "row-order", "source-folder"], rotation=25, ha="right")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def port_useful_existing_plots(paths: list[Path]) -> list[Path]:
    copied: list[Path] = []
    out_dir = TARGET / "ported_useful_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if src.exists():
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def write_manifest(paths: list[Path]) -> Path:
    rows = []
    for path in paths:
        rows.append(
            {
                "filename": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return write_df(TARGET / "manifest.csv", pd.DataFrame(rows))


def report_rows(score: pd.DataFrame, models: list[str], *, stage: str | None = None) -> list[dict]:
    rows = []
    subset = score[score["model"].isin(models)].copy()
    if stage is not None:
        subset = subset[subset["stage"].eq(stage)].copy()
    order = {m: i for i, m in enumerate(models)}
    subset["order"] = subset["model"].map(order)
    for _, r in subset.sort_values("order").iterrows():
        rows.append(
            {
                "model": r["label"],
                "features": int(r["feature_count"]),
                "R2": f6(r["r2"]),
                "RMSE lift": pct(r["rmse_improvement_pct"], 2),
                "left night gap": f6(r["remaining_night_gap"]),
                "left deep Q1-Q4": f6(r["remaining_deep_q1_minus_q4"]),
            }
        )
    return rows


def build_validation_appendix(validation: pd.DataFrame, cloud_increment: pd.DataFrame) -> str:
    keep_validation = validation[
        validation["model"].isin(
            [
                "server",
                "source_folder",
                "server_weekend_hour_night_share_linear",
                "night_share_q_hour",
                "server_hour_night_q_hour",
            ]
        )
    ].copy()
    validation_rows = []
    for _, r in keep_validation.sort_values(["split", "model"]).iterrows():
        validation_rows.append(
            {
                "split": r["split"],
                "model": r["label"],
                "R2": f6(r["r2"]),
                "RMSE lift": pct(r["rmse_improvement_pct_vs_split_mean"], 2),
                "left deep Q1-Q4": f6(r["remaining_deep_q1_minus_q4"]),
            }
        )

    cloud_rows = []
    for _, r in cloud_increment.iterrows():
        cloud_rows.append(
            {
                "comparison": r["comparison"],
                "base": r["base_model"],
                "test": r["test_model"],
                "delta R2": f6(r["delta_r2"]),
                "delta RMSE lift": pct(r["delta_rmse_lift_pct"], 3),
            }
        )

    return "\n".join(
        [
            "# Stepwise Validation Appendix",
            "",
            f"Updated: {UPDATED}",
            "",
            "## Selected Validation",
            "",
            "The selected models were rerun on three splits: random 25%, last 25% by `secondary_row_id`, and held-out source folders. This is not a new modeling branch; it checks whether the simple interpretation survives harder slices.",
            "",
            md_table(validation_rows, ["split", "model", "R2", "RMSE lift", "left deep Q1-Q4"]),
            "",
            "Interpretation: source-folder fixed effects are useful only when the same folders appear in train and test; they fail as an explanation when source folders are held out. Server remains the interpretable calibration layer. The exact night-share-by-hour correction is useful on random and row-order splits but less portable across held-out source folders, so use it as an active-dataset explanation rather than a fully transportable law.",
            "",
            "## Source Cloud Check",
            "",
            md_table(cloud_rows, ["comparison", "base", "test", "delta R2", "delta RMSE lift"]),
            "",
            "Interpretation: source folder is nearly tied with server as a standalone fit term because source folders are operationally tied to servers. Source cloud/batch suffix has visible standalone signal, but adding it to the already interpretable server/weekend/hour/night-share block barely changes fit. Treat it as QA/calibration context, not as the mechanism.",
        ]
    )


def build_report(df: pd.DataFrame, score: pd.DataFrame) -> str:
    single_top = (
        score[score["stage"].eq("single") & ~score["model"].eq("mean_only")]
        .sort_values("rmse_improvement_pct", ascending=False)
        .head(5)
    )
    additive_keep = [
        "server",
        "server_weekend",
        "server_weekend_night",
        "server_weekend_hour",
        "server_weekend_hour_night_share_linear",
        "server_weekend_hour_night_share",
        "server_weekend_hour_source_cloud_night_share",
    ]
    interaction_keep = [
        "server_weekend_hour_night_share",
        "server_hour",
        "server_night",
        "weekend_night",
        "night_share_q_hour",
        "server_hour_night_share_q",
        "server_hour_night_q_hour",
        "source_folder_hour_night_q",
    ]
    raw = score[score["model"].eq("mean_only")].iloc[0]
    best_add = score[score["stage"].eq("additive")].sort_values("remaining_deep_q1_minus_q4", key=lambda x: x.abs()).iloc[0]
    best_inter = score[score["stage"].eq("interaction")].sort_values("remaining_deep_q1_minus_q4", key=lambda x: x.abs()).iloc[0]

    top_single_rows = [
        {
            "variable": r["label"],
            "R2": f6(r["r2"]),
            "RMSE lift": pct(r["rmse_improvement_pct"], 2),
            "left night gap": f6(r["remaining_night_gap"]),
            "left deep Q1-Q4": f6(r["remaining_deep_q1_minus_q4"]),
        }
        for _, r in single_top.iterrows()
    ]

    return "\n".join(
        [
            "# Stepwise Residual Interpretation Reset",
            "",
            f"Updated: {UPDATED}",
            "",
            "## Goal",
            "",
            "This folder restarts the secondary residual analysis from step zero. The target is still:",
            "",
            "`residual = catboost_pred - actual_skill`",
            "",
            "Negative residuals mean the CatBoost model underpredicted the actual match quality. The goal here is interpretation, not maximum model complexity: test each variable separately, then additive blocks, then only small named interactions.",
            "",
            "## Data",
            "",
            f"- Rows: `{len(df):,}`.",
            f"- Servers: `{df['server'].nunique()}`.",
            f"- Source folders: `{df['source_folder'].nunique()}`.",
            f"- Source cloud/batch suffixes: `{df['source_cloud'].nunique()}`.",
            f"- Weekend share: `{pct(df['is_weekend'].mean(), 1)}`.",
            f"- Night-window share: `{pct(df['night_window'].mean(), 1)}`.",
            "",
            "## Step 0: Single Variables",
            "",
            md_table(top_single_rows, ["variable", "R2", "RMSE lift", "left night gap", "left deep Q1-Q4"]),
            "",
            "The single-variable check says `source_folder` has the highest raw holdout score, but the strongest interpretable simple variable is `server`. Source folder/cloud signal is operational/source diagnostic evidence and is confounded with server and collection batches. Weekend and the night flag are weak by themselves.",
            "",
            "## Additive Ladder",
            "",
            md_table(
                [
                    *report_rows(score, ["mean_only"], stage="single"),
                    *report_rows(score, additive_keep, stage="additive"),
                ],
                ["model", "features", "R2", "RMSE lift", "left night gap", "left deep Q1-Q4"],
            ),
            "",
            f"Raw holdout night gap starts at `{f6(raw['remaining_night_gap'])}` and raw deep-night Q1-Q4 starts at `{f6(raw['remaining_deep_q1_minus_q4'])}`. The best additive block here is `{best_add['label']}`, leaving deep Q1-Q4 `{f6(best_add['remaining_deep_q1_minus_q4'])}`.",
            "",
            "## Interaction Ladder",
            "",
            md_table(
                [
                    *report_rows(score, ["server_weekend_hour_night_share"], stage="additive"),
                    *report_rows(score, [m for m in interaction_keep if m != "server_weekend_hour_night_share"], stage="interaction"),
                ],
                ["model", "features", "R2", "RMSE lift", "left night gap", "left deep Q1-Q4"],
            ),
            "",
            f"The best limited interaction block is `{best_inter['label']}`, leaving deep Q1-Q4 `{f6(best_inter['remaining_deep_q1_minus_q4'])}`. This confirms that interactions should be used only after the simpler result is visible: server/hour structure gives broad calibration, and night-share-by-hour explains the specific deep-night Q1-Q4 shape.",
            "",
            "## Interpretation To Keep",
            "",
            "1. Server is the first useful variable. It explains broad residual baselines and should be treated as calibration/context, not a causal mechanism.",
            "2. Weekend is not a strong standalone driver. Keep it as a simple adjustment and moderation check.",
            "3. The night flag is descriptive but too blunt. The useful timing variable is local hour, especially when paired with historical night-share composition.",
            "4. Historical night-share is the most interpretable player-history variable. Rare-night groups look different, but the population-level story remains composition/context first.",
            "5. Source cloud/folder has enough signal to audit, but it is an operational artifact. Use it to check robustness, not as the thesis explanation.",
            "6. Only the final step should use interactions, and the only interaction worth keeping now is local hour by night-share composition, optionally after server-hour calibration.",
            "",
            "## Useful Plots Ported Forward",
            "",
            "The `ported_useful_plots/` subfolder contains only the existing plots that still fit this interpretation-first reset: the descriptive context ladder, the simple current-match hour profile, and the previous model gap decomposition for comparison.",
            "",
            "## Output Index",
            "",
            "| file | purpose |",
            "| --- | --- |",
            "| table_01_variable_inventory.csv | variables, types, and interpretation cautions |",
            "| table_02_single_variable_scoreboard.csv | one-variable holdout tests |",
            "| table_03_additive_ladder.csv | simple additive model ladder |",
            "| table_04_interaction_ladder.csv | limited named interaction checks |",
            "| plot_01_single_variable_scoreboard.png | simple variable ranking |",
            "| plot_02_additive_ladder.png | remaining night/deep-Q gap after additive blocks |",
            "| plot_03_interaction_ladder.png | fit/gap tradeoff for interactions |",
            "| plot_04_basic_context_means.png | raw means for server, source cloud, and binary flags |",
            "| plot_05_hour_by_night_share_quartile.png | simple hour profile by historical night-share group |",
            "| plot_06_selected_validation.png | selected-model validation across random, row-order, and source-folder splits |",
            "| table_10_selected_validation.csv | selected validation metrics |",
            "| table_11_cloud_increment.csv | source/cloud increment checks |",
            "| stepwise_validation_appendix.md | validation and source/cloud interpretation appendix |",
        ]
    )


def main() -> None:
    set_plot_style()
    TARGET.mkdir(parents=True, exist_ok=True)
    df = load_data()
    score = run_models(df)
    validation = run_selected_validations(df)
    cloud_increment = summarize_cloud_increment(score)
    tables = summarize_descriptives(df)

    paths: list[Path] = []
    paths.append(write_df(TARGET / "table_01_variable_inventory.csv", summarize_variable_inventory(df)))
    paths.append(write_df(TARGET / "table_02_single_variable_scoreboard.csv", score[score["stage"].eq("single")].sort_values("rmse")))
    paths.append(write_df(TARGET / "table_03_additive_ladder.csv", score[score["stage"].eq("additive")].sort_values("rmse")))
    paths.append(write_df(TARGET / "table_04_interaction_ladder.csv", score[score["stage"].eq("interaction")].sort_values("rmse")))
    paths.append(write_df(TARGET / "table_05_server_means.csv", tables["server"].sort_values("mean_residual")))
    paths.append(write_df(TARGET / "table_06_source_cloud_means.csv", tables["source_cloud"].sort_values("mean_residual")))
    paths.append(write_df(TARGET / "table_07_hour_profile.csv", tables["hour"]))
    paths.append(write_df(TARGET / "table_08_night_share_by_window.csv", tables["night_share_by_window"]))
    paths.append(write_df(TARGET / "table_09_source_folder_means.csv", tables["source_folder"].sort_values("mean_residual")))
    paths.append(write_df(TARGET / "table_10_selected_validation.csv", validation.sort_values(["split", "stage", "model"])))
    paths.append(write_df(TARGET / "table_11_cloud_increment.csv", cloud_increment))

    paths.append(plot_single_variable_scoreboard(score))
    paths.append(plot_additive_ladder(score))
    paths.append(plot_interaction_ladder(score))
    paths.append(plot_basic_context_means(tables))
    paths.append(plot_hour_q_profile(df))
    paths.append(plot_validation_summary(validation))

    ported = port_useful_existing_plots(
        [
            ROOT / "current_match_only" / "plot_01_hour_residual_profile.png",
            ROOT / "current_match_only" / "plot_07_context_ladder.png",
            ROOT / "current_match_only" / "modeling" / "plot_02_gap_decomposition.png",
            ROOT / "current_match_only" / "modeling" / "validation" / "plot_06_repeated_source_folder_validation.png",
        ]
    )
    paths.extend(ported)

    summary = {
        "updated": UPDATED,
        "rows": int(len(df)),
        "servers": int(df["server"].nunique()),
        "source_folders": int(df["source_folder"].nunique()),
        "source_clouds": int(df["source_cloud"].nunique()),
        "weekend_share": float(df["is_weekend"].mean()),
        "night_window_share": float(df["night_window"].mean()),
        "best_single": score[score["stage"].eq("single") & ~score["model"].eq("mean_only")]
        .sort_values("rmse_improvement_pct", ascending=False)
        .iloc[0][["model", "label", "r2", "rmse_improvement_pct"]]
        .to_dict(),
        "best_additive_deep_gap": score[score["stage"].eq("additive")]
        .sort_values("remaining_deep_q1_minus_q4", key=lambda x: x.abs())
        .iloc[0][["model", "label", "r2", "remaining_night_gap", "remaining_deep_q1_minus_q4"]]
        .to_dict(),
        "best_interaction_deep_gap": score[score["stage"].eq("interaction")]
        .sort_values("remaining_deep_q1_minus_q4", key=lambda x: x.abs())
        .iloc[0][["model", "label", "r2", "remaining_night_gap", "remaining_deep_q1_minus_q4"]]
        .to_dict(),
        "validation_splits": sorted(validation["split"].unique().tolist()),
        "cloud_increment_delta_r2": cloud_increment.set_index("comparison")["delta_r2"].to_dict(),
    }
    summary_path = TARGET / "stepwise_interpretation_summary.json"
    write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    paths.append(summary_path)

    report_path = write_text(TARGET / "stepwise_interpretation_report.md", build_report(df, score))
    paths.append(report_path)
    appendix_path = write_text(TARGET / "stepwise_validation_appendix.md", build_validation_appendix(validation, cloud_increment))
    paths.append(appendix_path)
    paths.append(write_manifest(paths))


if __name__ == "__main__":
    main()
