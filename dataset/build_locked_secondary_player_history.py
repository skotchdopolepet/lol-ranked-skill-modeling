from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

LOCK_DIR = (
    REPO_ROOT
    / "runtime"
    / "out_latest"
    / "reports"
    / "night_window"
    / "locked_secondary_downsample_larger_protected_2330_1300"
)
LOCKED_ROW_IDS = LOCK_DIR / "locked_secondary_row_ids.csv"
SECONDARY_DB = (
    REPO_ROOT
    / "runtime"
    / "out_prod_player"
    / "secondary_dataset"
    / "player_secondary_dataset.sqlite3"
)
CATBOOST_NPZ = (
    REPO_ROOT
    / "runtime"
    / "out_latest"
    / "reports"
    / "night_window"
    / "secondary_row_predictions_with_catboost.npz"
)
PLAYER_ROOT = REPO_ROOT / "runtime" / "out_prod_player"
DEFAULT_OUTPUT_DB = (
    REPO_ROOT
    / "runtime"
    / "out_prod_player"
    / "secondary_dataset"
    / "locked_secondary_with_player_history.sqlite3"
)
DEFAULT_WORK_DIR = (
    REPO_ROOT
    / "runtime"
    / "out_latest"
    / "reports"
    / "night_window"
    / "locked_secondary_player_history_build"
)

NIGHT_START = 1.0
NIGHT_END = 7.0
UNDERPERFORMING_START = 2.0
UNDERPERFORMING_END = 8.5
MIN_PLAYER_HISTORY_GAMES = 30
MIN_ELIGIBLE_PLAYERS = 8

FOLDER_TIMEZONES = {
    "br": "America/Sao_Paulo",
    "br2": "America/Sao_Paulo",
    "eune1": "Europe/Berlin",
    "eune2": "Europe/Berlin",
    "euw1": "Europe/Berlin",
    "euw2": "Europe/Berlin",
    "euw3": "Europe/Berlin",
    "euw4": "Europe/Berlin",
    "jp": "Asia/Tokyo",
    "jp2": "Asia/Tokyo",
    "jp3": "Asia/Tokyo",
    "kr": "Asia/Seoul",
    "kr2": "Asia/Seoul",
    "kr3": "Asia/Seoul",
    "kr4": "Asia/Seoul",
    "kr5": "Asia/Seoul",
    "la1": "America/Mexico_City",
    "la1_2": "America/Mexico_City",
    "la2": "America/Santiago",
    "na": "America/Los_Angeles",
    "na2": "America/Los_Angeles",
    "na3": "America/Los_Angeles",
    "sg": "Asia/Singapore",
    "sg2": "Asia/Singapore",
    "tr": "Europe/Istanbul",
    "tr2": "Europe/Istanbul",
    "tw": "Asia/Taipei",
    "tw2": "Asia/Taipei",
    "tw3": "Asia/Taipei",
    "vn": "Asia/Ho_Chi_Minh",
    "vn2": "Asia/Ho_Chi_Minh",
    "vn3": "Asia/Ho_Chi_Minh",
}


@dataclass(frozen=True)
class TargetMatch:
    secondary_row_id: int
    match_id: str


@dataclass
class PlayerProfile:
    timed_games: int = 0
    night_games: int = 0
    underperforming_games: int = 0

    @property
    def eligible(self) -> bool:
        return self.timed_games >= MIN_PLAYER_HISTORY_GAMES

    @property
    def night_share(self) -> float:
        return self.night_games / self.timed_games if self.timed_games else math.nan

    @property
    def underperforming_share(self) -> float:
        return self.underperforming_games / self.timed_games if self.timed_games else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build locked secondary dataset enriched with all-history player time-window features."
    )
    parser.add_argument("--locked-row-ids", default=str(LOCKED_ROW_IDS))
    parser.add_argument("--secondary-db", default=str(SECONDARY_DB))
    parser.add_argument("--catboost-npz", default=str(CATBOOST_NPZ))
    parser.add_argument("--player-root", default=str(PLAYER_ROOT))
    parser.add_argument("--output-db", default=str(DEFAULT_OUTPUT_DB))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--force", action="store_true", help="Rebuild existing shards and final DB.")
    parser.add_argument("--shards-only", action="store_true", help="Build folder shards but skip final merge.")
    parser.add_argument("--merge-only", action="store_true", help="Skip shard workers and only merge existing shards.")
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def infer_timezone(folder: str) -> str:
    name = folder.lower()
    for prefix in sorted(FOLDER_TIMEZONES, key=len, reverse=True):
        if name == prefix or name.startswith(f"{prefix}_"):
            return FOLDER_TIMEZONES[prefix]
    raise ValueError(f"Cannot infer timezone for folder {folder!r}")


def local_hour_from_utc_ms(game_creation_utc_ms: int, tz: ZoneInfo) -> float:
    dt = datetime.fromtimestamp(game_creation_utc_ms / 1000.0, tz=UTC).astimezone(tz)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3_600_000_000.0


def in_window(hour: float, start: float, end: float) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def load_locked_row_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "secondary_row_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path} is missing secondary_row_id column")
        for row in reader:
            ids.append(int(row["secondary_row_id"]))
    if len(ids) != len(set(ids)):
        raise RuntimeError("Locked row IDs contain duplicates")
    return ids


def fetch_locked_manifest(secondary_db: Path, row_ids: list[int], chunk_size: int) -> list[dict[str, Any]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    conn = sqlite3.connect(f"file:{secondary_db.as_posix()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        for start in range(0, len(row_ids), chunk_size):
            ids = row_ids[start : start + chunk_size]
            sqlite_rowids = [row_id + 1 for row_id in ids]
            placeholders = ",".join("?" for _ in sqlite_rowids)
            sql = (
                "SELECT rowid AS sqlite_rowid, match_id, source_folder, "
                "local_time_sin, local_time_cos, average_skill_level "
                f"FROM player_secondary_dataset_v1 WHERE rowid IN ({placeholders})"
            )
            for row in conn.execute(sql, sqlite_rowids):
                secondary_row_id = int(row["sqlite_rowid"]) - 1
                rows_by_id[secondary_row_id] = {
                    "secondary_row_id": secondary_row_id,
                    "match_id": str(row["match_id"]),
                    "source_folder": str(row["source_folder"]),
                    "average_skill_level": float(row["average_skill_level"]),
                    "local_time_sin": float(row["local_time_sin"]),
                    "local_time_cos": float(row["local_time_cos"]),
                }
    finally:
        conn.close()
    if len(rows_by_id) != len(row_ids):
        missing = sorted(set(row_ids).difference(rows_by_id))[:20]
        raise RuntimeError(f"Secondary DB did not return all locked row IDs; missing examples: {missing}")
    return [rows_by_id[row_id] for row_id in row_ids]


def add_catboost_and_windows(manifest: list[dict[str, Any]], npz_path: Path) -> None:
    raw = np.load(npz_path, allow_pickle=True)
    required = {"row_id", "hour", "y", "catboost_pred"}
    missing = required.difference(raw.files)
    if missing:
        raise RuntimeError(f"{npz_path} is missing arrays: {sorted(missing)}")
    row_id_array = raw["row_id"]
    catboost_pred = raw["catboost_pred"].astype(np.float64)
    hours = raw["hour"].astype(np.float64)
    y = raw["y"].astype(np.float64)
    if row_id_array.shape[0] <= max(row["secondary_row_id"] for row in manifest):
        raise RuntimeError("CatBoost NPZ is shorter than locked row IDs")
    for row in manifest:
        row_id = int(row["secondary_row_id"])
        if int(row_id_array[row_id]) != row_id:
            raise RuntimeError(f"CatBoost NPZ row alignment failed at row_id={row_id}")
        if abs(float(y[row_id]) - float(row["average_skill_level"])) > 1e-9:
            raise RuntimeError(f"Target mismatch between secondary DB and NPZ at row_id={row_id}")
        local_hour = float(hours[row_id])
        pred = float(catboost_pred[row_id])
        row["catboost_pred"] = pred
        row["residual"] = pred - float(row["average_skill_level"])
        row["local_start_hour"] = local_hour
        row["night_window"] = 1 if in_window(local_hour, NIGHT_START, NIGHT_END) else 0
        row["underperforming_window"] = 1 if in_window(local_hour, UNDERPERFORMING_START, UNDERPERFORMING_END) else 0


def write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "secondary_row_id",
        "match_id",
        "source_folder",
        "average_skill_level",
        "catboost_pred",
        "residual",
        "local_start_hour",
        "night_window",
        "underperforming_window",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row[key] for key in fieldnames})


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out: list[dict[str, Any]] = []
        for row in reader:
            out.append(
                {
                    "secondary_row_id": int(row["secondary_row_id"]),
                    "match_id": row["match_id"],
                    "source_folder": row["source_folder"],
                    "average_skill_level": float(row["average_skill_level"]),
                    "catboost_pred": float(row["catboost_pred"]),
                    "residual": float(row["residual"]),
                    "local_start_hour": float(row["local_start_hour"]),
                    "night_window": int(row["night_window"]),
                    "underperforming_window": int(row["underperforming_window"]),
                }
            )
        return out


def load_match_participants(
    conn: sqlite3.Connection, match_ids: list[str], chunk_size: int
) -> tuple[dict[str, list[str]], int]:
    match_to_puuids: dict[str, list[str]] = {match_id: [] for match_id in match_ids}
    for start in range(0, len(match_ids), chunk_size):
        chunk = match_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            "SELECT match_id, puuid FROM match_participants_cache "
            f"WHERE match_id IN ({placeholders}) ORDER BY match_id"
        )
        for match_id, puuid in conn.execute(sql, chunk):
            match_to_puuids[str(match_id)].append(str(puuid))
    missing = sum(1 for puuids in match_to_puuids.values() if not puuids)
    return match_to_puuids, missing


def compute_player_profiles(
    conn: sqlite3.Connection, puuids: list[str], tz_name: str, chunk_size: int
) -> dict[str, PlayerProfile]:
    tz = ZoneInfo(tz_name)
    profiles = {puuid: PlayerProfile() for puuid in puuids}
    for start in range(0, len(puuids), chunk_size):
        chunk = puuids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            SELECT pm.puuid, mt.game_creation_utc_ms
            FROM player_match_ids_cache AS pm
            INNER JOIN match_time_cache AS mt
                ON mt.match_id = pm.match_id
            WHERE pm.puuid IN ({placeholders})
              AND mt.game_creation_utc_ms IS NOT NULL
              AND mt.game_creation_utc_ms > 0
        """
        for puuid, game_creation_utc_ms in conn.execute(sql, chunk):
            profile = profiles[str(puuid)]
            hour = local_hour_from_utc_ms(int(game_creation_utc_ms), tz)
            profile.timed_games += 1
            if in_window(hour, NIGHT_START, NIGHT_END):
                profile.night_games += 1
            if in_window(hour, UNDERPERFORMING_START, UNDERPERFORMING_END):
                profile.underperforming_games += 1
    return profiles


def summarize_match_history(puuids: list[str], profiles: dict[str, PlayerProfile]) -> dict[str, Any]:
    eligible = [profiles[puuid] for puuid in puuids if puuid in profiles and profiles[puuid].eligible]
    eligible_count = len(eligible)
    if eligible_count < MIN_ELIGIBLE_PLAYERS:
        return {
            "history_features_available": 0,
            "history_eligible_count": eligible_count,
            "history_game_count": None,
            "night_game_share": None,
            "underperforming_game_share": None,
            "night_share_std": None,
            "underperforming_share_std": None,
        }

    history_game_count = sum(profile.timed_games for profile in eligible)
    night_games = sum(profile.night_games for profile in eligible)
    underperforming_games = sum(profile.underperforming_games for profile in eligible)
    night_shares = [profile.night_share for profile in eligible]
    underperforming_shares = [profile.underperforming_share for profile in eligible]
    return {
        "history_features_available": 1,
        "history_eligible_count": eligible_count,
        "history_game_count": history_game_count,
        "night_game_share": night_games / history_game_count if history_game_count else None,
        "underperforming_game_share": (
            underperforming_games / history_game_count if history_game_count else None
        ),
        "night_share_std": statistics.pstdev(night_shares) if len(night_shares) > 1 else 0.0,
        "underperforming_share_std": (
            statistics.pstdev(underperforming_shares) if len(underperforming_shares) > 1 else 0.0
        ),
    }


def shard_is_complete(shard_path: Path, expected_rows: int) -> bool:
    if not shard_path.exists() or shard_path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(f"file:{shard_path.as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM history_features").fetchone()
            return bool(row and int(row[0]) == expected_rows)
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def write_shard(shard_path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=shard_path.name, suffix=".tmp", dir=str(shard_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute(
                """
                CREATE TABLE history_features (
                    secondary_row_id INTEGER PRIMARY KEY,
                    history_features_available INTEGER NOT NULL,
                    history_eligible_count INTEGER NOT NULL,
                    history_game_count INTEGER,
                    night_game_share REAL,
                    underperforming_game_share REAL,
                    night_share_std REAL,
                    underperforming_share_std REAL
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO history_features VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(row["secondary_row_id"]),
                        int(row["history_features_available"]),
                        int(row["history_eligible_count"]),
                        row["history_game_count"],
                        row["night_game_share"],
                        row["underperforming_game_share"],
                        row["night_share_std"],
                        row["underperforming_share_std"],
                    )
                    for row in rows
                ],
            )
            conn.execute("CREATE TABLE shard_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO shard_metadata (key, value) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in stats.items()],
            )
            conn.commit()
        finally:
            conn.close()
        tmp_path.replace(shard_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_folder_shard(task: dict[str, Any]) -> dict[str, Any]:
    folder = str(task["folder"])
    player_root = Path(str(task["player_root"]))
    work_dir = Path(str(task["work_dir"]))
    chunk_size = int(task["chunk_size"])
    force = bool(task["force"])
    targets = [TargetMatch(int(row[0]), str(row[1])) for row in task["targets"]]
    shard_path = work_dir / "shards" / f"{folder}.sqlite3"

    if not force and shard_is_complete(shard_path, len(targets)):
        return {
            "folder": folder,
            "status": "skipped_existing",
            "target_rows": len(targets),
            "shard_path": str(shard_path),
        }

    t0 = time.time()
    cache_db = player_root / folder / "player_time_cache.sqlite3"
    if not cache_db.exists():
        raise RuntimeError(f"Missing cache DB for {folder}: {cache_db}")

    conn = sqlite3.connect(f"file:{cache_db.as_posix()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        match_ids = [target.match_id for target in targets]
        match_to_puuids, missing_participant_matches = load_match_participants(conn, match_ids, chunk_size)
        unique_puuids = sorted({puuid for puuids in match_to_puuids.values() for puuid in puuids})
        profiles = compute_player_profiles(conn, unique_puuids, infer_timezone(folder), chunk_size)
    finally:
        conn.close()

    rows: list[dict[str, Any]] = []
    available = 0
    for target in targets:
        puuids = match_to_puuids.get(target.match_id, [])
        row = {"secondary_row_id": target.secondary_row_id}
        row.update(summarize_match_history(puuids, profiles))
        available += int(row["history_features_available"])
        rows.append(row)

    stats = {
        "folder": folder,
        "target_rows": len(targets),
        "missing_participant_matches": missing_participant_matches,
        "unique_target_puuids": len(unique_puuids),
        "players_with_any_timed_history": sum(1 for p in profiles.values() if p.timed_games > 0),
        "eligible_players_unique": sum(1 for p in profiles.values() if p.eligible),
        "history_features_available_rows": available,
        "elapsed_sec": round(time.time() - t0, 3),
        "timezone": infer_timezone(folder),
    }
    write_shard(shard_path, rows, stats)
    stats.update({"status": "built", "shard_path": str(shard_path)})
    return stats


def create_base_tables(conn: sqlite3.Connection, manifest: list[dict[str, Any]]) -> None:
    conn.execute(
        """
        CREATE TABLE locked_base_enrichment (
            secondary_row_id INTEGER PRIMARY KEY,
            catboost_pred REAL NOT NULL,
            residual REAL NOT NULL,
            local_start_hour REAL NOT NULL,
            night_window INTEGER NOT NULL,
            underperforming_window INTEGER NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO locked_base_enrichment VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                int(row["secondary_row_id"]),
                float(row["catboost_pred"]),
                float(row["residual"]),
                float(row["local_start_hour"]),
                int(row["night_window"]),
                int(row["underperforming_window"]),
            )
            for row in manifest
        ],
    )
    conn.execute("CREATE TABLE locked_ids (secondary_row_id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO locked_ids VALUES (?)",
        [(int(row["secondary_row_id"]),) for row in manifest],
    )


def merge_shards(conn: sqlite3.Connection, shard_paths: list[Path]) -> None:
    conn.execute(
        """
        CREATE TABLE history_features (
            secondary_row_id INTEGER PRIMARY KEY,
            history_features_available INTEGER NOT NULL,
            history_eligible_count INTEGER NOT NULL,
            history_game_count INTEGER,
            night_game_share REAL,
            underperforming_game_share REAL,
            night_share_std REAL,
            underperforming_share_std REAL
        )
        """
    )
    for idx, shard_path in enumerate(shard_paths):
        if not shard_path.exists():
            raise RuntimeError(f"Missing shard for merge: {shard_path}")
        alias = f"shard_{idx}"
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_path),))
        try:
            conn.execute(
                """
                INSERT INTO history_features
                SELECT
                    secondary_row_id,
                    history_features_available,
                    history_eligible_count,
                    history_game_count,
                    night_game_share,
                    underperforming_game_share,
                    night_share_std,
                    underperforming_share_std
                FROM """ + alias + ".history_features"
            )
            conn.commit()
        finally:
            conn.execute(f"DETACH DATABASE {alias}")


def build_final_db(
    output_db: Path,
    secondary_db: Path,
    manifest: list[dict[str, Any]],
    shard_paths: list[Path],
    force: bool,
) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        if not force:
            raise RuntimeError(f"Output DB already exists; pass --force to replace: {output_db}")
        output_db.unlink()

    fd, tmp_name = tempfile.mkstemp(prefix=output_db.name, suffix=".tmp", dir=str(output_db.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            create_base_tables(conn, manifest)
            merge_shards(conn, shard_paths)
            conn.execute("ATTACH DATABASE ? AS src", (str(secondary_db),))
            try:
                src_cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA src.table_info(player_secondary_dataset_v1)"
                    ).fetchall()
                ]
                selected_src_cols = ", ".join(f"s.{quote_ident(col)} AS {quote_ident(col)}" for col in src_cols)
                sql = f"""
                    CREATE TABLE locked_secondary_with_player_history_v1 AS
                    SELECT
                        l.secondary_row_id AS secondary_row_id,
                        {selected_src_cols},
                        b.catboost_pred AS catboost_pred,
                        b.residual AS residual,
                        b.local_start_hour AS local_start_hour,
                        b.night_window AS night_window,
                        b.underperforming_window AS underperforming_window,
                        h.history_features_available AS history_features_available,
                        h.history_eligible_count AS history_eligible_count,
                        h.history_game_count AS history_game_count,
                        h.night_game_share AS night_game_share,
                        h.underperforming_game_share AS underperforming_game_share,
                        h.night_share_std AS night_share_std,
                        h.underperforming_share_std AS underperforming_share_std
                    FROM locked_ids AS l
                    INNER JOIN src.player_secondary_dataset_v1 AS s
                        ON s.rowid = l.secondary_row_id + 1
                    INNER JOIN locked_base_enrichment AS b
                        ON b.secondary_row_id = l.secondary_row_id
                    INNER JOIN history_features AS h
                        ON h.secondary_row_id = l.secondary_row_id
                    ORDER BY l.secondary_row_id
                """
                conn.execute(sql)
            finally:
                conn.execute("DETACH DATABASE src")
            conn.execute(
                "CREATE UNIQUE INDEX idx_locked_secondary_player_history_row_id "
                "ON locked_secondary_with_player_history_v1(secondary_row_id)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX idx_locked_secondary_player_history_match_id "
                "ON locked_secondary_with_player_history_v1(match_id)"
            )
            conn.commit()
        finally:
            conn.close()
        tmp_path.replace(output_db)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_shard_metadata(shard_paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        conn = sqlite3.connect(f"file:{shard_path.as_posix()}?mode=ro", uri=True)
        try:
            meta = {
                str(key): json.loads(value)
                for key, value in conn.execute("SELECT key, value FROM shard_metadata")
            }
            out.append(meta)
        finally:
            conn.close()
    return out


def build_qa(output_db: Path, work_dir: Path, manifest: list[dict[str, Any]], shard_paths: list[Path]) -> None:
    qa_dir = work_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    shard_meta = load_shard_metadata(shard_paths)
    write_csv_rows(
        qa_dir / "coverage_by_folder.csv",
        shard_meta,
        [
            "folder",
            "timezone",
            "target_rows",
            "missing_participant_matches",
            "unique_target_puuids",
            "players_with_any_timed_history",
            "eligible_players_unique",
            "history_features_available_rows",
            "elapsed_sec",
        ],
    )

    conn = sqlite3.connect(f"file:{output_db.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        total_rows = int(
            conn.execute("SELECT COUNT(*) FROM locked_secondary_with_player_history_v1").fetchone()[0]
        )
        distinct_row_ids = int(
            conn.execute(
                "SELECT COUNT(DISTINCT secondary_row_id) FROM locked_secondary_with_player_history_v1"
            ).fetchone()[0]
        )
        distinct_match_ids = int(
            conn.execute(
                "SELECT COUNT(DISTINCT match_id) FROM locked_secondary_with_player_history_v1"
            ).fetchone()[0]
        )
        history_available = int(
            conn.execute(
                "SELECT COUNT(*) FROM locked_secondary_with_player_history_v1 "
                "WHERE history_features_available = 1"
            ).fetchone()[0]
        )
        null_history_flags = int(
            conn.execute(
                "SELECT COUNT(*) FROM locked_secondary_with_player_history_v1 "
                "WHERE history_features_available IS NULL OR history_eligible_count IS NULL"
            ).fetchone()[0]
        )
        window_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    'all_locked' AS subset,
                    COUNT(*) AS n,
                    SUM(night_window) AS night_window_n,
                    SUM(underperforming_window) AS underperforming_window_n,
                    AVG(average_skill_level) AS actual_mean,
                    AVG(catboost_pred) AS catboost_pred_mean,
                    AVG(residual) AS residual_mean
                FROM locked_secondary_with_player_history_v1
                UNION ALL
                SELECT
                    'history_available' AS subset,
                    COUNT(*) AS n,
                    SUM(night_window) AS night_window_n,
                    SUM(underperforming_window) AS underperforming_window_n,
                    AVG(average_skill_level) AS actual_mean,
                    AVG(catboost_pred) AS catboost_pred_mean,
                    AVG(residual) AS residual_mean
                FROM locked_secondary_with_player_history_v1
                WHERE history_features_available = 1
                """
            )
        ]
        write_csv_rows(
            qa_dir / "window_summary.csv",
            window_rows,
            [
                "subset",
                "n",
                "night_window_n",
                "underperforming_window_n",
                "actual_mean",
                "catboost_pred_mean",
                "residual_mean",
            ],
        )
        history_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    COUNT(*) AS n,
                    AVG(history_eligible_count) AS history_eligible_count_mean,
                    MIN(history_eligible_count) AS history_eligible_count_min,
                    MAX(history_eligible_count) AS history_eligible_count_max,
                    AVG(history_game_count) AS history_game_count_mean,
                    MIN(history_game_count) AS history_game_count_min,
                    MAX(history_game_count) AS history_game_count_max,
                    AVG(night_game_share) AS night_game_share_mean,
                    AVG(underperforming_game_share) AS underperforming_game_share_mean,
                    AVG(night_share_std) AS night_share_std_mean,
                    AVG(underperforming_share_std) AS underperforming_share_std_mean
                FROM locked_secondary_with_player_history_v1
                WHERE history_features_available = 1
                """
            )
        ]
        write_csv_rows(
            qa_dir / "history_feature_summary.csv",
            history_rows,
            [
                "n",
                "history_eligible_count_mean",
                "history_eligible_count_min",
                "history_eligible_count_max",
                "history_game_count_mean",
                "history_game_count_min",
                "history_game_count_max",
                "night_game_share_mean",
                "underperforming_game_share_mean",
                "night_share_std_mean",
                "underperforming_share_std_mean",
            ],
        )
    finally:
        conn.close()

    manifest_night = sum(int(row["night_window"]) for row in manifest)
    manifest_under = sum(int(row["underperforming_window"]) for row in manifest)
    metadata = {
        "output_db": str(output_db),
        "table": "locked_secondary_with_player_history_v1",
        "locked_rows_expected": len(manifest),
        "final_rows": total_rows,
        "distinct_secondary_row_ids": distinct_row_ids,
        "distinct_match_ids": distinct_match_ids,
        "history_features_available_rows": history_available,
        "history_features_unavailable_rows": total_rows - history_available,
        "null_history_flag_rows": null_history_flags,
        "manifest_night_window_rows": manifest_night,
        "manifest_underperforming_window_rows": manifest_under,
        "residual_definition": "catboost_pred - average_skill_level",
        "catboost_source": str(CATBOOST_NPZ),
        "source_secondary_db": str(SECONDARY_DB),
        "locked_row_ids": str(LOCKED_ROW_IDS),
        "min_player_history_games": MIN_PLAYER_HISTORY_GAMES,
        "min_eligible_players_per_match": MIN_ELIGIBLE_PLAYERS,
        "uses_all_cached_player_history": True,
        "history_time_cutoff": None,
    }
    (qa_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (qa_dir / "README.md").write_text(
        "\n".join(
            [
                "# Locked Secondary Player History Dataset",
                "",
                f"Output DB: `{output_db}`",
                "",
                "Table: `locked_secondary_with_player_history_v1`",
                "",
                "This dataset keeps the locked secondary rows and appends CatBoost residuals,",
                "target-match time-window flags, and all-history player time-window features.",
                "",
                "Residual definition: `catboost_pred - average_skill_level`.",
                "",
                "History eligibility:",
                f"- eligible player: at least `{MIN_PLAYER_HISTORY_GAMES}` timed cached historical games",
                f"- eligible match: at least `{MIN_ELIGIBLE_PLAYERS}` eligible target-match players",
                "- all cached timed history is used; no as-of cutoff is applied",
                "",
                "QA files:",
                "- `coverage_by_folder.csv`",
                "- `window_summary.csv`",
                "- `history_feature_summary.csv`",
                "- `metadata.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if total_rows != len(manifest):
        raise RuntimeError(f"Final row count mismatch: {total_rows} != {len(manifest)}")
    if distinct_row_ids != total_rows:
        raise RuntimeError("secondary_row_id is not unique in final dataset")
    if distinct_match_ids != total_rows:
        raise RuntimeError("match_id is not unique in final dataset")
    if null_history_flags != 0:
        raise RuntimeError("Final dataset has null history coverage flags")
    if manifest_night != 40427:
        raise RuntimeError(f"Unexpected night_window count: {manifest_night}")
    if manifest_under != 30703:
        raise RuntimeError(f"Unexpected underperforming_window count: {manifest_under}")


def main() -> int:
    args = parse_args()
    locked_row_ids = Path(args.locked_row_ids).resolve()
    secondary_db = Path(args.secondary_db).resolve()
    catboost_npz = Path(args.catboost_npz).resolve()
    player_root = Path(args.player_root).resolve()
    output_db = Path(args.output_db).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "locked_manifest.csv"

    if args.merge_only:
        manifest = read_manifest(manifest_path)
    else:
        row_ids = load_locked_row_ids(locked_row_ids)
        manifest = fetch_locked_manifest(secondary_db, row_ids, int(args.chunk_size))
        add_catboost_and_windows(manifest, catboost_npz)
        write_manifest(manifest_path, manifest)

    by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_folder[str(row["source_folder"])].append(row)

    shard_paths = [work_dir / "shards" / f"{folder}.sqlite3" for folder in sorted(by_folder)]
    if not args.merge_only:
        tasks = [
            {
                "folder": folder,
                "player_root": str(player_root),
                "work_dir": str(work_dir),
                "chunk_size": int(args.chunk_size),
                "force": bool(args.force),
                "targets": [(int(row["secondary_row_id"]), str(row["match_id"])) for row in rows],
            }
            for folder, rows in sorted(by_folder.items())
        ]
        max_workers = max(1, int(args.workers))
        print(f"building {len(tasks)} folder shards with workers={max_workers}", flush=True)
        completed: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_folder = {executor.submit(process_folder_shard, task): task["folder"] for task in tasks}
            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                result = future.result()
                completed.append(result)
                print(
                    "folder_done "
                    + json.dumps(
                        {
                            "folder": folder,
                            "status": result.get("status"),
                            "target_rows": result.get("target_rows"),
                            "history_features_available_rows": result.get(
                                "history_features_available_rows"
                            ),
                            "elapsed_sec": result.get("elapsed_sec"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        (work_dir / "shard_run_results.json").write_text(
            json.dumps(sorted(completed, key=lambda row: str(row["folder"])), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    for folder, rows in sorted(by_folder.items()):
        shard_path = work_dir / "shards" / f"{folder}.sqlite3"
        if not shard_is_complete(shard_path, len(rows)):
            raise RuntimeError(f"Shard incomplete for {folder}: {shard_path}")

    if args.shards_only:
        print(f"shards_complete={len(shard_paths)}")
        return 0

    build_final_db(output_db, secondary_db, manifest, shard_paths, bool(args.force))
    build_qa(output_db, work_dir, manifest, shard_paths)
    print(f"wrote_output_db={output_db}")
    print(f"wrote_qa_dir={work_dir / 'qa'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
