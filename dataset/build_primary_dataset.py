from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "dataset", REPO_ROOT / "rank_mapping"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import build_match_feature_table_v1 as xbuilder
from finalize_and_extract_match_quality import infer_server_for_pipeline, load_ranks_by_puuid, safe_read_json
from probit_core import RankLpProbitMapper
from probit_settings import apex_lp_cutoffs_for_server, target_percentages_for_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final primary modeling dataset (X + y).")
    parser.add_argument("--input-root", default=str(REPO_ROOT / "runtime" / "out_prod"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runtime" / "out_prod" / "primary_dataset"))
    parser.add_argument("--required-unique-participants", type=int, default=10)
    parser.add_argument("--min-mapped-participants", type=int, default=9)
    parser.add_argument("--reuse-existing-x", action="store_true")
    parser.add_argument("--reuse-existing-y", action="store_true")
    parser.add_argument("--skip-csv", action="store_true")
    return parser.parse_args()


def discover_pipeline_dirs(input_root: Path) -> list[Path]:
    out: list[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "matches").exists() and (child / "player_ranks.sqlite3").exists() and (child / "participant_index_by_match.json").exists():
            out.append(child)
    return out


def infer_server_for_match(match_id: str, fallback_server: str) -> str:
    token = str(match_id or "")
    if "_" in token:
        prefix = token.split("_", 1)[0].upper()
        if prefix:
            return prefix
    return str(fallback_server).upper()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def count_match_files(input_root: Path) -> int:
    return sum(1 for _ in xbuilder.iter_match_files(input_root))


def build_eta_payload(*, started_at: float, done: int, total: int | None) -> dict[str, Any]:
    elapsed_sec = max(0.0, time.time() - float(started_at))
    files_per_sec = (float(done) / elapsed_sec) if elapsed_sec > 0.0 else None
    remaining = None
    if total is not None and files_per_sec and files_per_sec > 0.0 and done <= total:
        remaining = max(0.0, float(total - done) / files_per_sec)
    return {
        "elapsed_sec": elapsed_sec,
        "throughput_files_per_sec": files_per_sec,
        "estimated_remaining_sec": remaining,
        "estimated_completion_utc": (
            datetime.fromtimestamp(time.time() + remaining, tz=UTC).isoformat()
            if remaining is not None
            else None
        ),
    }


def match_id_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(".json.zst"):
        return name[: -len(".json.zst")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def summarize_existing_x_table(db_path: Path) -> dict[str, Any]:
    wal_path = db_path.with_name(db_path.name + "-wal")
    shm_path = db_path.with_name(db_path.name + "-shm")
    if wal_path.exists() or shm_path.exists():
        raise RuntimeError(
            f"Cannot reuse existing X DB while WAL/SHM sidecars exist: {db_path}. "
            "This usually means the previous X build is still running or did not checkpoint cleanly."
        )
    conn = sqlite3.connect(db_path)
    try:
        kept = int(conn.execute("SELECT COUNT(*) FROM match_table_v1").fetchone()[0])
        folder_kept = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT source_folder, COUNT(*) FROM match_table_v1 GROUP BY source_folder ORDER BY source_folder"
            ).fetchall()
        }
    finally:
        conn.close()
    if kept <= 0:
        raise RuntimeError(f"Existing X DB has 0 rows and is not safe to reuse: {db_path}")
    return {
        "x_db_path": str(db_path),
        "discovered_files": None,
        "loaded_payloads": None,
        "kept_matches": kept,
        "duplicate_match_ids_skipped": None,
        "dropped_matches": None,
        "drop_reasons": {},
        "folder_kept_counts": folder_kept,
        "total_column_count": len(xbuilder.schema_feature_columns()),
        "metadata_columns": [name for name, _ in xbuilder.METADATA_COLUMNS],
        "primary_model_feature_count": (
            len(xbuilder.PARTICIPANT_NUMERIC_FEATURES) * 5 * 2
            + len(xbuilder.PARTICIPANT_SHARE_FEATURES) * 5
            + len(xbuilder.PARTICIPANT_JUNGLE_ONLY_FEATURES) * 2
            + len(xbuilder.PARTICIPANT_DELTA_ONLY_FEATURES) * 5
            + sum(len(spec["positions"]) * 2 for spec in xbuilder.CHALLENGE_POSITION_FEATURES)
            + len(xbuilder.CHALLENGE_JUNGLE_FEATURES) * 2
            + sum(len(spec["positions"]) for spec in xbuilder.CHALLENGE_DELTA_ONLY_FEATURES)
            + len(xbuilder.TEAM_CONTEXT_COLUMNS)
            + len(xbuilder.OBJECTIVES_DTO_COLUMNS)
            + len(xbuilder.SOCIAL_CONTEXT_COLUMNS)
            + len(xbuilder.INFO_VARIABLE_COLUMNS)
        ),
        "secondary_analysis_feature_count": len(xbuilder.SECONDARY_ANALYSIS_COLUMNS),
        "reused_existing_db": True,
    }


def build_solution2_owner_db(input_root: Path, output_dir: Path) -> dict[str, Any]:
    compare_dir = output_dir / "y_targets"
    compare_dir.mkdir(parents=True, exist_ok=True)
    db_path = compare_dir / "solution2_match_owners.sqlite3"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE match_owners (match_id TEXT PRIMARY KEY, source_folder TEXT NOT NULL)")
    inserted = 0
    duplicate_actual_match_ids = 0
    duplicate_examples: list[dict[str, str]] = []
    try:
        for source_folder, match_path in xbuilder.iter_match_files(input_root):
            match_id = match_id_from_filename(match_path)
            cur = conn.execute(
                "INSERT OR IGNORE INTO match_owners(match_id, source_folder) VALUES (?, ?)",
                (match_id, source_folder),
            )
            if cur.rowcount:
                inserted += 1
                continue
            duplicate_actual_match_ids += 1
            existing = conn.execute(
                "SELECT source_folder FROM match_owners WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            if len(duplicate_examples) < 20:
                duplicate_examples.append(
                    {
                        "match_id": match_id,
                        "existing_source_folder": str(existing[0]) if existing else "",
                        "duplicate_source_folder": source_folder,
                    }
                )
        conn.commit()
    finally:
        conn.close()
    if duplicate_actual_match_ids > 0:
        raise RuntimeError(
            f"Solution 2 found {duplicate_actual_match_ids} duplicate actual match files. "
            f"Examples: {json.dumps(duplicate_examples[:3], ensure_ascii=False)}"
        )
    return {
        "owner_db_path": str(db_path),
        "inserted_match_ids": inserted,
        "duplicate_actual_match_ids": duplicate_actual_match_ids,
        "duplicate_examples": duplicate_examples,
    }


def summarize_existing_y_table(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        inserted = int(conn.execute("SELECT COUNT(*) FROM skill_targets").fetchone()[0])
        servers = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT skill_server, COUNT(*) FROM skill_targets GROUP BY skill_server ORDER BY skill_server"
            ).fetchall()
        }
    finally:
        conn.close()
    if inserted <= 0:
        raise RuntimeError(f"Existing Y DB has 0 rows and is not safe to reuse: {db_path}")
    return {
        "y_db_path": str(db_path),
        "inserted_rows": inserted,
        "duplicate_match_ids": 0,
        "conflicting_duplicates": 0,
        "conflict_examples": [],
        "pipeline_stats": [],
        "reused_existing_db": True,
        "servers_seen": servers,
    }


def compare_solution1_and_solution2(x_db_path: Path, owner_db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("ATTACH DATABASE ? AS xdb", (str(x_db_path),))
        conn.execute("ATTACH DATABASE ? AS odb", (str(owner_db_path),))
        compared = int(conn.execute("SELECT COUNT(*) FROM xdb.match_table_v1").fetchone()[0])
        missing = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM xdb.match_table_v1 AS x
                LEFT JOIN odb.match_owners AS o
                    ON o.match_id = x.match_id
                WHERE o.match_id IS NULL
                """
            ).fetchone()[0]
        )
        mismatches = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM xdb.match_table_v1 AS x
                JOIN odb.match_owners AS o
                    ON o.match_id = x.match_id
                WHERE x.source_folder != o.source_folder
                """
            ).fetchone()[0]
        )
        examples = [
            {
                "match_id": str(row[0]),
                "solution1_source_folder": str(row[1]),
                "solution2_source_folder": str(row[2]),
            }
            for row in conn.execute(
                """
                SELECT x.match_id, x.source_folder, o.source_folder
                FROM xdb.match_table_v1 AS x
                JOIN odb.match_owners AS o
                    ON o.match_id = x.match_id
                WHERE x.source_folder != o.source_folder
                ORDER BY x.match_id
                LIMIT 20
                """
            ).fetchall()
        ]
        missing_examples = [
            {
                "match_id": str(row[0]),
                "solution1_source_folder": str(row[1]),
            }
            for row in conn.execute(
                """
                SELECT x.match_id, x.source_folder
                FROM xdb.match_table_v1 AS x
                LEFT JOIN odb.match_owners AS o
                    ON o.match_id = x.match_id
                WHERE o.match_id IS NULL
                ORDER BY x.match_id
                LIMIT 20
                """
            ).fetchall()
        ]
    finally:
        conn.close()
    return {
        "x_rows_compared": compared,
        "solution2_missing_match_ids": missing,
        "source_folder_mismatches": mismatches,
        "mismatch_examples": examples,
        "missing_examples": missing_examples,
        "solutions_identical": (missing == 0 and mismatches == 0),
    }


def iter_x_match_ids_by_folder(x_db_path: Path, source_folder: str, chunk_size: int = 5000):
    conn = sqlite3.connect(x_db_path)
    try:
        cur = conn.execute(
            "SELECT match_id FROM match_table_v1 WHERE source_folder = ? ORDER BY match_id",
            (source_folder,),
        )
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            yield [str(row[0]) for row in rows]
    finally:
        conn.close()


def fetch_authoritative_participants(
    conn: sqlite3.Connection,
    match_ids: list[str],
) -> dict[str, list[str]]:
    if not match_ids:
        return {}
    placeholders = ", ".join("?" for _ in match_ids)
    rows = conn.execute(
        f"""
        SELECT mp.match_id, mp.puuid
        FROM match_participants AS mp
        JOIN matches AS m
            ON m.match_id = mp.match_id
        WHERE m.valid_for_pipeline = 1
          AND mp.match_id IN ({placeholders})
        ORDER BY mp.match_id
        """,
        match_ids,
    ).fetchall()
    out: dict[str, list[str]] = {}
    for match_id, puuid in rows:
        out.setdefault(str(match_id), []).append(str(puuid))
    return out


def compute_skill_rows_for_x_source_folder(
    *,
    source_folder: str,
    pipeline_dir: Path,
    x_db_path: Path,
    required_unique: int,
    min_mapped: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    db_path = pipeline_dir / "player_ranks.sqlite3"
    ranks_by_puuid = load_ranks_by_puuid(db_path)
    default_server = infer_server_for_pipeline(db_path) or pipeline_dir.name.split("_", 1)[0].upper()
    default_cutoffs = apex_lp_cutoffs_for_server(default_server)
    mapper_cache: dict[str, RankLpProbitMapper] = {}
    server_match_counts: Counter[str] = Counter()

    def mapper_for_server(server: str) -> RankLpProbitMapper:
        key = str(server).upper()
        mapper = mapper_cache.get(key)
        if mapper is None:
            mapper = RankLpProbitMapper(
                target_percentages=target_percentages_for_server(key),
                apex_lp_cutoffs=apex_lp_cutoffs_for_server(key),
                floor_epsilon_pct=0.01,
                ceil_epsilon_pct=0.01,
            )
            mapper_cache[key] = mapper
        return mapper

    skill_rows: list[dict[str, Any]] = []
    included_values: list[float] = []
    total_matches = 0
    excluded_not_exactly = 0
    excluded_low_mapped = 0
    missing_authoritative_participants = 0

    conn = sqlite3.connect(db_path)
    try:
        for chunk in iter_x_match_ids_by_folder(x_db_path, source_folder):
            participants_by_match = fetch_authoritative_participants(conn, chunk)
            for match_id in chunk:
                total_matches += 1
                raw_arr = participants_by_match.get(match_id)
                if not raw_arr:
                    missing_authoritative_participants += 1
                    excluded_not_exactly += 1
                    continue
                match_server = infer_server_for_match(match_id, default_server)
                server_match_counts[match_server] += 1
                mapper = mapper_for_server(match_server)
                unique_puuids = list(dict.fromkeys(str(x) for x in raw_arr if x))
                unique_count = len(unique_puuids)
                if unique_count != required_unique:
                    excluded_not_exactly += 1
                    continue
                z_values: list[float] = []
                for puuid in unique_puuids:
                    rank_tuple = ranks_by_puuid.get(puuid)
                    if rank_tuple is None:
                        continue
                    rank_name, lp = rank_tuple
                    try:
                        z = mapper.rank_lp_to_probit(rank_name, lp)
                    except Exception:
                        continue
                    if math.isfinite(z):
                        z_values.append(float(z))
                if len(z_values) < min_mapped:
                    excluded_low_mapped += 1
                    continue
                avg_skill = sum(z_values) / float(len(z_values))
                if not math.isfinite(avg_skill):
                    continue
                included_values.append(avg_skill)
                skill_rows.append(
                    {
                        "match_id": match_id,
                        "average_skill_level": avg_skill,
                        "mapped_participants": len(z_values),
                        "unique_participants": unique_count,
                        "skill_server": match_server,
                    }
                )
    finally:
        conn.close()

    stats = {
        "pipeline_dir": str(pipeline_dir),
        "source_folder": source_folder,
        "server": default_server,
        "servers_seen": dict(sorted(server_match_counts.items())),
        "total_matches_seen": total_matches,
        "included_matches": len(skill_rows),
        "dropped_not_exactly_10": excluded_not_exactly,
        "dropped_mapped_lt_min": excluded_low_mapped,
        "missing_authoritative_participants": missing_authoritative_participants,
        "gm_cutoff_lp": float(default_cutoffs["gm_cutoff_lp"]),
        "challenger_cutoff_lp": float(default_cutoffs["challenger_cutoff_lp"]),
        "rank1_lp_cap": float(default_cutoffs["rank1_lp"]),
        "avg_skill_stats": {
            "count": len(included_values),
            "min": (min(included_values) if included_values else None),
            "mean": (statistics.fmean(included_values) if included_values else None),
            "median": (statistics.median(included_values) if included_values else None),
            "max": (max(included_values) if included_values else None),
        },
    }
    return skill_rows, stats


def build_x_table(input_root: Path, output_dir: Path, reuse_existing: bool = False) -> dict[str, Any]:
    x_output_dir = output_dir / "x_features"
    x_output_dir.mkdir(parents=True, exist_ok=True)
    db_path = x_output_dir / "match_feature_table_v1.sqlite3"
    if reuse_existing and db_path.exists():
        return summarize_existing_x_table(db_path)
    conn = xbuilder.create_db(db_path)
    kept = duplicate_match_ids = loaded = 0
    drop_reasons: Counter[str] = Counter()
    folder_kept: Counter[str] = Counter()
    total_files = count_match_files(input_root)
    progress_path = output_dir / "build_progress.json"
    started_at = time.time()
    processed = 0
    last_progress_write = 0.0

    def emit_progress(final: bool = False) -> None:
        nonlocal last_progress_write
        if not final and (time.time() - last_progress_write) < 20.0 and processed % 2000 != 0:
            return
        eta = build_eta_payload(started_at=started_at, done=processed, total=total_files)
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "x_build",
                "processed_files": processed,
                "total_files": total_files,
                "loaded_payloads": loaded,
                "kept_matches": kept,
                "duplicate_match_ids_skipped": duplicate_match_ids,
                "dropped_matches": int(sum(drop_reasons.values())),
                "drop_reasons": dict(drop_reasons),
                "folder_kept_counts": dict(sorted(folder_kept.items())),
                **eta,
            },
        )
        last_progress_write = time.time()

    try:
        emit_progress()
        for source_folder, path in xbuilder.iter_match_files(input_root):
            processed += 1
            payload = xbuilder.load_payload(path)
            if payload is None:
                drop_reasons["load_error"] += 1
                emit_progress()
                continue
            loaded += 1
            row, status = xbuilder.extract_row(source_folder=source_folder, path=path, payload=payload)
            if row is None:
                drop_reasons[status] += 1
                emit_progress()
                continue
            if xbuilder.insert_row(conn, row):
                kept += 1
                folder_kept[source_folder] += 1
            else:
                duplicate_match_ids += 1
            emit_progress()
        conn.commit()
    finally:
        conn.close()
    emit_progress(final=True)
    summary = {
        "x_db_path": str(db_path),
        "discovered_files": total_files,
        "loaded_payloads": loaded,
        "kept_matches": kept,
        "duplicate_match_ids_skipped": duplicate_match_ids,
        "dropped_matches": int(sum(drop_reasons.values())),
        "drop_reasons": dict(drop_reasons),
        "folder_kept_counts": dict(sorted(folder_kept.items())),
        "total_column_count": len(xbuilder.schema_feature_columns()),
        "metadata_columns": [name for name, _ in xbuilder.METADATA_COLUMNS],
        "primary_model_feature_count": (
            len(xbuilder.PARTICIPANT_NUMERIC_FEATURES) * 5 * 2
            + len(xbuilder.PARTICIPANT_SHARE_FEATURES) * 5
            + len(xbuilder.PARTICIPANT_JUNGLE_ONLY_FEATURES) * 2
            + len(xbuilder.PARTICIPANT_DELTA_ONLY_FEATURES) * 5
            + sum(len(spec["positions"]) * 2 for spec in xbuilder.CHALLENGE_POSITION_FEATURES)
            + len(xbuilder.CHALLENGE_JUNGLE_FEATURES) * 2
            + sum(len(spec["positions"]) for spec in xbuilder.CHALLENGE_DELTA_ONLY_FEATURES)
            + len(xbuilder.TEAM_CONTEXT_COLUMNS)
            + len(xbuilder.OBJECTIVES_DTO_COLUMNS)
            + len(xbuilder.SOCIAL_CONTEXT_COLUMNS)
            + len(xbuilder.INFO_VARIABLE_COLUMNS)
        ),
        "secondary_analysis_feature_count": len(xbuilder.SECONDARY_ANALYSIS_COLUMNS),
        "reused_existing_db": False,
    }
    return summary


def compute_skill_rows_for_pipeline(
    pipeline_dir: Path,
    required_unique: int,
    min_mapped: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    participant_index_path = pipeline_dir / "participant_index_by_match.json"
    db_path = pipeline_dir / "player_ranks.sqlite3"
    participants_by_match = safe_read_json(participant_index_path)
    if not isinstance(participants_by_match, dict):
        raise RuntimeError(f"Invalid participant index format: {participant_index_path}")

    ranks_by_puuid = load_ranks_by_puuid(db_path)
    default_server = infer_server_for_pipeline(db_path) or pipeline_dir.name.split("_", 1)[0].upper()
    default_cutoffs = apex_lp_cutoffs_for_server(default_server)
    mapper_cache: dict[str, RankLpProbitMapper] = {}
    server_match_counts: Counter[str] = Counter()

    def mapper_for_server(server: str) -> RankLpProbitMapper:
        key = str(server).upper()
        mapper = mapper_cache.get(key)
        if mapper is None:
            mapper = RankLpProbitMapper(
                target_percentages=target_percentages_for_server(key),
                apex_lp_cutoffs=apex_lp_cutoffs_for_server(key),
                floor_epsilon_pct=0.01,
                ceil_epsilon_pct=0.01,
            )
            mapper_cache[key] = mapper
        return mapper

    skill_rows: list[dict[str, Any]] = []
    included_values: list[float] = []
    total_matches = 0
    excluded_not_exactly = 0
    excluded_low_mapped = 0

    for match_id in sorted(participants_by_match.keys()):
        raw_arr = participants_by_match.get(match_id)
        if not isinstance(raw_arr, list):
            continue
        total_matches += 1
        match_server = infer_server_for_match(match_id, default_server)
        server_match_counts[match_server] += 1
        mapper = mapper_for_server(match_server)
        unique_puuids = list(dict.fromkeys(str(x) for x in raw_arr if x))
        unique_count = len(unique_puuids)
        if unique_count != required_unique:
            excluded_not_exactly += 1
            continue
        z_values: list[float] = []
        for puuid in unique_puuids:
            rank_tuple = ranks_by_puuid.get(puuid)
            if rank_tuple is None:
                continue
            rank_name, lp = rank_tuple
            try:
                z = mapper.rank_lp_to_probit(rank_name, lp)
            except Exception:
                continue
            if math.isfinite(z):
                z_values.append(float(z))
        if len(z_values) < min_mapped:
            excluded_low_mapped += 1
            continue
        avg_skill = sum(z_values) / float(len(z_values))
        if not math.isfinite(avg_skill):
            continue
        included_values.append(avg_skill)
        skill_rows.append(
            {
                "match_id": match_id,
                "average_skill_level": avg_skill,
                "mapped_participants": len(z_values),
                "unique_participants": unique_count,
                "skill_server": match_server,
            }
        )

    stats = {
        "pipeline_dir": str(pipeline_dir),
        "server": default_server,
        "servers_seen": dict(sorted(server_match_counts.items())),
        "total_matches_seen": total_matches,
        "included_matches": len(skill_rows),
        "dropped_not_exactly_10": excluded_not_exactly,
        "dropped_mapped_lt_min": excluded_low_mapped,
        "gm_cutoff_lp": float(default_cutoffs["gm_cutoff_lp"]),
        "challenger_cutoff_lp": float(default_cutoffs["challenger_cutoff_lp"]),
        "rank1_lp_cap": float(default_cutoffs["rank1_lp"]),
        "avg_skill_stats": {
            "count": len(included_values),
            "min": (min(included_values) if included_values else None),
            "mean": (statistics.fmean(included_values) if included_values else None),
            "median": (statistics.median(included_values) if included_values else None),
            "max": (max(included_values) if included_values else None),
        },
    }
    return skill_rows, stats


def build_y_table(input_root: Path, pipeline_dirs: list[Path], output_dir: Path, required_unique: int, min_mapped: int) -> dict[str, Any]:
    y_output_dir = output_dir / "y_targets"
    y_output_dir.mkdir(parents=True, exist_ok=True)
    db_path = y_output_dir / "skill_targets.sqlite3"
    if db_path.exists():
        db_path.unlink()
    owner_summary = build_solution2_owner_db(input_root=input_root, output_dir=output_dir)
    x_db_path = output_dir / "x_features" / "match_feature_table_v1.sqlite3"
    compare_summary = compare_solution1_and_solution2(
        x_db_path=x_db_path,
        owner_db_path=Path(owner_summary["owner_db_path"]),
    )
    if not compare_summary["solutions_identical"]:
        raise RuntimeError(
            "Solution 1 and solution 2 do not agree on authoritative match ownership. "
            f"Summary: {json.dumps(compare_summary, ensure_ascii=False)}"
        )

    conn = sqlite3.connect(db_path)
    progress_path = output_dir / "build_progress.json"
    started_at = time.time()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_targets (
            match_id TEXT PRIMARY KEY,
            average_skill_level REAL NOT NULL,
            mapped_participants INTEGER NOT NULL,
            unique_participants INTEGER NOT NULL,
            skill_server TEXT NOT NULL
        )
        """
    )
    inserted = duplicate_rows = conflict_rows = 0
    conflict_examples: list[dict[str, Any]] = []
    pipeline_stats: list[dict[str, Any]] = []

    def emit_progress(*, current_pipeline: str | None = None, completed_pipelines: int = 0, final: bool = False) -> None:
        eta = build_eta_payload(started_at=started_at, done=completed_pipelines, total=len(pipeline_dirs))
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "y_build",
                "current_pipeline": current_pipeline,
                "completed_pipelines": completed_pipelines,
                "total_pipelines": len(pipeline_dirs),
                "inserted_rows": inserted,
                "duplicate_match_ids": duplicate_rows,
                "conflicting_duplicates": conflict_rows,
                "final": final,
                **eta,
            },
        )

    pipeline_map = {p.name: p for p in pipeline_dirs}
    x_conn = sqlite3.connect(x_db_path)
    try:
        source_folders = [
            str(row[0])
            for row in x_conn.execute(
                "SELECT DISTINCT source_folder FROM match_table_v1 ORDER BY source_folder"
            ).fetchall()
        ]
    finally:
        x_conn.close()

    try:
        emit_progress(completed_pipelines=0)
        for idx, source_folder in enumerate(source_folders, start=1):
            pipeline_dir = pipeline_map.get(source_folder)
            if pipeline_dir is None:
                raise RuntimeError(f"Missing pipeline dir for X source_folder={source_folder}")
            emit_progress(current_pipeline=str(pipeline_dir), completed_pipelines=idx - 1)
            rows, stats = compute_skill_rows_for_x_source_folder(
                source_folder=source_folder,
                pipeline_dir=pipeline_dir,
                x_db_path=x_db_path,
                required_unique=required_unique,
                min_mapped=min_mapped,
            )
            pipeline_stats.append(stats)
            for row in rows:
                existing = conn.execute(
                    "SELECT average_skill_level, mapped_participants, unique_participants, skill_server FROM skill_targets WHERE match_id = ?",
                    (row["match_id"],),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO skill_targets(match_id, average_skill_level, mapped_participants, unique_participants, skill_server) VALUES (?, ?, ?, ?, ?)",
                        (
                            row["match_id"],
                            float(row["average_skill_level"]),
                            int(row["mapped_participants"]),
                            int(row["unique_participants"]),
                            str(row["skill_server"]),
                        ),
                    )
                    inserted += 1
                    continue
                duplicate_rows += 1
                same = (
                    math.isclose(float(existing[0]), float(row["average_skill_level"]), rel_tol=0.0, abs_tol=1e-9)
                    and int(existing[1]) == int(row["mapped_participants"])
                    and int(existing[2]) == int(row["unique_participants"])
                    and str(existing[3]) == str(row["skill_server"])
                )
                if not same:
                    conflict_rows += 1
                    if len(conflict_examples) < 20:
                        conflict_examples.append(
                            {
                                "match_id": row["match_id"],
                                "existing": {
                                    "average_skill_level": float(existing[0]),
                                    "mapped_participants": int(existing[1]),
                                    "unique_participants": int(existing[2]),
                                    "skill_server": str(existing[3]),
                                },
                                "incoming": row,
                                "pipeline_dir": str(pipeline_dir),
                            }
                        )
            emit_progress(current_pipeline=str(pipeline_dir), completed_pipelines=idx)
        conn.commit()
    finally:
        conn.close()
    if conflict_rows > 0:
        raise RuntimeError(
            f"Found {conflict_rows} conflicting duplicate skill targets across solution-1 X-aligned folders. "
            f"Examples: {json.dumps(conflict_examples[:3], ensure_ascii=False)}"
        )
    return {
        "y_db_path": str(db_path),
        "inserted_rows": inserted,
        "duplicate_match_ids": duplicate_rows,
        "conflicting_duplicates": conflict_rows,
        "conflict_examples": conflict_examples,
        "pipeline_stats": pipeline_stats,
        "solution1_vs_solution2": compare_summary,
        "solution2_owner_build": owner_summary,
    }


def export_joined_dataset(output_dir: Path, skip_csv: bool) -> dict[str, Any]:
    x_db = output_dir / "x_features" / "match_feature_table_v1.sqlite3"
    y_db = output_dir / "y_targets" / "skill_targets.sqlite3"
    final_db = output_dir / "primary_dataset.sqlite3"
    table_name = "primary_dataset_v1"

    conn = sqlite3.connect(final_db)
    try:
        write_json(
            output_dir / "build_progress.json",
            {
                "updated_utc": utc_now_iso(),
                "phase": "join_export",
                "skip_csv": bool(skip_csv),
            },
        )
        conn.execute("ATTACH DATABASE ? AS xdb", (str(x_db),))
        conn.execute("ATTACH DATABASE ? AS ydb", (str(y_db),))
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} AS
            SELECT
                x.*,
                y.average_skill_level,
                y.mapped_participants,
                y.unique_participants,
                y.skill_server
            FROM xdb.match_table_v1 AS x
            INNER JOIN ydb.skill_targets AS y
                ON x.match_id = y.match_id
            """
        )
        conn.execute(f"CREATE UNIQUE INDEX idx_{table_name}_match_id ON {table_name}(match_id)")
        write_json(
            output_dir / "build_progress.json",
            {
                "updated_utc": utc_now_iso(),
                "phase": "null_audit",
                "skip_csv": bool(skip_csv),
            },
        )
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        x_count = conn.execute("SELECT COUNT(*) FROM xdb.match_table_v1").fetchone()[0]
        y_count = conn.execute("SELECT COUNT(*) FROM ydb.skill_targets").fetchone()[0]
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
        duplicate_count = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT match_id, COUNT(*) AS c FROM {table_name} GROUP BY match_id HAVING c > 1)"
        ).fetchone()[0]
        # Audit NULL presence in a single table scan instead of one scan per column.
        null_exprs = [
            f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "nulls__{idx}"'
            for idx, column in enumerate(columns)
        ]
        null_counts = conn.execute(
            f'SELECT {", ".join(null_exprs)} FROM {table_name}'
        ).fetchone()
        null_columns = [
            column for column, count in zip(columns, null_counts or ()) if int(count or 0) > 0
        ]
        if not skip_csv:
            csv_path = output_dir / "primary_dataset.csv"
            write_json(
                output_dir / "build_progress.json",
                {
                    "updated_utc": utc_now_iso(),
                    "phase": "csv_export",
                    "skip_csv": bool(skip_csv),
                    "final_row_count": int(row_count),
                },
            )
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(columns)
                cur = conn.execute(f"SELECT * FROM {table_name}")
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    writer.writerows(rows)
        else:
            csv_path = None
    finally:
        conn.close()

    return {
        "final_db_path": str(final_db),
        "final_table_name": table_name,
        "final_csv_path": (str(csv_path) if csv_path is not None else None),
        "x_row_count": int(x_count),
        "y_row_count": int(y_count),
        "final_row_count": int(row_count),
        "duplicate_match_id_rows": int(duplicate_count),
        "total_column_count": len(columns),
        "null_columns": null_columns,
    }


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dirs = discover_pipeline_dirs(input_root)
    if not pipeline_dirs:
        raise RuntimeError(f"No production pipeline folders found under: {input_root}")

    x_summary = build_x_table(
        input_root=input_root,
        output_dir=output_dir,
        reuse_existing=bool(args.reuse_existing_x),
    )
    y_db_path = output_dir / "y_targets" / "skill_targets.sqlite3"
    if bool(args.reuse_existing_y) and y_db_path.exists():
        y_summary = summarize_existing_y_table(y_db_path)
    else:
        y_summary = build_y_table(
            input_root=input_root,
            pipeline_dirs=pipeline_dirs,
            output_dir=output_dir,
            required_unique=int(args.required_unique_participants),
            min_mapped=int(args.min_mapped_participants),
        )
    join_summary = export_joined_dataset(output_dir=output_dir, skip_csv=bool(args.skip_csv))

    full_summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "pipeline_dirs": [str(p) for p in pipeline_dirs],
        "required_unique_participants": int(args.required_unique_participants),
        "min_mapped_participants": int(args.min_mapped_participants),
        "x_summary": x_summary,
        "y_summary": y_summary,
        "join_summary": join_summary,
    }
    (output_dir / "build_summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_json(
        output_dir / "build_progress.json",
        {
            "updated_utc": utc_now_iso(),
            "phase": "completed",
            "final_row_count": int(join_summary["final_row_count"]),
            "final_db_path": str(join_summary["final_db_path"]),
            "skip_csv": bool(args.skip_csv),
        },
    )
    text_lines = [
        f"pipeline_count={len(pipeline_dirs)}",
        f"x_kept_matches={x_summary['kept_matches']}",
        f"y_inserted_rows={y_summary['inserted_rows']}",
        f"y_duplicate_match_ids={y_summary['duplicate_match_ids']}",
        f"y_conflicting_duplicates={y_summary['conflicting_duplicates']}",
        f"final_row_count={join_summary['final_row_count']}",
        f"final_total_column_count={join_summary['total_column_count']}",
        f"final_duplicate_match_id_rows={join_summary['duplicate_match_id_rows']}",
        f"final_null_column_count={len(join_summary['null_columns'])}",
    ]
    (output_dir / "build_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"Built primary dataset at: {join_summary['final_db_path']}")
    print(f"rows={join_summary['final_row_count']} columns={join_summary['total_column_count']}")
    print(f"y_conflicting_duplicates={y_summary['conflicting_duplicates']}")
    print(f"null_columns={len(join_summary['null_columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
