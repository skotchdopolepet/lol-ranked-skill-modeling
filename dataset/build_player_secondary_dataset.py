from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "dataset",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import build_match_feature_table_v1 as xbuilder
from build_primary_dataset import (
    build_eta_payload,
    build_y_table,
    match_id_from_filename,
    summarize_existing_x_table,
    summarize_existing_y_table,
    utc_now_iso,
    write_json,
)


DEFAULT_INPUT_ROOT = REPO_ROOT / "runtime" / "out_prod_player"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "secondary_dataset"
ROOT_CONTROL_DB_NAME = "player_dataset_targets.sqlite3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final out_prod_player secondary-analysis dataset (X + y, no nightly_score yet)."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--required-unique-participants", type=int, default=10)
    parser.add_argument("--min-mapped-participants", type=int, default=9)
    parser.add_argument("--reuse-existing-x", action="store_true")
    parser.add_argument("--reuse-existing-y", action="store_true")
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument(
        "--allow-missing-target-jsons",
        action="store_true",
        help="Allow the build to continue even if some target match_ids are missing materialized match JSONs.",
    )
    return parser.parse_args()


def load_target_match_ids(input_root: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    control_db = input_root / ROOT_CONTROL_DB_NAME
    target_ids_by_folder: dict[str, set[str]] = defaultdict(set)
    source = "per_folder_csv"

    if control_db.exists():
        conn = sqlite3.connect(str(control_db))
        try:
            rows = conn.execute(
                "SELECT folder, match_id FROM target_matches ORDER BY folder, match_id"
            ).fetchall()
        finally:
            conn.close()
        if rows:
            source = "root_control_db"
            for folder, match_id in rows:
                if folder and match_id:
                    target_ids_by_folder[str(folder)].add(str(match_id))

    if not target_ids_by_folder:
        for child in sorted(input_root.iterdir()):
            csv_path = child / "player_dataset_targets.csv"
            if not child.is_dir() or not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    match_id = str(row.get("match_id") or "").strip()
                    if match_id:
                        target_ids_by_folder[child.name].add(match_id)

    out = {folder: ids for folder, ids in sorted(target_ids_by_folder.items()) if ids}
    summary = {
        "target_source": source,
        "folder_count": len(out),
        "target_match_count": int(sum(len(ids) for ids in out.values())),
        "folder_target_counts": {folder: len(ids) for folder, ids in out.items()},
        "control_db_path": str(control_db),
    }
    return out, summary


def discover_pipeline_dirs(input_root: Path, target_ids_by_folder: dict[str, set[str]]) -> list[Path]:
    out: list[Path] = []
    missing: list[str] = []
    for folder in sorted(target_ids_by_folder):
        pipeline_dir = input_root / folder
        if not pipeline_dir.is_dir():
            missing.append(folder)
            continue
        required = [
            pipeline_dir / "matches",
            pipeline_dir / "player_ranks.sqlite3",
            pipeline_dir / "participant_index_by_match.json",
        ]
        if not all(path.exists() for path in required):
            missing.append(folder)
            continue
        out.append(pipeline_dir)
    if missing:
        raise RuntimeError(f"Missing required out_prod_player pipeline folders/files: {missing[:10]}")
    return out


def export_joined_dataset(output_dir: Path, skip_csv: bool) -> dict[str, Any]:
    x_db = output_dir / "x_features" / "match_feature_table_v1.sqlite3"
    y_db = output_dir / "y_targets" / "skill_targets.sqlite3"
    final_db = output_dir / "player_secondary_dataset.sqlite3"
    table_name = "player_secondary_dataset_v1"

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
        null_exprs = [
            f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "nulls__{idx}"'
            for idx, column in enumerate(columns)
        ]
        null_counts = conn.execute(f'SELECT {", ".join(null_exprs)} FROM {table_name}').fetchone()
        null_columns = [
            column for column, count in zip(columns, null_counts or ()) if int(count or 0) > 0
        ]
        if not skip_csv:
            csv_path = output_dir / "player_secondary_dataset.csv"
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


def build_x_table(
    *,
    input_root: Path,
    output_dir: Path,
    target_ids_by_folder: dict[str, set[str]],
    reuse_existing: bool = False,
    allow_missing_target_jsons: bool = False,
) -> dict[str, Any]:
    x_output_dir = output_dir / "x_features"
    x_output_dir.mkdir(parents=True, exist_ok=True)
    db_path = x_output_dir / "match_feature_table_v1.sqlite3"
    if reuse_existing and db_path.exists():
        summary = summarize_existing_x_table(db_path)
        summary["target_match_count"] = int(sum(len(ids) for ids in target_ids_by_folder.values()))
        summary["reused_existing_db"] = True
        return summary

    conn = xbuilder.create_db(db_path)
    kept = duplicate_match_ids = loaded = ignored_non_target_files = 0
    duplicate_materialized_target_files = 0
    drop_reasons: Counter[str] = Counter()
    folder_kept: Counter[str] = Counter()
    target_file_counts: Counter[str] = Counter()
    seen_target_ids_by_folder: dict[str, set[str]] = defaultdict(set)
    duplicate_examples: list[dict[str, str]] = []
    total_target_ids = int(sum(len(ids) for ids in target_ids_by_folder.values()))
    progress_path = output_dir / "build_progress.json"
    started_at = time.time()
    processed = 0
    last_progress_write = 0.0

    def emit_progress(final: bool = False) -> None:
        nonlocal last_progress_write
        if not final and (time.time() - last_progress_write) < 20.0 and processed % 2000 != 0:
            return
        eta = build_eta_payload(started_at=started_at, done=processed, total=total_target_ids)
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "x_build",
                "processed_target_files": processed,
                "total_target_ids": total_target_ids,
                "loaded_payloads": loaded,
                "kept_matches": kept,
                "duplicate_match_ids_skipped": duplicate_match_ids,
                "ignored_non_target_files": ignored_non_target_files,
                "duplicate_materialized_target_files": duplicate_materialized_target_files,
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
            allowed_ids = target_ids_by_folder.get(source_folder)
            if not allowed_ids:
                ignored_non_target_files += 1
                continue
            match_id = match_id_from_filename(path)
            if match_id not in allowed_ids:
                ignored_non_target_files += 1
                continue
            if match_id in seen_target_ids_by_folder[source_folder]:
                duplicate_materialized_target_files += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append({"source_folder": source_folder, "match_id": match_id, "source_file": path.name})
                continue
            seen_target_ids_by_folder[source_folder].add(match_id)
            target_file_counts[source_folder] += 1
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

    missing_target_json_ids = {
        folder: sorted(ids - seen_target_ids_by_folder.get(folder, set()))
        for folder, ids in sorted(target_ids_by_folder.items())
        if ids - seen_target_ids_by_folder.get(folder, set())
    }
    missing_target_json_count = int(sum(len(ids) for ids in missing_target_json_ids.values()))
    if missing_target_json_count > 0 and not allow_missing_target_jsons:
        raise RuntimeError(
            "Some target match_ids are missing materialized JSON files under runtime/out_prod_player. "
            f"Missing count={missing_target_json_count}. Examples={json.dumps({k: v[:5] for k, v in list(missing_target_json_ids.items())[:3]}, ensure_ascii=False)}"
        )

    summary = {
        "x_db_path": str(db_path),
        "target_match_count": total_target_ids,
        "materialized_target_match_files": processed,
        "missing_target_json_count": missing_target_json_count,
        "missing_target_json_examples": {
            folder: ids[:10] for folder, ids in list(missing_target_json_ids.items())[:10]
        },
        "loaded_payloads": loaded,
        "kept_matches": kept,
        "duplicate_match_ids_skipped": duplicate_match_ids,
        "duplicate_materialized_target_files": duplicate_materialized_target_files,
        "duplicate_materialized_target_examples": duplicate_examples,
        "ignored_non_target_files": ignored_non_target_files,
        "dropped_matches": int(sum(drop_reasons.values())),
        "drop_reasons": dict(drop_reasons),
        "folder_kept_counts": dict(sorted(folder_kept.items())),
        "folder_target_file_counts": dict(sorted(target_file_counts.items())),
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


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_ids_by_folder, target_summary = load_target_match_ids(input_root)
    if not target_ids_by_folder:
        raise RuntimeError(f"No target match_ids found under: {input_root}")

    pipeline_dirs = discover_pipeline_dirs(input_root, target_ids_by_folder)
    x_summary = build_x_table(
        input_root=input_root,
        output_dir=output_dir,
        target_ids_by_folder=target_ids_by_folder,
        reuse_existing=bool(args.reuse_existing_x),
        allow_missing_target_jsons=bool(args.allow_missing_target_jsons),
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

    final_row_count = int(join_summary["final_row_count"])
    expected_target_rows = int(target_summary["target_match_count"])
    full_summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "pipeline_dirs": [str(p) for p in pipeline_dirs],
        "required_unique_participants": int(args.required_unique_participants),
        "min_mapped_participants": int(args.min_mapped_participants),
        "target_summary": target_summary,
        "x_summary": x_summary,
        "y_summary": y_summary,
        "join_summary": join_summary,
        "nightly_score_status": {
            "present": False,
            "column_name_reserved_for_future": "nightly_score",
            "note": "Planned post-build enrichment keyed by match_id.",
        },
        "row_count_vs_target": {
            "expected_target_rows": expected_target_rows,
            "final_row_count": final_row_count,
            "matches_expected_target_count": final_row_count == expected_target_rows,
        },
    }
    (output_dir / "build_summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_json(
        output_dir / "build_progress.json",
        {
            "updated_utc": utc_now_iso(),
            "phase": "completed",
            "final_row_count": final_row_count,
            "expected_target_rows": expected_target_rows,
            "matches_expected_target_count": final_row_count == expected_target_rows,
            "final_db_path": str(join_summary["final_db_path"]),
            "skip_csv": bool(args.skip_csv),
        },
    )
    text_lines = [
        f"pipeline_count={len(pipeline_dirs)}",
        f"target_match_count={expected_target_rows}",
        f"x_kept_matches={x_summary['kept_matches']}",
        f"y_inserted_rows={y_summary['inserted_rows']}",
        f"y_duplicate_match_ids={y_summary['duplicate_match_ids']}",
        f"y_conflicting_duplicates={y_summary['conflicting_duplicates']}",
        f"final_row_count={final_row_count}",
        f"final_matches_expected_target_count={int(final_row_count == expected_target_rows)}",
        f"final_total_column_count={join_summary['total_column_count']}",
        f"final_duplicate_match_id_rows={join_summary['duplicate_match_id_rows']}",
        f"final_null_column_count={len(join_summary['null_columns'])}",
    ]
    (output_dir / "build_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"Built player secondary dataset at: {join_summary['final_db_path']}")
    print(f"rows={final_row_count} columns={join_summary['total_column_count']}")
    print(f"expected_target_rows={expected_target_rows}")
    print(f"matches_expected_target_count={final_row_count == expected_target_rows}")
    print(f"null_columns={len(join_summary['null_columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
