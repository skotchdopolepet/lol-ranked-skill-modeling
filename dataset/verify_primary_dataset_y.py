from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "dataset",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_primary_dataset import compute_skill_rows_for_x_source_folder, discover_pipeline_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute Y targets for the final dataset and compare them exactly.")
    parser.add_argument("--input-root", default=str(REPO_ROOT / "runtime" / "out_prod"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runtime" / "out_prod" / "primary_dataset"))
    parser.add_argument("--required-unique-participants", type=int, default=10)
    parser.add_argument("--min-mapped-participants", type=int, default=9)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_final_rows_for_folder(final_db_path: Path, source_folder: str) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(final_db_path)
    try:
        rows = conn.execute(
            """
            SELECT match_id, average_skill_level, mapped_participants, unique_participants, skill_server
            FROM primary_dataset_v1
            WHERE source_folder = ?
            ORDER BY match_id
            """,
            (source_folder,),
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row[0]): {
            "average_skill_level": float(row[1]),
            "mapped_participants": int(row[2]),
            "unique_participants": int(row[3]),
            "skill_server": str(row[4]),
        }
        for row in rows
    }


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    final_db_path = output_dir / "primary_dataset.sqlite3"
    x_db_path = output_dir / "x_features" / "match_feature_table_v1.sqlite3"
    report_path = output_dir / "verify_y_report.json"
    progress_path = output_dir / "verify_y_progress.json"

    pipeline_dirs = discover_pipeline_dirs(input_root)
    pipeline_map = {p.name: p for p in pipeline_dirs}

    final_conn = sqlite3.connect(final_db_path)
    try:
        source_folders = [
            str(row[0])
            for row in final_conn.execute(
                "SELECT DISTINCT source_folder FROM primary_dataset_v1 ORDER BY source_folder"
            ).fetchall()
        ]
        final_row_count = int(final_conn.execute("SELECT COUNT(*) FROM primary_dataset_v1").fetchone()[0])
    finally:
        final_conn.close()

    started_at = time.time()
    per_folder: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    exact_float_mismatch_count = 0
    exact_float_match_count = 0
    tolerance_mismatch_count = 0
    max_abs_diff = 0.0
    compared_rows = 0
    missing_in_final = 0
    extra_in_final = 0
    mapped_mismatch = 0
    unique_mismatch = 0
    server_mismatch = 0
    final_server_counts: Counter[str] = Counter()
    recomputed_server_counts: Counter[str] = Counter()

    for idx, source_folder in enumerate(source_folders, start=1):
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "verify_y",
                "current_source_folder": source_folder,
                "completed_source_folders": idx - 1,
                "total_source_folders": len(source_folders),
                "compared_rows": compared_rows,
                "exact_float_match_count": exact_float_match_count,
                "exact_float_mismatch_count": exact_float_mismatch_count,
                "tolerance_mismatch_count": tolerance_mismatch_count,
                "missing_in_final": missing_in_final,
                "extra_in_final": extra_in_final,
                "mapped_mismatch": mapped_mismatch,
                "unique_mismatch": unique_mismatch,
                "server_mismatch": server_mismatch,
                "max_abs_diff": max_abs_diff,
                "elapsed_sec": time.time() - started_at,
            },
        )

        pipeline_dir = pipeline_map.get(source_folder)
        if pipeline_dir is None:
            raise RuntimeError(f"Missing pipeline dir for source_folder={source_folder}")

        recomputed_rows, stats = compute_skill_rows_for_x_source_folder(
            source_folder=source_folder,
            pipeline_dir=pipeline_dir,
            x_db_path=x_db_path,
            required_unique=int(args.required_unique_participants),
            min_mapped=int(args.min_mapped_participants),
        )
        recomputed = {str(row["match_id"]): row for row in recomputed_rows}
        final_rows = fetch_final_rows_for_folder(final_db_path, source_folder)

        recomputed_ids = set(recomputed)
        final_ids = set(final_rows)
        missing_ids = sorted(recomputed_ids - final_ids)
        extra_ids = sorted(final_ids - recomputed_ids)
        missing_in_final += len(missing_ids)
        extra_in_final += len(extra_ids)

        for mid in missing_ids[:5]:
            if len(mismatch_examples) >= 50:
                break
            mismatch_examples.append(
                {
                    "source_folder": source_folder,
                    "match_id": mid,
                    "issue": "missing_in_final",
                    "recomputed": recomputed[mid],
                }
            )
        for mid in extra_ids[:5]:
            if len(mismatch_examples) >= 50:
                break
            mismatch_examples.append(
                {
                    "source_folder": source_folder,
                    "match_id": mid,
                    "issue": "extra_in_final",
                    "final": final_rows[mid],
                }
            )

        shared_ids = sorted(recomputed_ids & final_ids)
        folder_exact_float_match_count = 0
        folder_exact_float_mismatch_count = 0
        folder_tolerance_mismatch_count = 0
        folder_mapped_mismatch = 0
        folder_unique_mismatch = 0
        folder_server_mismatch = 0
        folder_max_abs_diff = 0.0

        for match_id in shared_ids:
            final_row = final_rows[match_id]
            recomputed_row = recomputed[match_id]
            compared_rows += 1
            final_server_counts[str(final_row["skill_server"])] += 1
            recomputed_server_counts[str(recomputed_row["skill_server"])] += 1

            final_value = float(final_row["average_skill_level"])
            recomputed_value = float(recomputed_row["average_skill_level"])
            abs_diff = abs(final_value - recomputed_value)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
            if abs_diff > folder_max_abs_diff:
                folder_max_abs_diff = abs_diff

            if final_value == recomputed_value:
                exact_float_match_count += 1
                folder_exact_float_match_count += 1
            else:
                exact_float_mismatch_count += 1
                folder_exact_float_mismatch_count += 1

            if not math.isclose(final_value, recomputed_value, rel_tol=0.0, abs_tol=1e-12):
                tolerance_mismatch_count += 1
                folder_tolerance_mismatch_count += 1
                if len(mismatch_examples) < 50:
                    mismatch_examples.append(
                        {
                            "source_folder": source_folder,
                            "match_id": match_id,
                            "issue": "average_skill_level_mismatch",
                            "final_value": final_value,
                            "recomputed_value": recomputed_value,
                            "abs_diff": abs_diff,
                        }
                    )

            if int(final_row["mapped_participants"]) != int(recomputed_row["mapped_participants"]):
                mapped_mismatch += 1
                folder_mapped_mismatch += 1
                if len(mismatch_examples) < 50:
                    mismatch_examples.append(
                        {
                            "source_folder": source_folder,
                            "match_id": match_id,
                            "issue": "mapped_participants_mismatch",
                            "final_value": int(final_row["mapped_participants"]),
                            "recomputed_value": int(recomputed_row["mapped_participants"]),
                        }
                    )

            if int(final_row["unique_participants"]) != int(recomputed_row["unique_participants"]):
                unique_mismatch += 1
                folder_unique_mismatch += 1
                if len(mismatch_examples) < 50:
                    mismatch_examples.append(
                        {
                            "source_folder": source_folder,
                            "match_id": match_id,
                            "issue": "unique_participants_mismatch",
                            "final_value": int(final_row["unique_participants"]),
                            "recomputed_value": int(recomputed_row["unique_participants"]),
                        }
                    )

            if str(final_row["skill_server"]) != str(recomputed_row["skill_server"]):
                server_mismatch += 1
                folder_server_mismatch += 1
                if len(mismatch_examples) < 50:
                    mismatch_examples.append(
                        {
                            "source_folder": source_folder,
                            "match_id": match_id,
                            "issue": "skill_server_mismatch",
                            "final_value": str(final_row["skill_server"]),
                            "recomputed_value": str(recomputed_row["skill_server"]),
                        }
                    )

        per_folder.append(
            {
                "source_folder": source_folder,
                "pipeline_dir": str(pipeline_dir),
                "final_rows": len(final_rows),
                "recomputed_rows": len(recomputed),
                "shared_rows": len(shared_ids),
                "missing_in_final": len(missing_ids),
                "extra_in_final": len(extra_ids),
                "exact_float_match_count": folder_exact_float_match_count,
                "exact_float_mismatch_count": folder_exact_float_mismatch_count,
                "tolerance_mismatch_count": folder_tolerance_mismatch_count,
                "mapped_mismatch": folder_mapped_mismatch,
                "unique_mismatch": folder_unique_mismatch,
                "server_mismatch": folder_server_mismatch,
                "max_abs_diff": folder_max_abs_diff,
                "recompute_stats": stats,
            }
        )

    report = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "final_db_path": str(final_db_path),
        "x_db_path": str(x_db_path),
        "required_unique_participants": int(args.required_unique_participants),
        "min_mapped_participants": int(args.min_mapped_participants),
        "final_row_count": final_row_count,
        "compared_rows": compared_rows,
        "exact_float_match_count": exact_float_match_count,
        "exact_float_mismatch_count": exact_float_mismatch_count,
        "tolerance_abs_tol": 1e-12,
        "tolerance_mismatch_count": tolerance_mismatch_count,
        "mapped_participants_mismatch_count": mapped_mismatch,
        "unique_participants_mismatch_count": unique_mismatch,
        "skill_server_mismatch_count": server_mismatch,
        "missing_in_final_count": missing_in_final,
        "extra_in_final_count": extra_in_final,
        "max_abs_diff": max_abs_diff,
        "all_rows_verified_exact": (
            compared_rows == final_row_count
            and exact_float_mismatch_count == 0
            and mapped_mismatch == 0
            and unique_mismatch == 0
            and server_mismatch == 0
            and missing_in_final == 0
            and extra_in_final == 0
        ),
        "final_server_counts": dict(sorted(final_server_counts.items())),
        "recomputed_server_counts": dict(sorted(recomputed_server_counts.items())),
        "mismatch_examples": mismatch_examples,
        "per_folder": per_folder,
        "elapsed_sec": time.time() - started_at,
        "completed_utc": utc_now_iso(),
    }
    write_json(report_path, report)
    write_json(
        progress_path,
        {
            "updated_utc": utc_now_iso(),
            "phase": "completed",
            "report_path": str(report_path),
            "all_rows_verified_exact": bool(report["all_rows_verified_exact"]),
            "compared_rows": compared_rows,
            "elapsed_sec": report["elapsed_sec"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
