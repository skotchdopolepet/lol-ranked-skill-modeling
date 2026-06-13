from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "dataset",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import build_match_feature_table_v1 as xbuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final primary dataset readiness and match-id uniqueness.")
    parser.add_argument("--out-prod", default=str(REPO_ROOT / "runtime" / "out_prod"))
    parser.add_argument("--out-prod-player", default=str(REPO_ROOT / "runtime" / "out_prod_player"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runtime" / "out_prod" / "primary_dataset"))
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def match_id_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(".json.zst"):
        return name[: -len(".json.zst")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def discover_match_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "matches").exists():
            out.append(child)
    return out


def scan_root_match_ids(
    *,
    root: Path,
    table_name: str,
    conn: sqlite3.Connection,
    progress_path: Path,
    roots_done: int,
    roots_total: int,
) -> dict[str, Any]:
    dirs = discover_match_dirs(root)
    inserted = 0
    duplicates = 0
    duplicate_examples: list[dict[str, str]] = []

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"""
        CREATE TABLE {table_name} (
            match_id TEXT PRIMARY KEY,
            source_folder TEXT NOT NULL,
            source_file TEXT NOT NULL
        )
        """
    )

    processed_dirs = 0
    for pipeline_dir in dirs:
        processed_dirs += 1
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "scan_match_roots",
                "current_root": str(root),
                "current_pipeline": str(pipeline_dir),
                "roots_done": roots_done,
                "roots_total": roots_total,
                "processed_dirs": processed_dirs,
                "total_dirs": len(dirs),
                "table_name": table_name,
                "inserted": inserted,
                "duplicates": duplicates,
            },
        )
        for _, match_path in xbuilder.iter_match_files(pipeline_dir):
            match_id = match_id_from_filename(match_path)
            cur = conn.execute(
                f"INSERT OR IGNORE INTO {table_name}(match_id, source_folder, source_file) VALUES (?, ?, ?)",
                (match_id, pipeline_dir.name, str(match_path)),
            )
            if cur.rowcount:
                inserted += 1
                continue
            duplicates += 1
            if len(duplicate_examples) < 20:
                existing = conn.execute(
                    f"SELECT source_folder, source_file FROM {table_name} WHERE match_id = ?",
                    (match_id,),
                ).fetchone()
                duplicate_examples.append(
                    {
                        "match_id": match_id,
                        "existing_source_folder": str(existing[0]) if existing else "",
                        "existing_source_file": str(existing[1]) if existing else "",
                        "duplicate_source_folder": pipeline_dir.name,
                        "duplicate_source_file": str(match_path),
                    }
                )
        conn.commit()

    return {
        "root": str(root),
        "pipeline_count": len(dirs),
        "match_id_count": inserted,
        "duplicate_match_id_count": duplicates,
        "duplicate_examples": duplicate_examples,
    }


def load_final_dataset_into_temp(*, final_db_path: Path, conn: sqlite3.Connection) -> dict[str, Any]:
    conn.execute("DROP TABLE IF EXISTS final_ids")
    conn.execute(
        """
        CREATE TABLE final_ids (
            match_id TEXT PRIMARY KEY
        )
        """
    )
    src = sqlite3.connect(f"file:{final_db_path}?mode=ro", uri=True)
    try:
        cur = src.execute("SELECT match_id FROM primary_dataset_v1")
        inserted = 0
        while True:
            rows = cur.fetchmany(10000)
            if not rows:
                break
            conn.executemany("INSERT INTO final_ids(match_id) VALUES (?)", rows)
            inserted += len(rows)
        conn.commit()
        duplicate_count = int(
            src.execute(
                "SELECT COUNT(*) FROM (SELECT match_id, COUNT(*) AS c FROM primary_dataset_v1 GROUP BY match_id HAVING c > 1)"
            ).fetchone()[0]
        )
        row_count = int(src.execute("SELECT COUNT(*) FROM primary_dataset_v1").fetchone()[0])
        columns = [row[1] for row in src.execute("PRAGMA table_info(primary_dataset_v1)").fetchall()]
        null_exprs = [f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END)' for column in columns]
        null_counts = src.execute(f"SELECT {', '.join(null_exprs)} FROM primary_dataset_v1").fetchone()
        null_columns = [column for column, count in zip(columns, null_counts or ()) if int(count or 0) > 0]
    finally:
        src.close()
    return {
        "row_count": row_count,
        "inserted_into_temp": inserted,
        "duplicate_match_id_count": duplicate_count,
        "column_count": len(columns),
        "null_columns": null_columns,
    }


def audit_csv(csv_path: Path, expected_columns: int, progress_path: Path) -> dict[str, Any]:
    started = time.time()
    row_count = 0
    bad_width_rows = 0
    bad_width_examples: list[dict[str, Any]] = []
    header: list[str] | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for idx, row in enumerate(reader):
            if idx == 0:
                header = list(row)
                continue
            row_count += 1
            if len(row) != expected_columns:
                bad_width_rows += 1
                if len(bad_width_examples) < 20:
                    bad_width_examples.append(
                        {
                            "csv_row_number": idx + 1,
                            "field_count": len(row),
                        }
                    )
            if row_count % 100000 == 0:
                write_json(
                    progress_path,
                    {
                        "updated_utc": utc_now_iso(),
                        "phase": "audit_csv",
                        "csv_path": str(csv_path),
                        "rows_seen": row_count,
                        "expected_columns": expected_columns,
                        "bad_width_rows": bad_width_rows,
                        "elapsed_sec": time.time() - started,
                    },
                )
    return {
        "csv_path": str(csv_path),
        "header_columns": len(header or []),
        "row_count": row_count,
        "expected_columns": expected_columns,
        "bad_width_row_count": bad_width_rows,
        "bad_width_examples": bad_width_examples,
        "elapsed_sec": time.time() - started,
    }


def fetch_overlap_examples(conn: sqlite3.Connection, left_table: str, right_table: str, limit: int = 20) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT l.match_id
            FROM {left_table} AS l
            INNER JOIN {right_table} AS r
                ON r.match_id = l.match_id
            ORDER BY l.match_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    ]


def main() -> int:
    args = parse_args()
    out_prod = Path(args.out_prod)
    out_prod_player = Path(args.out_prod_player)
    output_dir = Path(args.output_dir)
    final_db_path = output_dir / "primary_dataset.sqlite3"
    csv_path = output_dir / "primary_dataset.csv"
    x_db_path = output_dir / "x_features" / "match_feature_table_v1.sqlite3"
    y_db_path = output_dir / "y_targets" / "skill_targets.sqlite3"
    report_path = output_dir / "dataset_readiness_report.json"
    progress_path = output_dir / "dataset_readiness_progress.json"
    temp_db_path = output_dir / "dataset_readiness_temp.sqlite3"

    if temp_db_path.exists():
        temp_db_path.unlink()
    conn = sqlite3.connect(temp_db_path)
    started = time.time()
    try:
        out_prod_scan = scan_root_match_ids(
            root=out_prod,
            table_name="out_prod_ids",
            conn=conn,
            progress_path=progress_path,
            roots_done=0,
            roots_total=2,
        )
        out_prod_player_scan = scan_root_match_ids(
            root=out_prod_player,
            table_name="out_prod_player_ids",
            conn=conn,
            progress_path=progress_path,
            roots_done=1,
            roots_total=2,
        )
        final_scan = load_final_dataset_into_temp(final_db_path=final_db_path, conn=conn)

        x_conn = sqlite3.connect(f"file:{x_db_path}?mode=ro", uri=True)
        try:
            x_row_count = int(x_conn.execute("SELECT COUNT(*) FROM match_table_v1").fetchone()[0])
        finally:
            x_conn.close()
        y_conn = sqlite3.connect(f"file:{y_db_path}?mode=ro", uri=True)
        try:
            y_row_count = int(y_conn.execute("SELECT COUNT(*) FROM skill_targets").fetchone()[0])
        finally:
            y_conn.close()

        out_prod_vs_player_overlap = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM out_prod_ids AS a
                INNER JOIN out_prod_player_ids AS b
                    ON b.match_id = a.match_id
                """
            ).fetchone()[0]
        )
        final_vs_out_prod_overlap = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM final_ids AS f
                INNER JOIN out_prod_ids AS o
                    ON o.match_id = f.match_id
                """
            ).fetchone()[0]
        )
        final_vs_out_prod_player_overlap = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM final_ids AS f
                INNER JOIN out_prod_player_ids AS p
                    ON p.match_id = f.match_id
                """
            ).fetchone()[0]
        )
        final_missing_from_out_prod = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM final_ids AS f
                LEFT JOIN out_prod_ids AS o
                    ON o.match_id = f.match_id
                WHERE o.match_id IS NULL
                """
            ).fetchone()[0]
        )
        csv_audit = audit_csv(csv_path=csv_path, expected_columns=int(final_scan["column_count"]), progress_path=progress_path)

        report = {
            "out_prod": out_prod_scan,
            "out_prod_player": out_prod_player_scan,
            "final_dataset": {
                "final_db_path": str(final_db_path),
                "final_csv_path": str(csv_path),
                "x_db_path": str(x_db_path),
                "y_db_path": str(y_db_path),
                "x_row_count": x_row_count,
                "y_row_count": y_row_count,
                "final_row_count": int(final_scan["row_count"]),
                "final_match_ids_loaded": int(final_scan["inserted_into_temp"]),
                "column_count": int(final_scan["column_count"]),
                "duplicate_match_id_count": int(final_scan["duplicate_match_id_count"]),
                "null_columns": list(final_scan["null_columns"]),
                "has_only_complete_xy_rows": (
                    int(final_scan["row_count"]) == int(y_row_count) == 1183637
                ),
                "x_minus_final_rows": int(x_row_count) - int(final_scan["row_count"]),
            },
            "csv_audit": csv_audit,
            "overlaps": {
                "out_prod_vs_out_prod_player_overlap_count": out_prod_vs_player_overlap,
                "out_prod_vs_out_prod_player_overlap_examples": fetch_overlap_examples(conn, "out_prod_ids", "out_prod_player_ids"),
                "final_vs_out_prod_overlap_count": final_vs_out_prod_overlap,
                "final_vs_out_prod_player_overlap_count": final_vs_out_prod_player_overlap,
                "final_vs_out_prod_player_overlap_examples": fetch_overlap_examples(conn, "final_ids", "out_prod_player_ids"),
                "final_missing_from_out_prod_count": final_missing_from_out_prod,
            },
            "all_checks_pass": (
                out_prod_scan["duplicate_match_id_count"] == 0
                and out_prod_player_scan["duplicate_match_id_count"] == 0
                and out_prod_vs_player_overlap == 0
                and int(final_scan["row_count"]) == 1183637
                and int(final_scan["row_count"]) == int(y_row_count)
                and int(final_scan["duplicate_match_id_count"]) == 0
                and len(final_scan["null_columns"]) == 0
                and csv_audit["row_count"] == 1183637
                and csv_audit["header_columns"] == int(final_scan["column_count"])
                and csv_audit["bad_width_row_count"] == 0
                and final_vs_out_prod_overlap == 1183637
                and final_vs_out_prod_player_overlap == 0
                and final_missing_from_out_prod == 0
            ),
            "elapsed_sec": time.time() - started,
            "completed_utc": utc_now_iso(),
        }
        write_json(report_path, report)
        write_json(
            progress_path,
            {
                "updated_utc": utc_now_iso(),
                "phase": "completed",
                "report_path": str(report_path),
                "all_checks_pass": bool(report["all_checks_pass"]),
                "elapsed_sec": report["elapsed_sec"],
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
