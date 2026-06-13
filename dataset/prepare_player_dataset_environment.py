from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


ROUTING_BY_GROUP = {
    "BR1": ("BR1", "americas"),
    "EUN1": ("EUN1", "europe"),
    "EUW1": ("EUW1", "europe"),
    "JP1": ("JP1", "asia"),
    "KR": ("KR", "asia"),
    "LA1": ("LA1", "americas"),
    "LA2": ("LA2", "americas"),
    "NA1": ("NA1", "americas"),
    "SG2": ("SG2", "sea"),
    "TR1": ("TR1", "europe"),
    "TW2": ("TW2", "sea"),
    "VN2": ("VN2", "sea"),
}

SUPPORT_FILES = [
    "seed_players.json",
    "participant_index_by_match.json",
    "match_ids_by_puuid.json",
    "crawl_stats.json",
]

SOURCE_SPLIT_DB_NAME = "player_dataset_split.sqlite3"
ROOT_CONTROL_DB_NAME = "player_dataset_targets.sqlite3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare out_prod_player for player_dataset from selected secondary match manifests."
    )
    p.add_argument("--source-root", default="runtime/out_prod")
    p.add_argument(
        "--allocation-dir",
        default="runtime/out_prod/secondary_seed_allocation_top40_buffer20",
        help="Directory produced by plan_secondary_seed_allocation.py",
    )
    p.add_argument("--output-root", default="runtime/out_prod_player")
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output root first if it already exists.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "linked"
    except Exception:
        shutil.copy2(src, dst)
        return "copied"


def safe_move(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    shutil.move(str(src), str(dst))
    return "moved"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_source_match_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT match_id
            FROM matches
            WHERE match_id IS NOT NULL
            ORDER BY match_id
            """
        ).fetchall()
        return [str(row[0]) for row in rows if str(row[0] or "").strip()]
    finally:
        conn.close()


def build_root_control_db(
    output_root: Path,
    folder_rows: list[dict[str, str]],
    selected_rows_by_folder: dict[str, list[dict[str, str]]],
    match_assignments_by_folder: dict[str, list[dict[str, str]]],
) -> Path:
    db_path = output_root / ROOT_CONTROL_DB_NAME
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE folder_summary (
                folder TEXT PRIMARY KEY,
                group_prefix TEXT NOT NULL,
                total_matches INTEGER NOT NULL,
                dominant_matches INTEGER NOT NULL,
                non_dominant_matches INTEGER NOT NULL,
                secondary_selected INTEGER NOT NULL,
                primary_remaining INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE target_matches (
                folder TEXT NOT NULL,
                group_prefix TEXT NOT NULL,
                match_id TEXT NOT NULL,
                rank_bucket TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE match_assignments (
                folder TEXT NOT NULL,
                group_prefix TEXT NOT NULL,
                match_id TEXT NOT NULL,
                assignment TEXT NOT NULL,
                json_home TEXT NOT NULL,
                json_status TEXT NOT NULL,
                rank_bucket TEXT,
                updated_utc INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_target_matches_folder_match ON target_matches(folder, match_id)"
        )
        conn.execute("CREATE INDEX idx_target_matches_match ON target_matches(match_id)")
        conn.execute(
            "CREATE UNIQUE INDEX idx_match_assignments_folder_match ON match_assignments(folder, match_id)"
        )
        conn.execute("CREATE INDEX idx_match_assignments_match ON match_assignments(match_id)")
        conn.execute("CREATE INDEX idx_match_assignments_assignment ON match_assignments(assignment)")

        for row in folder_rows:
            actual_secondary = len(selected_rows_by_folder.get(row["folder"], []))
            actual_primary = int(row["total_matches"]) - actual_secondary
            conn.execute(
                """
                INSERT INTO folder_summary (
                    folder, group_prefix, total_matches, dominant_matches,
                    non_dominant_matches, secondary_selected, primary_remaining
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["folder"],
                    row["dominant_prefix"],
                    int(row["total_matches"]),
                    int(row["dominant_matches"]),
                    int(row["non_dominant_matches"]),
                    int(actual_secondary),
                    int(actual_primary),
                ),
            )

        for folder, rows in selected_rows_by_folder.items():
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO target_matches (folder, group_prefix, match_id, rank_bucket)
                    VALUES (?, ?, ?, ?)
                    """,
                    (folder, row["group_prefix"], row["match_id"], row["rank_bucket"]),
                )

        for folder, rows in match_assignments_by_folder.items():
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO match_assignments (
                        folder, group_prefix, match_id, assignment, json_home,
                        json_status, rank_bucket, updated_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        folder,
                        row["group_prefix"],
                        row["match_id"],
                        row["assignment"],
                        row["json_home"],
                        row["json_status"],
                        row["rank_bucket"],
                        int(row["updated_utc"]),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


def write_source_split_db(
    source_folder_dir: Path,
    folder_row: dict[str, str],
    match_assignment_rows: list[dict[str, str]],
) -> Path:
    db_path = source_folder_dir / SOURCE_SPLIT_DB_NAME
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE split_summary (
                folder TEXT PRIMARY KEY,
                group_prefix TEXT NOT NULL,
                total_matches INTEGER NOT NULL,
                dominant_matches INTEGER NOT NULL,
                non_dominant_matches INTEGER NOT NULL,
                secondary_selected INTEGER NOT NULL,
                primary_remaining INTEGER NOT NULL,
                updated_utc INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE match_split (
                match_id TEXT PRIMARY KEY,
                assignment TEXT NOT NULL,
                json_home TEXT NOT NULL,
                json_status TEXT NOT NULL,
                rank_bucket TEXT,
                updated_utc INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_match_split_assignment ON match_split(assignment)")
        conn.execute("CREATE INDEX idx_match_split_json_home ON match_split(json_home)")

        updated_utc = int(time.time())
        conn.execute(
            """
            INSERT INTO split_summary (
                folder, group_prefix, total_matches, dominant_matches,
                non_dominant_matches, secondary_selected, primary_remaining, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                folder_row["folder"],
                folder_row["dominant_prefix"],
                int(folder_row["total_matches"]),
                int(folder_row["dominant_matches"]),
                int(folder_row["non_dominant_matches"]),
                int(folder_row["secondary_selected"]),
                int(folder_row["primary_remaining"]),
                updated_utc,
            ),
        )
        conn.executemany(
            """
            INSERT INTO match_split (
                match_id, assignment, json_home, json_status, rank_bucket, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["match_id"],
                    row["assignment"],
                    row["json_home"],
                    row["json_status"],
                    row["rank_bucket"],
                    int(row["updated_utc"]),
                )
                for row in match_assignment_rows
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def dedupe_selected_rows_by_match(
    folder_rows: list[dict[str, str]],
    selected_rows_by_folder: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    owner_by_match: dict[str, tuple[str, dict[str, str]]] = {}
    # Favor the later folder in planning order so the assignment is deterministic.
    for row in folder_rows:
        folder = row["folder"]
        for target in selected_rows_by_folder.get(folder, []):
            owner_by_match[str(target["match_id"])] = (folder, target)
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _match_id, (folder, target) in owner_by_match.items():
        out[folder].append(target)
    for folder in out:
        out[folder].sort(key=lambda row: str(row["match_id"]))
    return dict(out)


def build_match_assignment_rows(
    *,
    folder: str,
    group_prefix: str,
    all_match_ids: list[str],
    target_rows: list[dict[str, str]],
    materialized_secondary_match_ids: set[str],
    missing_secondary_match_ids: set[str],
) -> list[dict[str, str]]:
    rank_bucket_by_match = {
        str(row["match_id"]): str(row["rank_bucket"])
        for row in target_rows
    }
    updated_utc = int(time.time())
    out: list[dict[str, str]] = []
    for match_id in all_match_ids:
        if match_id in materialized_secondary_match_ids:
            assignment = "secondary"
            json_home = "runtime/out_prod_player"
            json_status = "materialized_secondary"
        elif match_id in missing_secondary_match_ids:
            assignment = "secondary"
            json_home = "missing"
            json_status = "missing_secondary_json"
        else:
            assignment = "primary"
            json_home = "runtime/out_prod"
            json_status = "primary_in_source"
        out.append(
            {
                "folder": folder,
                "group_prefix": group_prefix,
                "match_id": match_id,
                "assignment": assignment,
                "json_home": json_home,
                "json_status": json_status,
                "rank_bucket": rank_bucket_by_match.get(match_id, ""),
                "updated_utc": updated_utc,
            }
        )
    return out


def write_run_script(folder_dir: Path, group_prefix: str, target_count: int) -> None:
    platform, regional = ROUTING_BY_GROUP[group_prefix]
    script = f"""$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'

python player_dataset.py `
  --out-dir "{folder_dir.as_posix()}" `
  --platform-routing {platform} `
  --regional-routing {regional} `
  --candidate-match-ids-file "{(folder_dir / 'player_dataset_targets.csv').as_posix()}" `
  --source-matches-dir "{(folder_dir / 'matches').as_posix()}" `
  --slice-match-count {int(target_count)} `
  --slice-seed 42
"""
    write_text(folder_dir / "run_player_dataset.ps1", script)


def write_folder_manifest(
    folder_dir: Path,
    row: dict[str, str],
    copied_db_size: int,
    json_counts: dict[str, int],
    split_db_path: Path,
    output_split_db_path: Path,
) -> None:
    payload = {
        "folder": row["folder"],
        "group_prefix": row["dominant_prefix"],
        "total_matches": int(row["total_matches"]),
        "dominant_matches": int(row["dominant_matches"]),
        "non_dominant_matches": int(row["non_dominant_matches"]),
        "secondary_selected": int(row["secondary_selected"]),
        "primary_remaining": int(row["primary_remaining"]),
        "copied_db_size_bytes": int(copied_db_size),
        "selected_match_jsons_moved": int(json_counts.get("moved", 0)),
        "selected_match_jsons_existing": int(json_counts.get("existing", 0)),
        "selected_match_jsons_missing": int(json_counts.get("missing", 0)),
        "source_split_db_path": str(split_db_path),
        "output_split_db_path": str(output_split_db_path),
    }
    write_text(folder_dir / "player_dataset_bundle.json", json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    allocation_dir = Path(args.allocation_dir)
    output_root = Path(args.output_root)

    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")
    if not allocation_dir.exists():
        raise SystemExit(f"Missing allocation dir: {allocation_dir}")

    if output_root.exists():
        if not args.clean_output:
            raise SystemExit(
                f"Output root already exists: {output_root}. Re-run with --clean-output to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    folder_rows = read_csv(allocation_dir / "folder_allocation.csv")
    selected_dir = allocation_dir / "selected_secondary"
    selected_rows_by_folder: dict[str, list[dict[str, str]]] = {}

    for folder_csv in sorted(selected_dir.glob("*.csv")):
        selected_rows_by_folder[folder_csv.stem] = read_csv(folder_csv)
    selected_rows_by_folder = dedupe_selected_rows_by_match(folder_rows, selected_rows_by_folder)

    match_assignments_by_folder: dict[str, list[dict[str, str]]] = {}
    summary_rows: list[dict[str, object]] = []

    for row in folder_rows:
        folder = row["folder"]
        group_prefix = row["dominant_prefix"]
        src_dir = source_root / folder
        dst_dir = output_root / folder
        dst_matches_dir = dst_dir / "matches"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_matches_dir.mkdir(parents=True, exist_ok=True)

        db_path = src_dir / "player_ranks.sqlite3"
        if not db_path.exists():
            raise RuntimeError(f"Missing player_ranks.sqlite3 for {folder}")
        safe_copy(db_path, dst_dir / "player_ranks.sqlite3")

        for name in SUPPORT_FILES:
            src_path = src_dir / name
            if src_path.exists():
                safe_link_or_copy(src_path, dst_dir / name)

        all_match_ids = load_source_match_ids(db_path)
        target_rows = selected_rows_by_folder.get(folder, [])
        actual_secondary = len(target_rows)
        actual_primary = int(row["total_matches"]) - actual_secondary

        with (dst_dir / "player_dataset_targets.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_prefix", "match_id", "rank_bucket"])
            writer.writeheader()
            writer.writerows(target_rows)

        json_counts: dict[str, int] = defaultdict(int)
        matches_dir = src_dir / "matches"
        materialized_secondary_match_ids: set[str] = set()
        missing_match_jsons: list[str] = []
        for target in target_rows:
            match_id = str(target["match_id"])
            src_json = matches_dir / f"{match_id}.json"
            src_zst = matches_dir / f"{match_id}.json.zst"
            if src_json.exists():
                mode = safe_move(src_json, dst_matches_dir / src_json.name)
                json_counts[mode] += 1
                materialized_secondary_match_ids.add(match_id)
            elif src_zst.exists():
                mode = safe_move(src_zst, dst_matches_dir / src_zst.name)
                json_counts[mode] += 1
                materialized_secondary_match_ids.add(match_id)
            else:
                json_counts["missing"] += 1
                missing_match_jsons.append(match_id)

        if missing_match_jsons:
            write_text(dst_dir / "missing_selected_match_jsons.txt", "\n".join(missing_match_jsons) + "\n")

        match_assignments = build_match_assignment_rows(
            folder=folder,
            group_prefix=group_prefix,
            all_match_ids=all_match_ids,
            target_rows=target_rows,
            materialized_secondary_match_ids=materialized_secondary_match_ids,
            missing_secondary_match_ids=set(missing_match_jsons),
        )
        match_assignments_by_folder[folder] = match_assignments

        row_for_manifest = dict(row)
        row_for_manifest["secondary_selected"] = str(actual_secondary)
        row_for_manifest["primary_remaining"] = str(actual_primary)
        split_db_path = write_source_split_db(src_dir, row_for_manifest, match_assignments)
        output_split_db_path = dst_dir / SOURCE_SPLIT_DB_NAME
        safe_copy(split_db_path, output_split_db_path)
        write_run_script(dst_dir, group_prefix, actual_secondary)
        write_folder_manifest(
            dst_dir,
            row_for_manifest,
            (dst_dir / "player_ranks.sqlite3").stat().st_size,
            json_counts,
            split_db_path,
            output_split_db_path,
        )

        summary_rows.append(
            {
                "folder": folder,
                "group_prefix": group_prefix,
                "secondary_selected": int(actual_secondary),
                "primary_remaining": int(actual_primary),
                "player_ranks_db_mb": round((dst_dir / "player_ranks.sqlite3").stat().st_size / (1024 * 1024), 2),
                "secondary_jsons_moved": int(json_counts.get("moved", 0)),
                "secondary_jsons_existing": int(json_counts.get("existing", 0)),
                "missing_selected_match_jsons": len(missing_match_jsons),
                "source_split_db": split_db_path.name,
            }
        )
        print(
            f"Prepared {folder}: targets={int(actual_secondary)}, "
            f"moved_jsons={int(json_counts.get('moved', 0))}, "
            f"existing_jsons={int(json_counts.get('existing', 0))}, "
            f"missing_jsons={len(missing_match_jsons)}"
        )

    root_db_path = build_root_control_db(
        output_root,
        folder_rows,
        selected_rows_by_folder,
        match_assignments_by_folder,
    )

    with (output_root / "folder_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "group_prefix",
                "secondary_selected",
                "primary_remaining",
                "player_ranks_db_mb",
                "secondary_jsons_moved",
                "secondary_jsons_existing",
                "missing_selected_match_jsons",
                "source_split_db",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = "\n".join(
        [
            "Player dataset secondary environment",
            f"source_root: {source_root}",
            f"allocation_dir: {allocation_dir}",
            f"output_root: {output_root}",
            f"target_db: {root_db_path.name}",
            "",
            "Per folder contents in output_root:",
            "- player_ranks.sqlite3 (copied)",
            f"- {SOURCE_SPLIT_DB_NAME} (copied from source folder split assignment DB)",
            "- seed_players.json / participant_index_by_match.json / match_ids_by_puuid.json / crawl_stats.json (linked or copied if present)",
            "- matches/ with moved selected secondary match JSONs",
            "- player_dataset_targets.csv",
            "- run_player_dataset.ps1",
            "- player_dataset_bundle.json",
            "",
            "Per folder contents in source_root:",
            f"- {SOURCE_SPLIT_DB_NAME} with one row per match_id and assignment=primary/secondary",
            "- primary match JSONs remain in source_root/matches",
            "- secondary match JSONs are moved to output_root/matches",
        ]
    )
    write_text(output_root / "README.txt", readme + "\n")
    print(f"\nPrepared player environment at: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
