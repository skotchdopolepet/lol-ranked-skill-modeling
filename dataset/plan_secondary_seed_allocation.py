from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HOURS_IN_4_WEEKS = 24 * 28
DEFAULT_WINDOW = "top40"
DEFAULT_BUFFER = 0.20
API_SPEED_DIVISOR = 35.0
TOP_REPORT_NAME = "top30_active_hours_report.md"

TIER_ORDER = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]
DIVISION_ORDER = ["IV", "III", "II", "I"]


@dataclass(frozen=True)
class FolderInfo:
    name: str
    path: Path
    db_path: Path
    normalized_name: str
    dominant_prefix: str
    total_matches: int
    seed_rank_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan a grouped-server player_dataset secondary allocation from deduped out_prod data."
    )
    p.add_argument("--root-dir", type=str, default="runtime/out_prod")
    p.add_argument("--report-md", type=str, default="")
    p.add_argument("--window", choices=["top20", "top30", "top40", "top50", "top60"], default=DEFAULT_WINDOW)
    p.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def normalize_folder_name(folder_name: str) -> str:
    return re.sub(r"_[0-9a-f]{8}$", "", folder_name)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_bucket_sort_key(bucket: str) -> tuple[int, int, str]:
    if bucket == "UNKNOWN":
        return (999, 999, bucket)
    if "_" not in bucket:
        try:
            return (TIER_ORDER.index(bucket), 999, bucket)
        except ValueError:
            return (998, 999, bucket)
    tier, division = bucket.split("_", 1)
    try:
        tier_idx = TIER_ORDER.index(tier)
    except ValueError:
        tier_idx = 998
    try:
        div_idx = DIVISION_ORDER.index(division)
    except ValueError:
        div_idx = 998
    return (tier_idx, div_idx, bucket)


def normalize_seed_bucket(row: dict[str, Any]) -> str | None:
    explicit = row.get("seed_rank_bucket")
    if explicit:
        return str(explicit).upper()
    tier = str(row.get("source_tier") or row.get("solo_tier") or "").upper().strip()
    rank = str(row.get("source_rank") or row.get("solo_rank") or "").upper().strip()
    if not tier:
        return None
    if tier in {"MASTER", "GRANDMASTER", "CHALLENGER"}:
        return tier
    if rank in DIVISION_ORDER:
        return f"{tier}_{rank}"
    return tier


def load_seed_rank_map(seed_players_path: Path) -> dict[str, str]:
    if not seed_players_path.exists():
        return {}
    try:
        raw = json.loads(seed_players_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        puuid = str(row.get("puuid") or "").strip()
        if not puuid:
            continue
        bucket = normalize_seed_bucket(row)
        if bucket:
            out[puuid] = bucket
    return out


def parse_top_report(report_path: Path) -> dict[str, dict[str, float]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    in_table = False
    out: dict[str, dict[str, float]] = {}
    for line in lines:
        if line.startswith(
            "| Rank | Crawler | Bucket | overall_mph | top20_avg_mph | top30_avg_mph | top40_avg_mph | top50_avg_mph | top60_avg_mph |"
        ):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("| ---"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 9:
            continue
        _, crawler, bucket, overall, top20, top30, top40, top50, top60 = parts
        out[crawler] = {
            "bucket": bucket,
            "overall": float(overall),
            "top20": float(top20),
            "top30": float(top30),
            "top40": float(top40),
            "top50": float(top50),
            "top60": float(top60),
        }
    if not out:
        raise SystemExit(f"Could not parse peak-window throughput table from {report_path}")
    return out


def open_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def detect_dominant_prefix(db_path: Path) -> tuple[str, int]:
    conn = open_conn(db_path)
    try:
        row = conn.execute(
            """
            SELECT substr(match_id, 1, instr(match_id, '_') - 1) AS prefix, COUNT(*) AS c
            FROM matches
            GROUP BY prefix
            ORDER BY c DESC, prefix ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise RuntimeError(f"No matches in {db_path}")
        return str(row["prefix"]), int(row["c"] or 0)
    finally:
        conn.close()


def count_total_matches(db_path: Path) -> int:
    conn = open_conn(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM matches").fetchone()
        return int(row["c"] or 0)
    finally:
        conn.close()


def discover_folders(root_dir: Path) -> list[FolderInfo]:
    folders: list[FolderInfo] = []
    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        db_path = folder / "player_ranks.sqlite3"
        if not db_path.exists():
            continue
        total_matches = count_total_matches(db_path)
        dominant_prefix, _dominant_count = detect_dominant_prefix(db_path)
        folders.append(
            FolderInfo(
                name=folder.name,
                path=folder,
                db_path=db_path,
                normalized_name=normalize_folder_name(folder.name),
                dominant_prefix=dominant_prefix,
                total_matches=total_matches,
                seed_rank_map=load_seed_rank_map(folder / "seed_players.json"),
            )
        )
    if not folders:
        raise SystemExit(f"No player_ranks.sqlite3 folders found under {root_dir}")
    return folders


def folder_target_secondary(folder: FolderInfo, throughput_map: dict[str, dict[str, float]], window: str, buffer: float) -> int:
    mph_main = float(throughput_map[folder.normalized_name][window])
    mph_player = mph_main / API_SPEED_DIVISOR
    target = mph_player * HOURS_IN_4_WEEKS * (1.0 + buffer)
    return int(round(target))


def iter_dominant_match_buckets(folder: FolderInfo) -> tuple[tuple[str, str], ...]:
    conn = open_conn(folder.db_path)
    try:
        cur = conn.execute(
            """
            SELECT m.match_id, mp.puuid
            FROM matches m
            JOIN match_participants mp ON mp.match_id = m.match_id
            WHERE substr(m.match_id, 1, instr(m.match_id, '_') - 1) = ?
            ORDER BY m.match_id, mp.puuid
            """,
            (folder.dominant_prefix,),
        )
        current_match_id: str | None = None
        participants: list[str] = []
        rows: list[tuple[str, str]] = []
        for row in cur:
            match_id = str(row["match_id"])
            puuid = str(row["puuid"])
            if current_match_id is None:
                current_match_id = match_id
            if match_id != current_match_id:
                rows.append((current_match_id, bucket_from_participants(participants, folder.seed_rank_map)))
                current_match_id = match_id
                participants = []
            participants.append(puuid)
        if current_match_id is not None:
            rows.append((current_match_id, bucket_from_participants(participants, folder.seed_rank_map)))
        return tuple(rows)
    finally:
        conn.close()


def bucket_from_participants(participants: list[str], seed_rank_map: dict[str, str]) -> str:
    buckets = [seed_rank_map[p] for p in participants if p in seed_rank_map]
    if not buckets:
        return "UNKNOWN"
    counts = Counter(buckets)
    max_count = max(counts.values())
    return sorted((bucket for bucket, count in counts.items() if count == max_count), key=rank_bucket_sort_key)[0]


def apportion_targets(total_target: int, bucket_counts: dict[str, int]) -> dict[str, int]:
    if total_target <= 0 or not bucket_counts:
        return {bucket: 0 for bucket in bucket_counts}
    total_available = sum(bucket_counts.values())
    if total_available <= 0:
        return {bucket: 0 for bucket in bucket_counts}
    effective_target = min(int(total_target), int(total_available))
    raw_targets = {
        bucket: (count / total_available) * effective_target for bucket, count in bucket_counts.items()
    }
    base = {
        bucket: min(bucket_counts[bucket], int(raw_targets[bucket]))
        for bucket in bucket_counts
    }
    remaining = effective_target - sum(base.values())
    fractions = sorted(
        (
            raw_targets[bucket] - base[bucket],
            bucket,
        )
        for bucket in bucket_counts
        if base[bucket] < bucket_counts[bucket]
    )
    fractions.sort(key=lambda item: (-item[0], rank_bucket_sort_key(item[1])))
    for _frac, bucket in fractions:
        if remaining <= 0:
            break
        if base[bucket] < bucket_counts[bucket]:
            base[bucket] += 1
            remaining -= 1
    return base


def match_hash(match_id: str) -> int:
    digest = hashlib.sha256(match_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    report_path = Path(args.report_md) if args.report_md else root_dir / "fleet_throughput_analysis_db" / TOP_REPORT_NAME
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root_dir / f"secondary_seed_allocation_{args.window}_buffer{int(round(args.buffer * 100))}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    throughput_map = parse_top_report(report_path)
    folders = discover_folders(root_dir)

    group_to_folders: dict[str, list[FolderInfo]] = defaultdict(list)
    for folder in folders:
        group_to_folders[folder.dominant_prefix].append(folder)

    folder_match_buckets: dict[str, tuple[tuple[str, str], ...]] = {}
    folder_rows: list[dict[str, Any]] = []
    group_bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_targets: dict[str, int] = defaultdict(int)
    group_dominant_totals: dict[str, int] = defaultdict(int)
    group_unknown_totals: dict[str, int] = defaultdict(int)
    folder_ranked_counts: dict[str, Counter[str]] = defaultdict(Counter)
    folder_unknown_counts: dict[str, int] = defaultdict(int)

    for folder in folders:
        match_buckets = iter_dominant_match_buckets(folder)
        folder_match_buckets[folder.name] = match_buckets
        dominant_count = len(match_buckets)
        non_dominant_count = max(0, folder.total_matches - dominant_count)
        target = folder_target_secondary(folder, throughput_map, args.window, args.buffer)
        group_targets[folder.dominant_prefix] += min(target, dominant_count)
        group_dominant_totals[folder.dominant_prefix] += dominant_count
        for _match_id, bucket in match_buckets:
            if bucket == "UNKNOWN":
                folder_unknown_counts[folder.name] += 1
                group_unknown_totals[folder.dominant_prefix] += 1
            else:
                folder_ranked_counts[folder.name][bucket] += 1
                group_bucket_counts[folder.dominant_prefix][bucket] += 1
        folder_rows.append(
            {
                "folder": folder.name,
                "normalized_name": folder.normalized_name,
                "dominant_prefix": folder.dominant_prefix,
                "total_matches": folder.total_matches,
                "dominant_matches": dominant_count,
                "non_dominant_matches": non_dominant_count,
                "ranked_dominant_matches": sum(folder_ranked_counts[folder.name].values()),
                "unknown_seed_matches": folder_unknown_counts[folder.name],
                "secondary_target_cap": min(target, dominant_count),
                "planning_window": args.window,
                "buffer_pct": int(round(args.buffer * 100)),
            }
        )

    group_bucket_targets: dict[str, dict[str, int]] = {}
    for group, bucket_counts in group_bucket_counts.items():
        ranked_total = sum(bucket_counts.values())
        effective_target = min(group_targets[group], ranked_total)
        group_targets[group] = effective_target
        group_bucket_targets[group] = apportion_targets(effective_target, dict(bucket_counts))

    heaps: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    for group, targets in group_bucket_targets.items():
        for bucket, target in targets.items():
            if target > 0:
                heaps[(group, bucket)] = []

    for folder in folders:
        group = folder.dominant_prefix
        for match_id, bucket in folder_match_buckets[folder.name]:
            if bucket == "UNKNOWN":
                continue
            target = group_bucket_targets.get(group, {}).get(bucket, 0)
            if target <= 0:
                continue
            heap = heaps[(group, bucket)]
            h = match_hash(match_id)
            entry = (-h, match_id, folder.name)
            if len(heap) < target:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    selected_by_folder: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_bucket_selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (group, bucket), heap in heaps.items():
        items = [(-neg_hash, match_id, folder_name) for neg_hash, match_id, folder_name in heap]
        items.sort(key=lambda item: (item[0], item[1], item[2]))
        for _hash_value, match_id, folder_name in items:
            selected_by_folder[folder_name].append(
                {
                    "match_id": match_id,
                    "group_prefix": group,
                    "rank_bucket": bucket,
                }
            )
            group_bucket_selected_counts[group][bucket] += 1

    selected_dir = output_dir / "selected_secondary"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for folder_name, rows in selected_by_folder.items():
        rows.sort(key=lambda row: (rank_bucket_sort_key(row["rank_bucket"]), row["match_id"]))
        write_csv(selected_dir / f"{folder_name}.csv", rows)

    folder_summary_rows: list[dict[str, Any]] = []
    for row in folder_rows:
        folder_name = str(row["folder"])
        secondary_selected = len(selected_by_folder.get(folder_name, []))
        primary_remaining = int(row["total_matches"]) - secondary_selected
        folder_summary_rows.append(
            {
                **row,
                "secondary_selected": secondary_selected,
                "primary_remaining": primary_remaining,
                "primary_non_dominant_matches": int(row["non_dominant_matches"]),
                "primary_unknown_seed_matches": int(row["unknown_seed_matches"]),
                "primary_ranked_unselected": int(row["ranked_dominant_matches"]) - secondary_selected,
            }
        )

    group_summary_rows: list[dict[str, Any]] = []
    bucket_summary_rows: list[dict[str, Any]] = []
    for group in sorted(group_to_folders):
        ranked_total = sum(group_bucket_counts[group].values())
        selected_total = sum(group_bucket_selected_counts[group].values())
        dominant_total = group_dominant_totals[group]
        target_total = group_targets[group]
        group_summary_rows.append(
            {
                "group_prefix": group,
                "folders": ",".join(f.name for f in sorted(group_to_folders[group], key=lambda f: f.name)),
                "dominant_total": dominant_total,
                "non_dominant_total": sum(
                    max(0, f.total_matches - len(folder_match_buckets[f.name])) for f in group_to_folders[group]
                ),
                "ranked_dominant_total": ranked_total,
                "unknown_seed_total": group_unknown_totals[group],
                "secondary_target": target_total,
                "secondary_selected": selected_total,
                "primary_remaining": dominant_total + sum(
                    max(0, f.total_matches - len(folder_match_buckets[f.name])) for f in group_to_folders[group]
                )
                - selected_total,
            }
        )
        for bucket in sorted(group_bucket_counts[group], key=rank_bucket_sort_key):
            bucket_summary_rows.append(
                {
                    "group_prefix": group,
                    "rank_bucket": bucket,
                    "available_ranked_matches": group_bucket_counts[group][bucket],
                    "target_secondary_matches": group_bucket_targets[group].get(bucket, 0),
                    "selected_secondary_matches": group_bucket_selected_counts[group].get(bucket, 0),
                }
            )

    folder_summary_rows.sort(key=lambda row: row["folder"])
    group_summary_rows.sort(key=lambda row: row["group_prefix"])
    bucket_summary_rows.sort(key=lambda row: (row["group_prefix"], rank_bucket_sort_key(str(row["rank_bucket"]))))

    write_csv(output_dir / "folder_allocation.csv", folder_summary_rows)
    write_csv(output_dir / "group_allocation.csv", group_summary_rows)
    write_csv(output_dir / "group_bucket_curve.csv", bucket_summary_rows)

    total_matches = sum(folder.total_matches for folder in folders)
    total_secondary = sum(len(rows) for rows in selected_by_folder.values())
    summary = {
        "root_dir": str(root_dir),
        "report_md": str(report_path),
        "window": args.window,
        "buffer_pct": int(round(args.buffer * 100)),
        "hours_in_4_weeks": HOURS_IN_4_WEEKS,
        "api_speed_divisor": API_SPEED_DIVISOR,
        "total_matches": total_matches,
        "secondary_selected_total": total_secondary,
        "primary_remaining_total": total_matches - total_secondary,
        "dominant_only_total": sum(group_dominant_totals.values()),
        "non_dominant_primary_total": total_matches - sum(group_dominant_totals.values()),
        "unknown_seed_primary_total": sum(group_unknown_totals.values()),
        "group_count": len(group_to_folders),
        "folder_count": len(folders),
    }
    save_json(output_dir / "summary.json", summary)

    md_lines = [
        "# Secondary Seed Allocation Plan",
        "",
        f"- root: `{root_dir}`",
        f"- peak window: `{args.window}`",
        f"- buffer: `{int(round(args.buffer * 100))}%`",
        f"- total matches: `{total_matches}`",
        f"- secondary selected: `{total_secondary}`",
        f"- primary remaining: `{total_matches - total_secondary}`",
        f"- dominant-only matches considered for secondary: `{sum(group_dominant_totals.values())}`",
        f"- non-dominant matches kept on primary: `{total_matches - sum(group_dominant_totals.values())}`",
        f"- unknown-seed dominant matches kept on primary: `{sum(group_unknown_totals.values())}`",
        "",
        "## Group Summary",
        "",
        "| Group | Folders | Dominant Total | Ranked Dominant | Unknown Seed | Secondary Target | Secondary Selected | Primary Remaining |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_summary_rows:
        md_lines.append(
            f"| {row['group_prefix']} | {row['folders']} | {row['dominant_total']} | {row['ranked_dominant_total']} | {row['unknown_seed_total']} | {row['secondary_target']} | {row['secondary_selected']} | {row['primary_remaining']} |"
        )
    md_lines.extend(
        [
            "",
            "## Folder Summary",
            "",
            "| Folder | Group | Total | Dominant | Non-dominant Primary | Unknown-seed Primary | Secondary Selected | Primary Remaining |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in folder_summary_rows:
        md_lines.append(
            f"| {row['folder']} | {row['dominant_prefix']} | {row['total_matches']} | {row['dominant_matches']} | {row['primary_non_dominant_matches']} | {row['primary_unknown_seed_matches']} | {row['secondary_selected']} | {row['primary_remaining']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Secondary allocation plan written to: {output_dir}")
    print(f"Secondary selected total: {total_secondary}")
    print(f"Primary remaining total: {total_matches - total_secondary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
