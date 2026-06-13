from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from global_match_dedupe import (
    BundleInfo,
    ProgressTracker,
    cleanup_loser_json_files,
    delete_loser_match_ids,
    discover_bundles,
    json_health_for_match,
    open_conn,
    save_json,
)


def iter_matches_with_participants(conn: sqlite3.Connection):
    match_cur = conn.execute(
        """
        SELECT match_id, valid_for_pipeline, participant_count, game_creation_utc_ms
        FROM matches
        ORDER BY match_id
        """
    )
    part_cur = conn.execute(
        """
        SELECT match_id, puuid
        FROM match_participants
        ORDER BY match_id, puuid
        """
    )
    part_iter = iter(part_cur)
    current_part = next(part_iter, None)
    for row in match_cur:
        match_id = str(row["match_id"])
        puuids: set[str] = set()
        while current_part is not None and str(current_part["match_id"]) < match_id:
            current_part = next(part_iter, None)
        while current_part is not None and str(current_part["match_id"]) == match_id:
            puuid = str(current_part["puuid"] or "").strip()
            if puuid:
                puuids.add(puuid)
            current_part = next(part_iter, None)
        yield {
            "match_id": match_id,
            "valid_for_pipeline": int(row["valid_for_pipeline"] or 0),
            "participant_count": int(row["participant_count"] or 0),
            "game_creation_utc_ms": int(row["game_creation_utc_ms"] or 0),
            "participant_distinct_count": int(len(puuids)),
            "db_puuids": puuids,
        }


def evaluate_match(
    bundle: BundleInfo,
    row: dict[str, Any],
    *,
    require_valid_for_pipeline: bool,
    require_game_creation_positive: bool,
    min_participants: int,
) -> tuple[bool, str, dict[str, Any]]:
    checks: list[tuple[bool, str]] = []
    checks.append((int(row["participant_count"]) >= int(min_participants), "participant_count_lt_min"))
    checks.append(
        (int(row["participant_distinct_count"]) >= int(min_participants), "participant_distinct_lt_min")
    )
    if require_valid_for_pipeline:
        checks.append((int(row["valid_for_pipeline"]) == 1, "valid_for_pipeline_false"))
    if require_game_creation_positive:
        checks.append((int(row["game_creation_utc_ms"]) > 0, "game_creation_non_positive"))

    json_health = json_health_for_match(
        bundle.matches_dir,
        str(row["match_id"]),
        db_puuids=set(row["db_puuids"]),
        participant_count=int(row["participant_count"]),
        game_creation_utc_ms=int(row["game_creation_utc_ms"]),
    )

    for ok, reason in checks:
        if not ok:
            payload = dict(row)
            payload.update(json_health)
            return False, reason, payload

    if not bool(json_health["json_is_healthy"]):
        if not bool(json_health["json_has_file"]):
            reason = "json_missing"
        elif not bool(json_health["json_loaded"]):
            reason = "json_load_failed"
        elif not bool(json_health["json_match_id_ok"]):
            reason = "json_match_id_mismatch"
        elif not bool(json_health["json_participant_count_ok"]):
            reason = "json_participant_count_mismatch"
        elif not bool(json_health["json_puuid_set_ok"]):
            reason = "json_puuid_set_mismatch"
        elif not bool(json_health["json_game_creation_ok"]):
            reason = "json_game_creation_mismatch"
        else:
            reason = "json_unhealthy"
        payload = dict(row)
        payload.update(json_health)
        return False, reason, payload

    payload = dict(row)
    payload.update(json_health)
    return True, "", payload


def write_invalid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bundle",
                    "match_id",
                    "reason",
                    "valid_for_pipeline",
                    "participant_count",
                    "participant_distinct_count",
                    "game_creation_utc_ms",
                    "json_has_file",
                    "json_loaded",
                    "json_load_error",
                    "json_path",
                ]
            )
        return
    fieldnames = [
        "bundle",
        "match_id",
        "reason",
        "valid_for_pipeline",
        "participant_count",
        "participant_distinct_count",
        "game_creation_utc_ms",
        "json_has_file",
        "json_loaded",
        "json_load_error",
        "json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run(args: argparse.Namespace) -> int:
    inbox_root = Path(args.inbox_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(report_dir / "progress.txt")

    bundles, warnings = discover_bundles(inbox_root, args.glob)
    save_json(
        report_dir / "inventory.json",
        {
            "generated_utc": int(time.time()),
            "bundle_count": int(len(bundles)),
            "warnings": warnings,
            "bundles": [bundle.stack_name for bundle in bundles],
        },
    )

    total_matches = sum(int(b.db_matches_row_count) for b in bundles)
    checked = 0
    invalid_total = 0
    invalid_by_bundle: dict[str, list[str]] = {}
    invalid_rows_all: list[dict[str, Any]] = []
    bundle_summary_rows: list[dict[str, Any]] = []
    reason_counts_global: Counter[str] = Counter()

    for bundle in bundles:
        conn = open_conn(bundle.db_path)
        bundle_invalid_ids: list[str] = []
        bundle_invalid_rows: list[dict[str, Any]] = []
        reason_counts: Counter[str] = Counter()
        try:
            for row in iter_matches_with_participants(conn):
                ok, reason, payload = evaluate_match(
                    bundle,
                    row,
                    require_valid_for_pipeline=bool(args.require_valid_for_pipeline),
                    require_game_creation_positive=bool(args.require_game_creation_positive),
                    min_participants=int(args.min_participants),
                )
                checked += 1
                if not ok:
                    bundle_invalid_ids.append(str(row["match_id"]))
                    out_row = {
                        "bundle": bundle.stack_name,
                        "match_id": str(row["match_id"]),
                        "reason": reason,
                        **payload,
                    }
                    bundle_invalid_rows.append(out_row)
                    reason_counts[reason] += 1
                    reason_counts_global[reason] += 1
                if checked % 5000 == 0:
                    progress.update(
                        phase="audit",
                        checked=int(checked),
                        total=int(total_matches),
                        pct=round(100.0 * float(checked) / float(max(1, total_matches)), 2),
                        kept=int(checked - invalid_total),
                        skipped=int(invalid_total),
                    )
        finally:
            conn.close()

        invalid_total += len(bundle_invalid_ids)
        invalid_by_bundle[bundle.stack_name] = bundle_invalid_ids
        invalid_rows_all.extend(bundle_invalid_rows)
        bundle_summary_rows.append(
            {
                "bundle": bundle.stack_name,
                "total_matches": int(bundle.db_matches_row_count),
                "invalid_matches": int(len(bundle_invalid_ids)),
                "valid_matches": int(bundle.db_matches_row_count - len(bundle_invalid_ids)),
                "top_reason": reason_counts.most_common(1)[0][0] if reason_counts else "",
            }
        )
        write_invalid_csv(report_dir / f"invalid_{bundle.stack_name}.csv", bundle_invalid_rows)

    progress.update(
        phase="audit_complete",
        checked=int(checked),
        total=int(total_matches),
        pct=100.0,
        kept=int(total_matches - invalid_total),
        skipped=int(invalid_total),
    )

    summary = {
        "mode": str(args.mode),
        "generated_utc": int(time.time()),
        "bundle_count": int(len(bundles)),
        "total_matches": int(total_matches),
        "invalid_matches": int(invalid_total),
        "valid_matches": int(total_matches - invalid_total),
        "reason_counts": dict(reason_counts_global),
        "bundle_summaries": bundle_summary_rows,
    }
    save_json(report_dir / "audit_summary.json", summary)
    with (report_dir / "bundle_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bundle", "total_matches", "invalid_matches", "valid_matches", "top_reason"],
        )
        writer.writeheader()
        writer.writerows(bundle_summary_rows)
    write_invalid_csv(report_dir / "invalid_all.csv", invalid_rows_all)

    if args.mode == "dry-run":
        progress.update(
            phase="done",
            checked=int(total_matches),
            total=int(total_matches),
            pct=100.0,
            kept=int(total_matches - invalid_total),
            skipped=int(invalid_total),
            ok=True,
        )
        print(f"Dry-run complete. Invalid matches: {invalid_total}")
        print(f"Reports written to: {report_dir}")
        return 0

    db_delete_stats: list[dict[str, Any]] = []
    file_cleanup_stats: list[dict[str, Any]] = []
    apply_errors: list[str] = []
    bundle_map = {b.stack_name: b for b in bundles}
    to_mutate = {stack: mids for stack, mids in invalid_by_bundle.items() if mids}
    processed = 0
    total_to_delete = sum(len(v) for v in to_mutate.values())
    for stack, mids in to_mutate.items():
        bundle = bundle_map[stack]
        try:
            db_delete_stats.append(delete_loser_match_ids(bundle, mids))
            file_cleanup_stats.append(
                cleanup_loser_json_files(
                    bundle,
                    mids,
                    cleanup_mode=str(args.cleanup_json),
                    quarantine_dirname=str(args.json_quarantine_dirname),
                )
            )
        except Exception as exc:
            apply_errors.append(f"{stack}: {exc}")
            if not args.continue_on_error:
                break
        processed += len(mids)
        progress.update(
            phase="apply",
            checked=int(processed),
            total=int(total_to_delete),
            pct=round(100.0 * float(processed) / float(max(1, total_to_delete)), 2),
            kept=int(total_matches - invalid_total),
            skipped=int(invalid_total),
            errors=int(len(apply_errors)),
        )

    validation_errors = 0
    for stack, mids in to_mutate.items():
        bundle = bundle_map[stack]
        conn = open_conn(bundle.db_path)
        try:
            for mid in mids:
                c_matches = conn.execute("SELECT COUNT(*) FROM matches WHERE match_id = ?", (mid,)).fetchone()[0]
                c_parts = conn.execute(
                    "SELECT COUNT(*) FROM match_participants WHERE match_id = ?",
                    (mid,),
                ).fetchone()[0]
                if int(c_matches or 0) != 0 or int(c_parts or 0) != 0:
                    validation_errors += 1
        finally:
            conn.close()

    apply_result = {
        "mode": "apply",
        "generated_utc": int(time.time()),
        "bundle_count": int(len(bundles)),
        "total_matches": int(total_matches),
        "invalid_matches_deleted": int(invalid_total),
        "db_delete_stats": db_delete_stats,
        "file_cleanup_stats": file_cleanup_stats,
        "apply_errors": apply_errors,
        "validation_errors": int(validation_errors),
        "ok": not apply_errors and validation_errors == 0,
    }
    save_json(report_dir / "apply_result.json", apply_result)
    progress.update(
        phase="done",
        checked=int(total_to_delete),
        total=int(total_to_delete),
        pct=100.0,
        kept=int(total_matches - invalid_total),
        skipped=int(invalid_total),
        errors=int(len(apply_errors) + validation_errors),
        ok=bool(apply_result["ok"]),
    )
    print("Apply complete" if apply_result["ok"] else "Apply completed with errors")
    print(f"Reports written to: {report_dir}")
    return 0 if apply_result["ok"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit and optionally remove non-duplicate invalid match rows/files from bundle DBs."
    )
    p.add_argument("--inbox-root", type=str, default="runtime/out_prod")
    p.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    p.add_argument("--report-dir", type=str, required=True)
    p.add_argument("--glob", type=str, default="*_prod_*")
    p.add_argument(
        "--require-valid-for-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--require-game-creation-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--min-participants", type=int, default=10)
    p.add_argument("--cleanup-json", choices=["move", "delete", "none"], default="delete")
    p.add_argument("--json-quarantine-dirname", type=str, default="_invalid_match_quarantine")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
