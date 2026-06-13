from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional at runtime
    zstd = None


@dataclass(frozen=True)
class BundleInfo:
    stack_name: str
    path: Path
    db_path: Path
    matches_dir: Path
    meta_path: Path
    db_matches_row_count: int
    stack_parts: dict[str, str] | None
    meta: dict[str, Any]


class ProgressTracker:
    def __init__(self, path: Path) -> None:
        self.path = path

    def update(self, **payload: Any) -> None:
        snapshot = {"updated_utc": int(time.time()), **payload}
        simple_lines = [
            f"phase={snapshot.get('phase', '')}",
            f"updated_utc={snapshot.get('updated_utc', '')}",
        ]
        simple_keys = [
            "checked",
            "total",
            "pct",
            "kept",
            "skipped",
            "errors",
            "ok",
        ]
        for key in simple_keys:
            if key in snapshot:
                simple_lines.append(f"{key}={snapshot[key]}")
        self.path.write_text("\n".join(simple_lines) + "\n", encoding="utf-8")


def bundle_provenance(bundle: BundleInfo) -> dict[str, Any]:
    parts = bundle.stack_parts or {}
    meta = bundle.meta if isinstance(bundle.meta, dict) else {}
    return {
        "stack_name": bundle.stack_name,
        "bundle_path": str(bundle.path),
        "db_path": str(bundle.db_path),
        "matches_dir": str(bundle.matches_dir),
        "region": parts.get("region", ""),
        "vm": parts.get("vm", ""),
        "instance": parts.get("instance", ""),
        "crawler": parts.get("crawler", ""),
        "keyns": parts.get("keyns", ""),
        "snapshot_utc": parts.get("snapshot_utc", ""),
        "meta_service_name": str(meta.get("service_name", "") or meta.get("service", "") or ""),
        "meta_keyns": str(meta.get("keyns", "") or meta.get("api_key_fingerprint", "") or ""),
        "meta_region": str(meta.get("region", "") or ""),
        "meta_vm": str(meta.get("vm", "") or ""),
        "meta_instance": str(meta.get("instance", "") or ""),
        "meta_crawler": str(meta.get("crawler", "") or ""),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def infer_region_from_crawler(crawler: str) -> str:
    lowered = str(crawler).lower()
    if lowered.startswith("eune"):
        return "eune"
    if lowered.startswith("euw"):
        return "euw"
    if lowered.startswith("la1"):
        return "la1"
    if lowered.startswith("la2"):
        return "la2"
    if lowered.startswith("na"):
        return "na"
    if lowered.startswith("br"):
        return "br"
    if lowered.startswith("kr"):
        return "kr"
    if lowered.startswith("jp"):
        return "jp"
    if lowered.startswith("vn"):
        return "vn"
    if lowered.startswith("tw"):
        return "tw"
    if lowered.startswith("sg"):
        return "sg"
    if lowered.startswith("tr"):
        return "tr"
    return lowered


def parse_stack_name(stack_name: str) -> dict[str, str] | None:
    parts = stack_name.split("__")
    if len(parts) == 6:
        keys = ["region", "vm", "instance", "crawler", "keyns", "snapshot_utc"]
        return {k: str(v) for k, v in zip(keys, parts)}
    if len(parts) == 3:
        region, crawler, keyns = parts
        return {
            "region": str(region),
            "vm": "",
            "instance": "",
            "crawler": str(crawler),
            "keyns": str(keyns),
            "snapshot_utc": "",
        }
    prod_match = re.fullmatch(r"(?P<crawler>[a-z0-9_]+)_prod_(?P<keyns>[0-9a-f]{8})", stack_name, flags=re.IGNORECASE)
    if prod_match:
        crawler = str(prod_match.group("crawler"))
        return {
            "region": infer_region_from_crawler(crawler),
            "vm": "",
            "instance": "",
            "crawler": crawler,
            "keyns": str(prod_match.group("keyns")).lower(),
            "snapshot_utc": "",
        }
    return None


def open_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def safe_scalar(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> int:
    try:
        row = conn.execute(query, params).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)
    except Exception:
        return 0


def bundle_db_match_count(db_path: Path) -> int:
    conn = open_conn(db_path)
    try:
        return safe_scalar(conn, "SELECT COUNT(*) FROM matches")
    finally:
        conn.close()


def discover_bundles(inbox_root: Path, glob_pattern: str) -> tuple[list[BundleInfo], list[str]]:
    warnings: list[str] = []
    bundles: list[BundleInfo] = []
    if not inbox_root.exists():
        raise SystemExit(f"Inbox root does not exist: {inbox_root}")
    for bundle_dir in sorted([p for p in inbox_root.glob(glob_pattern) if p.is_dir()]):
        db_path = bundle_dir / "player_ranks.sqlite3"
        if not db_path.exists():
            warnings.append(f"Skipping {bundle_dir}: missing player_ranks.sqlite3")
            continue
        matches_dir = bundle_dir / "matches"
        if not matches_dir.exists():
            warnings.append(f"{bundle_dir}: missing matches/ directory")
        stack_name = bundle_dir.name
        stack_parts = parse_stack_name(stack_name)
        if stack_parts is None:
            warnings.append(
                f"{bundle_dir}: stack name does not match expected pattern "
                "<region>__<crawler>__<keyns> or "
                "<region>__<vm>__<instance>__<crawler>__<keyns>__<snapshot_utc> or "
                "<crawler>_prod_<keyns>"
            )
        meta_path = bundle_dir / "bundle_meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        bundles.append(
            BundleInfo(
                stack_name=stack_name,
                path=bundle_dir,
                db_path=db_path,
                matches_dir=matches_dir,
                meta_path=meta_path,
                db_matches_row_count=bundle_db_match_count(db_path),
                stack_parts=stack_parts,
                meta=meta,
            )
        )
    if not bundles:
        raise SystemExit(f"No bundles discovered under {inbox_root} with glob={glob_pattern!r}")
    return bundles, warnings


def build_occurrence_index(
    bundles: list[BundleInfo],
    work_db_path: Path,
    *,
    only_match_ids: set[str] | None = None,
    progress: ProgressTracker | None = None,
) -> tuple[int, int]:
    if work_db_path.exists():
        work_db_path.unlink()
    work_db_path.parent.mkdir(parents=True, exist_ok=True)
    work = sqlite3.connect(str(work_db_path))
    try:
        work.execute("PRAGMA journal_mode=WAL")
        work.execute("CREATE TABLE occurrences (match_id TEXT NOT NULL, bundle TEXT NOT NULL)")
        work.execute("CREATE INDEX idx_occ_match_id ON occurrences(match_id)")
        work.execute("CREATE INDEX idx_occ_bundle ON occurrences(bundle)")
        inserted_rows = 0
        total_rows_target = sum(
            b.db_matches_row_count if only_match_ids is None else 0
            for b in bundles
        )
        processed_bundles = 0
        for bundle in bundles:
            src = open_conn(bundle.db_path)
            try:
                cur = src.execute("SELECT match_id FROM matches")
                batch: list[tuple[str, str]] = []
                for row in cur:
                    match_id = str(row["match_id"])
                    if only_match_ids is not None and match_id not in only_match_ids:
                        continue
                    batch.append((match_id, bundle.stack_name))
                    if len(batch) >= 5000:
                        work.executemany("INSERT INTO occurrences(match_id, bundle) VALUES (?, ?)", batch)
                        inserted_rows += len(batch)
                        batch.clear()
                if batch:
                    work.executemany("INSERT INTO occurrences(match_id, bundle) VALUES (?, ?)", batch)
                    inserted_rows += len(batch)
            finally:
                src.close()
            processed_bundles += 1
            if progress is not None:
                denom = max(1, int(total_rows_target))
                progress.update(
                    phase="indexing",
                    checked=int(inserted_rows),
                    total=int(total_rows_target),
                    pct=round(100.0 * float(inserted_rows) / float(denom), 2),
                )
        work.commit()
        dup_groups = int(
            work.execute(
                "SELECT COUNT(*) FROM (SELECT match_id FROM occurrences GROUP BY match_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )
        if progress is not None:
            progress.update(
                phase="indexing_complete",
                checked=int(inserted_rows),
                total=int(inserted_rows),
                pct=100.0,
            )
        return inserted_rows, dup_groups
    finally:
        work.close()


def find_duplicate_match_ids(work_db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(work_db_path))
    try:
        rows = conn.execute(
            "SELECT match_id FROM occurrences GROUP BY match_id HAVING COUNT(*) > 1 ORDER BY match_id"
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def bundles_for_match(work_db_path: Path, match_id: str) -> list[str]:
    conn = sqlite3.connect(str(work_db_path))
    try:
        rows = conn.execute(
            "SELECT bundle FROM occurrences WHERE match_id = ? ORDER BY bundle",
            (match_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def json_file_presence(matches_dir: Path, match_id: str) -> dict[str, bool]:
    return {
        "json_zst_exists": (matches_dir / f"{match_id}.json.zst").exists(),
        "json_exists": (matches_dir / f"{match_id}.json").exists(),
    }


def load_match_payload(matches_dir: Path, match_id: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    candidates = [matches_dir / f"{match_id}.json.zst", matches_dir / f"{match_id}.json"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".zst":
                if zstd is None:
                    return None, str(candidate), "zstandard_unavailable"
                raw = zstd.ZstdDecompressor().stream_reader(candidate.open("rb")).read()
                payload = json.loads(raw.decode("utf-8"))
            else:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, str(candidate), None
            return None, str(candidate), "payload_not_object"
        except Exception as exc:
            return None, str(candidate), str(exc)
    return None, None, None


def json_health_for_match(
    matches_dir: Path,
    match_id: str,
    *,
    db_puuids: set[str],
    participant_count: int,
    game_creation_utc_ms: int,
) -> dict[str, Any]:
    presence = json_file_presence(matches_dir, match_id)
    payload, json_path, load_error = load_match_payload(matches_dir, match_id)
    has_any_file = bool(presence["json_zst_exists"] or presence["json_exists"])
    if not has_any_file:
        return {
            **presence,
            "json_path": "",
            "json_checked": False,
            "json_has_file": False,
            "json_loaded": False,
            "json_load_error": "json_missing",
            "json_match_id_ok": None,
            "json_participant_count_ok": None,
            "json_puuid_set_ok": None,
            "json_game_creation_ok": None,
            "json_is_healthy": False,
        }
    if payload is None:
        return {
            **presence,
            "json_path": str(json_path or ""),
            "json_checked": True,
            "json_has_file": True,
            "json_loaded": False,
            "json_load_error": str(load_error or "unknown_json_load_error"),
            "json_match_id_ok": False,
            "json_participant_count_ok": False,
            "json_puuid_set_ok": False,
            "json_game_creation_ok": False,
            "json_is_healthy": False,
        }

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    info = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
    info_participants = info.get("participants", [])
    if not isinstance(info_participants, list):
        info_participants = []
    json_puuids = {
        str(p.get("puuid"))
        for p in info_participants
        if isinstance(p, dict) and str(p.get("puuid") or "").strip()
    }
    json_match_id_ok = str(metadata.get("matchId") or "") == str(match_id)
    json_participant_count_ok = len(info_participants) == int(participant_count)
    json_puuid_set_ok = json_puuids == set(db_puuids)
    json_game_creation_ok = int(info.get("gameCreation") or 0) == int(game_creation_utc_ms)
    json_is_healthy = all(
        [
            json_match_id_ok,
            json_participant_count_ok,
            json_puuid_set_ok,
            json_game_creation_ok,
        ]
    )
    return {
        **presence,
        "json_path": str(json_path or ""),
        "json_checked": True,
        "json_has_file": True,
        "json_loaded": True,
        "json_load_error": "",
        "json_match_id_ok": bool(json_match_id_ok),
        "json_participant_count_ok": bool(json_participant_count_ok),
        "json_puuid_set_ok": bool(json_puuid_set_ok),
        "json_game_creation_ok": bool(json_game_creation_ok),
        "json_is_healthy": bool(json_is_healthy),
    }


def quality_for_match(
    bundle: BundleInfo,
    match_id: str,
    *,
    require_valid_for_pipeline: bool,
    require_game_creation_positive: bool,
    min_participants: int,
) -> dict[str, Any]:
    conn = open_conn(bundle.db_path)
    try:
        row = conn.execute(
            """
            SELECT
                m.valid_for_pipeline AS valid_for_pipeline,
                m.participant_count AS participant_count,
                m.game_creation_utc_ms AS game_creation_utc_ms,
                (
                    SELECT COUNT(DISTINCT mp.puuid)
                    FROM match_participants mp
                    WHERE mp.match_id = m.match_id
                ) AS participant_distinct_count
            FROM matches m
            WHERE m.match_id = ?
            """,
            (match_id,),
        ).fetchone()
        valid_for_pipeline = int(row["valid_for_pipeline"] or 0) if row else 0
        participant_count = int(row["participant_count"] or 0) if row else 0
        game_creation_utc_ms = int(row["game_creation_utc_ms"] or 0) if row else 0
        participant_distinct_count = int(row["participant_distinct_count"] or 0) if row else 0
        puuid_rows = conn.execute(
            "SELECT DISTINCT puuid FROM match_participants WHERE match_id = ?",
            (match_id,),
        ).fetchall()
        db_puuids = {str(r["puuid"]) for r in puuid_rows if str(r["puuid"] or "").strip()}
        has_game_creation = bool(game_creation_utc_ms > 0)
        checks = [
            participant_count >= int(min_participants),
            participant_distinct_count >= int(min_participants),
        ]
        if require_valid_for_pipeline:
            checks.append(valid_for_pipeline == 1)
        if require_game_creation_positive:
            checks.append(has_game_creation)
        json_health = json_health_for_match(
            bundle.matches_dir,
            match_id,
            db_puuids=db_puuids,
            participant_count=participant_count,
            game_creation_utc_ms=game_creation_utc_ms,
        )
        is_complete = bool(all(checks) and bool(json_health["json_is_healthy"]))
        return {
            "match_id": match_id,
            "bundle": bundle.stack_name,
            "bundle_db_matches_row_count": int(bundle.db_matches_row_count),
            "valid_for_pipeline": int(valid_for_pipeline),
            "participant_count": int(participant_count),
            "participant_distinct_count": int(participant_distinct_count),
            "game_creation_utc_ms": int(game_creation_utc_ms),
            "has_game_creation": bool(has_game_creation),
            "is_complete": bool(is_complete),
            **bundle_provenance(bundle),
            **json_health,
        }
    finally:
        conn.close()


def winner_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    capped_participant_distinct_count = min(10, int(candidate["participant_distinct_count"]))
    return (
        -int(bool(candidate["is_complete"])),
        -int(candidate["valid_for_pipeline"]),
        -int(capped_participant_distinct_count),
        -int(bool(candidate["has_game_creation"])),
        int(candidate["bundle_db_matches_row_count"]),
        str(candidate["bundle_path"]),
    )


def winner_primary_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    capped_participant_distinct_count = min(10, int(candidate["participant_distinct_count"]))
    return (
        -int(bool(candidate["is_complete"])),
        -int(candidate["valid_for_pipeline"]),
        -int(capped_participant_distinct_count),
        -int(bool(candidate["has_game_creation"])),
    )


def choose_tw_weighted_winner(match_id: str, tied_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    weights = {
        "tw2": 35,
        "tw3": 35,
        "tw": 30,
    }
    eligible: list[dict[str, Any]] = []
    total_weight = 0
    for candidate in tied_candidates:
        if str(candidate.get("region", "")).lower() != "tw":
            return None
        crawler = str(candidate.get("crawler", "")).lower()
        weight = int(weights.get(crawler, 0))
        if weight <= 0:
            return None
        eligible.append(candidate)
        total_weight += weight
    if not eligible or total_weight <= 0:
        return None

    digest = hashlib.sha256(str(match_id).encode("utf-8")).digest()
    pick = int.from_bytes(digest[:8], "big") % total_weight
    running = 0
    for candidate in sorted(eligible, key=winner_sort_key):
        running += int(weights[str(candidate.get("crawler", "")).lower()])
        if pick < running:
            return candidate
    return sorted(eligible, key=winner_sort_key)[0]


def order_candidates_for_match(match_id: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=winner_sort_key)
    if not ordered:
        return ordered
    top_key = winner_primary_key(ordered[0])
    tied_top = [candidate for candidate in ordered if winner_primary_key(candidate) == top_key]
    if len(tied_top) <= 1:
        return ordered

    tw_winner = choose_tw_weighted_winner(match_id, tied_top)
    if tw_winner is None:
        return ordered

    tied_rest = [candidate for candidate in tied_top if candidate is not tw_winner]
    tied_rest = sorted(tied_rest, key=winner_sort_key)
    remaining = [candidate for candidate in ordered if winner_primary_key(candidate) != top_key]
    return [tw_winner, *tied_rest, *remaining]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_dedupe_plan(
    bundles: list[BundleInfo],
    work_db_path: Path,
    *,
    require_valid_for_pipeline: bool,
    require_game_creation_positive: bool,
    min_participants: int,
    progress: ProgressTracker | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, str],
]:
    bundle_map = {b.stack_name: b for b in bundles}
    duplicate_match_ids = find_duplicate_match_ids(work_db_path)
    duplicate_groups: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    winners_rows: list[dict[str, Any]] = []
    losers_rows: list[dict[str, Any]] = []
    actions_by_bundle: dict[str, list[str]] = {}
    winner_by_match_id: dict[str, str] = {}
    total_duplicate_match_ids = len(duplicate_match_ids)
    processed_duplicate_match_ids = 0
    for match_id in duplicate_match_ids:
        candidate_bundles = bundles_for_match(work_db_path, match_id)
        candidates = [
            quality_for_match(
                bundle_map[bundle],
                match_id,
                require_valid_for_pipeline=require_valid_for_pipeline,
                require_game_creation_positive=require_game_creation_positive,
                min_participants=min_participants,
            )
            for bundle in candidate_bundles
        ]
        ordered = order_candidates_for_match(match_id, candidates)
        processed_duplicate_match_ids += 1
        if not any(bool(c["is_complete"]) for c in ordered):
            skipped_groups.append(
                {
                    "match_id": match_id,
                    "reason": "no_complete_candidates",
                    "candidate_bundles": [str(c["bundle"]) for c in ordered],
                    "candidate_count": len(candidates),
                }
            )
        else:
            winner = ordered[0]
            losers = ordered[1:]
            winner_by_match_id[match_id] = str(winner["bundle"])
            winners_rows.append(dict(winner))
            duplicate_groups.append(
                {
                    "match_id": match_id,
                    "winner_bundle": str(winner["bundle"]),
                    "loser_bundles": [str(loser["bundle"]) for loser in losers],
                    "candidate_count": len(candidates),
                }
            )
            for loser in losers:
                losers_rows.append(dict(loser))
                actions_by_bundle.setdefault(str(loser["bundle"]), []).append(match_id)
        if progress is not None and (
            processed_duplicate_match_ids == total_duplicate_match_ids
            or processed_duplicate_match_ids % 1000 == 0
        ):
            denom = max(1, int(total_duplicate_match_ids))
            progress.update(
                phase="planning",
                checked=int(processed_duplicate_match_ids),
                total=int(total_duplicate_match_ids),
                pct=round(100.0 * float(processed_duplicate_match_ids) / float(denom), 2),
                kept=int(len(duplicate_groups)),
                skipped=int(len(skipped_groups)),
            )
    for bundle, mids in actions_by_bundle.items():
        actions_by_bundle[bundle] = sorted(set(mids))
    if progress is not None:
        progress.update(
            phase="planning_complete",
            checked=int(processed_duplicate_match_ids),
            total=int(total_duplicate_match_ids),
            pct=100.0 if total_duplicate_match_ids else 100.0,
            kept=int(len(duplicate_groups)),
            skipped=int(len(skipped_groups)),
        )
    return duplicate_groups, skipped_groups, winners_rows, losers_rows, actions_by_bundle, winner_by_match_id


def backup_participating_dbs(
    bundles: list[BundleInfo],
    participating_stacks: set[str],
    backup_dir: Path,
    stamp: str,
) -> dict[str, str]:
    out: dict[str, str] = {}
    bundle_map = {b.stack_name: b for b in bundles}
    backup_dir.mkdir(parents=True, exist_ok=True)
    for stack in sorted(participating_stacks):
        bundle = bundle_map[stack]
        dst_dir = backup_dir / stack
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"player_ranks.sqlite3.{stamp}.bak"
        shutil.copy2(bundle.db_path, dst)
        out[stack] = str(dst)
    return out


def delete_loser_match_ids(
    bundle: BundleInfo,
    match_ids: list[str],
) -> dict[str, Any]:
    conn = open_conn(bundle.db_path)
    try:
        cur = conn.cursor()
        deleted_matches = 0
        deleted_participants = 0
        cur.execute("BEGIN")
        for match_id in match_ids:
            cur.execute("DELETE FROM match_participants WHERE match_id = ?", (match_id,))
            deleted_participants += int(cur.rowcount or 0)
            cur.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))
            deleted_matches += int(cur.rowcount or 0)
        conn.commit()
        return {
            "bundle": bundle.stack_name,
            "deleted_match_rows": int(deleted_matches),
            "deleted_participant_rows": int(deleted_participants),
            "match_ids": list(match_ids),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def move_with_unique_suffix(src: Path, dst: Path) -> Path:
    if not dst.exists():
        src.replace(dst)
        return dst
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    candidate = dst.with_name(f"{dst.name}.{stamp}")
    idx = 1
    while candidate.exists():
        candidate = dst.with_name(f"{dst.name}.{stamp}.{idx}")
        idx += 1
    src.replace(candidate)
    return candidate


def cleanup_loser_json_files(
    bundle: BundleInfo,
    match_ids: list[str],
    *,
    cleanup_mode: str,
    quarantine_dirname: str,
) -> dict[str, Any]:
    moved: list[dict[str, str]] = []
    deleted: list[str] = []
    missing = 0
    quarantine_dir = bundle.matches_dir / quarantine_dirname
    if cleanup_mode == "move":
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    for match_id in match_ids:
        for candidate in (bundle.matches_dir / f"{match_id}.json.zst", bundle.matches_dir / f"{match_id}.json"):
            if not candidate.exists():
                missing += 1
                continue
            if cleanup_mode == "none":
                continue
            if cleanup_mode == "delete":
                candidate.unlink()
                deleted.append(str(candidate))
            elif cleanup_mode == "move":
                dst = quarantine_dir / candidate.name
                final_dst = move_with_unique_suffix(candidate, dst)
                moved.append({"src": str(candidate), "dst": str(final_dst)})
    return {
        "bundle": bundle.stack_name,
        "cleanup_mode": cleanup_mode,
        "moved_count": len(moved),
        "deleted_count": len(deleted),
        "missing_count": int(missing),
        "moved": moved,
        "deleted": deleted,
    }


def validate_after_apply(
    *,
    bundles: list[BundleInfo],
    report_dir: Path,
    modified_match_ids: set[str],
    winner_rows_before: list[dict[str, Any]],
    loser_actions_by_bundle: dict[str, list[str]],
    cleanup_mode: str,
    apply_file_stats: list[dict[str, Any]],
    planned_file_ops: int,
    require_valid_for_pipeline: bool,
    require_game_creation_positive: bool,
    min_participants: int,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "modified_match_ids": int(len(modified_match_ids)),
        "checks": {},
        "errors": [],
    }
    work_post = report_dir / "_occurrences_post.sqlite3"
    _rows, dup_groups = build_occurrence_index(bundles, work_post, only_match_ids=modified_match_ids)
    dup_mids = set(find_duplicate_match_ids(work_post))
    validation["checks"]["duplicates_remaining_for_modified_ids"] = int(len(dup_mids))
    if dup_mids:
        validation["errors"].append(
            f"{len(dup_mids)} duplicate match_id groups remain after apply for modified set"
        )

    bundle_map = {b.stack_name: b for b in bundles}
    winner_regressions = 0
    for row in winner_rows_before:
        mid = str(row["match_id"])
        stack = str(row["bundle"])
        if stack not in bundle_map:
            winner_regressions += 1
            continue
        now = quality_for_match(
            bundle_map[stack],
            mid,
            require_valid_for_pipeline=require_valid_for_pipeline,
            require_game_creation_positive=require_game_creation_positive,
            min_participants=min_participants,
        )
        if int(now["is_complete"]) < int(bool(row["is_complete"])):
            winner_regressions += 1
    validation["checks"]["winner_completeness_regressions"] = int(winner_regressions)
    if winner_regressions > 0:
        validation["errors"].append(f"{winner_regressions} winner rows regressed in completeness")

    orphan_failures = 0
    for stack, mids in loser_actions_by_bundle.items():
        bundle = bundle_map.get(stack)
        if bundle is None:
            orphan_failures += len(mids)
            continue
        conn = open_conn(bundle.db_path)
        try:
            for mid in mids:
                c_matches = safe_scalar(conn, "SELECT COUNT(*) FROM matches WHERE match_id = ?", (mid,))
                c_parts = safe_scalar(
                    conn,
                    "SELECT COUNT(*) FROM match_participants WHERE match_id = ?",
                    (mid,),
                )
                if c_matches != 0 or c_parts != 0:
                    orphan_failures += 1
        finally:
            conn.close()
    validation["checks"]["orphan_delete_failures"] = int(orphan_failures)
    if orphan_failures > 0:
        validation["errors"].append(f"{orphan_failures} loser match_ids were not fully deleted")

    executed_file_ops = 0
    for stat in apply_file_stats:
        executed_file_ops += int(stat.get("moved_count", 0) or 0)
        executed_file_ops += int(stat.get("deleted_count", 0) or 0)
    validation["checks"]["planned_file_ops"] = int(planned_file_ops)
    validation["checks"]["executed_file_ops"] = int(executed_file_ops)
    if cleanup_mode in ("move", "delete") and executed_file_ops < planned_file_ops:
        validation["errors"].append(
            f"executed file ops ({executed_file_ops}) lower than planned ({planned_file_ops})"
        )

    validation["ok"] = len(validation["errors"]) == 0
    return validation


def build_inventory_payload(bundles: list[BundleInfo], warnings: list[str]) -> dict[str, Any]:
    return {
        "generated_utc": int(time.time()),
        "bundle_count": int(len(bundles)),
        "warnings": warnings,
        "bundles": [
            {
                "stack_name": b.stack_name,
                "path": str(b.path),
                "db_path": str(b.db_path),
                "matches_dir": str(b.matches_dir),
                "meta_path": str(b.meta_path),
                "db_matches_row_count": int(b.db_matches_row_count),
                "stack_parts": b.stack_parts,
                "meta": b.meta,
            }
            for b in bundles
        ],
    }


def run(args: argparse.Namespace) -> int:
    inbox_root = Path(args.inbox_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(report_dir / "progress.txt")

    bundles, warnings = discover_bundles(inbox_root, args.glob)
    progress.update(
        phase="discovered",
        checked=int(len(bundles)),
        total=int(len(bundles)),
        pct=100.0,
    )
    save_json(report_dir / "inventory.json", build_inventory_payload(bundles, warnings))

    occ_path = report_dir / "_occurrences.sqlite3"
    rows_indexed, duplicate_groups_count = build_occurrence_index(bundles, occ_path, progress=progress)
    duplicate_groups, skipped_groups, winners_rows, losers_rows, actions_by_bundle, winner_by_match_id = build_dedupe_plan(
        bundles,
        occ_path,
        require_valid_for_pipeline=bool(args.require_valid_for_pipeline),
        require_game_creation_positive=bool(args.require_game_creation_positive),
        min_participants=int(args.min_participants),
        progress=progress,
    )
    progress.update(
        phase="reporting",
        checked=int(len(duplicate_groups) + len(skipped_groups)),
        total=int(len(duplicate_groups) + len(skipped_groups)),
        pct=100.0,
        kept=int(len(duplicate_groups)),
        skipped=int(len(skipped_groups)),
    )

    save_json(
        report_dir / "duplicate_groups.json",
        {
            "generated_utc": int(time.time()),
            "rows_indexed": int(rows_indexed),
            "duplicate_groups_count": int(duplicate_groups_count),
            "duplicate_groups": duplicate_groups,
            "skipped_groups_count": int(len(skipped_groups)),
            "skipped_groups": skipped_groups,
        },
    )
    write_csv(report_dir / "winners.csv", winners_rows)
    write_csv(report_dir / "losers.csv", losers_rows)
    write_csv(report_dir / "skipped_duplicates.csv", skipped_groups)
    save_json(
        report_dir / "apply_actions_per_bundle.json",
        {
            "generated_utc": int(time.time()),
            "loser_match_ids_by_bundle": actions_by_bundle,
            "duplicate_groups_count": int(len(duplicate_groups)),
            "skipped_groups_count": int(len(skipped_groups)),
            "loser_rows_count": int(len(losers_rows)),
        },
    )

    if args.mode == "dry-run":
        validation = {
            "ok": True,
            "mode": "dry-run",
            "modified_match_ids": int(len(winner_by_match_id)),
            "checks": {
                "duplicate_groups_count": int(len(duplicate_groups)),
                "skipped_groups_count": int(len(skipped_groups)),
                "loser_rows_count": int(len(losers_rows)),
                "planned_bundle_mutations": int(len(actions_by_bundle)),
            },
            "errors": [],
        }
        save_json(report_dir / "validation.json", validation)
        save_json(
            report_dir / "apply_result.json",
            {
                "mode": "dry-run",
                "ok": True,
                "message": "No mutations executed in dry-run mode.",
                "duplicate_groups_count": int(len(duplicate_groups)),
                "skipped_groups_count": int(len(skipped_groups)),
                "loser_rows_count": int(len(losers_rows)),
            },
        )
        progress.update(
            phase="done",
            mode="dry-run",
            ok=True,
            checked=int(len(duplicate_groups) + len(skipped_groups)),
            total=int(len(duplicate_groups) + len(skipped_groups)),
            pct=100.0,
            kept=int(len(duplicate_groups)),
            skipped=int(len(skipped_groups)),
        )
        print(f"Dry-run complete. Duplicate groups: {len(duplicate_groups)}")
        print(f"Reports written to: {report_dir}")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    participating_stacks: set[str] = set()
    for g in duplicate_groups:
        participating_stacks.add(str(g["winner_bundle"]))
        for loser_bundle in g["loser_bundles"]:
            participating_stacks.add(str(loser_bundle))
    backup_map: dict[str, str] = {}
    if not args.skip_backup:
        if backup_dir is None:
            raise SystemExit("--backup-dir is required in apply mode unless --skip-backup is set")
        backup_map = backup_participating_dbs(bundles, participating_stacks, backup_dir, stamp)

    apply_errors: list[dict[str, Any]] = []
    db_delete_stats: list[dict[str, Any]] = []
    file_cleanup_stats: list[dict[str, Any]] = []
    planned_file_ops = 0
    if str(args.cleanup_json) in ("move", "delete"):
        for row in losers_rows:
            planned_file_ops += int(bool(row.get("json_zst_exists", False)))
            planned_file_ops += int(bool(row.get("json_exists", False)))
    bundle_map = {b.stack_name: b for b in bundles}

    for stack, mids in sorted(actions_by_bundle.items()):
        bundle = bundle_map.get(stack)
        if bundle is None:
            apply_errors.append(
                {"bundle": stack, "match_ids": mids, "error": "bundle_missing_from_inventory"}
            )
            if not args.continue_on_error:
                break
            continue
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
            apply_errors.append({"bundle": stack, "match_ids": mids, "error": str(exc)})
            if not args.continue_on_error:
                break
        progress.update(
            phase="apply",
            checked=int(sum(len(stat.get("match_ids", [])) for stat in db_delete_stats)),
            total=int(sum(len(v) for v in actions_by_bundle.values())),
            pct=round(
                100.0
                * float(sum(len(stat.get("match_ids", [])) for stat in db_delete_stats))
                / float(max(1, sum(len(v) for v in actions_by_bundle.values()))),
                2,
            ),
            errors=int(len(apply_errors)),
        )

    modified_match_ids = {mid for mids in actions_by_bundle.values() for mid in mids}
    validation = validate_after_apply(
        bundles=bundles,
        report_dir=report_dir,
        modified_match_ids=modified_match_ids,
        winner_rows_before=winners_rows,
        loser_actions_by_bundle=actions_by_bundle,
        cleanup_mode=str(args.cleanup_json),
        apply_file_stats=file_cleanup_stats,
        planned_file_ops=int(planned_file_ops),
        require_valid_for_pipeline=bool(args.require_valid_for_pipeline),
        require_game_creation_positive=bool(args.require_game_creation_positive),
        min_participants=int(args.min_participants),
    )
    save_json(report_dir / "validation.json", validation)

    ok = len(apply_errors) == 0 and bool(validation.get("ok", False))
    apply_result = {
        "mode": "apply",
        "ok": bool(ok),
        "generated_utc": int(time.time()),
        "backup_dir": str(backup_dir) if backup_dir is not None else "",
        "skip_backup": bool(args.skip_backup),
        "backup_map": backup_map,
        "duplicate_groups_count": int(len(duplicate_groups)),
        "skipped_groups_count": int(len(skipped_groups)),
        "loser_rows_count": int(len(losers_rows)),
        "bundle_mutation_count": int(len(actions_by_bundle)),
        "db_delete_stats": db_delete_stats,
        "file_cleanup_stats": file_cleanup_stats,
        "errors": apply_errors,
        "validation_ok": bool(validation.get("ok", False)),
    }
    save_json(report_dir / "apply_result.json", apply_result)
    progress.update(
        phase="done",
        mode="apply",
        ok=bool(ok),
        checked=int(sum(len(v) for v in actions_by_bundle.values())),
        total=int(sum(len(v) for v in actions_by_bundle.values())),
        pct=100.0 if actions_by_bundle else 100.0,
        kept=int(len(duplicate_groups)),
        skipped=int(len(skipped_groups)),
        errors=int(len(apply_errors)),
    )

    if ok:
        print("Apply complete: success")
        print(f"Reports written to: {report_dir}")
        return 0
    print("Apply completed with errors")
    print(f"Reports written to: {report_dir}")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global match_id dedupe across bundle DBs.")
    p.add_argument("--inbox-root", type=str, default="runtime/out_latest/analysis/dedupe_inbox")
    p.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    p.add_argument("--report-dir", type=str, required=True)
    p.add_argument("--backup-dir", type=str, default="")
    p.add_argument("--skip-backup", action="store_true")
    p.add_argument("--cleanup-json", choices=["move", "delete", "none"], default="move")
    p.add_argument("--json-quarantine-dirname", type=str, default="_dedupe_quarantine")
    p.add_argument("--glob", type=str, default="*")
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
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.min_participants) <= 0:
        raise SystemExit("--min-participants must be positive")
    rc = run(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
