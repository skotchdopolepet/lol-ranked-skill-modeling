from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import quote

from main_dataset import (
    DEFAULT_APP_LIMIT_REQUESTS,
    DEFAULT_APP_LIMIT_WINDOW_SEC,
    DEFAULT_PLATFORM_ROUTING,
    DEFAULT_QUEUE,
    REGIONAL_BASE,
    CrawlStats,
    FatalRiotAuthError,
    RateController,
    configure_api_bases,
    effective_workers,
    match_detail,
    match_ids_by_puuid as upstream_match_ids_by_puuid,
    request_json,
    should_skip_preflight,
    validate_api_key,
    write_preflight_cache,
)

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional at runtime
    zstd = None

NO_RESTART_EXIT_CODE = 64

UPSTREAM_MATCH_IDS_SUPPORTS_START = "start" in inspect.signature(
    upstream_match_ids_by_puuid
).parameters


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def stable_hash_hex(seed: int, key: str) -> str:
    return hashlib.sha1(f"{int(seed)}::{key}".encode("utf-8")).hexdigest()


def chunked(items: list[str], size: int) -> list[list[str]]:
    width = max(1, int(size))
    return [items[idx : idx + width] for idx in range(0, len(items), width)]


def dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def match_id_recency_key(match_id: str) -> tuple[int, str]:
    token = str(match_id or "")
    try:
        return (int(token.rsplit("_", 1)[-1]), token)
    except Exception:
        return (-1, token)


def sort_match_ids_recent_first(match_ids: list[str]) -> list[str]:
    return sorted((str(mid) for mid in match_ids if mid), key=match_id_recency_key, reverse=True)


def resolve_run_paths(args: argparse.Namespace, run_idx: int) -> tuple[Path, Path, Path]:
    base_dir = Path(args.run_out_base_dir)
    if str(args.run_id or "").strip():
        run_name = str(args.run_id).strip()
    else:
        run_name = f"run_{run_idx:04d}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "player_time_run.json"
    checkpoint_json = run_dir / "player_time_run.checkpoint.json"
    return run_dir, out_json, checkpoint_json


RESUME_STAGE_ORDER = {
    "slice_selected": 1,
    "slice_players_ready": 2,
    "match_id_top_up_pass1": 3,
    "detail_resolution_loop_pass1": 4,
    "match_id_top_up_pass2": 5,
    "detail_resolution_loop_pass2": 6,
    "match_id_top_up_pass3": 7,
    "detail_resolution_loop_pass3": 8,
}


def resolve_resume_context(args: argparse.Namespace) -> dict[str, Any] | None:
    if not bool(getattr(args, "resume_from_latest_checkpoint", False)):
        return None
    base_dir = Path(args.run_out_base_dir)
    if not base_dir.exists():
        return None

    candidate_dirs: list[Path]
    if str(args.run_id or "").strip():
        candidate_dirs = [base_dir / str(args.run_id).strip()]
    else:
        candidate_dirs = sorted(
            [path for path in base_dir.glob("run_*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    for run_dir in candidate_dirs:
        checkpoint_path = run_dir / "player_time_run.checkpoint.json"
        sampled_matches_path = run_dir / "sampled_matches.json"
        if not checkpoint_path.exists() or not sampled_matches_path.exists():
            continue
        checkpoint = load_json_if_exists(checkpoint_path)
        sampled_matches = load_json_if_exists(sampled_matches_path)
        if not isinstance(checkpoint, dict) or not isinstance(sampled_matches, list) or not sampled_matches:
            continue
        stage = str(checkpoint.get("stage") or "").strip()
        if stage not in RESUME_STAGE_ORDER:
            continue
        extra = checkpoint.get("extra", {}) or {}
        phase_times = extra.get("phase_times_sec", {}) or {}
        if not isinstance(phase_times, dict):
            phase_times = {}
        return {
            "run_dir": run_dir,
            "out_json_path": run_dir / "player_time_run.json",
            "checkpoint_path": checkpoint_path,
            "stage": stage,
            "started_utc": int(checkpoint.get("started_utc") or time.time()),
            "elapsed_sec": float(checkpoint.get("elapsed_sec") or 0.0),
            "phase_times_sec": {
                str(key): float(value)
                for key, value in phase_times.items()
                if isinstance(value, (int, float))
            },
            "sampled_matches": sampled_matches,
            "full_bucket_distribution": extra.get("full_bucket_distribution") or {},
            "slice_bucket_distribution": extra.get("slice_bucket_distribution") or {},
        }
    return None


def write_checkpoint(
    checkpoint_path: Path,
    started_utc: int,
    started_mono: float,
    stage: str,
    extra: dict[str, Any],
) -> None:
    payload = {
        "started_utc": int(started_utc),
        "updated_utc": int(time.time()),
        "elapsed_sec": float(time.monotonic() - started_mono),
        "stage": str(stage),
        "extra": dict(extra),
    }
    save_json(checkpoint_path, payload)


def write_run_summary(path: Path, result: dict[str, Any]) -> None:
    totals = result.get("api_stats", {}).get("totals", {}) or {}
    coverage = result.get("coverage", {}) or {}
    phases = result.get("phases", {}) or {}
    work = result.get("work", {}) or {}
    lines = [
        f"started_utc: {result.get('started_utc')}",
        f"ended_utc: {result.get('ended_utc')}",
        f"elapsed_sec: {float(result.get('elapsed_sec', 0.0)):.2f}",
        f"players_at_target: {int(coverage.get('players_at_target', 0))}",
        f"match_complete_10_of_10: {int(coverage.get('match_complete_10_of_10', 0))}",
        f"match_complete_8_of_10: {int(coverage.get('match_complete_8_of_10', 0))}",
        f"matches_per_hour_slice: {float(coverage.get('matches_per_hour_slice', 0.0)):.2f}",
        f"matches_per_hour_complete_10_of_10: {float(coverage.get('matches_per_hour_complete_10_of_10', 0.0)):.2f}",
        f"matches_per_hour_complete_8_of_10: {float(coverage.get('matches_per_hour_complete_8_of_10', 0.0)):.2f}",
        f"requests: {int(totals.get('requests', 0) or 0)}",
        f"success: {int(totals.get('success', 0) or 0)}",
        f"retries: {int(totals.get('retries', 0) or 0)}",
        f"http_429: {int(totals.get('http_429', 0) or 0)}",
        f"detail_api_resolved_pass1: {int(work.get('detail_api_resolved_pass1', 0) or 0)}",
        f"detail_api_resolved_pass2: {int(work.get('detail_api_resolved_pass2', 0) or 0)}",
        f"slowest_phase: {phases.get('slowest_phase', '')}",
        f"slowest_phase_sec: {float(phases.get('slowest_phase_sec', 0.0) or 0.0):.2f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def live_api_checkpoint_extra(stats: CrawlStats, controller: RateController) -> dict[str, Any]:
    api_stats = stats.to_dict()
    totals = api_stats.get("totals", {}) or {}
    requests = int(totals.get("requests", 0) or 0)
    http_429 = int(totals.get("http_429", 0) or 0)
    controller_snapshot = controller.snapshot()
    app_limiter = controller_snapshot.get("app_limiter", {}) or {}
    return {
        "api_requests_total": requests,
        "api_success_total": int(totals.get("success", 0) or 0),
        "api_retries_total": int(totals.get("retries", 0) or 0),
        "api_http_429": http_429,
        "api_http_5xx": int(totals.get("http_5xx", 0) or 0),
        "api_errors_total": int(totals.get("errors", 0) or 0),
        "api_rate_429": (float(http_429) / float(requests)) if requests > 0 else 0.0,
        "api_requests_per_sec": float(api_stats.get("requests_per_sec", 0.0) or 0.0),
        "api_success_per_sec": float(api_stats.get("success_per_sec", 0.0) or 0.0),
        "api_limiter_scale": float(app_limiter.get("scale", 0.0) or 0.0),
        "api_limiter_blocked_for_sec": float(app_limiter.get("blocked_for_sec", 0.0) or 0.0),
    }


def load_match_payload(path: Path) -> dict[str, Any]:
    if path.name.endswith(".json.zst"):
        if zstd is None:
            raise RuntimeError("zstandard is required to read .json.zst match files")
        raw = path.read_bytes()
        text = zstd.ZstdDecompressor().decompress(raw).decode("utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def extract_match_meta(payload: dict[str, Any]) -> tuple[int, list[str]]:
    info = payload.get("info", {})
    game_creation = int(info.get("gameCreation", 0) or 0)
    participants: list[str] = []
    seen: set[str] = set()
    for row in info.get("participants", []):
        puuid = row.get("puuid")
        if not puuid:
            continue
        token = str(puuid)
        if token in seen:
            continue
        seen.add(token)
        participants.append(token)
    return game_creation, participants


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def match_ids_by_puuid(
    api_key: str,
    puuid: str,
    count: int,
    queue_id: int | None,
    match_type: str | None,
    controller: RateController,
    stats: CrawlStats,
    start: int = 0,
) -> list[str]:
    start_i = max(0, int(start))
    count_i = max(1, int(count))

    if UPSTREAM_MATCH_IDS_SUPPORTS_START:
        upstream_with_start = cast(Any, upstream_match_ids_by_puuid)
        return upstream_with_start(
            api_key,
            puuid,
            count_i,
            queue_id,
            match_type,
            controller,
            stats,
            start=start_i,
        )

    if start_i <= 0:
        return upstream_match_ids_by_puuid(
            api_key,
            puuid,
            count_i,
            queue_id,
            match_type,
            controller,
            stats,
        )

    query_parts = [f"start={start_i}", f"count={count_i}"]
    if queue_id is not None:
        query_parts.append(f"queue={int(queue_id)}")
    if match_type:
        query_parts.append(f"type={quote(str(match_type))}")
    query = "&".join(query_parts)
    url = f"{REGIONAL_BASE}/lol/match/v5/matches/by-puuid/{quote(puuid)}/ids?{query}"
    payload = request_json(
        url,
        api_key,
        endpoint_class="match_ids",
        controller=controller,
        stats=stats,
    )
    return [str(x) for x in payload]


def with_retry(
    label: str,
    max_retries: int,
    fn: Callable[..., Any],
    *fn_args: Any,
    **fn_kwargs: Any,
) -> Any:
    retries = max(0, int(max_retries))
    for attempt in range(retries + 1):
        try:
            return fn(*fn_args, **fn_kwargs)
        except FatalRiotAuthError:
            raise
        except Exception as exc:
            if attempt >= retries:
                raise
            sleep_sec = min(5.0, 0.5 * (2**attempt))
            print(
                f"  retry {label}: attempt {attempt + 1}/{retries} failed ({exc}); sleeping {sleep_sec:.1f}s"
            )
            time.sleep(sleep_sec)
    raise RuntimeError(f"Unexpected retry loop termination for {label}")

def open_aux_cache_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_ids_cache (
            puuid TEXT NOT NULL,
            match_id TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at_utc INTEGER NOT NULL,
            PRIMARY KEY (puuid, match_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aux_player_match_ids_puuid ON player_match_ids_cache(puuid)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_time_cache (
            match_id TEXT PRIMARY KEY,
            game_creation_utc_ms INTEGER NOT NULL,
            source TEXT NOT NULL,
            updated_at_utc INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_participants_cache (
            match_id TEXT NOT NULL,
            puuid TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at_utc INTEGER NOT NULL,
            PRIMARY KEY (match_id, puuid)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aux_match_participants_puuid ON match_participants_cache(puuid)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_match_id_failures (
            puuid TEXT PRIMARY KEY,
            fail_count INTEGER NOT NULL,
            last_error TEXT,
            updated_at_utc INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_detail_failures (
            match_id TEXT PRIMARY KEY,
            fail_count INTEGER NOT NULL,
            last_error TEXT,
            updated_at_utc INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def resolve_aux_cache_db_path(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    return out_dir / "player_time_cache.sqlite3"


def load_aux_cache_snapshot_from_db_path(
    db_path: Path,
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, list[str]]]:
    if not db_path.exists():
        return {}, {}, {}
    conn = sqlite3.connect(str(db_path))
    try:
        return load_aux_cache_snapshot(conn)
    except sqlite3.Error:
        return {}, {}, {}
    finally:
        conn.close()


def load_union_aux_cache_snapshot(
    cache_db_paths: list[Path],
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, list[str]]]:
    player_match_ids: dict[str, set[str]] = defaultdict(set)
    match_times: dict[str, int] = {}
    match_participants: dict[str, set[str]] = defaultdict(set)

    for db_path in cache_db_paths:
        cache_player_ids, cache_match_times, cache_match_parts = load_aux_cache_snapshot_from_db_path(db_path)
        for puuid, match_ids in cache_player_ids.items():
            player_match_ids[str(puuid)].update(str(mid) for mid in match_ids if mid)
        for match_id, ts in cache_match_times.items():
            match_times.setdefault(str(match_id), int(ts))
        for match_id, participants in cache_match_parts.items():
            match_participants[str(match_id)].update(str(p) for p in participants if p)

    return (
        dict(player_match_ids),
        match_times,
        {match_id: sorted(participants) for match_id, participants in match_participants.items()},
    )


def refresh_aux_cache_into_memory(
    *,
    aux_cache_db_path: Path,
    player_match_id_sets: dict[str, set[str]],
    timestamp_cache: dict[str, int],
    match_participants_cache: dict[str, list[str]],
) -> dict[str, int]:
    union_player_ids, union_match_times, union_match_parts = load_aux_cache_snapshot_from_db_path(
        aux_cache_db_path
    )
    added_player_match_rows = 0
    added_match_times = 0
    added_match_participants = 0

    for puuid, match_ids in union_player_ids.items():
        before = len(player_match_id_sets.get(str(puuid), set()))
        player_match_id_sets.setdefault(str(puuid), set()).update(str(mid) for mid in match_ids if mid)
        added_player_match_rows += len(player_match_id_sets[str(puuid)]) - before

    for match_id, ts in union_match_times.items():
        if str(match_id) not in timestamp_cache:
            timestamp_cache[str(match_id)] = int(ts)
            added_match_times += 1

    for match_id, participants in union_match_parts.items():
        existing = set(match_participants_cache.get(str(match_id), []))
        merged = existing.union(str(p) for p in participants if p)
        if len(merged) > len(existing):
            match_participants_cache[str(match_id)] = sorted(merged)
            added_match_participants += len(merged) - len(existing)

    return {
        "cache_db_count": 1,
        "added_player_match_rows": int(added_player_match_rows),
        "added_match_times": int(added_match_times),
        "added_match_participants": int(added_match_participants),
    }


def load_aux_cache_snapshot(
    conn: sqlite3.Connection,
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, list[str]]]:
    player_match_ids: dict[str, set[str]] = defaultdict(set)
    for puuid, match_id in conn.execute("SELECT puuid, match_id FROM player_match_ids_cache"):
        if puuid and match_id:
            player_match_ids[str(puuid)].add(str(match_id))

    match_times: dict[str, int] = {}
    for match_id, game_creation in conn.execute(
        "SELECT match_id, game_creation_utc_ms FROM match_time_cache"
    ):
        if match_id:
            match_times[str(match_id)] = int(game_creation or 0)

    match_participants: dict[str, list[str]] = defaultdict(list)
    for match_id, puuid in conn.execute(
        "SELECT match_id, puuid FROM match_participants_cache ORDER BY match_id, puuid"
    ):
        if match_id and puuid:
            match_participants[str(match_id)].append(str(puuid))

    return player_match_ids, match_times, dict(match_participants)


def load_failure_counts(
    conn: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, int]]:
    player_failures: dict[str, int] = {}
    match_failures: dict[str, int] = {}
    try:
        for puuid, fail_count in conn.execute(
            "SELECT puuid, fail_count FROM player_match_id_failures"
        ):
            if puuid:
                player_failures[str(puuid)] = int(fail_count or 0)
    except sqlite3.Error:
        pass
    try:
        for match_id, fail_count in conn.execute(
            "SELECT match_id, fail_count FROM match_detail_failures"
        ):
            if match_id:
                match_failures[str(match_id)] = int(fail_count or 0)
    except sqlite3.Error:
        pass
    return player_failures, match_failures


def increment_player_match_id_failure(
    conn: sqlite3.Connection,
    puuid: str,
    error: str,
) -> int:
    now_utc = int(time.time())
    conn.execute(
        """
        INSERT INTO player_match_id_failures (puuid, fail_count, last_error, updated_at_utc)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            fail_count=player_match_id_failures.fail_count + 1,
            last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (str(puuid), str(error), now_utc),
    )
    row = conn.execute(
        "SELECT fail_count FROM player_match_id_failures WHERE puuid = ?",
        (str(puuid),),
    ).fetchone()
    return int((row or [0])[0] or 0)


def clear_player_match_id_failure(conn: sqlite3.Connection, puuid: str) -> None:
    conn.execute("DELETE FROM player_match_id_failures WHERE puuid = ?", (str(puuid),))


def increment_match_detail_failure(
    conn: sqlite3.Connection,
    match_id: str,
    error: str,
) -> int:
    now_utc = int(time.time())
    conn.execute(
        """
        INSERT INTO match_detail_failures (match_id, fail_count, last_error, updated_at_utc)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
            fail_count=match_detail_failures.fail_count + 1,
            last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (str(match_id), str(error), now_utc),
    )
    row = conn.execute(
        "SELECT fail_count FROM match_detail_failures WHERE match_id = ?",
        (str(match_id),),
    ).fetchone()
    return int((row or [0])[0] or 0)


def clear_match_detail_failure(conn: sqlite3.Connection, match_id: str) -> None:
    conn.execute("DELETE FROM match_detail_failures WHERE match_id = ?", (str(match_id),))


def upsert_aux_player_match_ids(
    conn: sqlite3.Connection,
    puuid: str,
    match_ids: list[str],
    source: str,
) -> None:
    now_utc = int(time.time())
    rows = [(str(puuid), str(match_id), str(source), now_utc) for match_id in match_ids if match_id]
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO player_match_ids_cache (puuid, match_id, source, updated_at_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(puuid, match_id) DO UPDATE SET
            source=excluded.source,
            updated_at_utc=excluded.updated_at_utc
        """,
        rows,
    )


def upsert_aux_match_detail(
    conn: sqlite3.Connection,
    match_id: str,
    game_creation_utc_ms: int,
    participants: list[str],
    source: str,
) -> None:
    now_utc = int(time.time())
    conn.execute(
        """
        INSERT INTO match_time_cache (match_id, game_creation_utc_ms, source, updated_at_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
            game_creation_utc_ms=excluded.game_creation_utc_ms,
            source=excluded.source,
            updated_at_utc=excluded.updated_at_utc
        """,
        (str(match_id), int(game_creation_utc_ms), str(source), now_utc),
    )
    if participants:
        conn.executemany(
            """
            INSERT INTO match_participants_cache (match_id, puuid, source, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(match_id, puuid) DO UPDATE SET
                source=excluded.source,
                updated_at_utc=excluded.updated_at_utc
            """,
            [(str(match_id), str(puuid), str(source), now_utc) for puuid in participants if puuid],
        )
        conn.executemany(
            """
            INSERT INTO player_match_ids_cache (puuid, match_id, source, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(puuid, match_id) DO UPDATE SET
                source=excluded.source,
                updated_at_utc=excluded.updated_at_utc
            """,
            [(str(puuid), str(match_id), str(source), now_utc) for puuid in participants if puuid],
        )


def add_player_match_ids(player_match_id_sets: dict[str, set[str]], puuid: str, match_ids: list[str]) -> int:
    bucket = player_match_id_sets.setdefault(str(puuid), set())
    before = len(bucket)
    bucket.update(str(mid) for mid in match_ids if mid)
    return len(bucket) - before


def load_rank_bucket_map(out_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    path = out_dir / "seed_players.json"
    raw = load_json_if_exists(path)
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            puuid = row.get("puuid")
            bucket = row.get("seed_rank_bucket")
            if puuid and bucket:
                out[str(puuid)] = str(bucket)

    ranks_db = out_dir / "player_ranks.sqlite3"
    if ranks_db.exists():
        conn = sqlite3.connect(str(ranks_db))
        try:
            for puuid, solo_tier, solo_rank in conn.execute(
                "SELECT puuid, solo_tier, solo_rank FROM player_ranks"
            ):
                if not puuid or not solo_tier:
                    continue
                tier = str(solo_tier).upper()
                rank = str(solo_rank or "").upper()
                bucket = tier if not rank else f"{tier}_{rank}"
                out.setdefault(str(puuid), bucket)
        finally:
            conn.close()

    return out


def load_participant_index_by_match(out_dir: Path) -> dict[str, list[str]]:
    path = out_dir / "participant_index_by_match.json"
    raw = load_json_if_exists(path)
    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for match_id, participants in raw.items():
        if not isinstance(participants, list):
            continue
        out[str(match_id)] = [str(p) for p in participants if p]
    return out


def load_external_match_times(out_dir: Path, cache_db_paths: list[Path]) -> tuple[dict[str, int], dict[str, int]]:
    match_times: dict[str, int] = {}
    source_counts: dict[str, int] = defaultdict(int)

    _aux_player_ids, aux_times, _aux_parts = load_union_aux_cache_snapshot(cache_db_paths)
    for match_id, ts in aux_times.items():
        match_times.setdefault(match_id, int(ts))
        source_counts["aux_cache"] += 1

    dataset_db = out_dir / "dataset.sqlite3"
    if dataset_db.exists():
        conn = sqlite3.connect(str(dataset_db))
        try:
            for match_id, ts in conn.execute(
                "SELECT match_id, game_creation_utc_ms FROM match_start_time_cache"
            ):
                if match_id and str(match_id) not in match_times:
                    match_times[str(match_id)] = int(ts or 0)
                    source_counts["dataset_match_start_time_cache"] += 1
        finally:
            conn.close()

    ranks_db = out_dir / "player_ranks.sqlite3"
    if ranks_db.exists():
        conn = sqlite3.connect(str(ranks_db))
        try:
            for match_id, ts in conn.execute(
                "SELECT match_id, game_creation_utc_ms FROM matches"
            ):
                if match_id and str(match_id) not in match_times:
                    match_times[str(match_id)] = int(ts or 0)
                    source_counts["player_ranks_matches"] += 1
        finally:
            conn.close()

    return match_times, dict(source_counts)

def load_external_match_participants(
    out_dir: Path,
    participant_index: dict[str, list[str]],
    cache_db_paths: list[Path],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    match_participants: dict[str, list[str]] = {}
    source_counts: dict[str, int] = defaultdict(int)

    _aux_player_ids, _aux_times, aux_parts = load_union_aux_cache_snapshot(cache_db_paths)
    for match_id, participants in aux_parts.items():
        if match_id and participants:
            match_participants.setdefault(str(match_id), dedupe_preserve(list(participants)))
            source_counts["aux_cache"] += 1

    for match_id, participants in participant_index.items():
        if match_id and participants and match_id not in match_participants:
            match_participants[match_id] = dedupe_preserve(list(participants))
            source_counts["participant_index_json"] += 1

    ranks_db = out_dir / "player_ranks.sqlite3"
    if ranks_db.exists():
        bucket: dict[str, list[str]] = defaultdict(list)
        conn = sqlite3.connect(str(ranks_db))
        try:
            for match_id, puuid in conn.execute(
                "SELECT match_id, puuid FROM match_participants ORDER BY match_id, puuid"
            ):
                if match_id and puuid and str(match_id) not in match_participants:
                    bucket[str(match_id)].append(str(puuid))
        finally:
            conn.close()
        for match_id, participants in bucket.items():
            if participants and match_id not in match_participants:
                match_participants[match_id] = dedupe_preserve(participants)
                source_counts["player_ranks_match_participants"] += 1

    return match_participants, dict(source_counts)


def load_external_player_match_ids(
    out_dir: Path,
    cache_db_paths: list[Path],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    player_match_ids: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, int] = defaultdict(int)

    aux_player_ids, _aux_times, _aux_parts = load_union_aux_cache_snapshot(cache_db_paths)
    for puuid, match_ids in aux_player_ids.items():
        if match_ids:
            player_match_ids[str(puuid)].update(str(mid) for mid in match_ids if mid)
            source_counts["aux_cache_rows"] += len(match_ids)

    dataset_db = out_dir / "dataset.sqlite3"
    if dataset_db.exists():
        conn = sqlite3.connect(str(dataset_db))
        try:
            for puuid, match_id in conn.execute("SELECT puuid, match_id FROM player_match_ids"):
                if puuid and match_id:
                    player_match_ids[str(puuid)].add(str(match_id))
                    source_counts["dataset_player_match_ids_rows"] += 1
        finally:
            conn.close()

    ranks_db = out_dir / "player_ranks.sqlite3"
    if ranks_db.exists():
        conn = sqlite3.connect(str(ranks_db))
        try:
            for puuid, match_ids_json in conn.execute(
                "SELECT puuid, match_ids_json FROM match_ids_cache"
            ):
                if not puuid or not match_ids_json:
                    continue
                try:
                    match_ids = json.loads(str(match_ids_json))
                except Exception:
                    continue
                if isinstance(match_ids, list):
                    player_match_ids[str(puuid)].update(str(mid) for mid in match_ids if mid)
                    source_counts["player_ranks_match_ids_cache_rows"] += len(match_ids)
        finally:
            conn.close()

    json_path = out_dir / "match_ids_by_puuid.json"
    raw = load_json_if_exists(json_path)
    if isinstance(raw, dict):
        for puuid, match_ids in raw.items():
            if isinstance(match_ids, list):
                player_match_ids[str(puuid)].update(str(mid) for mid in match_ids if mid)
                source_counts["match_ids_by_puuid_json_rows"] += len(match_ids)

    return player_match_ids, dict(source_counts)


def load_candidate_match_ids(
    out_dir: Path,
    match_participants_cache: dict[str, list[str]],
    timestamp_cache: dict[str, int],
    explicit_candidate_file: str = "",
) -> tuple[list[str], dict[str, int]]:
    candidate_ids: list[str] = []
    source_counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()

    explicit_paths: list[Path] = []
    if str(explicit_candidate_file or "").strip():
        explicit_paths.append(Path(str(explicit_candidate_file).strip()))
    explicit_paths.extend(
        [
            out_dir / "player_dataset_targets.csv",
            out_dir / "player_dataset_targets.txt",
        ]
    )
    for path in explicit_paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = json_reader = None
                    import csv

                    reader = csv.DictReader(f)
                    if reader.fieldnames and "match_id" in reader.fieldnames:
                        for row in reader:
                            token = str(row.get("match_id") or "").strip()
                            if token and token not in seen:
                                seen.add(token)
                                candidate_ids.append(token)
                                source_counts["explicit_candidate_csv_rows"] += 1
                    else:
                        f.seek(0)
                        for line in f:
                            token = str(line.split(",", 1)[0]).strip()
                            if token and token.lower() != "match_id" and token not in seen:
                                seen.add(token)
                                candidate_ids.append(token)
                                source_counts["explicit_candidate_csv_rows"] += 1
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    token = str(line).strip()
                    if token and token.lower() != "match_id" and token not in seen:
                        seen.add(token)
                        candidate_ids.append(token)
                        source_counts["explicit_candidate_text_rows"] += 1
        except Exception:
            continue
        if candidate_ids:
            return candidate_ids, dict(source_counts)

    ranks_db = out_dir / "player_ranks.sqlite3"
    if ranks_db.exists():
        conn = sqlite3.connect(str(ranks_db))
        try:
            rows = conn.execute(
                """
                SELECT match_id
                FROM matches
                WHERE match_id IS NOT NULL
                  AND COALESCE(valid_for_pipeline, 1) = 1
                  AND COALESCE(participant_count, 10) >= 10
                ORDER BY match_id
                """
            ).fetchall()
            for (match_id,) in rows:
                token = str(match_id)
                if token and token not in seen:
                    seen.add(token)
                    candidate_ids.append(token)
                    source_counts["player_ranks_matches"] += 1
        finally:
            conn.close()

    if not candidate_ids:
        for match_id in sorted(match_participants_cache.keys()):
            token = str(match_id)
            if token and token not in seen:
                seen.add(token)
                candidate_ids.append(token)
                source_counts["participant_cache_keys"] += 1

    if not candidate_ids:
        for match_id in sorted(timestamp_cache.keys()):
            token = str(match_id)
            if token and token not in seen:
                seen.add(token)
                candidate_ids.append(token)
                source_counts["timestamp_cache_keys"] += 1

    if not candidate_ids:
        source_matches_dir = Path(out_dir) / "matches"
        if source_matches_dir.exists():
            match_files = sorted(source_matches_dir.glob("*.json.zst")) + sorted(source_matches_dir.glob("*.json"))
            for path in match_files:
                token = path.name[: -len(".json.zst")] if path.name.endswith(".json.zst") else path.stem
                if token and token not in seen:
                    seen.add(token)
                    candidate_ids.append(token)
                    source_counts["match_file_names"] += 1

    return candidate_ids, dict(source_counts)


def match_bucket_from_participants(
    participants: list[str],
    seed_rank_bucket_by_puuid: dict[str, str],
) -> str:
    buckets = [seed_rank_bucket_by_puuid[p] for p in participants if p in seed_rank_bucket_by_puuid]
    if not buckets:
        return "UNKNOWN"
    counts = Counter(buckets)
    max_count = max(counts.values())
    return sorted([bucket for bucket, count in counts.items() if count == max_count])[0]


def select_balanced_slice(
    match_ids: list[str],
    match_participants: dict[str, list[str]],
    seed_rank_bucket_by_puuid: dict[str, str],
    slice_count: int,
    slice_seed: int,
    slice_selection: str = "balanced",
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    total_available = len(match_ids)
    effective_count = min(max(1, int(slice_count)), total_available)

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for match_id in match_ids:
        participants = match_participants.get(match_id, [])
        bucket = match_bucket_from_participants(participants, seed_rank_bucket_by_puuid)
        by_bucket[bucket].append(match_id)

    total = float(max(1, total_available))
    full_distribution = {
        bucket: len(ids) / total for bucket, ids in sorted(by_bucket.items())
    }

    raw_targets: dict[str, float] = {
        bucket: (len(ids) / total) * effective_count for bucket, ids in by_bucket.items()
    }
    base_targets: dict[str, int] = {
        bucket: min(len(by_bucket[bucket]), int(raw_targets[bucket])) for bucket in by_bucket
    }
    remaining = effective_count - sum(base_targets.values())
    fractions = sorted(
        [
            (raw_targets[bucket] - base_targets[bucket], bucket)
            for bucket in by_bucket
            if base_targets[bucket] < len(by_bucket[bucket])
        ],
        key=lambda item: (-item[0], item[1]),
    )
    for _fraction, bucket in fractions:
        if remaining <= 0:
            break
        if base_targets[bucket] < len(by_bucket[bucket]):
            base_targets[bucket] += 1
            remaining -= 1

    sampled: list[dict[str, Any]] = []
    participant_popularity: Counter[str] = Counter()
    use_overlap_dense = str(slice_selection or "balanced").strip().lower() == "overlap_dense"
    if use_overlap_dense:
        for participants in match_participants.values():
            for puuid in dedupe_preserve([str(p) for p in participants if p]):
                participant_popularity[puuid] += 1

    for bucket, ids in sorted(by_bucket.items()):
        if use_overlap_dense:
            def dense_key(match_id: str) -> tuple[int, str]:
                participants = dedupe_preserve(
                    [str(p) for p in match_participants.get(match_id, []) if p]
                )
                popularity = sum(int(participant_popularity.get(puuid, 0)) for puuid in participants)
                return (-popularity, stable_hash_hex(slice_seed, match_id))

            ordered = sorted(ids, key=dense_key)
        else:
            ordered = sorted(ids, key=lambda match_id: stable_hash_hex(slice_seed, match_id))
        for match_id in ordered[: base_targets[bucket]]:
            sampled.append({"match_id": match_id, "bucket": bucket})

    sampled = sorted(sampled, key=lambda row: stable_hash_hex(slice_seed, str(row["match_id"])))
    sampled = sampled[:effective_count]

    sampled_counter = Counter(str(row["bucket"]) for row in sampled)
    slice_distribution = {
        bucket: sampled_counter.get(bucket, 0) / float(max(1, len(sampled)))
        for bucket in sorted(by_bucket.keys())
    }
    return sampled, full_distribution, slice_distribution


def propagate_match_detail(
    match_id: str,
    game_creation_utc_ms: int,
    participants: list[str],
    timestamp_cache: dict[str, int],
    match_participants_cache: dict[str, list[str]],
    player_match_id_sets: dict[str, set[str]],
    aux_conn: sqlite3.Connection,
    source: str,
) -> None:
    timestamp_cache[str(match_id)] = int(game_creation_utc_ms)
    clean_participants = dedupe_preserve([str(p) for p in participants if p])
    if clean_participants:
        match_participants_cache[str(match_id)] = clean_participants
        for puuid in clean_participants:
            player_match_id_sets.setdefault(str(puuid), set()).add(str(match_id))
    upsert_aux_match_detail(aux_conn, str(match_id), int(game_creation_utc_ms), clean_participants, str(source))


def compute_live_completion(
    slice_match_participants: dict[str, list[str]],
    player_match_id_sets: dict[str, set[str]],
    timestamp_cache: dict[str, int],
    target_count_per_player: int,
) -> dict[str, Any]:
    target = max(1, int(target_count_per_player))
    player_coverage: dict[str, int] = {}
    for puuid, match_ids in player_match_id_sets.items():
        player_coverage[str(puuid)] = sum(1 for mid in match_ids if mid in timestamp_cache)

    complete_10 = 0
    complete_8 = 0
    partial_counts: dict[str, int] = {}
    for match_id, participants in slice_match_participants.items():
        covered = sum(1 for puuid in participants if int(player_coverage.get(puuid, 0)) >= target)
        partial_counts[str(match_id)] = int(covered)
        if covered >= 10:
            complete_10 += 1
        if covered >= 8:
            complete_8 += 1

    return {
        "players_at_target": int(sum(1 for v in player_coverage.values() if v >= target)),
        "match_complete_10_of_10": int(complete_10),
        "match_complete_8_of_10": int(complete_8),
        "player_coverage": player_coverage,
        "partial_counts": partial_counts,
    }


def build_player_to_slice_matches(
    slice_match_participants: dict[str, list[str]],
) -> dict[str, list[str]]:
    player_to_matches: dict[str, list[str]] = defaultdict(list)
    for match_id, participants in slice_match_participants.items():
        for puuid in participants:
            token = str(puuid)
            if token:
                player_to_matches[token].append(str(match_id))
    return dict(player_to_matches)


def prioritize_unresolved_players(
    unresolved_players: list[str],
    *,
    player_to_slice_matches: dict[str, list[str]],
    partial_counts: dict[str, int],
    player_coverage: dict[str, int],
    target_count: int,
) -> list[str]:
    target = max(1, int(target_count))

    def key_for(puuid: str) -> tuple[int, int, int, int, str]:
        coverage = int(player_coverage.get(puuid, 0))
        deficit = max(0, target - coverage)
        matches = player_to_slice_matches.get(puuid, [])
        best_partial = 0
        near_complete_count = 0
        incomplete_match_count = 0
        for match_id in matches:
            covered = int(partial_counts.get(match_id, 0))
            if covered < 10:
                incomplete_match_count += 1
                best_partial = max(best_partial, covered)
                if covered >= 8:
                    near_complete_count += 1
        # Priority:
        # 1) players that can push already-near-complete matches to 10/10
        # 2) players participating in more near-complete matches
        # 3) players that can improve more exact target matches at once
        # 4) players closest to target coverage (smaller deficit)
        return (-best_partial, -near_complete_count, -incomplete_match_count, deficit, puuid)

    return sorted(unresolved_players, key=key_for)


def prioritize_detail_candidate_ids(
    unresolved_players: list[str],
    *,
    player_match_id_sets: dict[str, set[str]],
    player_to_slice_matches: dict[str, list[str]],
    partial_counts: dict[str, int],
    player_coverage: dict[str, int],
    target_count: int,
    timestamp_cache: dict[str, int],
    attempted_detail_ids: set[str],
    match_failure_counts: dict[str, int],
    max_match_detail_failures: int,
) -> list[str]:
    target = max(1, int(target_count))
    candidate_scores: dict[str, int] = {}
    first_seen_rank: dict[str, int] = {}
    rank = 0
    for puuid in unresolved_players:
        coverage = int(player_coverage.get(puuid, 0))
        deficit = max(0, target - coverage)
        near_complete = 0
        incomplete = 0
        at_nine = 0
        at_eight = 0
        for match_id in player_to_slice_matches.get(puuid, []):
            covered = int(partial_counts.get(match_id, 0))
            if covered < 10:
                incomplete += 1
                if covered >= 8:
                    near_complete += 1
                    if covered == 9:
                        at_nine += 1
                    elif covered == 8:
                        at_eight += 1
        if incomplete <= 0:
            continue
        impact_score = (
            at_nine * 100_000
            + at_eight * 30_000
            + near_complete * 10_000
            + incomplete * 500
            + coverage * 200
            - deficit * 400
        )
        for match_id in sort_match_ids_recent_first(list(player_match_id_sets.get(puuid, set()))):
            if (
                not match_id
                or match_id in timestamp_cache
                or match_id in attempted_detail_ids
                or int(match_failure_counts.get(match_id, 0)) >= int(max_match_detail_failures)
            ):
                continue
            candidate_scores[match_id] = candidate_scores.get(match_id, 0) + impact_score
            first_seen_rank.setdefault(match_id, rank)
            rank += 1
    return sorted(
        candidate_scores,
        key=lambda mid: (-int(candidate_scores.get(mid, 0)), int(first_seen_rank.get(mid, 0))),
    )


def filter_players_by_partial_band(
    players: list[str],
    *,
    player_to_slice_matches: dict[str, list[str]],
    partial_counts: dict[str, int],
    min_partial: int,
    max_partial: int,
) -> list[str]:
    out: list[str] = []
    lo = int(min_partial)
    hi = int(max_partial)
    for puuid in players:
        if any(
            lo <= int(partial_counts.get(match_id, 0)) <= hi
            for match_id in player_to_slice_matches.get(puuid, [])
        ):
            out.append(puuid)
    return out


def resolve_stop_flag_path(args: argparse.Namespace) -> Path | None:
    raw = str(args.stop_flag_file or "").strip()
    if not raw:
        return None
    stop_path = Path(raw)
    if not stop_path.is_absolute():
        stop_path = Path(args.out_dir) / stop_path
    return stop_path


def apply_history_expansion_policy(args: argparse.Namespace) -> None:
    policy = str(getattr(args, "history_expansion_policy", "simple") or "simple").strip().lower()
    if policy not in {"simple", "completion_first", "hybrid", "fresh_first"}:
        policy = "simple"

    # The dataset runner now uses a single explicit flow:
    # pass 1 recent history -> detail resolution -> optional pass 2 -> detail resolution.
    # Older policy names are accepted for CLI compatibility, but they all resolve to the
    # same runtime behavior so the launch command once again describes the real execution.
    args.history_expansion_policy = "simple"
    args.third_match_ids_top_up = False
    args.detail_inline_match_id_refill = False
    first_page_width = max(1, int(args.target_matches_per_player) + int(args.search_buffer))
    args.second_match_ids_start = max(int(args.second_match_ids_start), first_page_width)


def fetch_match_id_top_up(
    *,
    label: str,
    args: argparse.Namespace,
    controller: RateController,
    stats: CrawlStats,
    aux_conn: sqlite3.Connection,
    aux_cache_db_path: Path,
    run_players: list[str],
    player_match_id_sets: dict[str, set[str]],
    slice_match_participants: dict[str, list[str]],
    player_to_slice_matches: dict[str, list[str]],
    timestamp_cache: dict[str, int],
    match_participants_cache: dict[str, list[str]],
    target_count: int,
    request_count: int,
    request_count_for_player: Callable[[str], int] | None,
    start_offset: int,
    stop_when_known_match_ids_reach: int | None,
    checkpoint_path: Path,
    started_utc: int,
    started_mono: float,
    phase_times_sec: dict[str, float],
    checkpoint_stage: str,
) -> dict[str, Any]:
    aux_conn.commit()
    player_failure_counts, _match_failure_counts = load_failure_counts(aux_conn)
    cache_refresh = refresh_aux_cache_into_memory(
        aux_cache_db_path=aux_cache_db_path,
        player_match_id_sets=player_match_id_sets,
        timestamp_cache=timestamp_cache,
        match_participants_cache=match_participants_cache,
    )
    live_before = compute_live_completion(
        slice_match_participants,
        {puuid: player_match_id_sets.get(puuid, set()) for puuid in run_players},
        timestamp_cache,
        target_count,
    )
    players_needing_api = [
        puuid
        for puuid in run_players
        if int(live_before["player_coverage"].get(puuid, 0)) < target_count
        and int(player_failure_counts.get(puuid, 0)) < int(args.max_player_match_id_failures)
        and (
            stop_when_known_match_ids_reach is None
            or len(player_match_id_sets.get(puuid, set())) < int(stop_when_known_match_ids_reach)
        )
    ]
    players_needing_api = prioritize_unresolved_players(
        players_needing_api,
        player_to_slice_matches=player_to_slice_matches,
        partial_counts=cast(dict[str, int], live_before["partial_counts"]),
        player_coverage=cast(dict[str, int], live_before["player_coverage"]),
        target_count=target_count,
    )
    player_limit = max(0, int(getattr(args, "match_id_top_up_player_limit", 0)))
    if player_limit > 0:
        players_needing_api = players_needing_api[:player_limit]

    failures: dict[str, str] = {}
    workers_ids = effective_workers(int(args.workers_match_ids), args.max_inflight_match_ids)
    submit_batch_size = max(
        max(1, workers_ids),
        int(getattr(args, "match_id_top_up_submit_batch_size", max(32, workers_ids * 8))),
    )
    progress_every = max(1, int(getattr(args, "match_id_top_up_progress_every", 100)))
    done = 0
    total = len(players_needing_api)

    def checkpoint_live(live: dict[str, Any]) -> None:
        write_checkpoint(
            checkpoint_path,
            started_utc,
            started_mono,
            stage=checkpoint_stage,
            extra={
                "done": int(done),
                "total": int(total),
                "start_offset": int(start_offset),
                "request_count": int(request_count),
                "max_player_match_id_failures": int(args.max_player_match_id_failures),
                "stop_when_known_match_ids_reach": (
                    None
                    if stop_when_known_match_ids_reach is None
                    else int(stop_when_known_match_ids_reach)
                ),
                "match_id_top_up_submit_batch_size": int(submit_batch_size),
                "match_id_top_up_progress_every": int(progress_every),
                "match_id_top_up_player_limit": int(player_limit),
                "match_id_top_up_plateau_break_triggered": False,
                "match_id_top_up_plateau_break_reason": "",
                "player_match_id_api_failures": int(len(failures)),
                "phase_times_sec": dict(phase_times_sec),
                "players_at_target": int(live["players_at_target"]),
                "match_complete_10_of_10": int(live["match_complete_10_of_10"]),
                "match_complete_8_of_10": int(live["match_complete_8_of_10"]),
                **live_api_checkpoint_extra(stats, controller),
            },
        )

    # Mark pass entry immediately so the checkpoint stage reflects the real phase
    # even before the first 25-player progress refresh.
    checkpoint_live(live_before)

    with ThreadPoolExecutor(max_workers=max(1, workers_ids)) as ex:
        for player_batch in chunked(players_needing_api, submit_batch_size):
            futures = {
                ex.submit(
                    with_retry,
                    f"{label}:{puuid}",
                    int(args.request_max_retries),
                    match_ids_by_puuid,
                    args.api_key,
                    puuid,
                    (
                        max(1, int(request_count_for_player(puuid)))
                        if request_count_for_player is not None
                        else int(request_count)
                    ),
                    int(args.queue_id),
                    args.match_type,
                    controller,
                    stats,
                    start_offset,
                ): puuid
                for puuid in player_batch
            }
            for fut in as_completed(futures):
                puuid = futures[fut]
                done += 1
                try:
                    mids = [str(x) for x in fut.result()]
                    add_player_match_ids(player_match_id_sets, puuid, mids)
                    upsert_aux_player_match_ids(aux_conn, puuid, mids, source=label)
                    clear_player_match_id_failure(aux_conn, puuid)
                except Exception as exc:
                    failures[puuid] = str(exc)
                    player_failure_counts[puuid] = increment_player_match_id_failure(
                        aux_conn, puuid, str(exc)
                    )
                if done % progress_every == 0 or done == total:
                    aux_conn.commit()
                    live = compute_live_completion(
                        slice_match_participants,
                        {player: player_match_id_sets.get(player, set()) for player in run_players},
                        timestamp_cache,
                        target_count,
                    )
                    checkpoint_live(live)
    aux_conn.commit()
    return {
        "players_considered": int(len(players_needing_api)),
        "failures": failures,
        "workers_ids": int(workers_ids),
        "cache_refresh": cache_refresh,
        "match_id_top_up_player_limit": int(player_limit),
        "match_id_top_up_submit_batch_size": int(submit_batch_size),
        "match_id_top_up_progress_every": int(progress_every),
        "match_id_top_up_plateau_break_triggered": False,
        "match_id_top_up_plateau_break_reason": "",
        "stop_when_known_match_ids_reach": (
            None if stop_when_known_match_ids_reach is None else int(stop_when_known_match_ids_reach)
        ),
    }


def run_detail_resolution_loop(
    *,
    args: argparse.Namespace,
    label: str,
    match_file_by_id: dict[str, Path],
    controller: RateController,
    stats: CrawlStats,
    aux_conn: sqlite3.Connection,
    aux_cache_db_path: Path,
    run_players: list[str],
    player_match_id_sets: dict[str, set[str]],
    slice_match_participants: dict[str, list[str]],
    player_to_slice_matches: dict[str, list[str]],
    timestamp_cache: dict[str, int],
    match_participants_cache: dict[str, list[str]],
    target_count: int,
    checkpoint_path: Path,
    started_utc: int,
    started_mono: float,
    phase_times_sec: dict[str, float],
    checkpoint_stage: str,
    stop_flag_path: Path | None,
) -> dict[str, Any]:
    attempted_detail_ids: set[str] = set()
    detail_local_resolved = 0
    detail_api_resolved = 0
    detail_api_failures: dict[str, str] = {}
    detail_loop_iterations = 0
    detail_chunks_total = 0
    workers_details = effective_workers(int(args.workers_match_details), args.max_inflight_match_details)
    detail_chunk_size_cfg = max(1, int(getattr(args, "detail_api_chunk_size", workers_details)))
    detail_refresh_every_chunks = max(1, int(getattr(args, "detail_progress_refresh_every", 1)))
    detail_cache_refresh_every_iterations = max(
        1, int(getattr(args, "detail_aux_cache_refresh_every_iterations", 1))
    )
    detail_commit_every_chunks = max(
        1,
        int(getattr(args, "detail_commit_every_chunks", detail_refresh_every_chunks)),
    )
    _player_failure_counts, match_failure_counts = load_failure_counts(aux_conn)
    detail_executor = ThreadPoolExecutor(max_workers=max(1, workers_details))

    def checkpoint_live(
        live: dict[str, Any],
        *,
        cache_refresh: dict[str, int],
        batch_size: int,
        chunk_size: int,
        stop_finish_mode: bool,
    ) -> None:
        write_checkpoint(
            checkpoint_path,
            started_utc,
            started_mono,
            stage=checkpoint_stage,
            extra={
                "label": str(label),
                "detail_loop_iterations": int(detail_loop_iterations),
                "detail_chunks_total": int(detail_chunks_total),
                "detail_batch_size": int(batch_size),
                "detail_chunk_size": int(chunk_size),
                "detail_progress_refresh_every": int(detail_refresh_every_chunks),
                "detail_commit_every_chunks": int(detail_commit_every_chunks),
                "detail_aux_cache_refresh_every_iterations": int(
                    detail_cache_refresh_every_iterations
                ),
                "detail_local_resolved": int(detail_local_resolved),
                "detail_api_resolved": int(detail_api_resolved),
                "detail_api_failures": int(len(detail_api_failures)),
                "attempted_detail_ids": int(len(attempted_detail_ids)),
                "max_match_detail_failures": int(args.max_match_detail_failures),
                "stop_finish_mode": bool(stop_finish_mode),
                "detail_plateau_break_triggered": False,
                "detail_plateau_break_reason": "",
                "cache_refresh": cache_refresh,
                "phase_times_sec": dict(phase_times_sec),
                "players_at_target": int(live["players_at_target"]),
                "match_complete_10_of_10": int(live["match_complete_10_of_10"]),
                "match_complete_8_of_10": int(live["match_complete_8_of_10"]),
                **live_api_checkpoint_extra(stats, controller),
            },
        )

    try:
        while True:
            aux_conn.commit()
            if detail_loop_iterations == 0 or (
                detail_loop_iterations % detail_cache_refresh_every_iterations
            ) == 0:
                cache_refresh = refresh_aux_cache_into_memory(
                    aux_cache_db_path=aux_cache_db_path,
                    player_match_id_sets=player_match_id_sets,
                    timestamp_cache=timestamp_cache,
                    match_participants_cache=match_participants_cache,
                )
            else:
                cache_refresh = {
                    "cache_db_count": 0,
                    "added_player_match_rows": 0,
                    "added_match_times": 0,
                    "added_match_participants": 0,
                }
            live = compute_live_completion(
                slice_match_participants,
                {player: player_match_id_sets.get(player, set()) for player in run_players},
                timestamp_cache,
                target_count,
            )
            unresolved_players = [
                puuid for puuid in run_players if int(live["player_coverage"].get(puuid, 0)) < target_count
            ]
            stop_finish_mode = bool(stop_flag_path is not None and stop_flag_path.exists())
            if stop_finish_mode:
                min_partial = max(1, int(args.stop_finish_min_covered_participants))
                unresolved_players = [
                    puuid
                    for puuid in unresolved_players
                    if any(
                        min_partial <= int(live["partial_counts"].get(match_id, 0)) < 10
                        for match_id in player_to_slice_matches.get(puuid, [])
                    )
                ]
            if not unresolved_players:
                break
            partial_counts = cast(dict[str, int], live["partial_counts"])
            player_coverage = cast(dict[str, int], live["player_coverage"])
            candidate_ids: list[str] = []
            detail_focus_bands = ((5, 9), (3, 4), (1, 2), (0, 0))
            for band_min, band_max in detail_focus_bands:
                band_players = filter_players_by_partial_band(
                    unresolved_players,
                    player_to_slice_matches=player_to_slice_matches,
                    partial_counts=partial_counts,
                    min_partial=band_min,
                    max_partial=band_max,
                )
                if not band_players:
                    continue
                prioritized_players = prioritize_unresolved_players(
                    band_players,
                    player_to_slice_matches=player_to_slice_matches,
                    partial_counts=partial_counts,
                    player_coverage=player_coverage,
                    target_count=target_count,
                )
                candidate_ids = prioritize_detail_candidate_ids(
                    prioritized_players,
                    player_match_id_sets=player_match_id_sets,
                    player_to_slice_matches=player_to_slice_matches,
                    partial_counts=partial_counts,
                    player_coverage=player_coverage,
                    target_count=target_count,
                    timestamp_cache=timestamp_cache,
                    attempted_detail_ids=attempted_detail_ids,
                    match_failure_counts=match_failure_counts,
                    max_match_detail_failures=int(args.max_match_detail_failures),
                )
                if candidate_ids:
                    unresolved_players = prioritized_players
                    break
            if not candidate_ids:
                break

            detail_loop_iterations += 1
            batch_ids = candidate_ids[: int(args.detail_batch_size)]
            batch_chunk_size = max(1, min(detail_chunk_size_cfg, len(batch_ids)))
            for chunk_idx in range(0, len(batch_ids), batch_chunk_size):
                chunk_ids = batch_ids[chunk_idx : chunk_idx + batch_chunk_size]
                api_chunk_ids: list[str] = []
                detail_chunks_total += 1

                for match_id in chunk_ids:
                    attempted_detail_ids.add(match_id)
                    local_path = match_file_by_id.get(match_id)
                    if local_path is None:
                        api_chunk_ids.append(match_id)
                        continue
                    try:
                        payload = load_match_payload(local_path)
                        ts, participants = extract_match_meta(payload)
                        propagate_match_detail(
                            match_id,
                            ts,
                            participants,
                            timestamp_cache,
                            match_participants_cache,
                            player_match_id_sets,
                            aux_conn,
                            source=f"{label}_local_match_file",
                        )
                        detail_local_resolved += 1
                    except Exception:
                        api_chunk_ids.append(match_id)

                if api_chunk_ids:
                    futures = {
                        detail_executor.submit(
                            match_detail,
                            args.api_key,
                            match_id,
                            controller,
                            stats,
                            int(args.request_timeout_sec),
                            int(args.request_max_retries),
                        ): match_id
                        for match_id in api_chunk_ids
                    }
                    for fut in as_completed(futures):
                        match_id = futures[fut]
                        try:
                            detail = fut.result()
                            ts, participants = extract_match_meta(detail)
                            propagate_match_detail(
                                match_id,
                                ts,
                                participants,
                                timestamp_cache,
                                match_participants_cache,
                                player_match_id_sets,
                                aux_conn,
                                source=f"{label}_riot_match_detail_api",
                            )
                            detail_api_resolved += 1
                            clear_match_detail_failure(aux_conn, match_id)
                            match_failure_counts.pop(match_id, None)
                        except Exception as exc:
                            detail_api_failures[match_id] = str(exc)
                            match_failure_counts[match_id] = increment_match_detail_failure(
                                aux_conn, match_id, str(exc)
                            )

                is_last_chunk = (chunk_idx + batch_chunk_size) >= len(batch_ids)
                should_commit = (
                    is_last_chunk or (detail_chunks_total % detail_commit_every_chunks) == 0
                )
                if should_commit:
                    aux_conn.commit()
                if is_last_chunk or (detail_chunks_total % detail_refresh_every_chunks) == 0:
                    if not should_commit:
                        aux_conn.commit()
                    live = compute_live_completion(
                        slice_match_participants,
                        {player: player_match_id_sets.get(player, set()) for player in run_players},
                        timestamp_cache,
                        target_count,
                    )
                    checkpoint_live(
                        live,
                        cache_refresh=cache_refresh,
                        batch_size=len(batch_ids),
                        chunk_size=len(chunk_ids),
                        stop_finish_mode=stop_finish_mode,
                    )
    finally:
        detail_executor.shutdown(wait=True)

    return {
        "detail_loop_iterations": int(detail_loop_iterations),
        "detail_chunks_total": int(detail_chunks_total),
        "detail_local_resolved": int(detail_local_resolved),
        "detail_api_resolved": int(detail_api_resolved),
        "detail_api_failures": detail_api_failures,
        "attempted_detail_ids": int(len(attempted_detail_ids)),
        "workers_details": int(workers_details),
        "plateau_break_triggered": False,
        "plateau_break_reason": "",
    }

def run_player_time_dataset(args: argparse.Namespace, run_idx: int) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    resume_ctx = resolve_resume_context(args)
    if resume_ctx is not None:
        run_dir = Path(resume_ctx["run_dir"])
        out_json_path = Path(resume_ctx["out_json_path"])
        checkpoint_path = Path(resume_ctx["checkpoint_path"])
    else:
        run_dir, out_json_path, checkpoint_path = resolve_run_paths(args, run_idx)
    aux_cache_db_path = resolve_aux_cache_db_path(args)
    aux_conn = open_aux_cache_db(aux_cache_db_path)
    stop_flag_path = resolve_stop_flag_path(args)
    print(f"Player-time run dir: {run_dir}")
    if resume_ctx is not None:
        print(f"Resume: enabled from {run_dir.name} at stage {resume_ctx['stage']}")
        started_utc = int(resume_ctx["started_utc"])
        started_mono = float(time.monotonic() - float(resume_ctx["elapsed_sec"]))
        phase_times_sec: dict[str, float] = dict(resume_ctx["phase_times_sec"])
        resume_stage = str(resume_ctx["stage"])
    else:
        started_utc = int(time.time())
        started_mono = time.monotonic()
        phase_times_sec = {}
        resume_stage = ""

    def phase_start(phase_name: str) -> float:
        print(f"\n[{phase_name}] started")
        return time.monotonic()

    def phase_end(phase_name: str, phase_mono: float) -> None:
        elapsed = float(time.monotonic() - phase_mono)
        phase_times_sec[phase_name] = elapsed
        print(f"[{phase_name}] finished in {elapsed:.2f}s")

    stats = CrawlStats()
    controller = RateController(
        profile=args.rate_profile,
        app_limit_requests=args.app_limit_requests,
        app_limit_window_sec=args.app_limit_window_sec,
    )

    preflight_cache_path = out_dir / ".preflight_cache.json"
    preflight_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force_preflight or not should_skip_preflight(
        preflight_cache_path, args.api_key, args.preflight_ttl_sec
    ):
        print("Preflight: validate API key")
        validate_api_key(args.api_key, controller=controller, stats=stats)
        write_preflight_cache(preflight_cache_path, args.api_key)
        print("  key validation OK")
    else:
        print(f"Preflight: skipped (cached within {args.preflight_ttl_sec}s for same key)")

    raw_source_matches_dir = str(args.source_matches_dir or "").strip()
    source_matches_dir = Path(raw_source_matches_dir) if raw_source_matches_dir else (out_dir / "matches")
    match_file_by_id: dict[str, Path] = {}
    if source_matches_dir.exists():
        match_files = sorted(source_matches_dir.glob("*.json.zst")) + sorted(source_matches_dir.glob("*.json"))
        match_file_by_id = {
            (path.name[: -len('.json.zst')] if path.name.endswith('.json.zst') else path.stem): path
            for path in match_files
        }

    phase = phase_start("phase_1_preload_local_sources")
    seed_rank_bucket_by_puuid = load_rank_bucket_map(out_dir)
    participant_index = load_participant_index_by_match(out_dir)
    timestamp_cache, timestamp_source_counts = load_external_match_times(out_dir, [aux_cache_db_path])
    match_participants_cache, participant_source_counts = load_external_match_participants(
        out_dir, participant_index, [aux_cache_db_path]
    )
    player_match_id_sets, player_id_source_counts = load_external_player_match_ids(
        out_dir, [aux_cache_db_path]
    )
    candidate_match_ids, candidate_match_id_sources = load_candidate_match_ids(
        out_dir,
        match_participants_cache,
        timestamp_cache,
        explicit_candidate_file=str(args.candidate_match_ids_file or ""),
    )
    phase_end("phase_1_preload_local_sources", phase)

    all_match_ids = sorted(candidate_match_ids)
    if not all_match_ids:
        raise RuntimeError(
            "No candidate matches available from DB/cache sources. "
            "Provide a populated player_ranks.sqlite3 or other local caches."
        )
    if resume_ctx is not None:
        sampled_matches = list(resume_ctx["sampled_matches"])
        full_bucket_distribution = dict(resume_ctx["full_bucket_distribution"])
        slice_bucket_distribution = dict(resume_ctx["slice_bucket_distribution"])
        print(f"Resume: reusing {len(sampled_matches)} sampled matches from {run_dir.name}")
    else:
        phase = phase_start("phase_2_select_slice")
        sampled_matches, full_bucket_distribution, slice_bucket_distribution = select_balanced_slice(
            match_ids=all_match_ids,
            match_participants=match_participants_cache,
            seed_rank_bucket_by_puuid=seed_rank_bucket_by_puuid,
            slice_count=int(args.slice_match_count),
            slice_seed=int(args.slice_seed),
            slice_selection=str(args.slice_selection or "balanced"),
        )
        save_json(run_dir / "sampled_matches.json", sampled_matches)
        phase_end("phase_2_select_slice", phase)

        write_checkpoint(
            checkpoint_path,
            started_utc,
            started_mono,
            stage="slice_selected",
            extra={
                "slice_match_count": int(len(sampled_matches)),
                "candidate_match_count": int(len(all_match_ids)),
                "candidate_match_id_sources": candidate_match_id_sources,
                "phase_times_sec": dict(phase_times_sec),
                "full_bucket_distribution": full_bucket_distribution,
                "slice_bucket_distribution": slice_bucket_distribution,
                **live_api_checkpoint_extra(stats, controller),
            },
        )

    phase = phase_start("phase_3_prepare_slice_players")
    slice_match_participants: dict[str, list[str]] = {}
    refresh_slice_api_success = 0
    refresh_slice_api_failures = 0

    for row in sampled_matches:
        match_id = str(row["match_id"])
        participants = match_participants_cache.get(match_id, [])
        timestamp = timestamp_cache.get(match_id)
        if (not participants or timestamp is None) and match_id in match_file_by_id:
            payload = load_match_payload(match_file_by_id[match_id])
            ts, file_participants = extract_match_meta(payload)
            propagate_match_detail(
                match_id,
                ts,
                file_participants,
                timestamp_cache,
                match_participants_cache,
                player_match_id_sets,
                aux_conn,
                source="local_match_file",
            )
            participants = match_participants_cache.get(match_id, [])
        elif (not participants or timestamp is None) and bool(args.refresh_slice_match_details_from_api):
            try:
                detail = with_retry(
                    f"slice_match_detail:{match_id}",
                    int(args.request_max_retries),
                    match_detail,
                    args.api_key,
                    match_id,
                    controller,
                    stats,
                    int(args.request_timeout_sec),
                )
                ts, api_participants = extract_match_meta(detail)
                propagate_match_detail(
                    match_id,
                    ts,
                    api_participants,
                    timestamp_cache,
                    match_participants_cache,
                    player_match_id_sets,
                    aux_conn,
                    source="slice_refresh_api",
                )
                refresh_slice_api_success += 1
                participants = match_participants_cache.get(match_id, [])
            except Exception:
                refresh_slice_api_failures += 1
                participants = participants or []
        slice_match_participants[match_id] = participants
        for puuid in participants:
            player_match_id_sets.setdefault(str(puuid), set()).add(match_id)

    aux_conn.commit()
    all_slice_players = sorted({p for participants in slice_match_participants.values() for p in participants if p})
    run_players = list(all_slice_players)
    player_to_slice_matches = build_player_to_slice_matches(slice_match_participants)
    phase_end("phase_3_prepare_slice_players", phase)

    write_checkpoint(
        checkpoint_path,
        started_utc,
        started_mono,
        stage="slice_players_ready",
        extra={
            "slice_player_count_total": int(len(all_slice_players)),
            "refresh_slice_api_success": int(refresh_slice_api_success),
            "refresh_slice_api_failures": int(refresh_slice_api_failures),
            "phase_times_sec": dict(phase_times_sec),
            **live_api_checkpoint_extra(stats, controller),
        },
    )

    target_count = int(args.target_matches_per_player)
    search_target = int(args.target_matches_per_player) + int(args.search_buffer)
    resume_rank = RESUME_STAGE_ORDER.get(resume_stage, 0)
    first_top_up = {
        "players_considered": 0,
        "failures": {},
        "workers_ids": int(effective_workers(int(args.workers_match_ids), args.max_inflight_match_ids)),
    }
    first_detail_pass = {
        "detail_loop_iterations": 0,
        "detail_chunks_total": 0,
        "detail_local_resolved": 0,
        "detail_api_resolved": 0,
        "detail_api_failures": {},
        "attempted_detail_ids": 0,
        "workers_details": int(effective_workers(int(args.workers_match_details), args.max_inflight_match_details)),
    }
    if resume_rank <= RESUME_STAGE_ORDER["match_id_top_up_pass1"]:
        phase = phase_start("phase_4_match_id_top_up")
        first_top_up = fetch_match_id_top_up(
            label="riot_match_ids_api_pass1",
            args=args,
            controller=controller,
            stats=stats,
            aux_conn=aux_conn,
            aux_cache_db_path=aux_cache_db_path,
            run_players=run_players,
            player_match_id_sets=player_match_id_sets,
            slice_match_participants=slice_match_participants,
            player_to_slice_matches=player_to_slice_matches,
            timestamp_cache=timestamp_cache,
            match_participants_cache=match_participants_cache,
            target_count=target_count,
            request_count=search_target,
            request_count_for_player=None,
            start_offset=0,
            stop_when_known_match_ids_reach=search_target,
            checkpoint_path=checkpoint_path,
            started_utc=started_utc,
            started_mono=started_mono,
            phase_times_sec=phase_times_sec,
            checkpoint_stage="match_id_top_up_pass1",
        )
        phase_end("phase_4_match_id_top_up", phase)
    else:
        print(f"Resume: skipping phase_4_match_id_top_up from stage {resume_stage}")

    if resume_rank <= RESUME_STAGE_ORDER["detail_resolution_loop_pass1"]:
        phase = phase_start("phase_5_detail_resolution_loop")
        first_detail_pass = run_detail_resolution_loop(
            args=args,
            label="detail_pass1",
            match_file_by_id=match_file_by_id,
            controller=controller,
            stats=stats,
            aux_conn=aux_conn,
            aux_cache_db_path=aux_cache_db_path,
            run_players=run_players,
            player_match_id_sets=player_match_id_sets,
            slice_match_participants=slice_match_participants,
            player_to_slice_matches=player_to_slice_matches,
            timestamp_cache=timestamp_cache,
            match_participants_cache=match_participants_cache,
            target_count=target_count,
            checkpoint_path=checkpoint_path,
            started_utc=started_utc,
            started_mono=started_mono,
            phase_times_sec=phase_times_sec,
            checkpoint_stage="detail_resolution_loop_pass1",
            stop_flag_path=stop_flag_path,
        )
        phase_end("phase_5_detail_resolution_loop", phase)
    else:
        print(f"Resume: skipping phase_5_detail_resolution_loop from stage {resume_stage}")

    second_top_up_enabled = bool(args.second_match_ids_top_up)
    if stop_flag_path is not None and stop_flag_path.exists():
        second_top_up_enabled = False
    second_top_up = {
        "players_considered": 0,
        "failures": {},
        "workers_ids": int(first_top_up["workers_ids"]),
    }
    second_detail_pass = {
        "detail_loop_iterations": 0,
        "detail_local_resolved": 0,
        "detail_api_resolved": 0,
        "detail_api_failures": {},
        "attempted_detail_ids": 0,
        "workers_details": int(first_detail_pass["workers_details"]),
    }
    if second_top_up_enabled and resume_rank <= RESUME_STAGE_ORDER["match_id_top_up_pass2"]:
        phase = phase_start("phase_5b_second_match_id_top_up")
        second_pass_request_count_for_player = lambda puuid: int(args.second_match_ids_count)
        # Pass 2 reuses the same unresolved-player definition as the rest of the
        # pipeline: players still below the resolved target. With the default
        # start/count, Riot's zero-based paging maps to the 35th-68th most recent matches.
        second_top_up = fetch_match_id_top_up(
            label="riot_match_ids_api_pass2",
            args=args,
            controller=controller,
            stats=stats,
            aux_conn=aux_conn,
            aux_cache_db_path=aux_cache_db_path,
            run_players=run_players,
            player_match_id_sets=player_match_id_sets,
            slice_match_participants=slice_match_participants,
            player_to_slice_matches=player_to_slice_matches,
            timestamp_cache=timestamp_cache,
            match_participants_cache=match_participants_cache,
            target_count=target_count,
            request_count=int(args.second_match_ids_count),
            request_count_for_player=second_pass_request_count_for_player,
            start_offset=int(args.second_match_ids_start),
            stop_when_known_match_ids_reach=None,
            checkpoint_path=checkpoint_path,
            started_utc=started_utc,
            started_mono=started_mono,
            phase_times_sec=phase_times_sec,
            checkpoint_stage="match_id_top_up_pass2",
        )
        phase_end("phase_5b_second_match_id_top_up", phase)
    elif second_top_up_enabled:
        print(f"Resume: skipping phase_5b_second_match_id_top_up from stage {resume_stage}")

    if second_top_up_enabled and resume_rank <= RESUME_STAGE_ORDER["detail_resolution_loop_pass2"]:
        phase = phase_start("phase_5c_second_detail_resolution_loop")
        second_detail_pass = run_detail_resolution_loop(
            args=args,
            label="detail_pass2",
            match_file_by_id=match_file_by_id,
            controller=controller,
            stats=stats,
            aux_conn=aux_conn,
            aux_cache_db_path=aux_cache_db_path,
            run_players=run_players,
            player_match_id_sets=player_match_id_sets,
            slice_match_participants=slice_match_participants,
            player_to_slice_matches=player_to_slice_matches,
            timestamp_cache=timestamp_cache,
            match_participants_cache=match_participants_cache,
            target_count=target_count,
            checkpoint_path=checkpoint_path,
            started_utc=started_utc,
            started_mono=started_mono,
            phase_times_sec=phase_times_sec,
            checkpoint_stage="detail_resolution_loop_pass2",
            stop_flag_path=stop_flag_path,
        )
        phase_end("phase_5c_second_detail_resolution_loop", phase)
    elif second_top_up_enabled:
        print(f"Resume: skipping phase_5c_second_detail_resolution_loop from stage {resume_stage}")

    phase = phase_start("phase_6_finalize")
    final_live = compute_live_completion(
        slice_match_participants,
        {player: player_match_id_sets.get(player, set()) for player in run_players},
        timestamp_cache,
        target_count,
    )
    coverage_snapshot = {
        puuid: {
            "ids_total": int(len(player_match_id_sets.get(puuid, set()))),
            "ids_with_time": int(final_live["player_coverage"].get(puuid, 0)),
        }
        for puuid in run_players
    }
    save_json(run_dir / "player_coverage_snapshot.json", coverage_snapshot)
    save_json(
        run_dir / "sampled_match_completion.json",
        [
            {
                "match_id": match_id,
                "covered_participants_count": int(final_live["partial_counts"].get(match_id, 0)),
                "is_complete_10_of_10": int(final_live["partial_counts"].get(match_id, 0)) >= 10,
                "is_complete_8_of_10": int(final_live["partial_counts"].get(match_id, 0)) >= 8,
            }
            for match_id in sorted(slice_match_participants.keys())
        ],
    )

    elapsed_sec = float(time.monotonic() - started_mono)
    elapsed_hours = elapsed_sec / 3600.0 if elapsed_sec > 0 else 0.0
    complete_10 = int(final_live["match_complete_10_of_10"])
    complete_8 = int(final_live["match_complete_8_of_10"])
    matches_per_hour_slice = float(len(sampled_matches)) / elapsed_hours if elapsed_hours > 0 else 0.0
    matches_per_hour_complete_10 = float(complete_10) / elapsed_hours if elapsed_hours > 0 else 0.0
    matches_per_hour_complete_8 = float(complete_8) / elapsed_hours if elapsed_hours > 0 else 0.0

    phase_end("phase_6_finalize", phase)
    slowest_phase = max(phase_times_sec.items(), key=lambda item: item[1])[0] if phase_times_sec else ""

    result = {
        "started_utc": int(started_utc),
        "ended_utc": int(time.time()),
        "elapsed_sec": elapsed_sec,
        "config": {
            "source_matches_dir": str(source_matches_dir),
            "source_match_files_available": int(len(match_file_by_id)),
            "out_dir": str(out_dir),
            "run_dir": str(run_dir),
            "resume_from_latest_checkpoint": bool(getattr(args, "resume_from_latest_checkpoint", False)),
            "resumed_from_run_dir": str(run_dir) if resume_ctx is not None else "",
            "resumed_from_stage": str(resume_stage),
            "slice_match_count_requested": int(args.slice_match_count),
            "slice_match_count_effective": int(len(sampled_matches)),
            "slice_seed": int(args.slice_seed),
            "target_matches_per_player": int(target_count),
            "search_buffer": int(args.search_buffer),
            "search_target_per_player": int(search_target),
            "detail_batch_size": int(args.detail_batch_size),
            "detail_api_chunk_size": int(args.detail_api_chunk_size),
            "detail_progress_refresh_every": int(args.detail_progress_refresh_every),
            "detail_commit_every_chunks": int(args.detail_commit_every_chunks),
            "detail_plateau_window_iterations": int(args.detail_plateau_window_iterations),
            "detail_plateau_min_completed_gain": int(args.detail_plateau_min_completed_gain),
            "detail_plateau_min_players_gain": int(args.detail_plateau_min_players_gain),
            "detail_plateau_min_attempted_ids": int(args.detail_plateau_min_attempted_ids),
            "history_expansion_policy": str(getattr(args, "history_expansion_policy", "hybrid")),
            "platform_routing": str(args.platform_routing),
            "regional_routing": str(args.regional_routing),
            "queue_id": int(args.queue_id),
            "match_type": str(args.match_type),
            "workers_match_ids": int(first_top_up["workers_ids"]),
            "workers_match_details": int(first_detail_pass["workers_details"]),
            "refresh_slice_match_details_from_api": bool(args.refresh_slice_match_details_from_api),
            "request_timeout_sec": int(args.request_timeout_sec),
            "request_max_retries": int(args.request_max_retries),
            "max_player_match_id_failures": int(args.max_player_match_id_failures),
            "max_match_detail_failures": int(args.max_match_detail_failures),
            "second_match_ids_top_up": bool(args.second_match_ids_top_up),
            "second_match_ids_start": int(args.second_match_ids_start),
            "second_match_ids_count": int(args.second_match_ids_count),
            "third_match_ids_top_up": bool(args.third_match_ids_top_up),
            "match_id_top_up_submit_batch_size": int(args.match_id_top_up_submit_batch_size),
            "match_id_top_up_progress_every": int(args.match_id_top_up_progress_every),
            "aux_cache_db": str(aux_cache_db_path),
        },
        "slice": {
            "full_bucket_distribution": full_bucket_distribution,
            "slice_bucket_distribution": slice_bucket_distribution,
            "slice_player_count_total": int(len(all_slice_players)),
            "refresh_slice_api_success": int(refresh_slice_api_success),
            "refresh_slice_api_failures": int(refresh_slice_api_failures),
        },
        "local_reuse": {
            "candidate_match_id_sources": candidate_match_id_sources,
            "timestamp_sources": timestamp_source_counts,
            "participant_sources": participant_source_counts,
            "player_match_id_sources": player_id_source_counts,
            "cached_match_times": int(len(timestamp_cache)),
            "cached_match_participants": int(len(match_participants_cache)),
        },
        "work": {
            "players_needing_match_id_api_pass1": int(first_top_up["players_considered"]),
            "player_match_id_api_failures_pass1": int(len(first_top_up["failures"])),
            "detail_loop_iterations_pass1": int(first_detail_pass["detail_loop_iterations"]),
            "detail_local_resolved_pass1": int(first_detail_pass["detail_local_resolved"]),
            "detail_api_resolved_pass1": int(first_detail_pass["detail_api_resolved"]),
            "detail_api_failures_pass1": int(len(first_detail_pass["detail_api_failures"])),
            "attempted_detail_ids_pass1": int(first_detail_pass["attempted_detail_ids"]),
            "players_needing_match_id_api_pass2": int(second_top_up["players_considered"]),
            "player_match_id_api_failures_pass2": int(len(second_top_up["failures"])),
            "detail_loop_iterations_pass2": int(second_detail_pass["detail_loop_iterations"]),
            "detail_local_resolved_pass2": int(second_detail_pass["detail_local_resolved"]),
            "detail_api_resolved_pass2": int(second_detail_pass["detail_api_resolved"]),
            "detail_api_failures_pass2": int(len(second_detail_pass["detail_api_failures"])),
            "attempted_detail_ids_pass2": int(second_detail_pass["attempted_detail_ids"]),
        },
        "coverage": {
            "players_at_target": int(final_live["players_at_target"]),
            "match_complete_10_of_10": int(complete_10),
            "match_complete_8_of_10": int(complete_8),
            "matches_per_hour_slice": matches_per_hour_slice,
            "matches_per_hour_complete_10_of_10": matches_per_hour_complete_10,
            "matches_per_hour_complete_8_of_10": matches_per_hour_complete_8,
        },
        "phases": {
            "phase_times_sec": dict(phase_times_sec),
            "slowest_phase": str(slowest_phase),
            "slowest_phase_sec": float(phase_times_sec.get(slowest_phase, 0.0)) if slowest_phase else 0.0,
        },
        "api_stats": stats.to_dict(),
    }

    save_json(out_json_path, result)
    write_run_summary(run_dir / "player_time_summary.txt", result)
    write_checkpoint(
        checkpoint_path,
        started_utc,
        started_mono,
        stage="done",
        extra={
            "out_json": str(out_json_path),
            "players_at_target": int(final_live["players_at_target"]),
            "match_complete_10_of_10": int(complete_10),
            "match_complete_8_of_10": int(complete_8),
            "matches_per_hour_slice": matches_per_hour_slice,
            "matches_per_hour_complete_10_of_10": matches_per_hour_complete_10,
            "matches_per_hour_complete_8_of_10": matches_per_hour_complete_8,
            "phase_times_sec": dict(phase_times_sec),
            "slowest_phase": str(slowest_phase),
            "slowest_phase_sec": float(phase_times_sec.get(slowest_phase, 0.0)) if slowest_phase else 0.0,
            **live_api_checkpoint_extra(stats, controller),
        },
    )
    aux_conn.commit()
    aux_conn.close()

    print("\nPlayer-time run complete:")
    print(f"  elapsed_sec: {elapsed_sec:.2f}")
    print(f"  matches/hour (slice): {matches_per_hour_slice:.2f}")
    print(f"  matches/hour (10/10 complete): {matches_per_hour_complete_10:.2f}")
    print(f"  matches/hour (8/10 complete): {matches_per_hour_complete_8:.2f}")
    print(f"  slowest phase: {slowest_phase} ({phase_times_sec.get(slowest_phase, 0.0):.2f}s)")
    print(f"  API requests: {result['api_stats']['totals']['requests']}")
    print(f"  API 429: {result['api_stats']['totals']['http_429']}")
    print(f"  Output: {out_json_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    return result

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache-first player time-of-day dataset runner with balanced slice sampling and throughput monitoring."
    )
    p.add_argument("--api-key", type=str, default=os.getenv("RIOT_API_KEY", ""))
    p.add_argument(
        "--platform-routing",
        "--platform",
        dest="platform_routing",
        type=str,
        default=DEFAULT_PLATFORM_ROUTING,
        help="Platform routing for platform APIs (EUW1, NA1, KR, JP1, ...).",
    )
    p.add_argument(
        "--regional-routing",
        "--region",
        dest="regional_routing",
        type=str,
        default=None,
        help="Regional routing for match APIs (americas, asia, europe, sea). If empty, inferred from platform.",
    )
    p.add_argument("--queue", type=str, default=DEFAULT_QUEUE)
    p.add_argument(
        "--out-dir",
        type=str,
        default="runtime/out_latest/runs/prod_like/prod_test_euw_600_120",
        help="Main dataset directory holding local DB/JSON caches and run outputs.",
    )
    p.add_argument(
        "--source-matches-dir",
        type=str,
        default="",
        help="Optional directory containing local .json/.json.zst match files for cheap local detail fallback.",
    )
    p.add_argument(
        "--run-out-base-dir",
        type=str,
        default="",
        help="Base output dir; each run writes to its own subfolder inside the main dir.",
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument(
        "--resume-from-latest-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume from the latest valid checkpoint/run folder in run-out-base-dir. "
            "Reuses the persisted sampled slice and skips already-finished phases when safe."
        ),
    )
    p.add_argument("--slice-match-count", type=int, default=10000)
    p.add_argument("--slice-seed", type=int, default=42)
    p.add_argument(
        "--slice-selection",
        type=str,
        choices=["balanced", "overlap_dense"],
        default="balanced",
        help="How to choose the sampled slice from candidate matches.",
    )
    p.add_argument(
        "--refresh-slice-match-details-from-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refresh selected slice match details from Riot API if local metadata is missing.",
    )
    p.add_argument("--target-matches-per-player", type=int, default=30)
    p.add_argument("--search-buffer", type=int, default=4)
    p.add_argument("--detail-batch-size", type=int, default=10)
    p.add_argument(
        "--detail-api-chunk-size",
        type=int,
        default=64,
        help="Internal chunk size for match-detail API work inside one detail batch.",
    )
    p.add_argument(
        "--detail-progress-refresh-every",
        type=int,
        default=4,
        help="Recompute completion/checkpoint every N internal detail chunks.",
    )
    p.add_argument(
        "--detail-commit-every-chunks",
        type=int,
        default=2,
        help="Commit resolved detail work to the local cache every N chunks.",
    )
    p.add_argument(
        "--detail-plateau-window-iterations",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility. The current runner does not plateau-break detail passes.",
    )
    p.add_argument(
        "--detail-plateau-min-completed-gain",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-plateau-min-players-gain",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-plateau-min-attempted-ids",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--history-expansion-policy",
        type=str,
        choices=["simple", "completion_first", "hybrid", "fresh_first"],
        default="simple",
        help=(
            "History expansion mode. The current runner always uses the simple two-pass flow; "
            "older policy names are accepted only so legacy launch scripts keep working."
        ),
    )
    p.add_argument(
        "--second-match-ids-top-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a second Riot match_ids pass for players still below target after the first detail loop.",
    )
    p.add_argument("--second-match-ids-start", type=int, default=34)
    p.add_argument("--second-match-ids-count", type=int, default=34)
    p.add_argument(
        "--third-match-ids-top-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy flag kept for CLI compatibility. The current runner ignores pass 3.",
    )
    p.add_argument(
        "--third-match-ids-start",
        type=int,
        default=134,
        help="Legacy pass-3 setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--third-match-ids-count",
        type=int,
        default=100,
        help="Legacy pass-3 setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--third-match-id-top-up-player-limit",
        type=int,
        default=1000,
        help="Legacy pass-3 setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--match-id-top-up-player-limit",
        type=int,
        default=0,
        help="If > 0, cap the number of prioritized players processed in a top-up pass.",
    )
    p.add_argument(
        "--match-id-top-up-submit-batch-size",
        type=int,
        default=64,
        help="How many players to queue into the executor at once during match-id top-up.",
    )
    p.add_argument(
        "--match-id-top-up-progress-every",
        type=int,
        default=100,
        help="Commit/checkpoint match-id top-up progress every N completed players.",
    )
    p.add_argument(
        "--match-id-top-up-plateau-window-checkpoints",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility. The current runner does not plateau-break top-up passes.",
    )
    p.add_argument(
        "--match-id-top-up-plateau-min-done",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--match-id-top-up-plateau-min-players-gain",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--match-id-top-up-plateau-min-completed-gain",
        type=int,
        default=0,
        help="Legacy setting kept for CLI compatibility.",
    )
    p.add_argument("--queue-id", type=int, default=420)
    p.add_argument("--match-type", type=str, default="ranked")
    p.add_argument("--request-timeout-sec", type=int, default=10)
    p.add_argument("--request-max-retries", type=int, default=1)
    p.add_argument("--max-player-match-id-failures", type=int, default=2)
    p.add_argument("--max-match-detail-failures", type=int, default=2)
    p.add_argument("--workers-match-ids", type=int, default=4)
    p.add_argument("--workers-match-details", type=int, default=12)
    p.add_argument("--max-inflight-match-ids", type=int, default=None)
    p.add_argument("--max-inflight-match-details", type=int, default=None)
    p.add_argument(
        "--rate-profile",
        type=str,
        choices=["auto", "conservative", "aggressive"],
        default="auto",
    )
    p.add_argument("--app-limit-requests", type=int, default=DEFAULT_APP_LIMIT_REQUESTS)
    p.add_argument("--app-limit-window-sec", type=float, default=DEFAULT_APP_LIMIT_WINDOW_SEC)
    p.add_argument("--preflight-ttl-sec", type=int, default=86400)
    p.add_argument("--force-preflight", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--loop-interval-sec", type=int, default=120)
    p.add_argument("--loop-max-runs", type=int, default=0)
    p.add_argument("--stop-flag-file", type=str, default="STOP")
    p.add_argument("--stop-finish-min-covered-participants", type=int, default=5)
    p.add_argument(
        "--detail-focus-min-covered-participants",
        type=int,
        default=5,
        help=(
            "Legacy compatibility flag. The current runner prioritizes detail work in bands "
            "5-9, then 3-4, then 1-2, then 0 covered participants."
        ),
    )
    p.add_argument(
        "--detail-inline-match-id-refill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy flag kept for CLI compatibility. The current runner does not refill inline.",
    )
    p.add_argument(
        "--detail-inline-match-ids-start",
        type=int,
        default=134,
        help="Legacy inline-refill setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-inline-match-ids-count",
        type=int,
        default=100,
        help="Legacy inline-refill setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-inline-match-id-player-limit",
        type=int,
        default=256,
        help="Legacy inline-refill setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-inline-max-refills",
        type=int,
        default=3,
        help="Legacy inline-refill setting kept for CLI compatibility.",
    )
    p.add_argument(
        "--detail-aux-cache-refresh-every-iterations",
        type=int,
        default=1,
        help=(
            "Reload the aux cache DB into memory every N detail-loop iterations. "
            "Raise this on IO-bound folders with very large cache DBs."
        ),
    )
    p.add_argument("--health-log-file", type=str, default="run_health_log.jsonl")
    p.add_argument(
        "--candidate-match-ids-file",
        type=str,
        default="",
        help="Optional CSV/TXT file with explicit match_id targets; if present, only these candidate matches are considered.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set RIOT_API_KEY.")
    if args.slice_match_count <= 0:
        raise SystemExit("slice-match-count must be positive.")
    if args.target_matches_per_player <= 0:
        raise SystemExit("target-matches-per-player must be positive.")
    if args.search_buffer < 0:
        raise SystemExit("search-buffer must be >= 0.")
    if args.detail_batch_size <= 0:
        raise SystemExit("detail-batch-size must be positive.")
    if args.detail_api_chunk_size <= 0:
        raise SystemExit("detail-api-chunk-size must be positive.")
    if args.detail_progress_refresh_every <= 0:
        raise SystemExit("detail-progress-refresh-every must be positive.")
    if args.detail_commit_every_chunks <= 0:
        raise SystemExit("detail-commit-every-chunks must be positive.")
    if args.detail_aux_cache_refresh_every_iterations <= 0:
        raise SystemExit("--detail-aux-cache-refresh-every-iterations must be positive.")
    if args.detail_plateau_window_iterations < 0:
        raise SystemExit("detail-plateau-window-iterations must be >= 0.")
    if args.detail_plateau_min_completed_gain < 0:
        raise SystemExit("detail-plateau-min-completed-gain must be >= 0.")
    if args.detail_plateau_min_players_gain < 0:
        raise SystemExit("detail-plateau-min-players-gain must be >= 0.")
    if args.detail_plateau_min_attempted_ids < 0:
        raise SystemExit("detail-plateau-min-attempted-ids must be >= 0.")
    if args.second_match_ids_start < 0:
        raise SystemExit("second-match-ids-start must be >= 0.")
    if args.second_match_ids_count <= 0:
        raise SystemExit("second-match-ids-count must be positive.")
    if args.third_match_ids_start < 0:
        raise SystemExit("third-match-ids-start must be >= 0.")
    if args.third_match_ids_count <= 0:
        raise SystemExit("third-match-ids-count must be positive.")
    if args.match_id_top_up_player_limit < 0:
        raise SystemExit("match-id-top-up-player-limit must be >= 0.")
    if args.match_id_top_up_submit_batch_size <= 0:
        raise SystemExit("match-id-top-up-submit-batch-size must be positive.")
    if args.match_id_top_up_progress_every <= 0:
        raise SystemExit("match-id-top-up-progress-every must be positive.")
    if args.match_id_top_up_plateau_window_checkpoints < 0:
        raise SystemExit("match-id-top-up-plateau-window-checkpoints must be >= 0.")
    if args.match_id_top_up_plateau_min_done < 0:
        raise SystemExit("match-id-top-up-plateau-min-done must be >= 0.")
    if args.match_id_top_up_plateau_min_players_gain < 0:
        raise SystemExit("match-id-top-up-plateau-min-players-gain must be >= 0.")
    if args.match_id_top_up_plateau_min_completed_gain < 0:
        raise SystemExit("match-id-top-up-plateau-min-completed-gain must be >= 0.")
    if args.request_timeout_sec <= 0:
        raise SystemExit("request-timeout-sec must be positive.")
    if args.request_max_retries < 0:
        raise SystemExit("request-max-retries must be >= 0.")
    if args.max_player_match_id_failures < 0:
        raise SystemExit("max-player-match-id-failures must be >= 0.")
    if args.max_match_detail_failures < 0:
        raise SystemExit("max-match-detail-failures must be >= 0.")
    if args.app_limit_requests <= 0:
        raise SystemExit("app-limit-requests must be positive.")
    if args.app_limit_window_sec <= 0:
        raise SystemExit("app-limit-window-sec must be positive.")
    if args.loop_interval_sec < 0:
        raise SystemExit("loop-interval-sec must be >= 0.")
    if args.loop_max_runs < 0:
        raise SystemExit("loop-max-runs must be >= 0.")
    if args.stop_finish_min_covered_participants < 0:
        raise SystemExit("stop-finish-min-covered-participants must be >= 0.")

    apply_history_expansion_policy(args)

    if not str(args.source_matches_dir or "").strip():
        args.source_matches_dir = str(Path(args.out_dir) / "matches")
    if not str(args.run_out_base_dir or "").strip():
        args.run_out_base_dir = str(Path(args.out_dir) / "player_time_runs")

    try:
        platform_routing, regional_routing = configure_api_bases(
            args.platform_routing,
            args.regional_routing,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.platform_routing = platform_routing
    args.regional_routing = regional_routing
    print(
        "Routing:"
        f" platform={args.platform_routing} (https://{args.platform_routing.lower()}.api.riotgames.com)"
        f", regional={args.regional_routing} (https://{args.regional_routing.lower()}.api.riotgames.com)"
    )

    run_idx = 0
    stop_flag_path = resolve_stop_flag_path(args)
    health_log_path = Path(args.out_dir) / args.health_log_file

    while True:
        if args.loop and run_idx == 0 and stop_flag_path is not None and stop_flag_path.exists():
            print(f"Stop flag present before first run, exiting loop: {stop_flag_path}")
            return NO_RESTART_EXIT_CODE

        run_idx += 1
        run_started = int(time.time())
        print(f"\n=== Player-Time Run {run_idx} ===")
        try:
            crawl_stats = run_player_time_dataset(args=args, run_idx=run_idx)
            health_entry = {
                "run_idx": int(run_idx),
                "run_started_utc": int(run_started),
                "run_finished_utc": int(time.time()),
                "ok": True,
                "api_totals": crawl_stats.get("api_stats", {}).get("totals", {}),
                "coverage": crawl_stats.get("coverage", {}),
                "phases": crawl_stats.get("phases", {}),
            }
        except FatalRiotAuthError as exc:
            health_entry = {
                "run_idx": int(run_idx),
                "run_started_utc": int(run_started),
                "run_finished_utc": int(time.time()),
                "ok": False,
                "fatal_auth_error": True,
                "error": str(exc),
            }
            append_jsonl(health_log_path, health_entry)
            print(f"Run {run_idx} fatal auth error: {exc}")
            print(f"Failure recorded to: {health_log_path}")
            if not args.loop:
                raise
            print("Stopping loop due to fatal auth error (401/403).")
            break
        except Exception as exc:
            health_entry = {
                "run_idx": int(run_idx),
                "run_started_utc": int(run_started),
                "run_finished_utc": int(time.time()),
                "ok": False,
                "error": str(exc),
            }
            append_jsonl(health_log_path, health_entry)
            if not args.loop:
                raise
            print(f"Run {run_idx} failed: {exc}")
            print(f"Failure recorded to: {health_log_path}")
        else:
            append_jsonl(health_log_path, health_entry)
            print(f"Health log updated: {health_log_path}")

        if not args.loop:
            return 0
        if args.loop and stop_flag_path is not None and stop_flag_path.exists():
            print(f"Stop flag detected. Exiting loop: {stop_flag_path}")
            return NO_RESTART_EXIT_CODE
        if args.loop_max_runs > 0 and run_idx >= args.loop_max_runs:
            print(f"Loop finished after {run_idx} runs.")
            return NO_RESTART_EXIT_CODE
        if args.loop_interval_sec > 0:
            print(f"Sleeping {args.loop_interval_sec}s before next run...")
            time.sleep(args.loop_interval_sec)
        else:
            print("Starting next run immediately (loop-interval-sec=0).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
