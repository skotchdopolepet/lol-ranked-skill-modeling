from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any


def _now_utc() -> int:
    return int(time.time())


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def ensure_spool_dirs(spool_dir: Path) -> dict[str, Path]:
    base = Path(spool_dir)
    dirs = {
        "base": base,
        "requests_pending": base / "requests" / "pending",
        "requests_claimed": base / "requests" / "claimed",
        "requests_done": base / "requests" / "done",
        "results_pending": base / "results" / "pending",
        "results_consumed": base / "results" / "consumed",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def spool_counts(spool_dir: Path) -> dict[str, int]:
    dirs = ensure_spool_dirs(spool_dir)
    return {
        "requests_pending": sum(1 for _ in dirs["requests_pending"].glob("*.json")),
        "requests_claimed": sum(1 for _ in dirs["requests_claimed"].glob("*.json")),
        "requests_done": sum(1 for _ in dirs["requests_done"].glob("*.json")),
        "results_pending": sum(1 for _ in dirs["results_pending"].glob("*.json")),
        "results_consumed": sum(1 for _ in dirs["results_consumed"].glob("*.json")),
    }


def default_actor_id(prefix: str) -> str:
    host = socket.gethostname().strip().lower() or "host"
    return f"{prefix}-{host}-{uuid.uuid4().hex[:8]}"


def submit_request_batches(
    *,
    spool_dir: Path,
    match_ids: list[str],
    origin_id: str,
    batch_size: int,
    band: str,
    run_id: str,
    stage: str,
) -> list[dict[str, Any]]:
    dirs = ensure_spool_dirs(spool_dir)
    clean_ids: list[str] = []
    seen: set[str] = set()
    for match_id in match_ids:
        token = str(match_id or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        clean_ids.append(token)

    out: list[dict[str, Any]] = []
    width = max(1, int(batch_size))
    for idx in range(0, len(clean_ids), width):
        chunk = clean_ids[idx : idx + width]
        now_utc = _now_utc()
        batch_id = f"{now_utc}_{uuid.uuid4().hex[:12]}"
        payload = {
            "batch_id": batch_id,
            "origin_id": str(origin_id),
            "band": str(band),
            "run_id": str(run_id),
            "stage": str(stage),
            "created_at_utc": now_utc,
            "claimed_at_utc": None,
            "claimed_by": "",
            "requeue_count": 0,
            "match_ids": chunk,
            "match_count": len(chunk),
        }
        _save_json(dirs["requests_pending"] / f"{batch_id}.json", payload)
        out.append(payload)
    return out


def load_pending_match_ids(spool_dir: Path) -> set[str]:
    dirs = ensure_spool_dirs(spool_dir)
    out: set[str] = set()
    for root in (dirs["requests_pending"], dirs["requests_claimed"]):
        for path in sorted(root.glob("*.json")):
            try:
                payload = _load_json(path)
            except Exception:
                continue
            for match_id in payload.get("match_ids", []):
                token = str(match_id or "").strip()
                if token:
                    out.add(token)
    return out


def requeue_stale_claims(spool_dir: Path, stale_after_sec: int) -> list[dict[str, Any]]:
    dirs = ensure_spool_dirs(spool_dir)
    reclaimed: list[dict[str, Any]] = []
    now_utc = _now_utc()
    stale_after = max(1, int(stale_after_sec))
    for path in sorted(dirs["requests_claimed"].glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        claimed_at = int(payload.get("claimed_at_utc") or payload.get("created_at_utc") or 0)
        if claimed_at <= 0 or (now_utc - claimed_at) < stale_after:
            continue
        payload["claimed_at_utc"] = None
        payload["claimed_by"] = ""
        payload["requeue_count"] = int(payload.get("requeue_count") or 0) + 1
        pending_path = dirs["requests_pending"] / f"{str(payload.get('batch_id') or path.stem)}.json"
        _save_json(pending_path, payload)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        reclaimed.append(payload)
    return reclaimed


def requeue_claims_for_helper(spool_dir: Path, helper_id: str) -> list[dict[str, Any]]:
    dirs = ensure_spool_dirs(spool_dir)
    reclaimed: list[dict[str, Any]] = []
    suffix = f"__{str(helper_id)}.json"
    for path in sorted(dirs["requests_claimed"].glob(f"*{suffix}")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        payload["claimed_at_utc"] = None
        payload["claimed_by"] = ""
        payload["requeue_count"] = int(payload.get("requeue_count") or 0) + 1
        pending_path = dirs["requests_pending"] / f"{str(payload.get('batch_id') or path.stem)}.json"
        _save_json(pending_path, payload)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        reclaimed.append(payload)
    return reclaimed


def claim_next_request(
    *,
    spool_dir: Path,
    helper_id: str,
    stale_after_sec: int,
) -> dict[str, Any] | None:
    dirs = ensure_spool_dirs(spool_dir)
    requeue_stale_claims(spool_dir, stale_after_sec)
    for path in sorted(dirs["requests_pending"].glob("*.json")):
        claimed_path = dirs["requests_claimed"] / f"{path.stem}__{helper_id}.json"
        try:
            path.replace(claimed_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        try:
            payload = _load_json(claimed_path)
        except Exception:
            try:
                claimed_path.unlink()
            except FileNotFoundError:
                pass
            continue
        payload["claimed_by"] = str(helper_id)
        payload["claimed_at_utc"] = _now_utc()
        _save_json(claimed_path, payload)
        return payload
    return None


def publish_result(
    *,
    spool_dir: Path,
    claimed_request: dict[str, Any],
    helper_id: str,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    processing_started_utc: int | None = None,
    processing_elapsed_sec: float | None = None,
) -> dict[str, Any]:
    dirs = ensure_spool_dirs(spool_dir)
    batch_id = str(claimed_request.get("batch_id") or "")
    success_count = len(successes)
    failure_count = len(failures)
    total_count = success_count + failure_count
    elapsed = float(processing_elapsed_sec or 0.0)
    match_ids_per_sec = (float(total_count) / elapsed) if elapsed > 0 else 0.0
    success_match_ids_per_sec = (float(success_count) / elapsed) if elapsed > 0 else 0.0
    sec_per_1000_total = (elapsed / float(total_count) * 1000.0) if total_count > 0 else None
    sec_per_1000_success = (elapsed / float(success_count) * 1000.0) if success_count > 0 else None
    payload = {
        "batch_id": batch_id,
        "origin_id": str(claimed_request.get("origin_id") or ""),
        "helper_id": str(helper_id),
        "band": str(claimed_request.get("band") or ""),
        "run_id": str(claimed_request.get("run_id") or ""),
        "stage": str(claimed_request.get("stage") or ""),
        "created_at_utc": int(claimed_request.get("created_at_utc") or 0),
        "claimed_at_utc": int(claimed_request.get("claimed_at_utc") or 0),
        "processing_started_utc": (
            int(processing_started_utc) if processing_started_utc is not None else int(_now_utc())
        ),
        "completed_at_utc": _now_utc(),
        "match_count": int(claimed_request.get("match_count") or len(claimed_request.get("match_ids", []))),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "processing_elapsed_sec": elapsed,
        "match_ids_per_sec": match_ids_per_sec,
        "success_match_ids_per_sec": success_match_ids_per_sec,
        "sec_per_1000_total": sec_per_1000_total,
        "sec_per_1000_success": sec_per_1000_success,
        "successes": list(successes),
        "failures": list(failures),
    }
    result_path = dirs["results_pending"] / f"{batch_id}__{helper_id}.json"
    _save_json(result_path, payload)

    claim_name = f"{batch_id}__{helper_id}.json"
    claimed_path = dirs["requests_claimed"] / claim_name
    done_path = dirs["requests_done"] / claim_name
    if claimed_path.exists():
        moved = False
        for _attempt in range(10):
            try:
                claimed_path.replace(done_path)
                moved = True
                break
            except FileNotFoundError:
                moved = True
                break
            except OSError:
                time.sleep(0.05)
        if not moved:
            _save_json(done_path, claimed_request)
    return payload


def consume_available_results(spool_dir: Path) -> list[dict[str, Any]]:
    dirs = ensure_spool_dirs(spool_dir)
    out: list[dict[str, Any]] = []
    for path in sorted(dirs["results_pending"].glob("*.json")):
        consumed_path = dirs["results_consumed"] / path.name
        try:
            path.replace(consumed_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        try:
            payload = _load_json(consumed_path)
        except Exception:
            continue
        out.append(payload)
    return out
