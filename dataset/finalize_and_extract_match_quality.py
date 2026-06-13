from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT / "dataset", REPO_ROOT / "rank_mapping"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from probit_core import RankLpProbitMapper
from probit_settings import (
    CHALLENGER_CUTOFF_LP,
    GM_CUTOFF_LP,
    RANK1_LP,
    apex_lp_cutoffs_for_server,
    target_percentages_for_server,
)


JOB_TABLES = ("jobs_match_ids", "jobs_match_details", "jobs_rank_lookup")
RANKED_DIVISION_TIERS = {"IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"}
RANKED_DIVISIONS = {"IV", "III", "II", "I"}


@dataclass
class StopResult:
    stopfile_path: Path
    stopfile_written_utc: int
    timed_out: bool
    forced_stop: bool
    force_killed_pids: list[int]
    graceful_exit: bool
    process_exit_utc: int
    initial_pids: list[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize running pipelines and export per-pipeline match quality + average skill CSVs."
    )
    p.add_argument(
        "--pipeline-dir",
        action="append",
        required=True,
        help="Pipeline output directory (repeat for multiple pipelines).",
    )
    p.add_argument("--stop-flag-name", type=str, default="STOP")
    p.add_argument("--stop-timeout-min", type=int, default=90)
    p.add_argument(
        "--kill-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force-kill a pipeline process if it does not stop in time.",
    )
    p.add_argument("--required-unique-participants", type=int, default=10)
    p.add_argument("--min-mapped-participants", type=int, default=9)
    p.add_argument("--gm-cutoff-lp", type=float, default=GM_CUTOFF_LP)
    p.add_argument("--challenger-cutoff-lp", type=float, default=CHALLENGER_CUTOFF_LP)
    p.add_argument("--rank1-lp", type=float, default=RANK1_LP)
    p.add_argument(
        "--poll-sec",
        type=float,
        default=5.0,
        help="Polling interval while waiting for graceful stop.",
    )
    return p.parse_args()


def utc_now() -> int:
    return int(time.time())


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_health_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def get_db_job_counts(db_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for table in JOB_TABLES:
            rows = cur.execute(f"SELECT status, COUNT(*) FROM {table} GROUP BY status").fetchall()
            row_map = {str(r[0]): int(r[1]) for r in rows}
            counts[table] = {
                "pending": row_map.get("pending", 0),
                "running": row_map.get("running", 0),
                "done": row_map.get("done", 0),
                "failed": row_map.get("failed", 0),
            }
    finally:
        conn.close()
    return counts


def list_python_processes() -> list[dict[str, Any]]:
    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$p = Get-CimInstance Win32_Process "
                "| Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'riot_euw_smoke.py|main_dataset.py') } "
                "| Select-Object ProcessId, CommandLine; "
                "if ($p) { $p | ConvertTo-Json -Compress }"
            ),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return []
        raw = proc.stdout.strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, dict):
            decoded = [decoded]
        out: list[dict[str, Any]] = []
        for item in decoded:
            if not isinstance(item, dict):
                continue
            pid = item.get("ProcessId")
            cmdline = item.get("CommandLine")
            if pid is None or cmdline is None:
                continue
            out.append({"pid": int(pid), "command_line": str(cmdline)})
        return out

    # Linux/Unix path: inspect process table.
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    out: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, cmdline = parts
        if not pid_text.isdigit():
            continue
        cmd_lower = cmdline.lower()
        if "python" not in cmd_lower:
            continue
        if "riot_euw_smoke.py" not in cmd_lower and "main_dataset.py" not in cmd_lower:
            continue
        out.append({"pid": int(pid_text), "command_line": cmdline})
    return out


def normalize_path_token(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


def detect_pipeline_pids(pipeline_dir: Path, repo_root: Path) -> list[int]:
    processes = list_python_processes()
    abs_token = normalize_path_token(pipeline_dir)
    rel_token = ""
    try:
        rel_token = normalize_path_token(pipeline_dir.relative_to(repo_root))
    except ValueError:
        rel_token = ""

    matched: list[int] = []
    for proc in processes:
        cmdline = str(proc["command_line"]).lower().replace("\\", "/")
        if abs_token and abs_token in cmdline:
            matched.append(int(proc["pid"]))
            continue
        if rel_token and rel_token in cmdline:
            matched.append(int(proc["pid"]))
            continue
    return sorted(set(matched))


def kill_pids(pids: list[int]) -> None:
    if os.name == "nt":
        for pid in pids:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        return

    # Linux/Unix: TERM first, then KILL fallback.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    time.sleep(1.0)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def write_stop_file(pipeline_dir: Path, stop_flag_name: str) -> tuple[Path, int]:
    stop_path = pipeline_dir / stop_flag_name
    ts = utc_now()
    stop_path.write_text(f"stop requested utc={ts}\n", encoding="utf-8")
    return stop_path, ts


def stop_pipeline(
    pipeline_dir: Path,
    stop_flag_name: str,
    timeout_min: int,
    kill_on_timeout: bool,
    poll_sec: float,
    repo_root: Path,
) -> StopResult:
    stopfile_path, stop_written = write_stop_file(pipeline_dir, stop_flag_name)
    initial_pids = detect_pipeline_pids(pipeline_dir, repo_root)
    deadline = time.monotonic() + max(0.0, float(timeout_min) * 60.0)
    force_killed: list[int] = []
    timed_out = False
    graceful_exit = False

    while True:
        current = detect_pipeline_pids(pipeline_dir, repo_root)
        if not current:
            graceful_exit = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            if kill_on_timeout:
                force_killed = current
                kill_pids(current)
                # Allow OS a brief moment to reap processes.
                time.sleep(1.0)
            break
        time.sleep(max(0.2, poll_sec))

    # One more check for final state.
    remaining = detect_pipeline_pids(pipeline_dir, repo_root)
    if not remaining:
        graceful_exit = not timed_out or (timed_out and kill_on_timeout)

    return StopResult(
        stopfile_path=stopfile_path,
        stopfile_written_utc=stop_written,
        timed_out=timed_out,
        forced_stop=bool(force_killed),
        force_killed_pids=force_killed,
        graceful_exit=graceful_exit,
        process_exit_utc=utc_now(),
        initial_pids=initial_pids,
    )


def normalize_rank_name(solo_tier: str | None, solo_rank: str | None) -> str | None:
    if solo_tier is None:
        return None
    tier = str(solo_tier).strip().upper()
    rank = "" if solo_rank is None else str(solo_rank).strip().upper()
    if tier in RANKED_DIVISION_TIERS:
        if rank not in RANKED_DIVISIONS:
            return None
        return f"{tier.title()} {rank}"
    if tier == "MASTER":
        return "Master"
    if tier == "GRANDMASTER":
        return "GrandMaster"
    if tier == "CHALLENGER":
        return "Challenger"
    return None


def load_ranks_by_puuid(db_path: Path) -> dict[str, tuple[str, float]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: dict[str, tuple[str, float]] = {}
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT puuid, solo_tier, solo_rank, solo_lp FROM player_ranks"
        ).fetchall()
        for row in rows:
            puuid = str(row["puuid"])
            rank_name = normalize_rank_name(row["solo_tier"], row["solo_rank"])
            if rank_name is None:
                continue
            lp_raw = row["solo_lp"]
            lp_value = 0.0 if lp_raw is None else max(0.0, float(lp_raw))
            out[puuid] = (rank_name, lp_value)
    finally:
        conn.close()
    return out


def infer_server_for_pipeline(db_path: Path) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT match_id FROM match_participants WHERE instr(match_id, '_') > 0 LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        match_id = str(row["match_id"] or "")
        if "_" not in match_id:
            return None
        return match_id.split("_", 1)[0]
    finally:
        conn.close()


def summarize_new_runs(
    health_before: list[dict[str, Any]],
    health_after: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    new_runs = health_after[len(health_before) :]
    totals = {
        "completed_runs": len(new_runs),
        "requests": 0,
        "success": 0,
        "retries": 0,
        "http_429": 0,
        "http_5xx": 0,
        "errors": 0,
        "failed_jobs": 0,
    }
    for run in new_runs:
        run_totals = run.get("totals", {}) if isinstance(run, dict) else {}
        run_health = run.get("health", {}) if isinstance(run, dict) else {}
        totals["requests"] += int(run_totals.get("requests", 0) or 0)
        totals["success"] += int(run_totals.get("success", 0) or 0)
        totals["retries"] += int(run_totals.get("retries", 0) or 0)
        totals["http_429"] += int(run_totals.get("http_429", 0) or 0)
        totals["http_5xx"] += int(run_totals.get("http_5xx", 0) or 0)
        totals["errors"] += int(run_totals.get("errors", 0) or 0)
        totals["failed_jobs"] += int(run_health.get("failed_jobs", 0) or 0)
    return new_runs, totals


def delta_job_counts(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for table in JOB_TABLES:
        out[table] = {}
        for status in ("pending", "running", "done", "failed"):
            out[table][status] = int(after.get(table, {}).get(status, 0)) - int(
                before.get(table, {}).get(status, 0)
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    seq = sorted(values)
    pos = (len(seq) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return seq[lo]
    frac = pos - lo
    return seq[lo] + frac * (seq[hi] - seq[lo])


def compute_outputs_for_pipeline(
    pipeline_dir: Path,
    mapper: RankLpProbitMapper,
    required_unique: int,
    min_mapped: int,
    export_dir: Path,
) -> dict[str, Any]:
    participant_index_path = pipeline_dir / "participant_index_by_match.json"
    db_path = pipeline_dir / "player_ranks.sqlite3"

    participants_by_match = safe_read_json(participant_index_path)
    if not isinstance(participants_by_match, dict):
        raise RuntimeError(f"Invalid participant index format: {participant_index_path}")

    ranks_by_puuid = load_ranks_by_puuid(db_path)

    quality_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    included_skills: list[float] = []

    total_matches = 0
    excluded_not_exactly = 0
    excluded_low_mapped = 0

    for match_id in sorted(participants_by_match.keys()):
        raw_arr = participants_by_match.get(match_id)
        if not isinstance(raw_arr, list):
            continue
        total_matches += 1
        unique_puuids = list(dict.fromkeys(str(x) for x in raw_arr if x))
        unique_count = len(unique_puuids)
        eligible_exactly_10 = unique_count == required_unique

        z_values: list[float] = []
        if eligible_exactly_10:
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

        mapped_count = len(z_values)
        eligible_mapped_ge_9 = mapped_count >= min_mapped
        included = eligible_exactly_10 and eligible_mapped_ge_9

        if not eligible_exactly_10:
            exclude_reason = "not_exactly_10"
            excluded_not_exactly += 1
        elif not eligible_mapped_ge_9:
            exclude_reason = "mapped_lt_9"
            excluded_low_mapped += 1
        else:
            exclude_reason = ""

        quality_rows.append(
            {
                "match_id": match_id,
                "unique_participants": unique_count,
                "mapped_participants": mapped_count,
                "eligible_exactly_10": eligible_exactly_10,
                "eligible_mapped_ge_9": eligible_mapped_ge_9,
                "included_in_skill_output": included,
                "exclude_reason": exclude_reason,
            }
        )

        if included:
            avg_skill = sum(z_values) / float(mapped_count)
            if not math.isfinite(avg_skill):
                continue
            included_skills.append(avg_skill)
            skill_rows.append(
                {
                    "match_id": match_id,
                    "average_skill_level": f"{avg_skill:.6f}",
                    "mapped_participants": mapped_count,
                    "unique_participants": unique_count,
                }
            )

    # Validation checks.
    quality_included_count = sum(1 for r in quality_rows if bool(r["included_in_skill_output"]))
    if quality_included_count != len(skill_rows):
        raise RuntimeError(
            "Validation failed: included rows in match_quality.csv do not match match_average_skill.csv count."
        )
    seen_match_ids: set[str] = set()
    for row in skill_rows:
        mid = str(row["match_id"])
        if mid in seen_match_ids:
            raise RuntimeError("Validation failed: duplicate match_id in match_average_skill.csv.")
        seen_match_ids.add(mid)
        if not math.isfinite(float(row["average_skill_level"])):
            raise RuntimeError("Validation failed: non-finite average_skill_level found.")

    # Write CSV outputs.
    write_csv(
        export_dir / "match_quality.csv",
        quality_rows,
        [
            "match_id",
            "unique_participants",
            "mapped_participants",
            "eligible_exactly_10",
            "eligible_mapped_ge_9",
            "included_in_skill_output",
            "exclude_reason",
        ],
    )
    write_csv(
        export_dir / "match_average_skill.csv",
        skill_rows,
        ["match_id", "average_skill_level", "mapped_participants", "unique_participants"],
    )

    stats = {
        "total_matches_seen": total_matches,
        "matches_exactly_10": total_matches - excluded_not_exactly,
        "matches_with_mapped_ge_min": quality_included_count,
        "included_matches": len(skill_rows),
        "dropped_not_exactly_10": excluded_not_exactly,
        "dropped_mapped_lt_min": excluded_low_mapped,
        "avg_skill_stats": {
            "count": len(included_skills),
            "min": (min(included_skills) if included_skills else None),
            "mean": (statistics.fmean(included_skills) if included_skills else None),
            "median": (statistics.median(included_skills) if included_skills else None),
            "p90": (percentile(included_skills, 0.90) if included_skills else None),
            "max": (max(included_skills) if included_skills else None),
        },
    }
    return stats


def write_pipeline_reports(
    export_dir: Path,
    report: dict[str, Any],
) -> None:
    json_path = export_dir / "finalization_report.json"
    txt_path = export_dir / "finalization_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"Pipeline: {report['pipeline_dir']}")
    lines.append(f"Export dir: {export_dir}")
    lines.append(f"Generated UTC: {report['generated_utc']}")
    lines.append("")
    stop = report["stop_behavior"]
    lines.append("Stop behavior:")
    lines.append(f"  graceful_exit: {stop['graceful_exit']}")
    lines.append(f"  timed_out: {stop['timed_out']}")
    lines.append(f"  forced_stop: {stop['forced_stop']}")
    lines.append(f"  force_killed_pids: {stop['force_killed_pids']}")
    lines.append("")
    run_window = report["run_window"]
    lines.append("Run window totals:")
    lines.append(
        "  completed_runs={completed_runs} requests={requests} success={success} "
        "retries={retries} http_429={http_429} errors={errors} failed_jobs={failed_jobs}".format(
            **run_window["totals"]
        )
    )
    lines.append("")
    q = report["quality_and_skill"]
    lines.append("Quality + skill output stats:")
    lines.append(
        "  total_matches_seen={total_matches_seen} matches_exactly_10={matches_exactly_10} "
        "included_matches={included_matches} dropped_not_exactly_10={dropped_not_exactly_10} "
        "dropped_mapped_lt_min={dropped_mapped_lt_min}".format(**q)
    )
    avg = q["avg_skill_stats"]
    lines.append(
        "  avg_skill count={count} min={min} mean={mean} median={median} p90={p90} max={max}".format(
            **avg
        )
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.required_unique_participants != 10:
        raise SystemExit("This workflow expects required-unique-participants=10.")
    if args.min_mapped_participants < 1:
        raise SystemExit("min-mapped-participants must be positive.")
    if args.stop_timeout_min < 0:
        raise SystemExit("stop-timeout-min must be >= 0.")
    if args.poll_sec <= 0:
        raise SystemExit("poll-sec must be > 0.")

    repo_root = Path(__file__).resolve().parents[1]
    export_stamp = utc_stamp()

    for pipeline in args.pipeline_dir:
        pipeline_dir = Path(pipeline).resolve()
        if not pipeline_dir.exists():
            raise SystemExit(f"Pipeline directory does not exist: {pipeline_dir}")
        participant_index = pipeline_dir / "participant_index_by_match.json"
        db_path = pipeline_dir / "player_ranks.sqlite3"
        health_path = pipeline_dir / "run_health_log.jsonl"
        if not participant_index.exists():
            raise SystemExit(f"Missing participant_index_by_match.json: {participant_index}")
        if not db_path.exists():
            raise SystemExit(f"Missing player_ranks.sqlite3: {db_path}")
        if not health_path.exists():
            raise SystemExit(f"Missing run_health_log.jsonl: {health_path}")

    for pipeline in args.pipeline_dir:
        pipeline_dir = Path(pipeline).resolve()
        print(f"\n=== Finalizing: {pipeline_dir} ===")
        export_dir = pipeline_dir / "final_exports" / export_stamp
        export_dir.mkdir(parents=True, exist_ok=True)

        health_before = read_health_entries(pipeline_dir / "run_health_log.jsonl")
        jobs_before = get_db_job_counts(pipeline_dir / "player_ranks.sqlite3")
        crawl_before = {}
        crawl_stats_path = pipeline_dir / "crawl_stats.json"
        if crawl_stats_path.exists():
            try:
                crawl_before = safe_read_json(crawl_stats_path)
            except Exception:
                crawl_before = {}

        stop_result = stop_pipeline(
            pipeline_dir=pipeline_dir,
            stop_flag_name=args.stop_flag_name,
            timeout_min=args.stop_timeout_min,
            kill_on_timeout=bool(args.kill_on_timeout),
            poll_sec=float(args.poll_sec),
            repo_root=repo_root,
        )
        print(
            f"Stop result: graceful={stop_result.graceful_exit}, "
            f"timed_out={stop_result.timed_out}, forced={stop_result.forced_stop}"
        )

        health_after = read_health_entries(pipeline_dir / "run_health_log.jsonl")
        jobs_after = get_db_job_counts(pipeline_dir / "player_ranks.sqlite3")
        crawl_after = {}
        if crawl_stats_path.exists():
            try:
                crawl_after = safe_read_json(crawl_stats_path)
            except Exception:
                crawl_after = {}

        new_runs, run_window_totals = summarize_new_runs(health_before, health_after)
        jobs_delta = delta_job_counts(jobs_before, jobs_after)

        server = infer_server_for_pipeline(pipeline_dir / "player_ranks.sqlite3")
        quality_stats = compute_outputs_for_pipeline(
            pipeline_dir=pipeline_dir,
            mapper=RankLpProbitMapper(
                target_percentages=target_percentages_for_server(server),
                floor_epsilon_pct=0.01,
                ceil_epsilon_pct=0.01,
                apex_lp_cutoffs=apex_lp_cutoffs_for_server(server),
            ),
            required_unique=int(args.required_unique_participants),
            min_mapped=int(args.min_mapped_participants),
            export_dir=export_dir,
        )

        actual_cutoffs = apex_lp_cutoffs_for_server(server)

        report = {
            "pipeline_dir": str(pipeline_dir),
            "generated_utc": utc_now(),
            "server": server,
            "config": {
                "required_unique_participants": int(args.required_unique_participants),
                "min_mapped_participants": int(args.min_mapped_participants),
                "gm_cutoff_lp": float(actual_cutoffs["gm_cutoff_lp"]),
                "challenger_cutoff_lp": float(actual_cutoffs["challenger_cutoff_lp"]),
                "rank1_lp": float(actual_cutoffs["rank1_lp"]),
                "epsilon_pct": 0.01,
            },
            "stop_behavior": {
                "stopfile_path": str(stop_result.stopfile_path),
                "stopfile_written_utc": stop_result.stopfile_written_utc,
                "timed_out": stop_result.timed_out,
                "forced_stop": stop_result.forced_stop,
                "force_killed_pids": stop_result.force_killed_pids,
                "graceful_exit": stop_result.graceful_exit,
                "process_exit_utc": stop_result.process_exit_utc,
                "initial_pids": stop_result.initial_pids,
            },
            "run_window": {
                "baseline_health_count": len(health_before),
                "final_health_count": len(health_after),
                "new_runs": new_runs,
                "totals": run_window_totals,
            },
            "db_jobs": {
                "before": jobs_before,
                "after": jobs_after,
                "delta": jobs_delta,
            },
            "crawl_stats": {
                "before": crawl_before,
                "after": crawl_after,
            },
            "quality_and_skill": quality_stats,
            "output_files": {
                "match_quality_csv": str(export_dir / "match_quality.csv"),
                "match_average_skill_csv": str(export_dir / "match_average_skill.csv"),
                "finalization_report_json": str(export_dir / "finalization_report.json"),
                "finalization_report_txt": str(export_dir / "finalization_report.txt"),
            },
        }
        write_pipeline_reports(export_dir, report)
        print(f"Wrote exports to: {export_dir}")

    print("\nAll pipelines finalized and exported.")


if __name__ == "__main__":
    main()
