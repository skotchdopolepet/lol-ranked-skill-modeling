from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_process_snapshot(command_pattern: str) -> Any:
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match "
        f"'{command_pattern}' }}"
        " | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def db_counts(db_path: Path) -> dict[str, int]:
    out = {
        "jobs_match_ids_done": 0,
        "jobs_match_details_done": 0,
        "matches": 0,
        "match_participants": 0,
        "match_ids_cache": 0,
    }
    if not db_path.exists():
        return out
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        out["matches"] = int(cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        out["match_participants"] = int(cur.execute("SELECT COUNT(*) FROM match_participants").fetchone()[0])
        out["match_ids_cache"] = int(cur.execute("SELECT COUNT(*) FROM match_ids_cache").fetchone()[0])
        out["jobs_match_ids_done"] = int(
            cur.execute("SELECT COUNT(*) FROM jobs_match_ids WHERE status='done'").fetchone()[0]
        )
        out["jobs_match_details_done"] = int(
            cur.execute("SELECT COUNT(*) FROM jobs_match_details WHERE status='done'").fetchone()[0]
        )
    finally:
        conn.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Launch player_dataset after bootstrap main_dataset is ready.")
    p.add_argument("--bootstrap-dir", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--main-command-pattern", required=True)
    p.add_argument("--player-run-id", required=True)
    p.add_argument("--min-ready-matches", type=int, default=20)
    p.add_argument("--poll-sec", type=int, default=120)
    p.add_argument("--stable-polls", type=int, default=3)
    args = p.parse_args()

    bootstrap_dir = Path(args.bootstrap_dir)
    db_path = bootstrap_dir / "player_ranks.sqlite3"
    run_out_base_dir = bootstrap_dir / "player_time_runs"
    monitor_dir = bootstrap_dir / "monitor_reports"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    state_path = monitor_dir / "auto_launch_state.json"

    last_done = None
    stable_count = 0

    while True:
        process = read_process_snapshot(str(args.main_command_pattern))
        counts = db_counts(db_path)

        current_done = int(counts["jobs_match_details_done"])
        if last_done is not None and current_done == last_done:
            stable_count += 1
        else:
            stable_count = 0
        last_done = current_done

        state = {
            "updated_utc": int(time.time()),
            "main_process_active": bool(process),
            "counts": counts,
            "stable_detail_done_polls": int(stable_count),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        ready_by_size = current_done >= int(args.min_ready_matches)
        ready_by_stall = ready_by_size and stable_count >= int(args.stable_polls)
        ready_by_exit = ready_by_size and not bool(process)

        if ready_by_stall or ready_by_exit:
            slice_count = max(int(args.min_ready_matches), current_done)
            stdout_log = bootstrap_dir / "player_dataset_auto.stdout.log"
            stderr_log = bootstrap_dir / "player_dataset_auto.stderr.log"
            cmd = [
                sys.executable,
                "-u",
                "player_dataset.py",
                "--api-key",
                str(args.api_key),
                "--out-dir",
                str(bootstrap_dir),
                "--source-matches-dir",
                str(bootstrap_dir / "matches"),
                "--run-out-base-dir",
                str(run_out_base_dir),
                "--run-id",
                str(args.player_run_id),
                "--slice-match-count",
                str(slice_count),
                "--slice-seed",
                "42",
                "--target-matches-per-player",
                "30",
                "--search-buffer",
                "6",
                "--detail-batch-size",
                "200",
                "--workers-match-ids",
                "4",
                "--workers-match-details",
                "12",
                "--rate-profile",
                "aggressive",
                "--app-limit-requests",
                "120",
                "--app-limit-window-sec",
                "120",
                "--request-timeout-sec",
                "8",
                "--request-max-retries",
                "1",
            ]
            with stdout_log.open("a", encoding="utf-8") as out_f, stderr_log.open(
                "a", encoding="utf-8"
            ) as err_f:
                subprocess.Popen(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    cwd=str(Path(__file__).resolve().parent),
                )

            monitor_cmd = [
                sys.executable,
                "monitor_player_run.py",
                "--run-dir",
                str(run_out_base_dir / str(args.player_run_id)),
                "--stdout-log",
                str(stdout_log),
                "--stderr-log",
                str(stderr_log),
                "--report-dir",
                str(monitor_dir),
                "--command-pattern",
                str(args.player_run_id),
                "--interval-sec",
                "1800",
            ]
            subprocess.Popen(monitor_cmd, cwd=str(Path(__file__).resolve().parent))

            launch_state = {
                "launched_utc": int(time.time()),
                "slice_count": int(slice_count),
                "player_run_id": str(args.player_run_id),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
            (monitor_dir / "auto_launch_result.json").write_text(
                json.dumps(launch_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            break

        time.sleep(max(10, int(args.poll_sec)))


if __name__ == "__main__":
    main()
