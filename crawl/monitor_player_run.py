from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def try_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_tail(path: Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception:
        return []


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


def build_report(
    *,
    run_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    command_pattern: str,
    monitor_started_utc: int,
) -> dict[str, Any]:
    checkpoint_path = run_dir / "player_time_run.checkpoint.json"
    run_json_path = run_dir / "player_time_run.json"
    checkpoint = try_load_json(checkpoint_path)
    run_json = try_load_json(run_json_path)
    process = read_process_snapshot(command_pattern)
    stdout_tail = read_tail(stdout_path, 120)
    stderr_tail = read_tail(stderr_path, 80)

    stage = None
    elapsed_sec = None
    requests = None
    http_429 = None
    est_hours_total = None
    complete_10 = None
    complete_8 = None

    if isinstance(run_json, dict):
        stage = "done"
        elapsed_sec = float(run_json.get("elapsed_sec", 0.0) or 0.0)
        requests = int(run_json.get("api_stats", {}).get("totals", {}).get("requests", 0) or 0)
        http_429 = int(run_json.get("api_stats", {}).get("totals", {}).get("http_429", 0) or 0)
        complete_10 = int(run_json.get("coverage", {}).get("match_complete_10_of_10", 0) or 0)
        complete_8 = int(run_json.get("coverage", {}).get("match_complete_8_of_10", 0) or 0)
    elif isinstance(checkpoint, dict):
        stage = str(checkpoint.get("stage", "unknown"))
        elapsed_sec = float(checkpoint.get("elapsed_sec", 0.0) or 0.0)
        extra = checkpoint.get("extra", {}) if isinstance(checkpoint.get("extra", {}), dict) else {}
        complete_10 = int(extra.get("match_complete_10_of_10", 0) or 0)
        complete_8 = int(extra.get("match_complete_8_of_10", 0) or 0)
        done = int(extra.get("done", 0) or 0)
        total = int(extra.get("total", 0) or 0)
        if stage.startswith("match_id_top_up") and done > 0 and total >= done:
            est_hours_total = (elapsed_sec / done * total) / 3600.0

    return {
        "generated_utc": int(time.time()),
        "monitor_started_utc": int(monitor_started_utc),
        "run_dir": str(run_dir),
        "process": process,
        "checkpoint": checkpoint,
        "run_json": run_json,
        "quick_summary": {
            "process_active": bool(process),
            "stage": stage,
            "elapsed_sec": elapsed_sec,
            "requests": requests,
            "http_429": http_429,
            "match_complete_10_of_10": complete_10,
            "match_complete_8_of_10": complete_8,
            "est_hours_total_if_current_rate_holds": est_hours_total,
            "stderr_nonempty": bool(stderr_tail),
        },
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def write_report_files(report_dir: Path, report: dict[str, Any], index: int) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime(int(report["generated_utc"])))
    stamped_json = report_dir / f"player_dataset_report_{index:03d}_{ts}.json"
    stamped_txt = report_dir / f"player_dataset_report_{index:03d}_{ts}.txt"
    latest_json = report_dir / "player_dataset_report_latest.json"
    latest_txt = report_dir / "player_dataset_report_latest.txt"

    stamped_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report.get("quick_summary", {}) if isinstance(report.get("quick_summary"), dict) else {}
    lines = [
        f"generated_utc: {report.get('generated_utc')}",
        f"process_active: {summary.get('process_active')}",
        f"stage: {summary.get('stage')}",
        f"elapsed_sec: {summary.get('elapsed_sec')}",
        f"requests: {summary.get('requests')}",
        f"http_429: {summary.get('http_429')}",
        f"match_complete_10_of_10: {summary.get('match_complete_10_of_10')}",
        f"match_complete_8_of_10: {summary.get('match_complete_8_of_10')}",
        f"est_hours_total_if_current_rate_holds: {summary.get('est_hours_total_if_current_rate_holds')}",
        f"stderr_nonempty: {summary.get('stderr_nonempty')}",
    ]
    stamped_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodic monitor for a running player_dataset.py job.")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--stdout-log", required=True, type=str)
    parser.add_argument("--stderr-log", required=True, type=str)
    parser.add_argument("--report-dir", required=True, type=str)
    parser.add_argument("--command-pattern", required=True, type=str)
    parser.add_argument("--interval-sec", type=int, default=1800)
    parser.add_argument("--max-reports", type=int, default=0, help="0 means keep going until process exits.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    monitor_started_utc = int(time.time())
    index = 0
    while True:
        report = build_report(
            run_dir=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command_pattern=str(args.command_pattern),
            monitor_started_utc=monitor_started_utc,
        )
        index += 1
        write_report_files(report_dir, report, index)

        process_active = bool(report.get("quick_summary", {}).get("process_active"))
        run_finished = isinstance(report.get("run_json"), dict)
        if not process_active and run_finished:
            break
        if int(args.max_reports) > 0 and index >= int(args.max_reports):
            break
        time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
    main()
