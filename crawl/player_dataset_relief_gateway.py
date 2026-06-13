from __future__ import annotations

import argparse
import os
from pathlib import Path

import player_dataset_relief_http as relief_http
import player_dataset_relief_spool as relief_spool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expose a local helper spool over HTTP so remote helper VMs can claim and publish batches."
    )
    p.add_argument("--spool-dir", type=str, required=True)
    p.add_argument("--bind-host", type=str, default=os.getenv("RELIEF_GATEWAY_BIND_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("RELIEF_GATEWAY_PORT", "18765")))
    p.add_argument(
        "--auth-token",
        type=str,
        default=os.getenv("RELIEF_GATEWAY_AUTH_TOKEN", ""),
        help="Shared secret required by remote helpers.",
    )
    p.add_argument(
        "--request-log-file",
        type=str,
        default="",
        help="Optional JSONL file path for gateway request logs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = str(args.auth_token or "").strip()
    if not token:
        raise SystemExit("Missing relief gateway auth token. Pass --auth-token or set RELIEF_GATEWAY_AUTH_TOKEN.")
    if int(args.port) <= 0:
        raise SystemExit("port must be positive.")

    spool_dir = Path(args.spool_dir)
    relief_spool.ensure_spool_dirs(spool_dir)
    request_log_path = Path(args.request_log_file) if str(args.request_log_file or "").strip() else None

    print(
        "Relief gateway:"
        f" bind={args.bind_host}:{int(args.port)}"
        f" spool_dir={spool_dir}"
        f" request_log={str(request_log_path) if request_log_path is not None else ''}"
    )
    relief_http.serve_gateway_forever(
        spool_dir=spool_dir,
        bind_host=str(args.bind_host),
        port=int(args.port),
        auth_token=token,
        request_log_path=request_log_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
