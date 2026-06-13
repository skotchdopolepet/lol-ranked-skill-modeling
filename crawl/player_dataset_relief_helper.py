from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from main_dataset import (
    DEFAULT_APP_LIMIT_REQUESTS,
    DEFAULT_APP_LIMIT_WINDOW_SEC,
    DEFAULT_PLATFORM_ROUTING,
    CrawlStats,
    RateController,
    configure_api_bases,
    effective_workers,
    match_detail,
    validate_api_key,
)
import player_dataset_relief_http as relief_http
import player_dataset_relief_spool as relief_spool


def extract_game_creation_utc_ms(payload: dict[str, Any]) -> int:
    info = payload.get("info", {})
    return int(info.get("gameCreation", 0) or 0)


class LocalSpoolTransport:
    def __init__(self, *, spool_dir: Path) -> None:
        self.spool_dir = Path(spool_dir)

    def claim_next_request(self, *, helper_id: str, stale_after_sec: int) -> dict[str, Any] | None:
        return relief_spool.claim_next_request(
            spool_dir=self.spool_dir,
            helper_id=helper_id,
            stale_after_sec=stale_after_sec,
        )

    def publish_result(
        self,
        *,
        claimed_request: dict[str, Any],
        helper_id: str,
        successes: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        processing_started_utc: int | None,
        processing_elapsed_sec: float | None,
    ) -> dict[str, Any]:
        return relief_spool.publish_result(
            spool_dir=self.spool_dir,
            claimed_request=claimed_request,
            helper_id=helper_id,
            successes=successes,
            failures=failures,
            processing_started_utc=processing_started_utc,
            processing_elapsed_sec=processing_elapsed_sec,
        )


def build_transport(
    args: argparse.Namespace,
) -> LocalSpoolTransport | relief_http.RemoteSpoolClient:
    remote_base_url = str(getattr(args, "remote_spool_base_url", "") or "").strip()
    if remote_base_url:
        return relief_http.RemoteSpoolClient(
            base_url=remote_base_url,
            auth_token=str(getattr(args, "remote_spool_auth_token", "") or ""),
            timeout_sec=float(getattr(args, "remote_spool_timeout_sec", 15.0)),
        )
    return LocalSpoolTransport(spool_dir=Path(str(getattr(args, "spool_dir", "") or "").strip()))


def run_helper_loop(
    *,
    args: argparse.Namespace,
    stop_event: Any | None = None,
    validate_key: bool = True,
) -> dict[str, Any]:
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set RIOT_API_KEY.")
    if args.request_timeout_sec <= 0:
        raise SystemExit("request-timeout-sec must be positive.")
    if args.request_max_retries < 0:
        raise SystemExit("request-max-retries must be >= 0.")
    if args.poll_interval_sec < 0:
        raise SystemExit("poll-interval-sec must be >= 0.")
    if args.claim_stale_after_sec <= 0:
        raise SystemExit("claim-stale-after-sec must be positive.")
    if args.max_batches < 0:
        raise SystemExit("max-batches must be >= 0.")
    if float(getattr(args, "remote_spool_timeout_sec", 15.0)) <= 0:
        raise SystemExit("remote-spool-timeout-sec must be positive.")

    controller = RateController(
        profile=args.rate_profile,
        app_limit_requests=args.app_limit_requests,
        app_limit_window_sec=args.app_limit_window_sec,
    )
    stats = CrawlStats()
    if validate_key and not bool(getattr(args, "skip_preflight", False)):
        validate_api_key(args.api_key, controller=controller, stats=stats)
    transport = build_transport(args)

    handled_batches = 0
    while True:
        claimed = transport.claim_next_request(
            helper_id=str(args.helper_id),
            stale_after_sec=int(args.claim_stale_after_sec),
        )
        if claimed is None:
            if stop_event is not None and bool(stop_event.is_set()):
                break
            if bool(getattr(args, "once", False)):
                break
            if int(getattr(args, "max_batches", 0)) > 0 and handled_batches >= int(args.max_batches):
                break
            if float(args.poll_interval_sec) > 0:
                time.sleep(float(args.poll_interval_sec))
            continue

        result = process_request(
            args=args,
            controller=controller,
            stats=stats,
            claimed_request=claimed,
            transport=transport,
        )
        handled_batches += 1
        print(
            f"helper batch={result['batch_id']} success={len(result['successes'])} "
            f"failures={len(result['failures'])}"
        )
        if int(getattr(args, "max_batches", 0)) > 0 and handled_batches >= int(args.max_batches):
            break
        if bool(getattr(args, "once", False)):
            break

    totals = stats.to_dict().get("totals", {})
    print(
        "helper totals:"
        f" requests={int(totals.get('requests', 0) or 0)}"
        f" success={int(totals.get('success', 0) or 0)}"
        f" retries={int(totals.get('retries', 0) or 0)}"
        f" http_429={int(totals.get('http_429', 0) or 0)}"
    )
    return {
        "handled_batches": int(handled_batches),
        "api_totals": totals,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Helper worker that claims offloaded match-id detail batches and returns only gameCreation."
    )
    p.add_argument("--api-key", type=str, default=os.getenv("RIOT_API_KEY", ""))
    p.add_argument("--spool-dir", type=str, default="")
    p.add_argument(
        "--remote-spool-base-url",
        type=str,
        default=os.getenv("RELIEF_SPOOL_BASE_URL", ""),
        help="Optional HTTP base URL for a remote relief spool gateway.",
    )
    p.add_argument(
        "--remote-spool-auth-token",
        type=str,
        default=os.getenv("RELIEF_GATEWAY_AUTH_TOKEN", ""),
        help="Shared auth token for the remote relief spool gateway.",
    )
    p.add_argument("--remote-spool-timeout-sec", type=float, default=15.0)
    p.add_argument(
        "--platform-routing",
        "--platform",
        dest="platform_routing",
        type=str,
        default=DEFAULT_PLATFORM_ROUTING,
    )
    p.add_argument(
        "--regional-routing",
        "--region",
        dest="regional_routing",
        type=str,
        default=None,
    )
    p.add_argument("--request-timeout-sec", type=int, default=10)
    p.add_argument("--request-max-retries", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=None)
    p.add_argument("--poll-interval-sec", type=float, default=2.0)
    p.add_argument("--claim-stale-after-sec", type=int, default=900)
    p.add_argument("--app-limit-requests", type=int, default=DEFAULT_APP_LIMIT_REQUESTS)
    p.add_argument("--app-limit-window-sec", type=float, default=DEFAULT_APP_LIMIT_WINDOW_SEC)
    p.add_argument("--rate-profile", type=str, choices=["auto", "conservative", "aggressive"], default="auto")
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--helper-id", type=str, default="")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args()


def process_request(
    *,
    args: argparse.Namespace,
    controller: RateController,
    stats: CrawlStats,
    claimed_request: dict[str, Any],
    transport: LocalSpoolTransport | relief_http.RemoteSpoolClient | None = None,
) -> dict[str, Any]:
    started_utc = int(time.time())
    started_mono = time.monotonic()
    match_ids = [str(match_id) for match_id in claimed_request.get("match_ids", []) if match_id]
    workers = effective_workers(int(args.workers), args.max_inflight)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(
                match_detail,
                args.api_key,
                match_id,
                controller,
                stats,
                int(args.request_timeout_sec),
                int(args.request_max_retries),
            ): match_id
            for match_id in match_ids
        }
        for fut in as_completed(futures):
            match_id = futures[fut]
            try:
                detail = fut.result()
                game_creation_utc_ms = extract_game_creation_utc_ms(detail)
                if game_creation_utc_ms <= 0:
                    raise RuntimeError("missing gameCreation")
                successes.append(
                    {
                        "match_id": str(match_id),
                        "game_creation_utc_ms": int(game_creation_utc_ms),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "match_id": str(match_id),
                        "error": str(exc),
                    }
                )
    active_transport = transport or build_transport(args)
    return active_transport.publish_result(
        claimed_request=claimed_request,
        helper_id=str(args.helper_id),
        successes=successes,
        failures=failures,
        processing_started_utc=started_utc,
        processing_elapsed_sec=float(time.monotonic() - started_mono),
    )


def main() -> int:
    args = parse_args()
    try:
        platform_routing, regional_routing = configure_api_bases(
            args.platform_routing,
            args.regional_routing,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.platform_routing = platform_routing
    args.regional_routing = regional_routing
    if not str(args.helper_id or "").strip():
        args.helper_id = relief_spool.default_actor_id("helper")

    remote_base_url = str(args.remote_spool_base_url or "").strip()
    spool_dir = str(args.spool_dir or "").strip()
    if bool(remote_base_url) == bool(spool_dir):
        raise SystemExit("Pass exactly one of --spool-dir or --remote-spool-base-url.")
    if spool_dir:
        relief_spool.ensure_spool_dirs(Path(spool_dir))
    if remote_base_url and not str(args.remote_spool_auth_token or "").strip():
        raise SystemExit("Missing --remote-spool-auth-token for remote relief spool mode.")

    run_helper_loop(args=args, stop_event=None, validate_key=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
