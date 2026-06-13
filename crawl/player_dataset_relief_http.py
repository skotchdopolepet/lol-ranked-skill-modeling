from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import player_dataset_relief_spool as relief_spool


AUTH_HEADER = "X-Relief-Token"


def _auth_headers(auth_token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(auth_token or "").strip()
    if token:
        headers[AUTH_HEADER] = token
    return headers


def http_json_request(
    *,
    base_url: str,
    path: str,
    method: str,
    payload: dict[str, Any] | None,
    auth_token: str,
    timeout_sec: float,
) -> dict[str, Any]:
    url = urllib.parse.urljoin(str(base_url).rstrip("/") + "/", path.lstrip("/"))
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method=str(method).upper(),
        headers=_auth_headers(auth_token),
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach relief gateway {url}: {exc}") from exc
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}


class RemoteSpoolClient:
    def __init__(self, *, base_url: str, auth_token: str, timeout_sec: float = 15.0) -> None:
        token = str(auth_token or "").strip()
        if not token:
            raise ValueError("remote relief gateway auth token is required")
        self.base_url = str(base_url or "").strip()
        if not self.base_url:
            raise ValueError("remote relief gateway base url is required")
        self.auth_token = token
        self.timeout_sec = max(1.0, float(timeout_sec))

    def claim_next_request(self, *, helper_id: str, stale_after_sec: int) -> dict[str, Any] | None:
        payload = http_json_request(
            base_url=self.base_url,
            path="/claim",
            method="POST",
            payload={
                "helper_id": str(helper_id),
                "stale_after_sec": int(stale_after_sec),
            },
            auth_token=self.auth_token,
            timeout_sec=self.timeout_sec,
        )
        if not bool(payload.get("claimed")):
            return None
        claimed = payload.get("request")
        return claimed if isinstance(claimed, dict) else None

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
        payload = http_json_request(
            base_url=self.base_url,
            path="/publish-result",
            method="POST",
            payload={
                "claimed_request": claimed_request,
                "helper_id": str(helper_id),
                "successes": list(successes),
                "failures": list(failures),
                "processing_started_utc": processing_started_utc,
                "processing_elapsed_sec": processing_elapsed_sec,
            },
            auth_token=self.auth_token,
            timeout_sec=max(self.timeout_sec, 30.0),
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("relief gateway returned an invalid publish-result payload")
        return result

    def health(self) -> dict[str, Any]:
        return http_json_request(
            base_url=self.base_url,
            path="/health",
            method="GET",
            payload=None,
            auth_token=self.auth_token,
            timeout_sec=self.timeout_sec,
        )


def _json_response(
    handler: BaseHTTPRequestHandler,
    *,
    status: HTTPStatus,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _load_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    data = json.loads(raw.decode("utf-8") or "{}")
    return data if isinstance(data, dict) else {}


def _append_request_log(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_gateway_handler(
    *,
    spool_dir: Path,
    auth_token: str,
    request_log_path: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ReliefGatewayHandler(BaseHTTPRequestHandler):
        server_version = "ReliefGateway/1.0"

        def _check_auth(self) -> bool:
            header_token = str(self.headers.get(AUTH_HEADER, "") or "").strip()
            if header_token != auth_token:
                _json_response(
                    self,
                    status=HTTPStatus.UNAUTHORIZED,
                    payload={"ok": False, "error": "unauthorized"},
                )
                return False
            return True

        def _log_request(self, *, route: str, status: int, extra: dict[str, Any] | None = None) -> None:
            payload = {
                "ts_utc": int(time.time()),
                "method": str(self.command),
                "route": str(route),
                "status": int(status),
                "client": str(self.client_address[0] if self.client_address else ""),
            }
            if extra:
                payload.update(extra)
            _append_request_log(request_log_path, payload)

        def do_GET(self) -> None:  # noqa: N802
            route = urllib.parse.urlsplit(self.path).path
            if route != "/health":
                _json_response(self, status=HTTPStatus.NOT_FOUND, payload={"ok": False, "error": "not_found"})
                self._log_request(route=route, status=int(HTTPStatus.NOT_FOUND))
                return
            if not self._check_auth():
                self._log_request(route=route, status=int(HTTPStatus.UNAUTHORIZED))
                return
            payload = {
                "ok": True,
                "spool_dir": str(spool_dir),
                **relief_spool.spool_counts(spool_dir),
            }
            _json_response(self, status=HTTPStatus.OK, payload=payload)
            self._log_request(route=route, status=int(HTTPStatus.OK), extra=payload)

        def do_POST(self) -> None:  # noqa: N802
            route = urllib.parse.urlsplit(self.path).path
            if not self._check_auth():
                self._log_request(route=route, status=int(HTTPStatus.UNAUTHORIZED))
                return
            try:
                data = _load_request_json(self)
                if route == "/claim":
                    helper_id = str(data.get("helper_id") or "").strip()
                    if not helper_id:
                        raise ValueError("helper_id is required")
                    stale_after_sec = int(data.get("stale_after_sec") or 0)
                    if stale_after_sec <= 0:
                        raise ValueError("stale_after_sec must be positive")
                    claimed = relief_spool.claim_next_request(
                        spool_dir=spool_dir,
                        helper_id=helper_id,
                        stale_after_sec=stale_after_sec,
                    )
                    payload = {"ok": True, "claimed": claimed is not None, "request": claimed}
                    _json_response(self, status=HTTPStatus.OK, payload=payload)
                    self._log_request(
                        route=route,
                        status=int(HTTPStatus.OK),
                        extra={
                            "helper_id": helper_id,
                            "claimed": bool(claimed is not None),
                            "batch_id": str((claimed or {}).get("batch_id") or ""),
                        },
                    )
                    return
                if route == "/publish-result":
                    claimed_request = data.get("claimed_request")
                    if not isinstance(claimed_request, dict):
                        raise ValueError("claimed_request is required")
                    helper_id = str(data.get("helper_id") or "").strip()
                    if not helper_id:
                        raise ValueError("helper_id is required")
                    successes = data.get("successes")
                    failures = data.get("failures")
                    if not isinstance(successes, list) or not isinstance(failures, list):
                        raise ValueError("successes and failures must be lists")
                    payload = relief_spool.publish_result(
                        spool_dir=spool_dir,
                        claimed_request=claimed_request,
                        helper_id=helper_id,
                        successes=list(successes),
                        failures=list(failures),
                        processing_started_utc=(
                            int(data["processing_started_utc"])
                            if data.get("processing_started_utc") is not None
                            else None
                        ),
                        processing_elapsed_sec=(
                            float(data["processing_elapsed_sec"])
                            if data.get("processing_elapsed_sec") is not None
                            else None
                        ),
                    )
                    _json_response(self, status=HTTPStatus.OK, payload={"ok": True, "result": payload})
                    self._log_request(
                        route=route,
                        status=int(HTTPStatus.OK),
                        extra={
                            "helper_id": helper_id,
                            "batch_id": str(payload.get("batch_id") or ""),
                            "success_count": int(payload.get("success_count") or 0),
                            "failure_count": int(payload.get("failure_count") or 0),
                        },
                    )
                    return
                _json_response(
                    self,
                    status=HTTPStatus.NOT_FOUND,
                    payload={"ok": False, "error": "not_found"},
                )
                self._log_request(route=route, status=int(HTTPStatus.NOT_FOUND))
            except Exception as exc:  # noqa: BLE001
                _json_response(
                    self,
                    status=HTTPStatus.BAD_REQUEST,
                    payload={"ok": False, "error": str(exc)},
                )
                self._log_request(route=route, status=int(HTTPStatus.BAD_REQUEST), extra={"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return ReliefGatewayHandler


def create_gateway_server(
    *,
    spool_dir: Path,
    bind_host: str,
    port: int,
    auth_token: str,
    request_log_path: Path | None = None,
) -> ThreadingHTTPServer:
    relief_spool.ensure_spool_dirs(spool_dir)
    handler = build_gateway_handler(
        spool_dir=spool_dir,
        auth_token=str(auth_token),
        request_log_path=request_log_path,
    )
    return ThreadingHTTPServer((str(bind_host), int(port)), handler)


def serve_gateway_forever(
    *,
    spool_dir: Path,
    bind_host: str,
    port: int,
    auth_token: str,
    request_log_path: Path | None = None,
) -> None:
    httpd = create_gateway_server(
        spool_dir=spool_dir,
        bind_host=bind_host,
        port=port,
        auth_token=auth_token,
        request_log_path=request_log_path,
    )
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
