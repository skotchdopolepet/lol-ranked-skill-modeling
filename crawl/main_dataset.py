from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - runtime optional dependency
    zstd = None


DEFAULT_PLATFORM_ROUTING = "EUW1"
DEFAULT_REGIONAL_ROUTING = "EUROPE"
PLATFORM_BASE = f"https://{DEFAULT_PLATFORM_ROUTING.lower()}.api.riotgames.com"
REGIONAL_BASE = f"https://{DEFAULT_REGIONAL_ROUTING.lower()}.api.riotgames.com"
REGIONAL_ROUTINGS = ("AMERICAS", "ASIA", "EUROPE", "SEA")
PLATFORM_TO_REGIONAL = {
    "BR1": "AMERICAS",
    "LA1": "AMERICAS",
    "LA2": "AMERICAS",
    "NA1": "AMERICAS",
    "OC1": "AMERICAS",
    "JP1": "ASIA",
    "KR": "ASIA",
    "EUN1": "EUROPE",
    "EUW1": "EUROPE",
    "TR1": "EUROPE",
    "RU": "EUROPE",
    # Optional compatibility mapping; not listed in LoL routing table but
    # commonly referenced by users as "ME".
    "ME1": "EUROPE",
    "ME": "EUROPE",
    "PH2": "SEA",
    "SG2": "SEA",
    "TH2": "SEA",
    "TW2": "SEA",
    "VN2": "SEA",
}
DEFAULT_QUEUE = "RANKED_SOLO_5x5"
APEX_TIERS = ("CHALLENGER", "GRANDMASTER", "MASTER")
TIER_PRIORITY = {"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}
DIVISION_TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
DIVISION_RANKS = ("IV", "III", "II", "I")
ALL_SOLO_TIERS = DIVISION_TIERS + APEX_TIERS
SOLO_TIER_ORDER = {tier: idx for idx, tier in enumerate(ALL_SOLO_TIERS)}
SOLO_RANK_ORDER = {"IV": 0, "III": 1, "II": 2, "I": 3}
JOB_TABLES = ("jobs_match_ids", "jobs_match_details", "jobs_rank_lookup")
MAX_RETRY_AFTER_SEC = 30.0
MIN_RETRY_JITTER_SEC = 1.0
MAX_RETRY_JITTER_SEC = 5.0
DEFAULT_APEX_STOP_CHALLENGER_COUNT = 260
DEFAULT_APEX_STOP_GRANDMASTER_COUNT = 650
DEFAULT_APP_LIMIT_REQUESTS = 200
DEFAULT_APP_LIMIT_WINDOW_SEC = 120.0
DEFAULT_APP_LIMIT_SHORT_REQUESTS = 20
DEFAULT_APP_LIMIT_SHORT_WINDOW_SEC = 1.0
HTTP_CONNECT_TIMEOUT_SEC = 3.05
HTTP_POOL_SIZE = 32
_HTTP_SESSION_LOCAL = threading.local()


class FatalRiotAuthError(RuntimeError):
    """Fatal auth failure (401/403): stop the loop and require operator action."""


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    try:
        seconds = float(token)
        return seconds if seconds >= 0 else None
    except ValueError:
        try:
            dt = parsedate_to_datetime(token)
        except Exception:
            return None
        now_utc = time.time()
        if dt.tzinfo is None:
            return None
        retry_after = dt.timestamp() - now_utc
        return retry_after if retry_after >= 0 else 0.0


def _parse_rate_pairs(value: str | None) -> dict[int, float]:
    if not value:
        return {}
    pairs: dict[int, float] = {}
    for part in value.split(","):
        token = part.strip()
        if ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            n1 = float(left.strip())
            n2 = int(float(right.strip()))
        except ValueError:
            continue
        pairs[n2] = n1
    return pairs


def _rate_usage_values_from_headers(headers: dict[str, str]) -> list[float]:
    hdr = {k.lower(): v for k, v in headers.items()}
    usage_values: list[float] = []
    windows = [
        ("x-app-rate-limit", "x-app-rate-limit-count"),
        ("x-method-rate-limit", "x-method-rate-limit-count"),
    ]
    for limit_key, count_key in windows:
        limits = _parse_rate_pairs(hdr.get(limit_key))
        counts = _parse_rate_pairs(hdr.get(count_key))
        for sec, limit in limits.items():
            count = counts.get(sec)
            if count is None or limit <= 0:
                continue
            usage_values.append(count / limit)
    return usage_values


def _rate_windows_from_headers(headers: dict[str, str], limit_key: str) -> list[tuple[int, float]]:
    hdr = {k.lower(): v for k, v in headers.items()}
    raw = _parse_rate_pairs(hdr.get(limit_key.lower()))
    out: list[tuple[int, float]] = []
    for sec, limit in sorted(raw.items(), key=lambda item: item[0]):
        if limit <= 0 or sec <= 0:
            continue
        out.append((max(1, int(limit)), float(sec)))
    return out


def _build_http_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=HTTP_POOL_SIZE,
        pool_maxsize=HTTP_POOL_SIZE,
        max_retries=0,
        pool_block=False,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    return session


def _http_session() -> requests.Session:
    session = getattr(_HTTP_SESSION_LOCAL, "session", None)
    if session is None:
        session = _build_http_session()
        _HTTP_SESSION_LOCAL.session = session
    return session


def resolve_api_routing(
    platform_routing: str,
    regional_routing: str | None = None,
) -> tuple[str, str]:
    platform = str(platform_routing or "").strip().upper()
    if not platform:
        raise ValueError("platform-routing must be a non-empty routing value (e.g. EUW1).")
    if regional_routing is None or not str(regional_routing).strip():
        regional = PLATFORM_TO_REGIONAL.get(platform)
        if regional is None:
            supported = ", ".join(sorted(PLATFORM_TO_REGIONAL.keys()))
            raise ValueError(
                f"Unsupported platform routing: {platform}. "
                f"Provide --regional-routing explicitly or use one of: {supported}"
            )
        return platform, regional
    regional = str(regional_routing).strip().upper()
    if regional not in REGIONAL_ROUTINGS:
        supported_regions = ", ".join(REGIONAL_ROUTINGS)
        raise ValueError(
            f"Unsupported regional routing: {regional}. Use one of: {supported_regions}"
        )
    return platform, regional


def configure_api_bases(
    platform_routing: str,
    regional_routing: str | None = None,
) -> tuple[str, str]:
    global PLATFORM_BASE, REGIONAL_BASE
    platform, regional = resolve_api_routing(platform_routing, regional_routing)
    PLATFORM_BASE = f"https://{platform.lower()}.api.riotgames.com"
    REGIONAL_BASE = f"https://{regional.lower()}.api.riotgames.com"
    return platform, regional


class CrawlStats:
    def __init__(self) -> None:
        self.start_monotonic = time.monotonic()
        self.start_utc = int(time.time())
        self.endpoint: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "requests": 0,
                "success": 0,
                "retries": 0,
                "http_429": 0,
                "http_5xx": 0,
                "errors": 0,
            }
        )

    def mark_request(self, endpoint_class: str, retry: bool) -> None:
        row = self.endpoint[endpoint_class]
        row["requests"] += 1
        if retry:
            row["retries"] += 1

    def mark_success(self, endpoint_class: str) -> None:
        self.endpoint[endpoint_class]["success"] += 1

    def mark_429(self, endpoint_class: str) -> None:
        self.endpoint[endpoint_class]["http_429"] += 1

    def mark_5xx(self, endpoint_class: str) -> None:
        self.endpoint[endpoint_class]["http_5xx"] += 1

    def mark_error(self, endpoint_class: str) -> None:
        self.endpoint[endpoint_class]["errors"] += 1

    def to_dict(self) -> dict[str, Any]:
        elapsed_sec = max(1e-9, time.monotonic() - self.start_monotonic)
        totals = {
            "requests": 0,
            "success": 0,
            "retries": 0,
            "http_429": 0,
            "http_5xx": 0,
            "errors": 0,
        }
        for row in self.endpoint.values():
            for key in totals:
                totals[key] += int(row.get(key, 0) or 0)
        return {
            "start_utc": self.start_utc,
            "end_utc": int(time.time()),
            "elapsed_sec": elapsed_sec,
            "requests_per_sec": totals["requests"] / elapsed_sec,
            "success_per_sec": totals["success"] / elapsed_sec,
            "totals": totals,
            "by_endpoint": self.endpoint,
        }


class AdaptiveLimiter:
    def __init__(
        self,
        windows: list[tuple[int, float]],
        initial_scale: float,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
    ) -> None:
        self.windows = [(max(1, int(limit)), max(0.001, float(sec))) for limit, sec in windows]
        self.initial_scale = max(min(initial_scale, max_scale), min_scale)
        self.scale = self.initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blocked_until = 0.0
        self.smooth_window_threshold_sec = 2.0
        self.history = [deque() for _ in self.windows]
        self.next_ready = [0.0 for _ in self.windows]
        self.lock = threading.Lock()

    def _recovery_step(self, max_usage: float | None) -> float:
        scale = float(self.scale)
        if max_usage is None:
            if scale < 0.35:
                return 0.03
            if scale < 0.55:
                return 0.02
            return 0.005
        if max_usage < 0.40:
            if scale < 0.35:
                return 0.04
            if scale < 0.55:
                return 0.03
            if scale < 0.75:
                return 0.015
            return 0.01
        if max_usage < 0.55:
            if scale < 0.35:
                return 0.03
            if scale < 0.55:
                return 0.02
            return 0.01
        if max_usage < 0.75 and scale < 0.45:
            return 0.01
        return 0.0

    def _allowed(self, limit: int) -> float:
        return max(1.0, float(limit) * float(self.scale))

    def _interval(self, limit: int, window_sec: float) -> float:
        return max(0.001, float(window_sec) / self._allowed(limit))

    def update_windows(self, windows: list[tuple[int, float]]) -> None:
        normalized = [(max(1, int(limit)), max(0.001, float(sec))) for limit, sec in windows]
        if not normalized:
            return
        with self.lock:
            if normalized == self.windows:
                return
            self.windows = normalized
            self.history = [deque() for _ in self.windows]
            self.next_ready = [max(time.monotonic(), self.blocked_until) for _ in self.windows]

    def _prune(self, now: float) -> None:
        for idx, (_, window_sec) in enumerate(self.windows):
            if window_sec <= self.smooth_window_threshold_sec:
                continue
            queue = self.history[idx]
            while queue and (now - queue[0]) >= window_sec:
                queue.popleft()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                wait_for = max(0.0, self.blocked_until - now)
                for idx, (limit, window_sec) in enumerate(self.windows):
                    if window_sec <= self.smooth_window_threshold_sec:
                        if self.next_ready[idx] > now:
                            wait_for = max(wait_for, self.next_ready[idx] - now)
                    else:
                        allowed = max(1, int(self._allowed(limit)))
                        queue = self.history[idx]
                        if len(queue) >= allowed:
                            wait_for = max(wait_for, (queue[0] + window_sec) - now)
                if wait_for <= 0:
                    stamp = max(now, self.blocked_until)
                    for idx, (limit, window_sec) in enumerate(self.windows):
                        if window_sec <= self.smooth_window_threshold_sec:
                            interval = self._interval(limit, window_sec)
                            self.next_ready[idx] = max(self.next_ready[idx], stamp) + interval
                        else:
                            self.history[idx].append(stamp)
                    return
                sleep_for = min(max(0.001, wait_for), 0.5)
            time.sleep(sleep_for)

    def on_429(self, retry_after: float | None, rate_limit_type: str | None = None) -> None:
        with self.lock:
            now = time.monotonic()
            cooldown = float(retry_after) if retry_after and retry_after > 0 else 2.0
            self.blocked_until = max(self.blocked_until, now + cooldown)
            if (rate_limit_type or "").lower() == "service":
                decay = 0.55
            elif (rate_limit_type or "").lower() == "method":
                decay = 0.65
            else:
                decay = 0.7
            self.scale = max(self.min_scale, self.scale * decay)
            self.next_ready = [max(nxt, self.blocked_until) for nxt in self.next_ready]

    def on_success(self, headers: dict[str, str]) -> None:
        usage_values = _rate_usage_values_from_headers(headers)

        with self.lock:
            if usage_values:
                max_usage = max(usage_values)
                if max_usage > 0.9 and self.scale > 0.45:
                    self.scale = max(self.min_scale, self.scale * 0.92)
                else:
                    self.scale = min(self.max_scale, self.scale + self._recovery_step(max_usage))
            else:
                self.scale = min(self.max_scale, self.scale + self._recovery_step(None))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "windows": [{"limit": limit, "seconds": sec} for limit, sec in self.windows],
                "scale": self.scale,
                "initial_scale": self.initial_scale,
                "blocked_for_sec": max(0.0, self.blocked_until - time.monotonic()),
            }


class RateController:
    def __init__(
        self,
        profile: str,
        app_limit_requests: int = DEFAULT_APP_LIMIT_REQUESTS,
        app_limit_window_sec: float = DEFAULT_APP_LIMIT_WINDOW_SEC,
    ) -> None:
        profile_scale = {
            "conservative": 0.45,
            "auto": 0.65,
            "aggressive": 0.85,
        }
        start_scale = profile_scale.get(profile, 0.65)
        self.profile = profile
        self.app_limit_requests = max(1, int(app_limit_requests))
        self.app_limit_window_sec = max(1.0, float(app_limit_window_sec))
        self.app_windows = [
            (self.app_limit_requests, self.app_limit_window_sec),
            (DEFAULT_APP_LIMIT_SHORT_REQUESTS, DEFAULT_APP_LIMIT_SHORT_WINDOW_SEC),
        ]
        self.app_limiter = AdaptiveLimiter(
            self.app_windows,
            start_scale,
        )
        self.limiters: dict[str, AdaptiveLimiter] = {
            "platform_status": AdaptiveLimiter([(30, 10.0), (500, 600.0)], start_scale),
            "league_apex": AdaptiveLimiter([(30, 10.0), (500, 600.0)], start_scale),
            "league_division_entries": AdaptiveLimiter([(50, 10.0)], start_scale),
            "match_ids": AdaptiveLimiter([(2000, 10.0)], start_scale),
            "match_details": AdaptiveLimiter([(2000, 10.0)], start_scale),
            "rank_by_puuid": AdaptiveLimiter([(20000, 10.0), (1200000, 600.0)], start_scale),
            "default": AdaptiveLimiter([(100, 1.0)], start_scale),
        }

    def _limiter(self, endpoint_class: str) -> AdaptiveLimiter:
        return self.limiters.get(endpoint_class, self.limiters["default"])

    def acquire(self, endpoint_class: str) -> None:
        self.app_limiter.acquire()
        self._limiter(endpoint_class).acquire()

    def on_429(
        self,
        endpoint_class: str,
        retry_after: float | None,
        rate_limit_type: str | None = None,
    ) -> None:
        self.app_limiter.on_429(retry_after, rate_limit_type=rate_limit_type)
        self._limiter(endpoint_class).on_429(retry_after, rate_limit_type=rate_limit_type)

    def on_success(self, endpoint_class: str, headers: dict[str, str]) -> None:
        app_windows = _rate_windows_from_headers(headers, "x-app-rate-limit")
        if app_windows:
            self.app_limiter.update_windows(app_windows)
        method_windows = _rate_windows_from_headers(headers, "x-method-rate-limit")
        if method_windows:
            self._limiter(endpoint_class).update_windows(method_windows)
        self.app_limiter.on_success(headers)
        self._limiter(endpoint_class).on_success(headers)

    def snapshot(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "app_limiter": {
                "configured_windows": [
                    {"limit": limit, "seconds": seconds} for limit, seconds in self.app_windows
                ],
                **self.app_limiter.snapshot(),
            },
            "limiters": {
                name: limiter.snapshot() for name, limiter in self.limiters.items()
            },
        }


def request_json(
    url: str,
    api_key: str,
    endpoint_class: str,
    controller: RateController,
    stats: CrawlStats,
    retries: int = 3,
    timeout_sec: int = 10,
) -> Any:
    last_err: Exception | None = None
    session = _http_session()
    for attempt in range(retries + 1):
        controller.acquire(endpoint_class)
        stats.mark_request(endpoint_class, retry=attempt > 0)
        try:
            resp = session.get(
                url,
                headers={"X-Riot-Token": api_key},
                timeout=(HTTP_CONNECT_TIMEOUT_SEC, timeout_sec),
            )
            err_headers = dict(resp.headers.items())
            if resp.status_code == 429:
                stats.mark_429(endpoint_class)
                retry_after = _parse_retry_after_seconds(err_headers.get("Retry-After"))
                cooldown = retry_after if retry_after is not None else 2.0
                cooldown = min(MAX_RETRY_AFTER_SEC, max(0.5, cooldown))
                if err_headers:
                    controller.on_success(endpoint_class, err_headers)
                lower_err_headers = {k.lower(): v for k, v in err_headers.items()}
                controller.on_429(
                    endpoint_class,
                    cooldown,
                    rate_limit_type=lower_err_headers.get("x-rate-limit-type"),
                )
                if attempt < retries:
                    jitter = random.uniform(MIN_RETRY_JITTER_SEC, MAX_RETRY_JITTER_SEC)
                    time.sleep(min(jitter, cooldown))
                    last_err = RuntimeError(f"HTTP 429 for {url}")
                    continue
                last_err = RuntimeError(f"HTTP 429 for {url}")
                continue
            if 500 <= resp.status_code < 600 and attempt < retries:
                stats.mark_5xx(endpoint_class)
                time.sleep(1.0 + attempt)
                last_err = RuntimeError(f"HTTP {resp.status_code} for {url}")
                continue
            if resp.status_code >= 400:
                stats.mark_error(endpoint_class)
                err_body = resp.text
                if resp.status_code in (401, 403):
                    if resp.status_code == 403 and "1010" in err_body:
                        raise FatalRiotAuthError(
                            "Riot API rejected the key (403/1010). "
                            "The key is usually expired or invalid. "
                            "Generate a fresh key in the Riot Developer Portal and retry."
                        )
                    raise FatalRiotAuthError(
                        f"Riot API auth failure HTTP {resp.status_code} for {url}. "
                        "Stop crawler, refresh/verify API key or access, then restart."
                    )
                raise RuntimeError(f"HTTP {resp.status_code} for {url}\n{err_body}")
            payload = resp.json()
            controller.on_success(endpoint_class, err_headers)
            stats.mark_success(endpoint_class)
            return payload
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.5 + attempt)
                last_err = e
                continue
            stats.mark_error(endpoint_class)
            raise RuntimeError(f"Network error for {url}: {e}") from e
        except ValueError as e:
            stats.mark_error(endpoint_class)
            raise RuntimeError(f"Invalid JSON for {url}: {e}") from e
        except FatalRiotAuthError:
            raise
        except Exception as e:
            if str(e).startswith("HTTP 429"):
                last_err = e
                continue
            if str(e).startswith("HTTP 5"):
                last_err = e
                continue
            if "auth failure" in str(e).lower() or "rejected the key" in str(e).lower():
                raise
            if attempt < retries:
                time.sleep(0.5 + attempt)
                last_err = e
                continue
            stats.mark_error(endpoint_class)
            raise
    stats.mark_error(endpoint_class)
    raise RuntimeError(f"Failed request after retries for {url}: {last_err}")


def _normalize_seed_entry(entry: dict[str, Any], fallback_tier: str) -> dict[str, Any] | None:
    puuid = entry.get("puuid")
    if not puuid:
        return None
    tier = str(entry.get("tier") or fallback_tier).upper()
    rank = str(entry.get("rank", "I") or "I")
    lp = int(entry.get("leaguePoints", 0) or 0)
    return {
        "puuid": str(puuid),
        "solo_tier": tier,
        "solo_rank": rank,
        "solo_lp": lp,
        "league_points": lp,
    }


def _sort_seed_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda x: (
            TIER_PRIORITY.get(str(x.get("solo_tier", "MASTER")), 99),
            -int(x.get("solo_lp", 0) or 0),
            str(x.get("puuid", "")),
        ),
    )


def _sort_seed_entries_general(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda x: (
            SOLO_TIER_ORDER.get(str(x.get("solo_tier", "")).upper(), 99),
            SOLO_RANK_ORDER.get(str(x.get("solo_rank", "I")).upper(), 99),
            -int(x.get("solo_lp", 0) or 0),
            str(x.get("puuid", "")),
        ),
    )


def get_league_seed(
    api_key: str,
    queue: str,
    tier: str,
    controller: RateController,
    stats: CrawlStats,
) -> list[dict[str, Any]]:
    tier_upper = tier.upper()
    endpoint_by_tier = {
        "CHALLENGER": "challengerleagues",
        "GRANDMASTER": "grandmasterleagues",
        "MASTER": "masterleagues",
    }
    endpoint = endpoint_by_tier[tier_upper]
    url = f"{PLATFORM_BASE}/lol/league/v4/{endpoint}/by-queue/{quote(queue)}"
    payload = request_json(
        url,
        api_key,
        endpoint_class="league_apex",
        controller=controller,
        stats=stats,
    )
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []
    for entry in raw_entries:
        merged = dict(entry)
        merged["tier"] = payload.get("tier", tier_upper)
        row = _normalize_seed_entry(merged, tier_upper)
        if row is not None:
            normalized.append(row)
    return sorted(
        normalized,
        key=lambda x: (
            -int(x.get("solo_lp", 0) or 0),
            str(x.get("puuid", "")),
        ),
    )


def resolve_apex_quotas(
    players: int,
    seed_challenger: int | None,
    seed_grandmaster: int | None,
    seed_master: int | None,
) -> tuple[dict[str, int], int, bool]:
    overrides = {
        "CHALLENGER": seed_challenger,
        "GRANDMASTER": seed_grandmaster,
        "MASTER": seed_master,
    }
    has_override = any(v is not None for v in overrides.values())
    if has_override:
        quotas = {tier: max(0, int(overrides[tier] or 0)) for tier in APEX_TIERS}
        total = sum(quotas.values())
        if total <= 0:
            raise ValueError("Seed overrides were provided but all are zero.")
        return quotas, total, True

    base = players // len(APEX_TIERS)
    remainder = players % len(APEX_TIERS)
    quotas = {tier: base for tier in APEX_TIERS}
    for tier in APEX_TIERS[:remainder]:
        quotas[tier] += 1
    return quotas, players, False


def select_apex_seed_players(
    entries_by_tier: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    target_total: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for tier in APEX_TIERS:
        need = max(0, int(quotas.get(tier, 0)))
        if need <= 0:
            continue
        taken = 0
        for entry in entries_by_tier.get(tier, []):
            puuid = str(entry["puuid"])
            if puuid in selected:
                continue
            selected[puuid] = entry
            taken += 1
            if taken >= need:
                break

    if len(selected) < target_total:
        leftovers: list[dict[str, Any]] = []
        for tier in APEX_TIERS:
            for entry in entries_by_tier.get(tier, []):
                puuid = str(entry["puuid"])
                if puuid in selected:
                    continue
                leftovers.append(entry)
        leftovers = _sort_seed_entries(leftovers)
        for entry in leftovers:
            if len(selected) >= target_total:
                break
            selected[str(entry["puuid"])] = entry

    return _sort_seed_entries(list(selected.values()))[:target_total]


def get_division_entries_page(
    api_key: str,
    queue: str,
    tier: str,
    division: str,
    page: int,
    controller: RateController,
    stats: CrawlStats,
) -> list[dict[str, Any]]:
    url = (
        f"{PLATFORM_BASE}/lol/league/v4/entries/{quote(queue)}/{quote(tier)}/{quote(division)}"
        f"?page={int(page)}"
    )
    payload = request_json(
        url,
        api_key,
        endpoint_class="league_division_entries",
        controller=controller,
        stats=stats,
    )
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def select_division_pages_for_bucket(
    tier: str,
    division: str,
    run_idx: int,
    selection_rng_seed: int,
) -> list[int]:
    pages = [1]
    bucket_hash = sum(ord(ch) for ch in f"{tier}:{division}")
    rng = random.Random(int(selection_rng_seed) + int(run_idx) * 1000 + bucket_hash)
    pages.append(int(rng.randint(2, 8)))
    return pages


def collect_division_seed_entries(
    api_key: str,
    queue: str,
    controller: RateController,
    stats: CrawlStats,
    run_idx: int,
    selection_rng_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    metrics = {
        "division_pages_fetched": 0,
        "division_entries_seen": 0,
        "rows_without_puuid": 0,
    }
    warnings: list[str] = []
    out: list[dict[str, Any]] = []
    for tier in DIVISION_TIERS:
        for division in DIVISION_RANKS:
            pages = select_division_pages_for_bucket(
                tier=tier,
                division=division,
                run_idx=run_idx,
                selection_rng_seed=selection_rng_seed,
            )
            for page in pages:
                try:
                    raw_entries = get_division_entries_page(
                        api_key,
                        queue,
                        tier,
                        division,
                        page,
                        controller,
                        stats,
                    )
                except Exception as exc:
                    warnings.append(f"{tier}-{division}-p{page}: {exc}")
                    continue
                metrics["division_pages_fetched"] += 1
                if not raw_entries:
                    continue
                metrics["division_entries_seen"] += len(raw_entries)
                for raw_entry in raw_entries:
                    row = _normalize_seed_entry(raw_entry, tier)
                    if row is None:
                        metrics["rows_without_puuid"] += 1
                        continue
                    row["source_kind"] = "division"
                    row["source_tier"] = tier
                    row["source_rank"] = division
                    row["source_page"] = page
                    out.append(row)
    return out, metrics, warnings


def collect_apex_seed_entries(
    api_key: str,
    queue: str,
    controller: RateController,
    stats: CrawlStats,
    tiers: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    active_tiers = [str(t).upper() for t in (tiers or APEX_TIERS)]
    entries_by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in active_tiers}
    warnings: list[str] = []
    if not active_tiers:
        return [], warnings
    with ThreadPoolExecutor(max_workers=len(active_tiers)) as ex:
        futures = {
            ex.submit(
                get_league_seed,
                api_key,
                queue,
                tier,
                controller,
                stats,
            ): tier
            for tier in active_tiers
        }
        for fut in as_completed(futures):
            tier = futures[fut]
            try:
                entries_by_tier[tier] = fut.result()
            except Exception as exc:
                warnings.append(f"{tier}: {exc}")
                entries_by_tier[tier] = []
    flattened: list[dict[str, Any]] = []
    for tier in active_tiers:
        for entry in entries_by_tier.get(tier, []):
            row = dict(entry)
            row["source_kind"] = "apex"
            row["source_tier"] = tier
            row["source_rank"] = str(row.get("solo_rank", "I"))
            row["source_page"] = None
            flattened.append(row)
    return flattened, warnings


def _sample_entries(
    rng: random.Random,
    entries: list[dict[str, Any]],
    take_n: int,
) -> list[dict[str, Any]]:
    if take_n <= 0 or not entries:
        return []
    ordered = sorted(
        entries,
        key=lambda row: (
            str(row.get("puuid", "")),
            -int(row.get("solo_lp", 0) or 0),
            str(row.get("solo_tier", "")),
            str(row.get("solo_rank", "")),
        ),
    )
    if len(ordered) <= take_n:
        return ordered
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    chosen_idx = sorted(indices[:take_n])
    return [ordered[i] for i in chosen_idx]


def resolve_leaderboard_demo_apex_quotas(
    seed_apex_demo_count: int,
    seed_apex_demo_challenger: int | None,
    seed_apex_demo_grandmaster: int | None,
    seed_apex_demo_master: int | None,
) -> tuple[dict[str, int] | None, int]:
    overrides = {
        "CHALLENGER": seed_apex_demo_challenger,
        "GRANDMASTER": seed_apex_demo_grandmaster,
        "MASTER": seed_apex_demo_master,
    }
    has_override = any(v is not None for v in overrides.values())
    if not has_override:
        return None, max(0, int(seed_apex_demo_count))
    quotas = {tier: max(0, int(overrides[tier] or 0)) for tier in APEX_TIERS}
    return quotas, sum(quotas.values())


def select_leaderboard_demo_seed_players(
    division_entries: list[dict[str, Any]],
    apex_entries: list[dict[str, Any]],
    seed_per_division: int,
    apex_demo_count: int,
    selection_rng_seed: int,
    run_idx: int,
    apex_quotas: dict[str, int] | None = None,
    division_seen_puuids_by_bucket: dict[str, set[str]] | None = None,
    apex_seen_puuids_by_tier: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(int(selection_rng_seed) + int(run_idx))
    selected: dict[str, dict[str, Any]] = {}
    division_selected = 0
    apex_selected = 0
    division_unseen_selected = 0
    apex_unseen_selected = 0

    def _sample_unseen_only(
        rows: list[dict[str, Any]],
        take_n: int,
        seen_predicate: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        if take_n <= 0 or not rows:
            return [], 0
        unseen_rows = [r for r in rows if not seen_predicate(r)]
        chosen = _sample_entries(rng, unseen_rows, take_n)
        return chosen, len(chosen)

    for tier in DIVISION_TIERS:
        for division in DIVISION_RANKS:
            bucket = [
                row
                for row in division_entries
                if str(row.get("solo_tier", "")).upper() == tier
                and str(row.get("solo_rank", "")).upper() == division
            ]
            bucket_key = f"{tier}:{division}"
            seen_bucket = (
                division_seen_puuids_by_bucket.get(bucket_key, set())
                if division_seen_puuids_by_bucket is not None
                else set()
            )
            chosen_rows, unseen_count = _sample_unseen_only(
                bucket,
                seed_per_division,
                lambda r: str(r.get("puuid", "")) in seen_bucket,
            )
            division_unseen_selected += unseen_count
            for row in chosen_rows:
                puuid = str(row["puuid"])
                if puuid in selected:
                    continue
                selected[puuid] = row
                division_selected += 1

    if apex_quotas is not None:
        target_total = max(0, int(sum(apex_quotas.values())))
        apex_by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in APEX_TIERS}
        for row in apex_entries:
            tier = str(row.get("solo_tier", "")).upper()
            if tier in apex_by_tier:
                apex_by_tier[tier].append(row)
        for tier in APEX_TIERS:
            seen_tier = (
                apex_seen_puuids_by_tier.get(tier, set())
                if apex_seen_puuids_by_tier is not None
                else set()
            )
            chosen_rows, unseen_count = _sample_unseen_only(
                apex_by_tier[tier],
                int(apex_quotas.get(tier, 0)),
                lambda r: str(r.get("puuid", "")) in seen_tier,
            )
            apex_unseen_selected += unseen_count
            for row in chosen_rows:
                puuid = str(row["puuid"])
                if puuid in selected:
                    continue
                selected[puuid] = row
                apex_selected += 1
        if apex_selected < target_total:
            leftovers: list[dict[str, Any]] = []
            for tier in APEX_TIERS:
                leftovers.extend(apex_by_tier[tier])
            chosen_rows, unseen_count = _sample_unseen_only(
                leftovers,
                target_total,
                lambda r: str(r.get("puuid", "")) in (
                    apex_seen_puuids_by_tier.get(str(r.get("solo_tier", "")).upper(), set())
                    if apex_seen_puuids_by_tier is not None
                    else set()
                ),
            )
            apex_unseen_selected += unseen_count
            for row in chosen_rows:
                if apex_selected >= target_total:
                    break
                puuid = str(row["puuid"])
                if puuid in selected:
                    continue
                selected[puuid] = row
                apex_selected += 1
    else:
        chosen_rows, unseen_count = _sample_unseen_only(
            apex_entries,
            apex_demo_count,
            lambda r: str(r.get("puuid", "")) in (
                apex_seen_puuids_by_tier.get(str(r.get("solo_tier", "")).upper(), set())
                if apex_seen_puuids_by_tier is not None
                else set()
            ),
        )
        apex_unseen_selected += unseen_count
        for row in chosen_rows:
            puuid = str(row["puuid"])
            if puuid in selected:
                continue
            selected[puuid] = row
            apex_selected += 1

    return _sort_seed_entries_general(list(selected.values())), {
        "division_selected": division_selected,
        "apex_selected": apex_selected,
        "division_unseen_selected": division_unseen_selected,
        "apex_unseen_selected": apex_unseen_selected,
    }


def should_refill_player_pool(
    cached_player_count: int,
    unused_player_count: int,
    min_unused: int,
) -> bool:
    if int(cached_player_count) <= 0:
        return True
    if int(min_unused) <= 0:
        return int(unused_player_count) <= 0
    return int(unused_player_count) < int(min_unused)


def should_fetch_match_ids_for_player(
    job_status: str | None,
    cache_known: bool,
    cached_ids: list[str] | None,
    requested_count: int,
    revisit_done_players: bool,
) -> bool:
    if job_status == "done" and not revisit_done_players and cache_known:
        return False
    if (
        job_status == "done"
        and cached_ids is not None
        and len(cached_ids) >= int(requested_count)
    ):
        return False
    return True


def load_player_source_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_player_source_state(path: Path, state: dict[str, Any]) -> None:
    save_json(path, state)


def apply_apex_count_stop(
    seen_by_tier: dict[str, set[str]],
    saturated_by_tier: dict[str, bool],
    count_targets: dict[str, int],
) -> tuple[dict[str, int], dict[str, bool]]:
    seen_counts: dict[str, int] = {}
    for tier in APEX_TIERS:
        seen_count = len(seen_by_tier.get(tier, set()))
        seen_counts[tier] = seen_count
        target = int(count_targets.get(tier, 0) or 0)
        if target > 0:
            # Explicit count thresholds override previous saturation state.
            saturated_by_tier[tier] = seen_count >= target
        elif tier not in saturated_by_tier:
            saturated_by_tier[tier] = False
    return seen_counts, saturated_by_tier


def load_seed_players_cache(seed_cache_path: Path) -> list[dict[str, Any]] | None:
    if not seed_cache_path.exists():
        return None
    try:
        raw = json.loads(seed_cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        puuid = item.get("puuid")
        if not puuid:
            continue
        rows.append(dict(item))
    return rows or None


def dedupe_players_by_puuid(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    for row in rows:
        puuid = row.get("puuid")
        if not puuid:
            dropped += 1
            continue
        puuid_s = str(puuid)
        if puuid_s in seen:
            dropped += 1
            continue
        seen.add(puuid_s)
        out.append(row)
    return out, dropped


def should_refresh_seed(run_idx: int, seed_refresh_every_runs: int) -> bool:
    if int(seed_refresh_every_runs) <= 0:
        return False
    cadence = max(1, int(seed_refresh_every_runs))
    return (int(run_idx) - 1) % cadence == 0


def init_seed_stats() -> dict[str, Any]:
    return {
        "refresh_executed": False,
        "division_pages_fetched": 0,
        "division_entries_seen": 0,
        "division_selected": 0,
        "apex_selected": 0,
        "seed_players_total": 0,
        "rows_without_puuid": 0,
    }


def match_ids_by_puuid(
    api_key: str,
    puuid: str,
    count: int,
    queue_id: int | None,
    match_type: str | None,
    controller: RateController,
    stats: CrawlStats,
) -> list[str]:
    query_parts = [f"start=0", f"count={count}"]
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


def match_detail(
    api_key: str,
    match_id: str,
    controller: RateController,
    stats: CrawlStats,
    timeout_sec: int = 10,
    retries: int = 1,
) -> dict[str, Any]:
    url = f"{REGIONAL_BASE}/lol/match/v5/matches/{quote(match_id)}"
    payload = request_json(
        url,
        api_key,
        endpoint_class="match_details",
        controller=controller,
        stats=stats,
        retries=max(0, int(retries)),
        timeout_sec=int(timeout_sec),
    )
    if isinstance(payload, dict):
        return payload
    return {}


def rank_entries_by_puuid(
    api_key: str,
    puuid: str,
    controller: RateController,
    stats: CrawlStats,
) -> list[dict[str, Any]]:
    url = f"{PLATFORM_BASE}/lol/league/v4/entries/by-puuid/{quote(puuid)}"
    payload = request_json(
        url,
        api_key,
        endpoint_class="rank_by_puuid",
        controller=controller,
        stats=stats,
    )
    if isinstance(payload, list):
        return payload
    return []


def validate_api_key(api_key: str, controller: RateController, stats: CrawlStats) -> None:
    url = f"{PLATFORM_BASE}/lol/status/v4/platform-data"
    request_json(
        url,
        api_key,
        endpoint_class="platform_status",
        controller=controller,
        stats=stats,
    )


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def match_detail_file_path(
    matches_dir: Path,
    match_id: str,
    compression: str,
) -> Path:
    if compression == "zstd":
        return matches_dir / f"{match_id}.json.zst"
    return matches_dir / f"{match_id}.json"


def save_match_detail(
    matches_dir: Path,
    match_id: str,
    detail: dict[str, Any],
    compression: str,
    zstd_level: int,
) -> Path:
    path = match_detail_file_path(matches_dir, match_id, compression)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstandard is not installed. Install it with `python -m pip install zstandard` "
                "or run with `--match-json-compression none`."
            )
        payload = json.dumps(detail, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressor = zstd.ZstdCompressor(level=int(zstd_level))
        path.write_bytes(compressor.compress(payload))
        return path
    save_json(path, detail)
    return path


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def should_skip_preflight(cache_path: Path, api_key: str, ttl_sec: int) -> bool:
    if ttl_sec <= 0 or not cache_path.exists():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("key_fp") != key_fingerprint(api_key):
        return False
    validated_at = int(payload.get("validated_at_utc", 0) or 0)
    return (int(time.time()) - validated_at) < ttl_sec


def write_preflight_cache(cache_path: Path, api_key: str) -> None:
    payload = {"key_fp": key_fingerprint(api_key), "validated_at_utc": int(time.time())}
    save_json(cache_path, payload)


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_ranks (
            puuid TEXT PRIMARY KEY,
            solo_tier TEXT,
            solo_rank TEXT,
            solo_lp INTEGER,
            fetched_at_utc INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            game_version TEXT,
            game_creation_utc_ms INTEGER,
            participant_count INTEGER,
            valid_for_pipeline INTEGER NOT NULL,
            reason TEXT,
            fetched_at_utc INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_participants (
            match_id TEXT NOT NULL,
            puuid TEXT NOT NULL,
            PRIMARY KEY (match_id, puuid)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_ids_cache (
            puuid TEXT PRIMARY KEY,
            match_ids_json TEXT NOT NULL,
            fetched_at_utc INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_match_participants_puuid ON match_participants(puuid)"
    )
    for job_table in JOB_TABLES:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {job_table} (
                entity_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                last_error TEXT,
                updated_at_utc INTEGER NOT NULL
            )
            """
        )
    return conn


def upsert_ranks(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    now_ts = int(time.time())
    for r in rows:
        conn.execute(
            """
            INSERT INTO player_ranks
            (puuid, solo_tier, solo_rank, solo_lp, fetched_at_utc)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(puuid) DO UPDATE SET
                solo_tier=excluded.solo_tier,
                solo_rank=excluded.solo_rank,
                solo_lp=excluded.solo_lp,
                fetched_at_utc=excluded.fetched_at_utc
            """,
            (
                r.get("puuid"),
                r.get("solo_tier"),
                r.get("solo_rank"),
                r.get("solo_lp"),
                now_ts,
            ),
        )
    conn.commit()


def get_cached_rank(conn: sqlite3.Connection, puuid: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT puuid, solo_tier, solo_rank, solo_lp
        FROM player_ranks
        WHERE puuid = ?
        """,
        (puuid,),
    ).fetchone()
    if row is None:
        return None
    return {
        "puuid": row["puuid"],
        "solo_tier": row["solo_tier"],
        "solo_rank": row["solo_rank"],
        "solo_lp": row["solo_lp"],
    }


def upsert_match_ids_cache(
    conn: sqlite3.Connection,
    puuid: str,
    match_ids: list[str],
    commit: bool = True,
) -> None:
    now_ts = int(time.time())
    conn.execute(
        """
        INSERT INTO match_ids_cache
        (puuid, match_ids_json, fetched_at_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            match_ids_json=excluded.match_ids_json,
            fetched_at_utc=excluded.fetched_at_utc
        """,
        (
            puuid,
            json.dumps([str(x) for x in match_ids], ensure_ascii=False),
            now_ts,
        ),
    )
    if commit:
        conn.commit()


def get_cached_match_ids_map(
    conn: sqlite3.Connection,
    puuids: list[str],
) -> dict[str, list[str]]:
    if not puuids:
        return {}
    placeholders = ",".join("?" for _ in puuids)
    rows = conn.execute(
        f"""
        SELECT puuid, match_ids_json
        FROM match_ids_cache
        WHERE puuid IN ({placeholders})
        """,
        tuple(puuids),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        puuid = str(row["puuid"])
        payload = row["match_ids_json"]
        mids: list[str] = []
        try:
            decoded = json.loads(str(payload))
            if isinstance(decoded, list):
                mids = [str(x) for x in decoded]
        except Exception:
            mids = []
        out[puuid] = mids
    return out


def get_match_record(conn: sqlite3.Connection, match_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM matches WHERE match_id = ?",
        (match_id,),
    ).fetchone()


def get_match_participants(
    conn: sqlite3.Connection, match_id: str
) -> list[str]:
    rows = conn.execute(
        """
        SELECT puuid
        FROM match_participants
        WHERE match_id = ?
        ORDER BY puuid ASC
        """,
        (match_id,),
    ).fetchall()
    return [str(r["puuid"]) for r in rows]


def get_dataset_match_totals(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_matches,
            COALESCE(SUM(CASE WHEN valid_for_pipeline = 1 THEN 1 ELSE 0 END), 0) AS kept_matches
        FROM matches
        """
    ).fetchone()
    if row is None:
        return {"total_matches": 0, "kept_matches": 0}
    return {
        "total_matches": int(row["total_matches"] or 0),
        "kept_matches": int(row["kept_matches"] or 0),
    }


def get_kept_matches_total_from_db(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN valid_for_pipeline = 1 THEN 1 ELSE 0 END), 0) AS kept_matches
            FROM matches
            """
        ).fetchone()
        if row is None:
            return 0
        return int(row["kept_matches"] or 0)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def reached_kept_match_target(current_kept: int, target: int) -> bool:
    return int(target) > 0 and int(current_kept) >= int(target)


def _validate_job_table(table: str) -> str:
    if table not in JOB_TABLES:
        raise ValueError(f"Unsupported job table: {table}")
    return table


def enqueue_jobs(
    conn: sqlite3.Connection,
    table: str,
    entity_ids: list[str],
    commit: bool = True,
) -> None:
    table_name = _validate_job_table(table)
    if not entity_ids:
        return
    now_ts = int(time.time())
    rows = [
        (entity_id, "pending", 0, None, now_ts)
        for entity_id in entity_ids
    ]
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO {table_name}
        (entity_id, status, attempt_count, last_error, updated_at_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()


def get_job_statuses(
    conn: sqlite3.Connection,
    table: str,
    entity_ids: list[str],
) -> dict[str, str]:
    table_name = _validate_job_table(table)
    if not entity_ids:
        return {}
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"""
        SELECT entity_id, status
        FROM {table_name}
        WHERE entity_id IN ({placeholders})
        """,
        tuple(entity_ids),
    ).fetchall()
    return {str(r["entity_id"]): str(r["status"]) for r in rows}


def mark_jobs_running(
    conn: sqlite3.Connection,
    table: str,
    entity_ids: list[str],
    commit: bool = True,
) -> None:
    table_name = _validate_job_table(table)
    if not entity_ids:
        return
    now_ts = int(time.time())
    rows = [(now_ts, entity_id) for entity_id in entity_ids]
    conn.executemany(
        f"""
        UPDATE {table_name}
        SET status = 'running', attempt_count = attempt_count + 1, updated_at_utc = ?
        WHERE entity_id = ?
        """,
        rows,
    )
    if commit:
        conn.commit()


def mark_job_done(
    conn: sqlite3.Connection,
    table: str,
    entity_id: str,
    commit: bool = True,
) -> None:
    table_name = _validate_job_table(table)
    now_ts = int(time.time())
    conn.execute(
        f"""
        INSERT INTO {table_name}
        (entity_id, status, attempt_count, last_error, updated_at_utc)
        VALUES (?, 'done', 1, NULL, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            status = 'done',
            last_error = NULL,
            updated_at_utc = excluded.updated_at_utc
        """,
        (entity_id, now_ts),
    )
    if commit:
        conn.commit()


def mark_job_failed(
    conn: sqlite3.Connection,
    table: str,
    entity_id: str,
    error: str,
    commit: bool = True,
) -> None:
    table_name = _validate_job_table(table)
    now_ts = int(time.time())
    error_trim = (error or "")[:400]
    conn.execute(
        f"""
        INSERT INTO {table_name}
        (entity_id, status, attempt_count, last_error, updated_at_utc)
        VALUES (?, 'failed', 1, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            status = 'failed',
            attempt_count = attempt_count + 1,
            last_error = excluded.last_error,
            updated_at_utc = excluded.updated_at_utc
        """,
        (entity_id, error_trim, now_ts),
    )
    if commit:
        conn.commit()


def get_job_status_counts(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    table_name = _validate_job_table(table)
    rows = conn.execute(
        f"""
        SELECT status, COUNT(*) AS c
        FROM {table_name}
        GROUP BY status
        """
    ).fetchall()
    out: dict[str, int] = {}
    for row in rows:
        out[str(row["status"])] = int(row["c"])
    return out


def reset_stale_running_jobs(
    conn: sqlite3.Connection,
    table: str,
    stale_sec: int,
) -> int:
    table_name = _validate_job_table(table)
    if stale_sec <= 0:
        return 0
    now_ts = int(time.time())
    cutoff = now_ts - int(stale_sec)
    result = conn.execute(
        f"""
        UPDATE {table_name}
        SET status = 'pending', last_error = 'reset_stale_running', updated_at_utc = ?
        WHERE status = 'running' AND updated_at_utc < ?
        """,
        (now_ts, cutoff),
    )
    conn.commit()
    return int(result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0)


def requeue_failed_jobs(
    conn: sqlite3.Connection,
    table: str,
    cooldown_sec: int,
) -> int:
    table_name = _validate_job_table(table)
    if cooldown_sec <= 0:
        return 0
    now_ts = int(time.time())
    cutoff = now_ts - int(cooldown_sec)
    result = conn.execute(
        f"""
        UPDATE {table_name}
        SET status = 'pending', last_error = 'retry_after_cooldown', updated_at_utc = ?
        WHERE status = 'failed' AND updated_at_utc < ?
        """,
        (now_ts, cutoff),
    )
    conn.commit()
    return int(result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0)


def get_failed_entities_since(
    conn: sqlite3.Connection,
    table: str,
    since_utc: int,
) -> list[str]:
    table_name = _validate_job_table(table)
    rows = conn.execute(
        f"""
        SELECT entity_id
        FROM {table_name}
        WHERE status = 'failed' AND updated_at_utc >= ?
        """,
        (int(since_utc),),
    ).fetchall()
    return [str(row["entity_id"]) for row in rows]


def get_all_failed_entities(
    conn: sqlite3.Connection,
    table: str,
) -> list[str]:
    table_name = _validate_job_table(table)
    rows = conn.execute(
        f"""
        SELECT entity_id
        FROM {table_name}
        WHERE status = 'failed'
        """
    ).fetchall()
    return [str(row["entity_id"]) for row in rows]


def requeue_failed_entities(
    conn: sqlite3.Connection,
    table: str,
    entity_ids: list[str],
    reason: str,
) -> int:
    table_name = _validate_job_table(table)
    if not entity_ids:
        return 0
    now_ts = int(time.time())
    placeholders = ",".join("?" for _ in entity_ids)
    params: list[Any] = [now_ts, str(reason)[:120], *entity_ids]
    result = conn.execute(
        f"""
        UPDATE {table_name}
        SET status = 'pending', updated_at_utc = ?, last_error = ?
        WHERE status = 'failed' AND entity_id IN ({placeholders})
        """,
        tuple(params),
    )
    conn.commit()
    return int(result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0)


def compute_health_alerts(
    crawl_stats: dict[str, Any],
    alert_429_rate: float,
    alert_failed_job_rate: float,
) -> dict[str, Any]:
    totals = crawl_stats.get("totals", {})
    requests = int(totals.get("requests", 0) or 0)
    n_429 = int(totals.get("http_429", 0) or 0)
    rate_429 = n_429 / max(1, requests)

    jobs = crawl_stats.get("jobs", {})
    job_tables = ("jobs_match_ids", "jobs_match_details", "jobs_rank_lookup")
    job_failed = 0
    job_total = 0
    for table in job_tables:
        counts = jobs.get(table, {})
        job_failed += int(counts.get("failed", 0) or 0)
        for status in ("pending", "running", "done", "failed"):
            job_total += int(counts.get(status, 0) or 0)
    failed_job_rate = job_failed / max(1, job_total)

    alerts: list[str] = []
    if rate_429 >= alert_429_rate:
        alerts.append(
            f"429 rate {rate_429:.3f} >= threshold {alert_429_rate:.3f}"
        )
    if job_total > 0 and failed_job_rate >= alert_failed_job_rate:
        alerts.append(
            f"failed-job rate {failed_job_rate:.3f} >= threshold {alert_failed_job_rate:.3f}"
        )
    if int(totals.get("errors", 0) or 0) > 0:
        alerts.append(f"errors > 0 ({int(totals.get('errors', 0) or 0)})")
    return {
        "rate_429": rate_429,
        "failed_job_rate": failed_job_rate,
        "failed_jobs": job_failed,
        "job_total": job_total,
        "alerts": alerts,
    }


def upsert_match_with_participants(
    conn: sqlite3.Connection,
    match_id: str,
    game_version: str,
    game_creation_utc_ms: int,
    participant_rows: list[dict[str, Any]],
    valid_for_pipeline: bool,
    reason: str,
    commit: bool = True,
) -> None:
    now_ts = int(time.time())
    conn.execute(
        """
        INSERT INTO matches
        (match_id, game_version, game_creation_utc_ms, participant_count, valid_for_pipeline, reason, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
            game_version=excluded.game_version,
            game_creation_utc_ms=excluded.game_creation_utc_ms,
            participant_count=excluded.participant_count,
            valid_for_pipeline=excluded.valid_for_pipeline,
            reason=excluded.reason,
            fetched_at_utc=excluded.fetched_at_utc
        """,
        (
            match_id,
            game_version,
            game_creation_utc_ms,
            len(participant_rows),
            1 if valid_for_pipeline else 0,
            reason,
            now_ts,
        ),
    )
    conn.execute("DELETE FROM match_participants WHERE match_id = ?", (match_id,))
    for p in participant_rows:
        conn.execute(
            """
            INSERT INTO match_participants
            (match_id, puuid)
            VALUES (?, ?)
            """,
            (
                match_id,
                p.get("puuid"),
            ),
        )
    if commit:
        conn.commit()


def effective_workers(requested: int, max_inflight: int | None) -> int:
    workers = max(1, int(requested))
    if max_inflight is not None:
        workers = min(workers, max(1, int(max_inflight)))
    return workers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Riot API data crawl pipeline")
    p.add_argument("--api-key", type=str, default=os.getenv("RIOT_API_KEY", ""))
    p.add_argument(
        "--platform-routing",
        "--platform",
        dest="platform_routing",
        type=str,
        default=DEFAULT_PLATFORM_ROUTING,
        help="Platform routing value for platform-host APIs (e.g. EUW1, NA1, KR, JP1).",
    )
    p.add_argument(
        "--regional-routing",
        "--region",
        dest="regional_routing",
        type=str,
        default=None,
        help="Regional routing value for match-host APIs (americas|asia|europe|sea). "
        "If omitted, it is inferred from --platform-routing.",
    )
    p.add_argument("--queue", type=str, default=DEFAULT_QUEUE)
    p.add_argument("--players", type=int, default=3, help="Player target count when quotas are not explicitly set")
    p.set_defaults(seed_scope="leaderboard_demo")
    p.add_argument("--players-challenger", "--seed-challenger", dest="seed_challenger", type=int, default=None, help="Explicit Challenger player count")
    p.add_argument("--players-grandmaster", "--seed-grandmaster", dest="seed_grandmaster", type=int, default=None, help="Explicit Grandmaster player count")
    p.add_argument("--players-master", "--seed-master", dest="seed_master", type=int, default=None, help="Explicit Master player count")
    p.add_argument("--players-per-division", "--seed-per-division", dest="seed_per_division", type=int, default=2, help="Demo mode: random players to sample per division bucket")
    p.add_argument("--players-apex-demo-count", "--seed-apex-demo-count", dest="seed_apex_demo_count", type=int, default=6, help="Demo mode: additional random apex players to include")
    p.add_argument("--players-apex-demo-challenger", "--seed-apex-demo-challenger", dest="seed_apex_demo_challenger", type=int, default=None, help="Demo mode: explicit Challenger player count override")
    p.add_argument("--players-apex-demo-grandmaster", "--seed-apex-demo-grandmaster", dest="seed_apex_demo_grandmaster", type=int, default=None, help="Demo mode: explicit Grandmaster player count override")
    p.add_argument("--players-apex-demo-master", "--seed-apex-demo-master", dest="seed_apex_demo_master", type=int, default=None, help="Demo mode: explicit Master player count override")
    p.add_argument("--player-selection-rng-seed", "--seed-selection-rng-seed", dest="seed_selection_rng_seed", type=int, default=42, help="Base RNG seed for deterministic demo player selection")
    p.add_argument("--seed-refresh-every-runs", type=int, default=0, help="Cadence fallback for player refill (0 disables cadence policy)")
    p.add_argument(
        "--player-refill-min-unused",
        type=int,
        default=0,
        help="When unused players fall below this, fetch and append more players (0 = refill only when unused is 0)",
    )
    p.add_argument(
        "--apex-stop-challenger-count",
        type=int,
        default=DEFAULT_APEX_STOP_CHALLENGER_COUNT,
        help="Stop refreshing Challenger tier once seen unique Challenger players reaches this count (0 disables)",
    )
    p.add_argument(
        "--apex-stop-grandmaster-count",
        type=int,
        default=DEFAULT_APEX_STOP_GRANDMASTER_COUNT,
        help="Stop refreshing Grandmaster tier once seen unique GM players reaches this count (0 disables)",
    )
    p.add_argument("--matches-per-player", type=int, default=2)
    p.add_argument("--queue-id", type=int, default=420, help="Match queue filter for match-id search (420 = Ranked Solo SR)")
    p.add_argument("--match-type", type=str, default="ranked", help="Match type filter for match-id search (e.g. ranked)")
    p.add_argument(
        "--revisit-done-players",
        action="store_true",
        help="Allow re-fetching match IDs for players already marked done (default: off)",
    )
    p.add_argument("--season-major", type=int, default=16, help="Keep only matches where gameVersion starts with this major, e.g. 16")
    p.add_argument("--preflight-ttl-sec", type=int, default=86400, help="Skip key preflight if validated recently with same key")
    p.add_argument("--force-preflight", action="store_true", help="Always run preflight check now")
    p.add_argument("--workers-match-ids", type=int, default=8, help="Worker threads for by-puuid match-id calls")
    p.add_argument("--workers-match-details", type=int, default=8, help="Worker threads for match detail calls")
    p.add_argument("--workers-ranks", type=int, default=8, help="Worker threads for rank by-puuid calls")
    p.add_argument(
        "--db-commit-every",
        type=int,
        default=100,
        help="Commit SQLite writes every N write operations during crawl stages",
    )
    p.add_argument(
        "--timeout-match-details-sec",
        type=int,
        default=10,
        help="HTTP timeout in seconds for match-detail calls",
    )
    p.add_argument("--max-inflight-match-ids", type=int, default=None, help="Hard cap on in-flight match-id calls")
    p.add_argument("--max-inflight-match-details", type=int, default=None, help="Hard cap on in-flight match-detail calls")
    p.add_argument("--max-inflight-ranks", type=int, default=None, help="Hard cap on in-flight rank calls")
    p.add_argument("--rate-profile", type=str, choices=["auto", "conservative", "aggressive"], default="auto", help="Adaptive rate profile")
    p.add_argument(
        "--app-limit-requests",
        type=int,
        default=DEFAULT_APP_LIMIT_REQUESTS,
        help="Global app-level request limit for adaptive limiter window",
    )
    p.add_argument(
        "--app-limit-window-sec",
        type=float,
        default=DEFAULT_APP_LIMIT_WINDOW_SEC,
        help="Global app-level limiter window size in seconds",
    )
    p.add_argument(
        "--stale-running-sec",
        type=int,
        default=900,
        help="At startup, reset running jobs older than this many seconds back to pending (0 disables)",
    )
    p.add_argument(
        "--failed-retry-cooldown-sec",
        type=int,
        default=1800,
        help="At startup, requeue failed jobs older than this many seconds (0 disables)",
    )
    p.add_argument(
        "--alert-429-rate",
        type=float,
        default=0.15,
        help="Health alert threshold for HTTP 429 rate",
    )
    p.add_argument(
        "--alert-failed-job-rate",
        type=float,
        default=0.10,
        help="Health alert threshold for failed job ratio",
    )
    p.add_argument(
        "--health-log-file",
        type=str,
        default="run_health_log.jsonl",
        help="JSONL file (under out-dir) for per-run health records",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously and start a new crawl every loop-interval-sec",
    )
    p.add_argument(
        "--loop-interval-sec",
        type=int,
        default=120,
        help="Sleep duration between loop runs (0 = run back-to-back)",
    )
    p.add_argument(
        "--stop-flag-file",
        type=str,
        default="",
        help="If this file exists, loop mode exits cleanly before the next run",
    )
    p.add_argument(
        "--loop-max-runs",
        type=int,
        default=0,
        help="Stop loop after N runs (0 = infinite)",
    )
    p.add_argument(
        "--stop-after-kept-matches",
        type=int,
        default=0,
        help="Auto-stop when total kept matches in DB reach this value (0 disables)",
    )
    p.add_argument(
        "--match-json-compression",
        type=str,
        choices=["none", "zstd"],
        default="zstd",
        help="Compression format for saved match detail files",
    )
    p.add_argument(
        "--match-zstd-level",
        type=int,
        default=3,
        help="Zstandard compression level for match files (when enabled)",
    )
    p.add_argument("--out-dir", type=str, default="runtime/out_latest/runs/smokes/riot_smoke")
    return p.parse_args()


def run_single(
    args: argparse.Namespace,
    run_idx: int = 1,
    retry_failed_since_utc: int | None = None,
    retry_failed_all: bool = False,
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    matches_dir = out_dir / "matches"
    db_path = out_dir / "player_ranks.sqlite3"
    preflight_cache_path = out_dir / ".preflight_cache.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)
    conn = open_db(db_path)
    stats = CrawlStats()
    controller = RateController(
        profile=args.rate_profile,
        app_limit_requests=args.app_limit_requests,
        app_limit_window_sec=args.app_limit_window_sec,
    )
    db_commit_every = max(1, int(args.db_commit_every))
    pending_db_writes = 0
    recent_failed_entities: dict[str, set[str]] = {t: set() for t in JOB_TABLES}

    def maybe_commit_db(write_ops: int = 1, force: bool = False) -> None:
        nonlocal pending_db_writes
        if force:
            if pending_db_writes > 0:
                conn.commit()
                pending_db_writes = 0
            return
        pending_db_writes += max(1, int(write_ops))
        if pending_db_writes >= db_commit_every:
            conn.commit()
            pending_db_writes = 0

    try:
        if args.force_preflight or not should_skip_preflight(
            preflight_cache_path, args.api_key, args.preflight_ttl_sec
        ):
            print("Preflight: validate API key")
            validate_api_key(args.api_key, controller=controller, stats=stats)
            write_preflight_cache(preflight_cache_path, args.api_key)
            print("  key validation OK")
        else:
            print(
                f"Preflight: skipped (cached within {args.preflight_ttl_sec}s for same key)"
            )

        stale_reset_counts = {}
        for job_table in JOB_TABLES:
            stale_reset_counts[job_table] = reset_stale_running_jobs(
                conn,
                job_table,
                args.stale_running_sec,
            )
        if sum(stale_reset_counts.values()) > 0:
            print("Recovered stale running jobs:")
            for job_table, recovered in stale_reset_counts.items():
                if recovered > 0:
                    print(f"  {job_table}: {recovered}")

        failed_requeue_counts = {}
        if retry_failed_all:
            for job_table in JOB_TABLES:
                ids = get_all_failed_entities(conn, job_table)
                recent_failed_entities[job_table] = set(ids)
                failed_requeue_counts[job_table] = requeue_failed_entities(
                    conn,
                    job_table,
                    ids,
                    reason="retry_stop_finalize_all",
                )
            if sum(failed_requeue_counts.values()) > 0:
                print("Requeued all failed jobs for STOP final cleanup pass:")
                for job_table, recovered in failed_requeue_counts.items():
                    if recovered > 0:
                        print(f"  {job_table}: {recovered}")
        elif retry_failed_since_utc is not None:
            for job_table in JOB_TABLES:
                ids = get_failed_entities_since(conn, job_table, int(retry_failed_since_utc))
                recent_failed_entities[job_table] = set(ids)
                failed_requeue_counts[job_table] = requeue_failed_entities(
                    conn,
                    job_table,
                    ids,
                    reason="retry_stop_finalize_recent",
                )
            if sum(failed_requeue_counts.values()) > 0:
                print("Requeued recent failed jobs for final cleanup pass:")
                for job_table, recovered in failed_requeue_counts.items():
                    if recovered > 0:
                        print(f"  {job_table}: {recovered}")

        print("Step 1/4: player sourcing -> puuid")
        seed_cache_path = out_dir / "seed_players.json"
        player_source_state_path = out_dir / "player_source_state.json"
        seed_stats = init_seed_stats()
        normalized_seed_players: list[dict[str, Any]] = []
        seed_rank_rows: list[dict[str, Any]] = []
        quota_override_mode = False
        quotas = {"CHALLENGER": 0, "GRANDMASTER": 0, "MASTER": 0}
        apex_count_stop_targets = {
            "CHALLENGER": max(0, int(args.apex_stop_challenger_count)),
            "GRANDMASTER": max(0, int(args.apex_stop_grandmaster_count)),
        }
        source_state = load_player_source_state(player_source_state_path)
        source_seen_raw = source_state.get("apex_seen_puuids", {})
        source_saturated_raw = source_state.get("apex_saturated", {})
        source_division_selected_seen_raw = source_state.get(
            "division_selected_seen_puuids", {}
        )
        source_apex_selected_seen_raw = source_state.get(
            "apex_selected_seen_puuids", {}
        )
        seen_apex_by_tier: dict[str, set[str]] = {
            tier: {
                str(x)
                for x in (source_seen_raw.get(tier, []) if isinstance(source_seen_raw, dict) else [])
                if x
            }
            for tier in APEX_TIERS
        }
        selected_seen_division_by_bucket: dict[str, set[str]] = {
            f"{tier}:{division}": {
                str(x)
                for x in (
                    source_division_selected_seen_raw.get(f"{tier}:{division}", [])
                    if isinstance(source_division_selected_seen_raw, dict)
                    else []
                )
                if x
            }
            for tier in DIVISION_TIERS
            for division in DIVISION_RANKS
        }
        selected_seen_apex_by_tier: dict[str, set[str]] = {
            tier: {
                str(x)
                for x in (
                    source_apex_selected_seen_raw.get(tier, [])
                    if isinstance(source_apex_selected_seen_raw, dict)
                    else []
                )
                if x
            }
            for tier in APEX_TIERS
        }
        saturated_apex_tiers: dict[str, bool] = {
            tier: bool(source_saturated_raw.get(tier, False))
            if isinstance(source_saturated_raw, dict)
            else False
            for tier in APEX_TIERS
        }
        # Stop thresholds should track selected apex seed players, not full endpoint rosters.
        apex_seen_count_by_tier, saturated_apex_tiers = apply_apex_count_stop(
            seen_by_tier=selected_seen_apex_by_tier,
            saturated_by_tier=saturated_apex_tiers,
            count_targets=apex_count_stop_targets,
        )
        cached_seed_players = load_seed_players_cache(seed_cache_path) or []
        cached_seed_players, dropped_cached = dedupe_players_by_puuid(cached_seed_players)
        if dropped_cached > 0:
            print(f"  dropped duplicate/invalid cached players: {dropped_cached}")
            save_json(seed_cache_path, cached_seed_players)

        existing_seed_puuids = [str(p["puuid"]) for p in cached_seed_players if p.get("puuid")]
        existing_seed_statuses = get_job_statuses(conn, "jobs_match_ids", existing_seed_puuids)
        unused_cached_players = [
            p for p in cached_seed_players if existing_seed_statuses.get(str(p.get("puuid"))) != "done"
        ]
        cadence_refresh = should_refresh_seed(run_idx, args.seed_refresh_every_runs)
        should_refill_by_pool = should_refill_player_pool(
            cached_player_count=len(cached_seed_players),
            unused_player_count=len(unused_cached_players),
            min_unused=args.player_refill_min_unused,
        )
        refresh_executed = cadence_refresh or should_refill_by_pool

        if refresh_executed:
            if cadence_refresh:
                print("  player refill triggered by cadence policy")
            else:
                if args.player_refill_min_unused <= 0:
                    print(
                        f"  player refill triggered by empty unused pool ({len(unused_cached_players)} <= 0)"
                    )
                else:
                    print(
                        f"  player refill triggered by low unused pool ({len(unused_cached_players)} < {args.player_refill_min_unused})"
                    )

            seed_players: list[dict[str, Any]] = []
            if args.seed_scope == "challenger":
                challenger_entries = get_league_seed(
                    args.api_key,
                    args.queue,
                    "CHALLENGER",
                    controller=controller,
                    stats=stats,
                )
                for row in challenger_entries:
                    row["source_kind"] = "apex"
                    row["source_tier"] = "CHALLENGER"
                    row["source_rank"] = str(row.get("solo_rank", "I"))
                    row["source_page"] = None
                seed_players = challenger_entries[: args.players]
                quotas = {"CHALLENGER": args.players, "GRANDMASTER": 0, "MASTER": 0}
            elif args.seed_scope == "leaderboard_demo":
                division_entries, division_metrics, division_warnings = collect_division_seed_entries(
                    args.api_key,
                    args.queue,
                    controller,
                    stats,
                    run_idx,
                    args.seed_selection_rng_seed,
                )
                active_apex_tiers = [
                    tier
                    for tier in APEX_TIERS
                    if not saturated_apex_tiers.get(tier, False)
                ]
                apex_entries, apex_warnings = collect_apex_seed_entries(
                    args.api_key,
                    args.queue,
                    controller,
                    stats,
                    tiers=active_apex_tiers,
                )
                for row in apex_entries:
                    tier = str(row.get("solo_tier", "")).upper()
                    puuid = row.get("puuid")
                    if puuid and tier in seen_apex_by_tier:
                        # Keep endpoint-level visibility for diagnostics only.
                        seen_apex_by_tier[tier].add(str(puuid))
                demo_apex_quotas, demo_apex_target = resolve_leaderboard_demo_apex_quotas(
                    args.seed_apex_demo_count,
                    args.seed_apex_demo_challenger,
                    args.seed_apex_demo_grandmaster,
                    args.seed_apex_demo_master,
                )
                seed_players, selection_counts = select_leaderboard_demo_seed_players(
                    division_entries=division_entries,
                    apex_entries=apex_entries,
                    seed_per_division=args.seed_per_division,
                    apex_demo_count=demo_apex_target,
                    selection_rng_seed=args.seed_selection_rng_seed,
                    run_idx=run_idx,
                    apex_quotas=demo_apex_quotas,
                    division_seen_puuids_by_bucket=selected_seen_division_by_bucket,
                    apex_seen_puuids_by_tier=selected_seen_apex_by_tier,
                )
                seed_stats["division_pages_fetched"] = int(division_metrics["division_pages_fetched"])
                seed_stats["division_entries_seen"] = int(division_metrics["division_entries_seen"])
                seed_stats["rows_without_puuid"] = int(division_metrics["rows_without_puuid"])
                seed_stats["division_selected"] = int(selection_counts["division_selected"])
                seed_stats["apex_selected"] = int(selection_counts["apex_selected"])
                seed_stats["division_unseen_selected"] = int(
                    selection_counts.get("division_unseen_selected", 0)
                )
                seed_stats["apex_unseen_selected"] = int(
                    selection_counts.get("apex_unseen_selected", 0)
                )
                seed_stats["apex_active_tiers"] = active_apex_tiers
                warnings = division_warnings + apex_warnings
                if warnings:
                    print("  warning: partial seed endpoint failures")
                    for warning in warnings:
                        print(f"    - {warning}")
            else:
                quotas, target_total, quota_override_mode = resolve_apex_quotas(
                    args.players,
                    args.seed_challenger,
                    args.seed_grandmaster,
                    args.seed_master,
                )
                entries_by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in APEX_TIERS}
                warnings: list[str] = []
                with ThreadPoolExecutor(max_workers=len(APEX_TIERS)) as ex:
                    futures = {
                        ex.submit(
                            get_league_seed,
                            args.api_key,
                            args.queue,
                            tier,
                            controller,
                            stats,
                        ): tier
                        for tier in APEX_TIERS
                    }
                    for fut in as_completed(futures):
                        tier = futures[fut]
                        try:
                            entries_by_tier[tier] = fut.result()
                        except Exception as exc:
                            warnings.append(f"{tier}: {exc}")
                            entries_by_tier[tier] = []

                seed_players = select_apex_seed_players(
                    entries_by_tier=entries_by_tier,
                    quotas=quotas,
                    target_total=target_total,
                )
                for row in seed_players:
                    row["source_kind"] = "apex"
                    row["source_tier"] = str(row.get("solo_tier", "MASTER"))
                    row["source_rank"] = str(row.get("solo_rank", "I"))
                    row["source_page"] = None
                if warnings:
                    print("  warning: partial seed endpoint failures")
                    for warning in warnings:
                        print(f"    - {warning}")

            normalized_new_players: list[dict[str, Any]] = []
            for s in seed_players:
                puuid = s.get("puuid")
                if not puuid:
                    continue
                puuid_s = str(puuid)
                tier_u = str(s.get("solo_tier", "")).upper()
                rank_u = str(s.get("solo_rank", "")).upper()
                if tier_u in DIVISION_TIERS and rank_u in DIVISION_RANKS:
                    selected_seen_division_by_bucket[f"{tier_u}:{rank_u}"].add(puuid_s)
                if tier_u in APEX_TIERS:
                    selected_seen_apex_by_tier[tier_u].add(puuid_s)
                row = {
                    "puuid": puuid,
                    "solo_tier": s.get("solo_tier"),
                    "solo_rank": s.get("solo_rank"),
                    "solo_lp": s.get("solo_lp"),
                    "league_points": s.get("solo_lp"),
                    "source_kind": s.get("source_kind"),
                    "source_tier": s.get("source_tier"),
                    "source_rank": s.get("source_rank"),
                    "source_page": s.get("source_page"),
                    "source_scope": args.seed_scope,
                }
                normalized_new_players.append(row)

            merged_players = cached_seed_players + normalized_new_players
            normalized_seed_players, dropped_merged = dedupe_players_by_puuid(merged_players)
            if dropped_merged > 0:
                print(f"  dropped duplicate/invalid players during merge: {dropped_merged}")
            save_json(seed_cache_path, normalized_seed_players)
        else:
            normalized_seed_players = cached_seed_players
            print(
                f"  reusing cached players ({len(normalized_seed_players)} total, {len(unused_cached_players)} unused)"
            )

        save_player_source_state(
            player_source_state_path,
            {
                "updated_at_utc": int(time.time()),
                "apex_seen_puuids": {
                    tier: sorted(list(puuids))
                    for tier, puuids in seen_apex_by_tier.items()
                },
                "apex_saturated": saturated_apex_tiers,
                "apex_count_stop_targets": {
                    "CHALLENGER": int(args.apex_stop_challenger_count),
                    "GRANDMASTER": int(args.apex_stop_grandmaster_count),
                },
                "division_selected_seen_puuids": {
                    bucket: sorted(list(puuids))
                    for bucket, puuids in selected_seen_division_by_bucket.items()
                },
                "apex_selected_seen_puuids": {
                    tier: sorted(list(puuids))
                    for tier, puuids in selected_seen_apex_by_tier.items()
                },
            },
        )

        # Recompute apex saturation based only on selected apex seed players.
        apex_seen_count_by_tier, saturated_apex_tiers = apply_apex_count_stop(
            seen_by_tier=selected_seen_apex_by_tier,
            saturated_by_tier=saturated_apex_tiers,
            count_targets=apex_count_stop_targets,
        )
        if args.seed_scope == "leaderboard_demo":
            saturated_list = [tier for tier, sat in saturated_apex_tiers.items() if sat]
            seed_counts = {
                tier: int(apex_seen_count_by_tier.get(tier, 0)) for tier in APEX_TIERS
            }
            seed_stats["apex_saturated_tiers"] = saturated_list
            seed_stats["apex_seed_seen_count_by_tier"] = seed_counts
            # Keep legacy key populated for compatibility.
            seed_stats["apex_seen_count_by_tier"] = seed_counts
            seed_stats["apex_count_stop_targets"] = {
                tier: int(apex_count_stop_targets.get(tier, 0))
                for tier in APEX_TIERS
            }

        for row in normalized_seed_players:
            seed_rank_rows.append(
                {
                    "puuid": row["puuid"],
                    "solo_tier": row.get("solo_tier"),
                    "solo_rank": row.get("solo_rank"),
                    "solo_lp": row.get("solo_lp"),
                }
            )
        upsert_ranks(conn, seed_rank_rows)
        seed_stats["refresh_executed"] = bool(refresh_executed)
        seed_stats["seed_players_total"] = len(normalized_seed_players)
        print(f"  total players in source pool: {len(normalized_seed_players)}")
        if args.seed_scope == "apex" and refresh_executed:
            mode = "explicit" if quota_override_mode else "auto-even"
            print(f"  apex quotas ({mode}): {quotas}")
        if args.seed_scope == "leaderboard_demo":
            print(
                "  leaderboard demo selection:"
                f" division={seed_stats['division_selected']}, apex={seed_stats['apex_selected']}"
            )
            print(
                "  unseen-first selected:"
                f" division={seed_stats.get('division_unseen_selected', 0)}"
                f", apex={seed_stats.get('apex_unseen_selected', 0)}"
            )
            saturated_list = seed_stats.get("apex_saturated_tiers", [])
            if saturated_list:
                print(f"  apex stop active for tiers: {saturated_list}")
            seen_counts = seed_stats.get("apex_seed_seen_count_by_tier", {})
            print(
                "  apex selected-seed seen counts:"
                f" CHALLENGER={seen_counts.get('CHALLENGER', 0)}"
                f", GRANDMASTER={seen_counts.get('GRANDMASTER', 0)}"
            )

        print(
            f"Step 2/4: puuid -> match ids (queue={args.queue_id}, type={args.match_type})"
        )
        match_ids_map: dict[str, list[str]] = {}
        all_match_ids: set[str] = set()
        seed_puuids = [str(p["puuid"]) for p in normalized_seed_players if p.get("puuid")]
        match_ids_cache_path = out_dir / "match_ids_by_puuid.json"
        prev_match_ids_map: dict[str, list[str]] = {}
        if match_ids_cache_path.exists():
            try:
                raw_prev = json.loads(match_ids_cache_path.read_text(encoding="utf-8"))
                if isinstance(raw_prev, dict):
                    prev_match_ids_map = {
                        str(k): [str(x) for x in (v or [])]
                        for k, v in raw_prev.items()
                        if isinstance(v, list)
                    }
            except Exception:
                prev_match_ids_map = {}
        db_match_ids_map = get_cached_match_ids_map(conn, seed_puuids)

        enqueue_jobs(conn, "jobs_match_ids", seed_puuids, commit=False)
        maybe_commit_db(len(seed_puuids))
        job_statuses = get_job_statuses(conn, "jobs_match_ids", seed_puuids)
        to_fetch_seed_puuids: list[str] = []
        skipped_done_no_revisit = 0
        recovered_done_missing_cache = 0

        print("Step 3/4: keep only season matches, skip low-participant matches, dedupe by DB")
        print("  mode: streaming overlap (match-id fetch -> match-detail fetch)")
        participants: dict[str, dict[str, Any]] = {}
        participant_index_by_match: dict[str, list[str]] = {}
        kept = 0
        skipped_already = 0
        skipped_season = 0
        skipped_low_participants = 0
        detail_failures = 0

        workers_details = effective_workers(
            args.workers_match_details,
            args.max_inflight_match_details,
        )
        detail_executor = ThreadPoolExecutor(max_workers=workers_details)
        detail_futures: dict[Any, str] = {}
        detail_submitted: set[str] = set()

        def handle_match_detail(mid: str, detail: dict[str, Any]) -> None:
            nonlocal kept, skipped_season, skipped_low_participants
            info = detail.get("info", {})
            game_version = str(info.get("gameVersion", ""))
            game_creation = int(info.get("gameCreation", 0) or 0)
            raw_parts = info.get("participants", [])
            participant_rows: list[dict[str, Any]] = []
            seen_puuids: set[str] = set()
            for part in raw_parts:
                puuid = part.get("puuid")
                if not puuid or puuid in seen_puuids:
                    continue
                seen_puuids.add(puuid)
                participant_rows.append({"puuid": puuid})

            if not game_version.startswith(f"{args.season_major}."):
                skipped_season += 1
                upsert_match_with_participants(
                    conn,
                    match_id=mid,
                    game_version=game_version,
                    game_creation_utc_ms=game_creation,
                    participant_rows=[],
                    valid_for_pipeline=False,
                    reason=f"not_season_{args.season_major}",
                    commit=False,
                )
                mark_job_done(conn, "jobs_match_details", mid, commit=False)
                maybe_commit_db(2)
                return

            if len(seen_puuids) < 10:
                skipped_low_participants += 1
                upsert_match_with_participants(
                    conn,
                    match_id=mid,
                    game_version=game_version,
                    game_creation_utc_ms=game_creation,
                    participant_rows=participant_rows,
                    valid_for_pipeline=False,
                    reason="less_than_10_unique_puuids",
                    commit=False,
                )
                mark_job_done(conn, "jobs_match_details", mid, commit=False)
                maybe_commit_db(2)
                return

            save_match_detail(
                matches_dir=matches_dir,
                match_id=mid,
                detail=detail,
                compression=args.match_json_compression,
                zstd_level=args.match_zstd_level,
            )
            upsert_match_with_participants(
                conn,
                match_id=mid,
                game_version=game_version,
                game_creation_utc_ms=game_creation,
                participant_rows=participant_rows,
                valid_for_pipeline=True,
                reason="ok",
                commit=False,
            )
            mark_job_done(conn, "jobs_match_details", mid, commit=False)
            maybe_commit_db(2)
            participant_index_by_match[mid] = [str(p["puuid"]) for p in participant_rows]
            kept += 1
            for p in participant_rows:
                puuid = p.get("puuid")
                if not puuid:
                    continue
                participants[puuid] = {"puuid": puuid}

        def schedule_match_detail_if_needed(mid: str) -> None:
            nonlocal skipped_already
            if mid in detail_submitted:
                return
            detail_submitted.add(mid)
            enqueue_jobs(conn, "jobs_match_details", [mid], commit=False)
            maybe_commit_db()
            existing = get_match_record(conn, mid)
            if existing is not None:
                mark_job_done(conn, "jobs_match_details", mid, commit=False)
                maybe_commit_db()
                skipped_already += 1
                if int(existing["valid_for_pipeline"]) != 1:
                    return
                cached_parts = get_match_participants(conn, mid)
                participant_index_by_match[mid] = cached_parts
                for puuid in cached_parts:
                    participants[puuid] = {"puuid": puuid}
                return
            mark_jobs_running(conn, "jobs_match_details", [mid], commit=False)
            maybe_commit_db()
            fut = detail_executor.submit(
                match_detail,
                args.api_key,
                mid,
                controller,
                stats,
                int(args.timeout_match_details_sec),
            )
            detail_futures[fut] = mid

        def drain_completed_detail_futures(wait_for_one: bool = False) -> None:
            nonlocal detail_failures
            if not detail_futures:
                return
            done_list: list[Any]
            if wait_for_one:
                done, _ = wait(
                    set(detail_futures.keys()),
                    return_when=FIRST_COMPLETED,
                )
                done_list = list(done)
            else:
                done_list = [f for f in list(detail_futures.keys()) if f.done()]
            for fut in done_list:
                mid = detail_futures.pop(fut)
                try:
                    detail = fut.result()
                    handle_match_detail(mid, detail)
                except Exception as exc:
                    detail_failures += 1
                    mark_job_failed(conn, "jobs_match_details", mid, str(exc), commit=False)
                    maybe_commit_db()

        for puuid in seed_puuids:
            cache_known = False
            cached_ids: list[str] | None = None
            if puuid in prev_match_ids_map:
                cached_ids = prev_match_ids_map[puuid]
                cache_known = True
            elif puuid in db_match_ids_map:
                cached_ids = db_match_ids_map[puuid]
                cache_known = True
            requested_count = int(args.matches_per_player)
            job_status = job_statuses.get(puuid)
            if not should_fetch_match_ids_for_player(
                job_status=job_status,
                cache_known=cache_known,
                cached_ids=cached_ids,
                requested_count=requested_count,
                revisit_done_players=args.revisit_done_players,
            ):
                if cached_ids is not None:
                    match_ids_map[puuid] = cached_ids[:requested_count]
                    all_match_ids.update(match_ids_map[puuid])
                    for mid in match_ids_map[puuid]:
                        schedule_match_detail_if_needed(mid)
                else:
                    match_ids_map[puuid] = []
                if job_status == "done" and not args.revisit_done_players:
                    skipped_done_no_revisit += 1
                continue
            if job_status == "done" and not args.revisit_done_players and not cache_known:
                recovered_done_missing_cache += 1
            to_fetch_seed_puuids.append(puuid)
        forced_retry_seed_puuids = recent_failed_entities.get("jobs_match_ids", set())
        for puuid in sorted(forced_retry_seed_puuids):
            if puuid in seed_puuids and puuid not in to_fetch_seed_puuids:
                to_fetch_seed_puuids.append(puuid)
        if forced_retry_seed_puuids:
            print(
                "  forced match-id retry set size (recent failed): "
                f"{len(forced_retry_seed_puuids)}"
            )

        workers_ids = effective_workers(args.workers_match_ids, args.max_inflight_match_ids)
        mark_jobs_running(conn, "jobs_match_ids", to_fetch_seed_puuids, commit=False)
        if to_fetch_seed_puuids:
            maybe_commit_db(len(to_fetch_seed_puuids))
        try:
            if workers_ids == 1 or len(to_fetch_seed_puuids) <= 1:
                for puuid in to_fetch_seed_puuids:
                    try:
                        mids = match_ids_by_puuid(
                            args.api_key,
                            puuid,
                            int(args.matches_per_player),
                            queue_id=args.queue_id,
                            match_type=args.match_type,
                            controller=controller,
                            stats=stats,
                        )
                        match_ids_map[puuid] = mids
                        all_match_ids.update(mids)
                        upsert_match_ids_cache(conn, puuid, mids, commit=False)
                        mark_job_done(conn, "jobs_match_ids", puuid, commit=False)
                        maybe_commit_db(2)
                    except Exception as exc:
                        mark_job_failed(conn, "jobs_match_ids", puuid, str(exc), commit=False)
                        maybe_commit_db()
                        match_ids_map[puuid] = []
                        mids = []
                    for mid in mids:
                        schedule_match_detail_if_needed(mid)
                    drain_completed_detail_futures(wait_for_one=False)
            else:
                with ThreadPoolExecutor(max_workers=workers_ids) as ex:
                    id_futures = {
                        ex.submit(
                            match_ids_by_puuid,
                            args.api_key,
                            puuid,
                            int(args.matches_per_player),
                            args.queue_id,
                            args.match_type,
                            controller,
                            stats,
                        ): puuid
                        for puuid in to_fetch_seed_puuids
                    }
                    while id_futures:
                        done_ids, _ = wait(
                            set(id_futures.keys()),
                            timeout=0.2,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done_ids:
                            drain_completed_detail_futures(wait_for_one=False)
                            continue
                        for fut in done_ids:
                            puuid = id_futures.pop(fut)
                            try:
                                mids = fut.result()
                                upsert_match_ids_cache(conn, puuid, mids, commit=False)
                                mark_job_done(conn, "jobs_match_ids", puuid, commit=False)
                                maybe_commit_db(2)
                            except Exception as exc:
                                mark_job_failed(conn, "jobs_match_ids", puuid, str(exc), commit=False)
                                maybe_commit_db()
                                mids = []
                            match_ids_map[puuid] = mids
                            all_match_ids.update(mids)
                            for mid in mids:
                                schedule_match_detail_if_needed(mid)
                        drain_completed_detail_futures(wait_for_one=False)

            forced_retry_detail_ids = recent_failed_entities.get("jobs_match_details", set())
            for mid in sorted(forced_retry_detail_ids):
                schedule_match_detail_if_needed(mid)
            if forced_retry_detail_ids:
                print(
                    "  forced match-detail retry set size (recent failed): "
                    f"{len(forced_retry_detail_ids)}"
                )

            while detail_futures:
                drain_completed_detail_futures(wait_for_one=True)
        finally:
            detail_executor.shutdown(wait=True)

        maybe_commit_db(force=True)
        save_json(match_ids_cache_path, match_ids_map)
        print(f"  unique match ids: {len(all_match_ids)}")
        if skipped_done_no_revisit > 0:
            print(f"  skipped done players (no revisit policy): {skipped_done_no_revisit}")
        if recovered_done_missing_cache > 0:
            print(f"  recovered done players with missing cache by refetch: {recovered_done_missing_cache}")
        save_json(out_dir / "participant_index_by_match.json", participant_index_by_match)
        print(f"  kept matches: {kept}")
        print(f"  skipped already in DB: {skipped_already}")
        print(f"  skipped non-season-{args.season_major}: {skipped_season}")
        print(f"  skipped <10 unique puuids: {skipped_low_participants}")
        print(f"  detail fetch failures: {detail_failures}")
        print(f"  distinct participants seen: {len(participants)}")

        print("Step 4/4: participant puuid -> rank (with DB rank cache)")
        rank_rows: list[dict[str, Any]] = []
        rank_cache_hits = 0
        rank_cache_ranked_hits = 0
        rank_cache_unranked_hits = 0
        rank_api_calls = 0
        rank_api_failures = 0
        to_fetch_rank: list[str] = []
        participant_candidates = list(participants.keys())
        forced_retry_rank_puuids = recent_failed_entities.get("jobs_rank_lookup", set())
        if forced_retry_rank_puuids:
            participant_candidates = sorted(
                set(participant_candidates).union(forced_retry_rank_puuids)
            )
            print(
                "  forced rank retry set size (recent failed): "
                f"{len(forced_retry_rank_puuids)}"
            )
        enqueue_jobs(conn, "jobs_rank_lookup", participant_candidates, commit=False)
        if participant_candidates:
            maybe_commit_db(len(participant_candidates))

        for puuid in participant_candidates:

            cached = get_cached_rank(conn, puuid)
            if cached is not None:
                rank_cache_hits += 1
                if cached.get("solo_tier") is None or cached.get("solo_rank") is None:
                    rank_cache_unranked_hits += 1
                else:
                    rank_cache_ranked_hits += 1
                rank_rows.append(cached)
                mark_job_done(conn, "jobs_rank_lookup", puuid, commit=False)
                maybe_commit_db()
                continue
            to_fetch_rank.append(puuid)
        maybe_commit_db(force=True)

        fetched_rank_entries: dict[str, list[dict[str, Any]]] = {}
        failed_rank_puuids: set[str] = set()
        workers_ranks = effective_workers(args.workers_ranks, args.max_inflight_ranks)
        mark_jobs_running(conn, "jobs_rank_lookup", to_fetch_rank, commit=False)
        if to_fetch_rank:
            maybe_commit_db(len(to_fetch_rank))
        if workers_ranks == 1 or len(to_fetch_rank) <= 1:
            for puuid in to_fetch_rank:
                try:
                    fetched_rank_entries[puuid] = rank_entries_by_puuid(
                        args.api_key,
                        puuid,
                        controller=controller,
                        stats=stats,
                    )
                    rank_api_calls += 1
                except Exception:
                    failed_rank_puuids.add(puuid)
                    rank_api_failures += 1
        else:
            with ThreadPoolExecutor(max_workers=workers_ranks) as ex:
                futures = {
                    ex.submit(rank_entries_by_puuid, args.api_key, puuid, controller, stats): puuid
                    for puuid in to_fetch_rank
                }
                for fut in as_completed(futures):
                    puuid = futures[fut]
                    try:
                        fetched_rank_entries[puuid] = fut.result()
                        rank_api_calls += 1
                    except Exception:
                        failed_rank_puuids.add(puuid)
                        rank_api_failures += 1

        for puuid in to_fetch_rank:
            if puuid in failed_rank_puuids:
                mark_job_failed(
                    conn,
                    "jobs_rank_lookup",
                    puuid,
                    "rank lookup failed",
                    commit=False,
                )
                maybe_commit_db()
                continue
            entries = fetched_rank_entries.get(puuid, [])
            solo = next((e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"), None)
            rank_rows.append(
                {
                    "puuid": puuid,
                    "solo_tier": None if not solo else solo.get("tier"),
                    "solo_rank": None if not solo else solo.get("rank"),
                    "solo_lp": None if not solo else solo.get("leaguePoints"),
                }
            )
            mark_job_done(conn, "jobs_rank_lookup", puuid, commit=False)
            maybe_commit_db()
        maybe_commit_db(force=True)

        save_json(out_dir / "player_ranks_by_puuid.json", rank_rows)
        upsert_ranks(conn, rank_rows)
        print(f"  rank rows saved: {len(rank_rows)}")
        print(f"  rank cache hits: {rank_cache_hits}")
        print(f"    ranked cache hits: {rank_cache_ranked_hits}")
        print(f"    no-solo-rank cache hits: {rank_cache_unranked_hits}")
        print(f"  rank API calls: {rank_api_calls}")
        print(f"  rank API failures (not cached): {rank_api_failures}")

        crawl_stats = stats.to_dict()
        crawl_stats["cache"] = {
            "rank_cache_hits": rank_cache_hits,
            "rank_cache_ranked_hits": rank_cache_ranked_hits,
            "rank_cache_unranked_hits": rank_cache_unranked_hits,
            "rank_api_calls": rank_api_calls,
            "rank_api_failures": rank_api_failures,
            "rank_cache_hit_ratio": rank_cache_hits / max(1, rank_cache_hits + rank_api_calls),
        }
        crawl_stats["rate_controller"] = controller.snapshot()
        crawl_stats["config"] = {
            "run_idx": run_idx,
            "retry_failed_since_utc": retry_failed_since_utc,
            "retry_failed_all": retry_failed_all,
            "platform_routing": args.platform_routing,
            "regional_routing": args.regional_routing,
            "platform_base": PLATFORM_BASE,
            "regional_base": REGIONAL_BASE,
            "player_scope": args.seed_scope,
            "players": args.players,
            "queue": args.queue,
            "queue_id": args.queue_id,
            "match_type": args.match_type,
            "revisit_done_players": args.revisit_done_players,
            "matches_per_player": args.matches_per_player,
            "players_per_division": args.seed_per_division,
            "players_apex_demo_count": args.seed_apex_demo_count,
            "players_apex_demo_challenger": args.seed_apex_demo_challenger,
            "players_apex_demo_grandmaster": args.seed_apex_demo_grandmaster,
            "players_apex_demo_master": args.seed_apex_demo_master,
            "player_selection_rng_seed": args.seed_selection_rng_seed,
            "seed_refresh_every_runs": args.seed_refresh_every_runs,
            "player_refill_min_unused": args.player_refill_min_unused,
            "apex_stop_challenger_count": args.apex_stop_challenger_count,
            "apex_stop_grandmaster_count": args.apex_stop_grandmaster_count,
            "rate_profile": args.rate_profile,
            "workers_match_ids": workers_ids,
            "workers_match_details": workers_details,
            "workers_ranks": workers_ranks,
            "db_commit_every": args.db_commit_every,
            "timeout_match_details_sec": args.timeout_match_details_sec,
            "max_inflight_match_ids": args.max_inflight_match_ids,
            "max_inflight_match_details": args.max_inflight_match_details,
            "max_inflight_ranks": args.max_inflight_ranks,
            "app_limit_requests": args.app_limit_requests,
            "app_limit_window_sec": args.app_limit_window_sec,
            "match_json_compression": args.match_json_compression,
            "match_zstd_level": args.match_zstd_level,
            "stale_running_sec": args.stale_running_sec,
            "failed_retry_cooldown_sec": args.failed_retry_cooldown_sec,
            "stop_after_kept_matches": args.stop_after_kept_matches,
        }
        crawl_stats["seed"] = seed_stats
        crawl_stats["jobs"] = {
            "jobs_match_ids": get_job_status_counts(conn, "jobs_match_ids"),
            "jobs_match_details": get_job_status_counts(conn, "jobs_match_details"),
            "jobs_rank_lookup": get_job_status_counts(conn, "jobs_rank_lookup"),
            "stale_resets": stale_reset_counts,
            "failed_requeues": failed_requeue_counts,
        }
        crawl_stats["health"] = compute_health_alerts(
            crawl_stats,
            alert_429_rate=args.alert_429_rate,
            alert_failed_job_rate=args.alert_failed_job_rate,
        )
        dataset_totals = get_dataset_match_totals(conn)
        crawl_stats["dataset"] = {
            "matches_total": dataset_totals["total_matches"],
            "kept_matches_total": dataset_totals["kept_matches"],
        }
        save_json(out_dir / "crawl_stats.json", crawl_stats)

        totals = crawl_stats["totals"]
        print("\nRun stats:")
        print(f"  requests: {totals['requests']}")
        print(f"  successes: {totals['success']}")
        print(f"  retries: {totals['retries']}")
        print(f"  HTTP 429: {totals['http_429']}")
        print(f"  HTTP 5xx: {totals['http_5xx']}")
        print(f"  request rate (req/s): {crawl_stats['requests_per_sec']:.2f}")
        print(f"  success rate (req/s): {crawl_stats['success_per_sec']:.2f}")
        print(f"  429 rate: {crawl_stats['health']['rate_429']:.3f}")
        print(f"  failed job rate: {crawl_stats['health']['failed_job_rate']:.3f}")
        print(f"  kept matches total (DB): {crawl_stats['dataset']['kept_matches_total']}")
        print("  job status snapshot:")
        for table_name, table_counts in crawl_stats["jobs"].items():
            print(f"    {table_name}: {table_counts}")
        if crawl_stats["health"]["alerts"]:
            print("  HEALTH ALERTS:")
            for alert in crawl_stats["health"]["alerts"]:
                print(f"    - {alert}")

        print("\nDone.")
        print(f"Outputs:\n  {out_dir / 'seed_players.json'}")
        print(f"  {out_dir / 'match_ids_by_puuid.json'}")
        match_glob = "*.json.zst" if args.match_json_compression == "zstd" else "*.json"
        print(f"  {matches_dir}/{match_glob}")
        print(f"  {out_dir / 'participant_index_by_match.json'}")
        print(f"  {out_dir / 'player_ranks_by_puuid.json'}")
        print(f"  {out_dir / 'crawl_stats.json'}")
        print(f"  {out_dir / 'player_ranks.sqlite3'}")
        return crawl_stats
    finally:
        conn.close()


def resolve_stop_flag_path(args: argparse.Namespace) -> Path | None:
    if not args.stop_flag_file:
        return None
    p = Path(args.stop_flag_file)
    if p.is_absolute():
        return p
    return Path(args.out_dir) / p


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set RIOT_API_KEY.")
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
        f" platform={args.platform_routing} ({PLATFORM_BASE})"
        f", regional={args.regional_routing} ({REGIONAL_BASE})"
    )
    if args.players <= 0 or args.matches_per_player <= 0:
        raise SystemExit("players and matches-per-player must be positive.")
    if args.seed_apex_demo_challenger is not None and args.seed_apex_demo_challenger < 0:
        raise SystemExit("seed-apex-demo-challenger must be >= 0.")
    if args.seed_apex_demo_grandmaster is not None and args.seed_apex_demo_grandmaster < 0:
        raise SystemExit("seed-apex-demo-grandmaster must be >= 0.")
    if args.seed_apex_demo_master is not None and args.seed_apex_demo_master < 0:
        raise SystemExit("seed-apex-demo-master must be >= 0.")
    if args.seed_per_division <= 0:
        raise SystemExit("seed-per-division must be positive.")
    if args.seed_apex_demo_count < 0:
        raise SystemExit("seed-apex-demo-count must be >= 0.")
    if args.seed_refresh_every_runs < 0:
        raise SystemExit("seed-refresh-every-runs must be >= 0.")
    if args.player_refill_min_unused < 0:
        raise SystemExit("player-refill-min-unused must be >= 0.")
    if args.apex_stop_challenger_count < 0:
        raise SystemExit("apex-stop-challenger-count must be >= 0.")
    if args.apex_stop_grandmaster_count < 0:
        raise SystemExit("apex-stop-grandmaster-count must be >= 0.")
    if args.timeout_match_details_sec <= 0:
        raise SystemExit("timeout-match-details-sec must be positive.")
    if args.db_commit_every <= 0:
        raise SystemExit("db-commit-every must be positive.")
    if args.stale_running_sec < 0:
        raise SystemExit("stale-running-sec must be >= 0.")
    if args.failed_retry_cooldown_sec < 0:
        raise SystemExit("failed-retry-cooldown-sec must be >= 0.")
    if args.alert_429_rate < 0 or args.alert_failed_job_rate < 0:
        raise SystemExit("alert thresholds must be >= 0.")
    if args.app_limit_requests <= 0:
        raise SystemExit("app-limit-requests must be positive.")
    if args.app_limit_window_sec <= 0:
        raise SystemExit("app-limit-window-sec must be positive.")
    if args.match_json_compression not in {"none", "zstd"}:
        raise SystemExit("match-json-compression must be one of: none, zstd.")
    if args.match_zstd_level < -7 or args.match_zstd_level > 22:
        raise SystemExit("match-zstd-level must be between -7 and 22.")
    if args.match_json_compression == "zstd" and zstd is None:
        raise SystemExit(
            "zstandard is not installed. Install it with `python -m pip install zstandard` "
            "or use `--match-json-compression none`."
        )
    if args.loop_interval_sec < 0:
        raise SystemExit("loop-interval-sec must be >= 0.")
    if args.loop_max_runs < 0:
        raise SystemExit("loop-max-runs must be >= 0.")
    if args.stop_after_kept_matches < 0:
        raise SystemExit("stop-after-kept-matches must be >= 0.")
    run_idx = 0
    stop_flag_path = resolve_stop_flag_path(args)
    target_kept_matches = int(args.stop_after_kept_matches)
    db_path = Path(args.out_dir) / "player_ranks.sqlite3"
    stop_cleanup_scheduled = False
    previous_run_started_utc: int | None = None
    while True:
        if (
            args.loop
            and run_idx == 0
            and stop_flag_path is not None
            and stop_flag_path.exists()
        ):
            print(f"Stop flag present before first run, exiting loop: {stop_flag_path}")
            break
        if args.loop and reached_kept_match_target(
            get_kept_matches_total_from_db(db_path),
            target_kept_matches,
        ):
            current = get_kept_matches_total_from_db(db_path)
            print(
                "Kept match target reached before next run "
                f"({current} >= {target_kept_matches}), exiting loop."
            )
            break
        run_idx += 1
        run_started = int(time.time())
        retry_all_for_run = bool(stop_cleanup_scheduled)
        retry_since_for_run: int | None = None
        if not retry_all_for_run and args.loop and previous_run_started_utc is not None:
            retry_since_for_run = previous_run_started_utc
        print(f"\n=== Crawl Run {run_idx} ===")
        if retry_all_for_run:
            print(
                "STOP cleanup mode: retrying all failed jobs once, then exiting."
            )
        elif retry_since_for_run is not None:
            print(
                "Recent-failed retry mode: retrying only failed jobs from previous run "
                f"(updated_at_utc >= {retry_since_for_run})"
            )
        health_log_path = Path(args.out_dir) / args.health_log_file
        try:
            crawl_stats = run_single(
                args,
                run_idx=run_idx,
                retry_failed_since_utc=retry_since_for_run,
                retry_failed_all=retry_all_for_run,
            )
            health_entry = {
                "run_idx": run_idx,
                "run_started_utc": run_started,
                "run_finished_utc": int(time.time()),
                "ok": True,
                "totals": crawl_stats.get("totals", {}),
                "jobs": crawl_stats.get("jobs", {}),
                "health": crawl_stats.get("health", {}),
                "seed": crawl_stats.get("seed", {}),
            }
        except FatalRiotAuthError as exc:
            health_entry = {
                "run_idx": run_idx,
                "run_started_utc": run_started,
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
                "run_idx": run_idx,
                "run_started_utc": run_started,
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
            kept_total = int(crawl_stats.get("dataset", {}).get("kept_matches_total", 0))
            if reached_kept_match_target(kept_total, target_kept_matches):
                print(
                    "Kept match target reached after run "
                    f"({kept_total} >= {target_kept_matches}), exiting."
                )
                break

        previous_run_started_utc = run_started
        if not args.loop:
            break
        if args.loop and stop_flag_path is not None and stop_flag_path.exists():
            if stop_cleanup_scheduled:
                print(f"Stop flag detected, final cleanup pass completed. Exiting loop: {stop_flag_path}")
                break
            stop_cleanup_scheduled = True
            print(
                "Stop flag detected. Scheduling one final cleanup run for all failed jobs."
            )
            continue
        if args.loop_max_runs > 0 and run_idx >= args.loop_max_runs:
            print(f"Loop finished after {run_idx} runs.")
            break
        if args.loop_interval_sec > 0:
            print(f"Sleeping {args.loop_interval_sec}s before next run...")
            time.sleep(args.loop_interval_sec)
        else:
            print("Starting next run immediately (loop-interval-sec=0).")


if __name__ == "__main__":
    main()
