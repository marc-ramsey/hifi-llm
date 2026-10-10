"""Backend health probes — collect status of all configured backends."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Shared mutable state — updated by the periodic probe task
_current_report_value: HealthReport | None = None
_health_task: asyncio.Task | None = None
_deferred_models: list[Any] | None = None  # models to probe when event loop starts
_previous_status: dict[str, HealthStatus] | None = None  # name -> status for change detection


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class BackendHealth:
    name: str
    url: str
    status: HealthStatus
    error: str | None = None
    latency_ms: float | None = None
    model_count: int | None = None


@dataclass
class HealthReport:
    """Immutable snapshot of all backend health states."""
    backends: list[BackendHealth] = field(default_factory=list)
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    total: int = 0


# ── Probe function ──────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = httpx.Timeout(
    connect=2.0,   # quick to detect unreachable backends
    read=5.0,      # generous for slow backends (e.g., cold model loads)
    write=5.0,
    pool=10.0,
)
STARTUP_TIMEOUT = httpx.Timeout(
    connect=1.0,
    read=2.0,
    write=2.0,
    pool=5.0,
)


def _make_timeout(timeout: float | httpx.Timeout) -> httpx.Timeout:
    """Normalise a timeout value to an httpx.Timeout.

    Accepts either a plain float (legacy: all timeouts equal) or an
    httpx.Timeout object.  Returns the value unchanged if already structured;
    otherwise wraps it as ``httpx.Timeout(connect=v, read=v, write=v, pool=v*2)``
    so that callers passing a bare float still get differentiated connect vs.
    read timeouts.
    """
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(connect=timeout, read=timeout, write=timeout, pool=timeout * 2)


async def _probe_single(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
    verify_ssl: bool = True,
) -> BackendHealth:
    """Probe a single backend and return its health state."""
    start = time.monotonic()
    try:
        resp = await client.get(f"{url}/v1/models", timeout=_make_timeout(timeout), verify=verify_ssl)
        elapsed_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 200:
            body = resp.json()
            models = body.get("data", [])
            model_count = len(models) if isinstance(models, list) else None

            # Degraded if it responds but has no models (slow, or partial)
            if model_count is not None and model_count == 0:
                status = HealthStatus.DEGRADED
                error = "backend returned empty model list"
            else:
                status = HealthStatus.HEALTHY
                error = None

            return BackendHealth(
                name=name, url=url, status=status, latency_ms=elapsed_ms,
                model_count=model_count,
            )
        else:
            return BackendHealth(
                name=name, url=url, status=HealthStatus.UNHEALTHY,
                error=f"HTTP {resp.status_code}", latency_ms=elapsed_ms,
            )

    except httpx.TimeoutException:
        elapsed_ms = (time.monotonic() - start) * 1000
        return BackendHealth(name=name, url=url, status=HealthStatus.UNHEALTHY,
                             error="timeout", latency_ms=elapsed_ms)
    except httpx.ConnectError as e:
        return BackendHealth(name=name, url=url, status=HealthStatus.UNHEALTHY,
                             error=f"connection refused: {e}")
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return BackendHealth(name=name, url=url, status=HealthStatus.UNHEALTHY,
                             error=str(e), latency_ms=elapsed_ms)


async def collect_health(
    models: list[Any],
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
) -> HealthReport:
    """Probe all configured backends and return a HealthReport.

    Probes are run in parallel — total time is roughly the slowest single probe.

    Args:
        models: List of ModelConfig objects (each must have .name and .url).
        timeout: Per-backend probe timeout.  Accepts a plain float (legacy)
                 or an ``httpx.Timeout`` with separate connect/read/write values.

    Returns:
        HealthReport with status for each backend.
    """
    if not models:
        return HealthReport()

    async with httpx.AsyncClient(timeout=_make_timeout(timeout)) as client:
        tasks = [_probe_single(client, m.name, m.url, timeout, verify_ssl=m.verify_ssl) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    backends = []
    healthy = degraded = unhealthy = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            model = models[i]
            bh = BackendHealth(name=model.name, url=model.url,
                               status=HealthStatus.UNHEALTHY, error=str(result))
        else:
            bh = result

        backends.append(bh)

        if bh.status == HealthStatus.HEALTHY:
            healthy += 1
        elif bh.status == HealthStatus.DEGRADED:
            degraded += 1
        else:
            unhealthy += 1

    return HealthReport(
        backends=backends,
        healthy_count=healthy,
        degraded_count=degraded,
        unhealthy_count=unhealthy,
        total=len(models),
    )


# ── Logging helper ──────────────────────────────────────────────────────────

def log_health_report(report: HealthReport) -> None:
    """Log backend health. First call always logs full report; subsequent calls
    only warn on status changes."""
    global _previous_status

    current = {bh.name: bh.status for bh in report.backends}

    if _previous_status is None:
        # Startup — always log everything
        lines = [f"Backend health ({report.total} models):"]
        for bh in report.backends:
            tag = {HealthStatus.HEALTHY: "OK", HealthStatus.DEGRADED: "WARN",
                   HealthStatus.UNHEALTHY: "FAIL"}[bh.status]
            parts = [f"  [{tag:4s}] {bh.name} -> {bh.url}"]
            if bh.status == HealthStatus.HEALTHY and bh.latency_ms is not None:
                parts.append(f"{bh.latency_ms:.0f}ms")
                if bh.model_count is not None:
                    parts.append(f"({bh.model_count} models)")
            if bh.error:
                parts.append(f"— {bh.error}")
            lines.append(" ".join(parts))
        logger.info("\n" + "\n".join(lines))
    else:
        # Detect changes — log only deltas
        changed = [
            (name, _previous_status[name], new_s)
            for name, new_s in current.items()
            if _previous_status.get(name) != new_s
        ]

        if changed:
            lines = ["Backend health changes:"]
            # Build a name→BackendHealth map for error lookups
            bh_map = {bh.name: bh for bh in report.backends}
            for name, old_s, new_s in changed:
                direction = "DOWN" if new_s != HealthStatus.HEALTHY else "UP"
                bh = bh_map.get(name)
                err_msg = f" — {bh.error}" if bh and bh.error else ""
                lines.append(f"  [{direction}] {name}: {old_s.value} -> {new_s.value}{err_msg}")
            logger.warning("\n" + "\n".join(lines))

    _previous_status = current


# ── Periodic probe task ─────────────────────────────────────────────────────

async def _probe_loop(models: list[Any], interval: float) -> None:
    """Continuously probe backends at the given interval."""
    timeout = STARTUP_TIMEOUT  # first probe uses short structured timeout
    while True:
        try:
            report = await collect_health(models, timeout=timeout)
            global _current_report_value
            _current_report_value = report
            log_health_report(report)
        except Exception:
            logger.exception("Health probe failed")
        # After first probe, use the configured interval as a plain float
        # (which _make_timeout will wrap into a structured Timeout).
        timeout = interval
        await asyncio.sleep(interval)


def start_health_probe(
    models: list[Any],
    interval: float = 2.0,
) -> None:
    """Start the periodic health probe background task.

    Must be called from within a running event loop (e.g. after app creation).
    If called outside an event loop, defers to on-first-request via _models.

    Args:
        models: List of ModelConfig objects to probe.
        interval: Seconds between probes (default 2.0).
    """
    global _health_task
    if _health_task is not None and not _health_task.done():
        return  # already running
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop — defer startup; the probe will start lazily on first health request
        global _deferred_models
        _deferred_models = models
        logger.info("Health probe deferred until event loop starts (interval=%ds)", interval)
        return
    _health_task = asyncio.create_task(_probe_loop(models, interval))
    logger.info("Health probe started (interval=%ds)")


def stop_health_probe() -> None:
    """Cancel the periodic health probe task."""
    global _health_task
    if _health_task is not None and not _health_task.done():
        _health_task.cancel()
        try:
            _health_task.result(timeout=2.0)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # task may have already finished
        _health_task = None
        logger.info("Health probe stopped")


def restart_health_probe(
    models: list[Any],
    interval: float = 2.0,
) -> None:
    """Stop the current probe and start a new one with updated models.

    Used after config reload (SIGHUP) to refresh the model list being probed.

    The old task is cancelled asynchronously — we do NOT block waiting for
    it to finish.  ``start_health_probe`` immediately creates a fresh task
    with the new model list, so there is zero gap in probing.
    """
    global _health_task
    old = _health_task
    if old is not None and not old.done():
        old.cancel()
        # Do NOT await old.result() here — SIGHUP handlers run on the main
        # thread and blocking would stall signal processing.  The cancelled
        # task will exit on its next sleep().
    # Mark as done so start_health_probe's guard doesn't return early,
    # but don't set to None (which would allow a double-call race).
    _health_task = old


def _current_report() -> HealthReport:
    """Return the latest health report snapshot."""
    if _current_report_value is not None:
        return _current_report_value
    return HealthReport()
