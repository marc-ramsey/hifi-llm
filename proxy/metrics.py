"""Prometheus-style metrics endpoint — /metrics (text/plain).

Exposes counters and histograms derived from in-memory state.  Designed to
be scraped by Prometheus, Grafana, or any other monitoring tool.

Endpoints::

    GET /metrics   →  text/plain Prometheus exposition format

Metrics exposed::

    proxy_requests_total{path="/v1/chat/completions",status="200"} 142
    proxy_request_duration_seconds_bucket{path="/v1/chat/completions",le="0.5"} 80
    proxy_request_duration_seconds_count{path="/v1/chat/completions"} 142
    proxy_rate_limits_total{ip="10.0.0.1"} 3

"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Module-level singleton — survives across create_app() calls during SIGHUP
# reloads. Prometheus counters must be monotonically increasing; resetting
# them on reload would break dashboards and alerting.
_instance: MetricsStore | None = None


def get_metrics_store() -> MetricsStore:
    """Return the singleton MetricsStore, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = MetricsStore()
    return _instance


class MetricsStore:
    """In-memory counters for the /metrics endpoint.

    Thread-safe for CPython (GIL).  Not safe across processes — if you run
    multiple proxy workers, each will have its own counters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # Protects concurrent reads during scrapes
        # path + status -> count
        self._request_counts: dict[tuple[str, str], int] = defaultdict(int)
        # path -> ring buffer of recent durations (last N per path)
        self._max_durations = 10_000  # per-path ring buffer size
        self._durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self._max_durations))
        # ip -> rate-limit hit count
        self._rate_limit_counts: dict[str, int] = defaultdict(int)
        # (model, parameter) -> times the proxy supplied that parameter instead
        # of receiving it.  Per-parameter, not one total: "which knob is this
        # firing on, and how often" is the question actually asked at 2am.
        self._default_param_counts: dict[tuple[str, str], int] = defaultdict(int)
        # Track total requests for cleanup
        self._total_requests = 0

    def record_request(self, path: str, status: int, duration_ms: float) -> None:
        """Record a completed request."""
        status_str = str(status)
        self._request_counts[(path, status_str)] += 1
        self._durations[path].append(duration_ms / 1000.0)
        self._total_requests += 1

    def record_rate_limit(self, ip: str) -> None:
        """Record a rate-limit hit."""
        self._rate_limit_counts[ip] += 1

    def record_default_param(self, model: str, param: str) -> None:
        """Record that ``default_params`` supplied *param* for *model*."""
        self._default_param_counts[(model, param)] += 1

    def generate(self) -> str:
        """Return metrics in Prometheus exposition format."""
        lines: list[str] = []
        
        # Snapshot state under lock to avoid RuntimeError during iteration
        with self._lock:
            request_counts = dict(self._request_counts)
            durations = {k: list(v) for k, v in self._durations.items()}
            rate_limit_counts = dict(self._rate_limit_counts)
            default_param_counts = dict(self._default_param_counts)

        # -- proxy_requests_total (counter) --
        lines.append("# HELP proxy_requests_total Total HTTP requests by path and status.")
        lines.append("# TYPE proxy_requests_total counter")
        for (path, status), count in sorted(request_counts.items()):
            lines.append(f'proxy_requests_total{{path="{_esc(path)}",status="{status}"}} {count}')

        # -- proxy_request_duration_seconds (histogram) --
        lines.append("# HELP proxy_request_duration_seconds Request duration in seconds.")
        lines.append("# TYPE proxy_request_duration_seconds histogram")

        for path, dur_list in sorted(durations.items()):
            count = len(dur_list)
            total_sum = sum(dur_list)
            # Build buckets with standard le values
            buckets: dict[str, int] = defaultdict(int)
            for d in dur_list:
                # Prometheus histogram buckets are cumulative; use independent `if`
                # statements so each threshold counts all durations <= it.
                if d <= 0.05:
                    buckets["0.05"] += 1
                if d <= 0.1:
                    buckets["0.1"] += 1
                if d <= 0.25:
                    buckets["0.25"] += 1
                if d <= 0.5:
                    buckets["0.5"] += 1
                if d <= 1.0:
                    buckets["1.0"] += 1
                if d <= 2.5:
                    buckets["2.5"] += 1
                if d <= 5.0:
                    buckets["5.0"] += 1
                else:
                    buckets["+Inf"] += 1

            lines.append(
                f'proxy_request_duration_seconds_count{{path="{_esc(path)}"}} {count}'
            )
            lines.append(
                f'proxy_request_duration_seconds_sum{{path="{_esc(path)}"}} {_fmt(total_sum)}'
            )
            for le in ["0.05", "0.1", "0.25", "0.5", "1.0", "2.5", "5.0", "+Inf"]:
                lines.append(
                    f'proxy_request_duration_seconds_bucket{{path="{_esc(path)}",le="{le}"}} '
                    f'{buckets[le]}'
                )

        # -- proxy_rate_limits_total (counter) --
        lines.append("# HELP proxy_rate_limits_total Total rate-limit hits by IP.")
        lines.append("# TYPE proxy_rate_limits_total counter")
        for ip, count in sorted(rate_limit_counts.items()):
            lines.append(f'proxy_rate_limits_total{{ip="{_esc(ip)}"}} {count}')

        # -- proxy_default_params_total (counter) --
        lines.append(
            "# HELP proxy_default_params_total Times default_params supplied a "
            "parameter the request omitted, by model and parameter."
        )
        lines.append("# TYPE proxy_default_params_total counter")
        for (model, param), count in sorted(default_param_counts.items()):
            lines.append(
                f'proxy_default_params_total{{model="{_esc(model)}",'
                f'param="{_esc(param)}"}} {count}'
            )

        # -- proxy_requests_in_flight (gauge) — placeholder
        lines.append("# HELP proxy_requests_in_flight Current in-flight requests.")
        lines.append("# TYPE proxy_requests_in_flight gauge")
        lines.append("proxy_requests_in_flight 0")

        return "\n".join(lines) + "\n"


def _esc(s: str) -> str:
    """Escape special chars for Prometheus label values."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(v: float) -> str:
    """Format a number for Prometheus (avoid scientific notation)."""
    if v.is_integer():
        return f"{v:.1f}"
    return f"{v:.6g}"
