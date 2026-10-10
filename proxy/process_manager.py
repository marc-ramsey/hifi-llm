"""Process manager — tracks and restarts managed server subprocesses."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _ProcessEntry:
    proc: subprocess.Popen[bytes]
    last_restart: float = field(default_factory=time.monotonic)


class ProcessManager:
    """Manages lifecycle of managed server subprocesses.

    Tracks processes by model name, provides lazy start with restart
    backoff (5-second cooldown to prevent thrashing), and clean shutdown.

    Singleton — use the class methods; do not instantiate directly.
    """

    _processes: dict[str, _ProcessEntry] = {}
    COOLDOWN_SECONDS = 5.0

    @classmethod
    def is_running(cls, name: str) -> bool:
        """Check whether a managed process is currently alive."""
        entry = cls._processes.get(name)
        if entry is None:
            return False
        if entry.proc.poll() is not None:
            return False
        return True

    @classmethod
    def start(cls, name: str, cmd: list[str]) -> bool:
        """Start a managed process. Returns True on success."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            cls._processes[name] = _ProcessEntry(proc=proc)
            logger.info("Started managed process '%s': %s", name, " ".join(cmd))
            return True
        except Exception as e:
            logger.error("Failed to start managed process '%s': %s", name, e)
            return False

    @classmethod
    def stop(cls, name: str) -> None:
        """Stop a managed process if running."""
        entry = cls._processes.pop(name, None)
        if entry is not None and entry.proc.poll() is None:
            entry.proc.terminate()
            logger.info("Stopped managed process '%s'", name)

    @classmethod
    def ensure_running(
        cls,
        name: str,
        cmd: list[str],
        cooldown: float | None = None,
    ) -> bool:
        """Start the process if not already running, with restart backoff.

        If the process was restarted within *cooldown* seconds, skip and
        return False — avoids thrashing on a persistently failing server.

        Args:
            name: Model name used as the process key.
            cmd: Command list to launch the server.
            cooldown: Seconds to wait before retrying after a failure.

        Returns:
            True if the process is running (or just started), False if
            still in cooldown from a recent restart attempt.
        """
        entry = cls._processes.get(name)
        if entry is not None and entry.proc.poll() is None:
            return True  # already running

        now = time.monotonic()
        if entry is not None and (now - entry.last_restart) < (cooldown or cls.COOLDOWN_SECONDS):
            logger.debug(
                "Managed process '%s' in cooldown (%.1fs/<%.1fs)",
                name, now - entry.last_restart, cooldown or cls.COOLDOWN_SECONDS,
            )
            return False

        success = cls.start(name, cmd)
        if success:
            cls._processes[name].last_restart = now
        return success

    @classmethod
    def shutdown(cls) -> None:
        """Terminate all managed processes."""
        for name, entry in list(cls._processes.items()):
            if entry.proc.poll() is None:
                entry.proc.terminate()
                logger.info("Stopped managed process '%s' at shutdown", name)
        cls._processes.clear()
