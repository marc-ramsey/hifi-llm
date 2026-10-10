"""Process manager — tracks and restarts managed server subprocesses."""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_KILL_TIMEOUT_S = 20.0
_MAX_RESTARTS = 5          # max consecutive crashes before giving up
_RESTART_WINDOW_S = 60     # time window for counting restarts


@dataclass
class _ProcessEntry:
    proc: subprocess.Popen[bytes]
    last_restart: float = field(default_factory=time.monotonic)
    restart_times: list[float] = field(default_factory=list)


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
            # Preserve restart history if the key already exists (e.g. after
            # a crash — start() is called by ensure_running on the same name).
            existing = cls._processes.get(name)
            if existing is not None:
                existing.proc = proc
                existing.last_restart = time.monotonic()
            else:
                cls._processes[name] = _ProcessEntry(proc=proc)
            logger.info("Started managed process '%s': %s", name, " ".join(cmd))
            return True
        except Exception as e:
            logger.error("Failed to start managed process '%s': %s", name, e)
            return False

    @classmethod
    def _prune_restart_times(cls, entry: _ProcessEntry) -> None:
        """Remove restart timestamps older than the window."""
        cutoff = time.monotonic() - _RESTART_WINDOW_S
        entry.restart_times = [t for t in entry.restart_times if t > cutoff]

    @classmethod
    def _exceeded_restart_limit(cls, entry: _ProcessEntry) -> bool:
        """Return True if the process has crashed too many times recently."""
        cls._prune_restart_times(entry)
        return len(entry.restart_times) >= _MAX_RESTARTS

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

        If the process has crashed more than _MAX_RESTARTS times within
        _RESTART_WINDOW_S seconds, stop restarting entirely (returns False)
        to prevent a self-DoS loop.  The counter resets when the process
        stays alive for the full window duration.

        Args:
            name: Model name used as the process key.
            cmd: Command list to launch the server.
            cooldown: Seconds to wait before retrying after a failure.

        Returns:
            True if the process is running (or just started), False if
            still in cooldown or max restarts exceeded.
        """
        entry = cls._processes.get(name)
        if entry is not None and entry.proc.poll() is None:
            return True  # already running

        now = time.monotonic()

        # Check cooldown first.
        if entry is not None and (now - entry.last_restart) < (cooldown or cls.COOLDOWN_SECONDS):
            logger.debug(
                "Managed process '%s' in cooldown (%.1fs/<%.1fs)",
                name, now - entry.last_restart, cooldown or cls.COOLDOWN_SECONDS,
            )
            return False

        # Check restart limit — prevents self-DoS from a perpetually crashing server.
        if entry is not None and cls._exceeded_restart_limit(entry):
            logger.error(
                "Managed process '%s' exceeded %d restarts in %.0fs — giving up",
                name, _MAX_RESTARTS, _RESTART_WINDOW_S,
            )
            return False

        success = cls.start(name, cmd)
        if success:
            entry = cls._processes[name]
            entry.last_restart = now
            entry.restart_times.append(now)
        return success

    @classmethod
    async def shutdown(cls) -> None:
        """Terminate all managed processes with SIGTERM→SIGKILL escalation.

        No blocking calls — everything runs via asyncio.create_task so the
        caller returns immediately.  Each process is terminated in parallel;
        after *kill_timeout* seconds any still-alive processes are SIGKILLed.
        """
        entries = list(cls._processes.items())
        cls._processes.clear()

        if not entries:
            return

        async def _kill_one(name: str, proc: subprocess.Popen[bytes]) -> None:
            if proc.poll() is not None:
                return  # already dead
            logger.info("Stopping managed process '%s' at shutdown", name)
            proc.send_signal(signal.SIGTERM)
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, proc.communicate),
                    timeout=_KILL_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                logger.warning(
                    "Managed process '%s' did not exit after %ds — SIGKILLing",
                    name, _KILL_TIMEOUT_S,
                )
                proc.kill()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, proc.communicate)

        tasks = [_kill_one(name, entry.proc) for name, entry in entries]
        await asyncio.gather(*tasks)
