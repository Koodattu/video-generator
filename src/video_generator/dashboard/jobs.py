from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

from ..errors import CheckpointError, ErrorKind, VideoGeneratorError
from ..run_store import RunStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    run_id: str
    status: str = "queued"
    queued_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    log_handle: IO[str] | None = field(default=None, repr=False)
    stop_watchdog: threading.Thread | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "return_code": self.return_code,
            "error": self.error,
            "pid": self.process.pid if self.process and self.process.poll() is None else None,
        }


class RunSupervisor:
    """Serialize Run worker processes so one dashboard cannot oversubscribe the GPU or budget."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._jobs: dict[str, Job] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="video-generator-dashboard-worker",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, run_id: str) -> dict[str, Any]:
        with self._condition:
            existing = self._jobs.get(run_id)
            if existing and existing.status in {"queued", "running", "stopping"}:
                return existing.public()
            job = Job(run_id=run_id)
            self._jobs[run_id] = job
            self._queue.append(run_id)
            self._condition.notify()
            return job.public()

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(run_id)
            return job.public() if job else None

    def all_snapshots(self) -> dict[str, dict[str, Any]]:
        with self._condition:
            return {run_id: job.public() for run_id, job in self._jobs.items()}

    def stop(self, run_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            if job.status == "queued":
                self._queue = deque(item for item in self._queue if item != run_id)
                job.status = "stopped"
                job.completed_at = _now()
                return job.public()
            retry_failed_stop = (
                job.status == "stopping"
                and job.error is not None
                and job.process is not None
                and job.process.poll() is None
            )
            if job.status == "stopping" and not retry_failed_stop:
                return job.public()
            if retry_failed_stop:
                job.error = None
                job.stop_watchdog = None
                process = job.process
            else:
                if job.status != "running":
                    return job.public()
                job.status = "stopping"
                process = job.process

            # The worker publishes "running" just before it starts Popen. Preserve a
            # stop requested in that narrow window; the worker will signal the process
            # as soon as it exists.
            if process is None:
                return job.public()

        watchdog = self._signal_stop(run_id, process)
        if watchdog is not None:
            with self._condition:
                if job.process is process and job.status == "stopping":
                    job.stop_watchdog = watchdog
        elif process.poll() is None:
            with self._condition:
                if job.process is process and job.status == "stopping":
                    job.error = "Dashboard could not terminate the worker process tree."
        return self.snapshot(run_id)

    def close(self) -> None:
        queued: list[Job] = []
        active: list[tuple[str, subprocess.Popen[str]]] = []
        watchdogs: list[threading.Thread] = []
        with self._condition:
            if self._closed:
                return
            self._closed = True
            while self._queue:
                job = self._jobs[self._queue.popleft()]
                job.status = "stopped"
                job.completed_at = _now()
                queued.append(job)
            for run_id, job in self._jobs.items():
                if job.status == "running":
                    job.status = "stopping"
                elif job.status != "stopping":
                    continue

                watchdog = job.stop_watchdog
                if watchdog is not None and watchdog.is_alive():
                    watchdogs.append(watchdog)
                elif job.process is not None and job.process.poll() is None:
                    job.error = None
                    job.stop_watchdog = None
                    active.append((run_id, job.process))
            self._condition.notify_all()

        for job in queued:
            final_status, reconcile_error = self._reconcile_manifest(
                self.project_root / "runs" / job.run_id,
                "stopped",
                return_code=0,
                error=None,
            )
            with self._condition:
                job.status = final_status
                job.error = reconcile_error
        for run_id, process in active:
            watchdog = self._signal_stop(run_id, process)
            if watchdog is not None:
                with self._condition:
                    job = self._jobs.get(run_id)
                    if job is not None and job.process is process:
                        job.stop_watchdog = watchdog
                watchdogs.append(watchdog)
            elif process.poll() is None:
                with self._condition:
                    self._jobs[run_id].error = (
                        "Dashboard could not terminate the worker process tree."
                    )
        for watchdog in watchdogs:
            watchdog.join(timeout=30)

        # A watchdog or synchronous tree kill can fail (for example because of an
        # OS permission race). Make one final forced attempt during shutdown and
        # retain an actionable error if the process still survives it.
        with self._condition:
            unresolved = [
                (run_id, job.process)
                for run_id, job in self._jobs.items()
                if job.status == "stopping"
                and job.process is not None
                and job.process.poll() is None
            ]
        for run_id, process in unresolved:
            self._terminate_process_tree(process)
            if process.poll() is None:
                with self._condition:
                    job = self._jobs.get(run_id)
                    if job is not None and job.process is process:
                        job.error = "Dashboard could not terminate the worker process tree."
        self._thread.join(timeout=30)

        # Cover the narrow window where the worker was marked running before Popen
        # published its process handle.
        with self._condition:
            late_processes = [
                (run_id, job.process)
                for run_id, job in self._jobs.items()
                if job.status == "stopping"
                and job.process is not None
                and job.process.poll() is None
            ]
        for run_id, process in late_processes:
            self._terminate_process_tree(process)
            if process.poll() is None:
                with self._condition:
                    job = self._jobs.get(run_id)
                    if job is not None and job.process is process:
                        job.error = "Dashboard could not terminate the worker process tree."
        if late_processes:
            self._thread.join(timeout=30)

    @classmethod
    def _force_stop_after_grace(cls, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        # The parent can exit while a child ignores the graceful signal. Always target the
        # process group after the grace period instead of treating parent exit as completion.
        cls._terminate_process_tree(process)

    def _force_stop_and_record(
        self,
        run_id: str,
        process: subprocess.Popen[str],
    ) -> None:
        self._force_stop_after_grace(process)
        if process.poll() is not None:
            return
        with self._condition:
            job = self._jobs.get(run_id)
            if job is not None and job.process is process:
                job.error = "Dashboard could not terminate the worker process tree."

    @staticmethod
    def _terminate_process_tree(
        process: subprocess.Popen[str],
        *,
        platform: str | None = None,
    ) -> None:
        platform = platform or os.name
        if platform == "nt":
            def fallback_group_signal() -> None:
                try:
                    process.send_signal(
                        getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)
                    )
                except (OSError, ValueError, AttributeError):
                    pass
                if process.poll() is None:
                    try:
                        process.kill()
                    except (OSError, ValueError, AttributeError):
                        pass

            try:
                completed = subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                if completed.returncode and process.poll() is None:
                    fallback_group_signal()
            except (OSError, subprocess.TimeoutExpired):
                fallback_group_signal()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                if process.poll() is None:
                    try:
                        process.kill()
                    except (OSError, ValueError, AttributeError):
                        pass
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (OSError, ValueError, AttributeError):
                return
            try:
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                return

    def _signal_stop(
        self,
        run_id: str,
        process: subprocess.Popen[str],
    ) -> threading.Thread | None:
        if os.name == "nt":
            # taskkill snapshots and terminates the full tree while the parent still exists.
            self._terminate_process_tree(process)
            return None
        try:
            os.killpg(process.pid, signal.SIGINT)
        except OSError:
            self._terminate_process_tree(process)
            return None
        watchdog = threading.Thread(
            target=self._force_stop_and_record,
            args=(run_id, process),
            name=f"stop-{run_id}",
            daemon=False,
        )
        watchdog.start()
        return watchdog

    @staticmethod
    def _reconcile_manifest(
        run_root: Path,
        desired_status: str,
        *,
        return_code: int,
        error: str | None,
    ) -> tuple[str, str | None]:
        try:
            store = RunStore.open(run_root)
        except (OSError, ValueError, VideoGeneratorError) as exc:
            return (
                desired_status,
                f"Dashboard could not open the Run manifest for reconciliation: {exc}",
            )
        lock_acquired = False
        try:
            with store.execution_lock():
                lock_acquired = True
                store = RunStore.open(run_root)
                if store.manifest.status in {"complete", "failed", "stopped"}:
                    return store.manifest.status, None
                if desired_status == "stopped":
                    store.set_status("stopped")
                    return "stopped", None
                message = error or (
                    f"dashboard worker exited with code {return_code} without writing a terminal Run status"
                )
                store.set_status(
                    "failed",
                    VideoGeneratorError(message, kind=ErrorKind.INTERNAL),
                )
                return "failed", None
        except CheckpointError as exc:
            if not lock_acquired:
                return (
                    "running_external",
                    "Run status was left unchanged because another executor owns it.",
                )
            return (
                desired_status,
                f"Dashboard could not reconcile the Run manifest: {exc}",
            )
        except Exception as exc:
            return (
                desired_status,
                f"Dashboard could not reconcile the Run manifest: {exc}",
            )

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                run_id = self._queue.popleft()
                job = self._jobs[run_id]
                job.status = "running"
                job.started_at = _now()

            run_root = self.project_root / "runs" / run_id
            log_path = run_root / "logs" / "dashboard-worker.log"
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:
                error = f"Dashboard could not open the worker log: {exc}"
                final_status, reconcile_error = self._reconcile_manifest(
                    run_root,
                    "failed",
                    return_code=1,
                    error=error,
                )
                with self._condition:
                    job.return_code = 1
                    job.completed_at = _now()
                    job.error = error or reconcile_error
                    job.status = final_status
                continue
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            )
            error = None
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "video_generator", "resume", run_id],
                    cwd=self.project_root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creation_flags,
                    start_new_session=os.name != "nt",
                )
                with self._condition:
                    job.process = process
                    job.log_handle = log_handle
                    stop_requested = job.status == "stopping"
                if stop_requested:
                    watchdog = self._signal_stop(run_id, process)
                    if watchdog is not None:
                        with self._condition:
                            job.stop_watchdog = watchdog
                    elif process.poll() is None:
                        with self._condition:
                            job.error = (
                                "Dashboard could not terminate the worker process tree."
                            )
                return_code = process.wait()
            except OSError as exc:
                error = f"Dashboard could not start worker: {exc}"
                try:
                    log_handle.write(error + "\n")
                except OSError as log_exc:
                    error += f"; worker log write failed: {log_exc}"
                return_code = 1
            finally:
                try:
                    log_handle.close()
                except OSError as exc:
                    if error is None:
                        error = f"Dashboard could not close the worker log: {exc}"

            with self._condition:
                desired_status = (
                    "stopped"
                    if job.status == "stopping"
                    else "complete" if return_code == 0 else "failed"
                )
            final_status, reconcile_error = self._reconcile_manifest(
                run_root,
                desired_status,
                return_code=return_code,
                error=error,
            )
            with self._condition:
                job.return_code = return_code
                job.completed_at = _now()
                job.process = None
                job.log_handle = None
                job.stop_watchdog = None
                job.error = error or reconcile_error
                job.status = final_status
