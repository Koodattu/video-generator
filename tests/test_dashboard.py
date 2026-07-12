from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from video_generator.contracts import CreativeBrief, TASK_IDS
from video_generator.dashboard import create_dashboard_app
from video_generator.dashboard import views as dashboard_views
from video_generator.dashboard.jobs import Job, RunSupervisor
from video_generator.dashboard.views import list_runs, resolve_artifact_path, run_detail
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.run_store import RunStore


class FakeSupervisor:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.jobs: dict[str, dict[str, Any]] = {}

    def enqueue(self, run_id: str) -> dict[str, Any]:
        job = {"run_id": run_id, "status": "queued", "pid": None}
        self.jobs[run_id] = job
        return job

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        return self.jobs.get(run_id)

    def stop(self, run_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(run_id)
        if job:
            job = {**job, "status": "stopped"}
            self.jobs[run_id] = job
        return job

    def close(self) -> None:
        pass


@pytest.fixture
def dashboard_run(tmp_path: Path, resolved_config):
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "offline": True,
            "task_bindings": dict(PROFILES["deterministic-test"]),
        }
    )
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=CreativeBrief(idea_direction="A lantern crosses the snow"),
        frozen_assets=build_frozen_assets(config),
    )
    preview = store.root / "previews" / "identity.svg"
    preview.parent.mkdir(parents=True)
    preview.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    return store


def test_dashboard_lists_run_without_exposing_voice_configuration(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        data = bootstrap.json()
        assert "voice" not in data["defaults"]
        assert data["runs"][0]["run_id"] == dashboard_run.manifest.run_id
        assert {item["task_id"] for item in data["tasks"]} == set(TASK_IDS)
        assert {
            task_id
            for task_ids in data["task_groups"].values()
            for task_id in task_ids
        } == set(TASK_IDS)

        detail = client.get(f"/api/runs/{dashboard_run.manifest.run_id}")
        assert detail.status_code == 200
        assert "voice" not in detail.json()["config"]
        assert any(item["path"] == "previews/identity.svg" for item in detail.json()["files"])


def test_dashboard_initial_models_include_project_task_overrides(dashboard_run) -> None:
    (dashboard_run.root.parents[1] / "config.toml").write_text(
        (
            'profile = "local"\n'
            '[voice]\nname = "test-voice"\n'
            '[task_overrides]\nscript_draft = "openai:gpt-5.6-terra"\n'
        ),
        encoding="utf-8",
    )
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/bootstrap")

    assert response.status_code == 200
    bootstrap = response.json()
    assert bootstrap["default_task_bindings"]["script_draft"] == "openai:gpt-5.6-terra"
    assert bootstrap["default_task_bindings"]["image_generate"] == (
        PROFILES["local"]["image_generate"]
    )


def test_dashboard_preflight_and_create_start_a_new_run(
    dashboard_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (dashboard_run.root.parents[1] / "config.toml").write_text(
        'profile = "local"\n[voice]\nname = "test-voice"\n',
        encoding="utf-8",
    )
    report = SimpleNamespace(
        ready=True,
        model_dump=lambda *, mode: {
            "schema_version": 1,
            "ready": True,
            "profile": "local",
            "checks": [],
            "backend_reports": [],
        },
    )
    monkeypatch.setattr("video_generator.dashboard.app.run_preflight", lambda **kwargs: report)
    monkeypatch.setattr(
        "video_generator.dashboard.app.build_runtime_snapshot",
        lambda config: {"schema_version": 1, "snapshot_hash": "test"},
    )
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    payload = {
        "brief": {"idea_direction": "A fox follows a lantern through the snow"},
        "options": {
            "profile": "local",
            "offline": True,
            "research_query_limit": 0,
        },
    }

    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        headers = {"X-Dashboard-Token": token}
        preflight = client.post("/api/preflight", headers=headers, json=payload)
        created = client.post("/api/runs", headers=headers, json=payload)

    assert preflight.status_code == 200
    assert preflight.json()["ready"] is True
    assert created.status_code == 201
    run_id = created.json()["run_id"]
    assert created.json()["job"]["status"] == "queued"
    created_store = RunStore.open(dashboard_run.root.parents[1] / "runs" / run_id)
    assert created_store.manifest.status == "created"
    assert created_store.brief.idea_direction == payload["brief"]["idea_direction"]


def test_dashboard_mutations_require_token_and_resume_queues_run(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post(
            f"/api/runs/{dashboard_run.manifest.run_id}/resume", json={}
        ).status_code == 403
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        response = client.post(
            f"/api/runs/{dashboard_run.manifest.run_id}/resume",
            headers={"X-Dashboard-Token": token},
            json={},
        )
        assert response.status_code == 200
        assert response.json()["job"]["status"] == "queued"


def test_dashboard_does_not_queue_a_run_owned_by_another_executor(dashboard_run) -> None:
    dashboard_run.set_status("running")
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        with dashboard_run.execution_lock():
            response = client.post(
                f"/api/runs/{dashboard_run.manifest.run_id}/resume",
                headers={"X-Dashboard-Token": token},
                json={},
            )

        assert response.status_code == 409
        assert response.json()["detail"] == (
            "This Run is already executing outside this dashboard."
        )
        assert app.state.supervisor.jobs == {}


def test_dashboard_queued_stop_does_not_overwrite_an_external_run(dashboard_run) -> None:
    dashboard_run.set_status("running")
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        app.state.supervisor.enqueue(dashboard_run.manifest.run_id)
        with dashboard_run.execution_lock():
            response = client.post(
                f"/api/runs/{dashboard_run.manifest.run_id}/stop",
                headers={"X-Dashboard-Token": token},
                json={},
            )

    assert response.status_code == 200
    assert "another executor owns this Run" in response.json()["warning"]
    assert RunStore.open(dashboard_run.root).manifest.status == "running"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/runs/bad!/resume"),
        ("post", "/api/runs/bad!/stop"),
        ("get", "/api/runs/bad!/events"),
    ],
)
def test_dashboard_rejects_invalid_run_ids_before_dispatch(
    dashboard_run,
    method: str,
    path: str,
) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        response = client.request(
            method,
            path,
            headers={"X-Dashboard-Token": token},
            json={} if method == "post" else None,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid Run ID"


def test_dashboard_resume_reports_a_malformed_manifest(dashboard_run) -> None:
    dashboard_run.manifest_path.write_text("[]", encoding="utf-8")
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        response = client.post(
            f"/api/runs/{dashboard_run.manifest.run_id}/resume",
            headers={"X-Dashboard-Token": token},
            json={},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Run manifest is invalid."


def test_dashboard_events_reject_a_malformed_manifest_before_streaming(
    dashboard_run,
) -> None:
    dashboard_run.manifest_path.write_text("[]", encoding="utf-8")
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(f"/api/runs/{dashboard_run.manifest.run_id}/events")

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


def test_dashboard_stops_a_supervised_job_even_when_manifest_is_malformed(
    dashboard_run,
) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        app.state.supervisor.enqueue(dashboard_run.manifest.run_id)
        dashboard_run.manifest_path.write_text("[]", encoding="utf-8")
        response = client.post(
            f"/api/runs/{dashboard_run.manifest.run_id}/stop",
            headers={"X-Dashboard-Token": token},
            json={},
        )

    assert response.status_code == 200
    assert response.json()["job"]["status"] == "stopped"
    assert response.json()["warning"]


def test_dashboard_forces_svg_artifacts_to_download(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(
            f"/api/runs/{dashboard_run.manifest.run_id}/files/previews/identity.svg"
        )
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert response.headers["x-content-type-options"] == "nosniff"


def test_artifact_path_cannot_leave_run(dashboard_run) -> None:
    with pytest.raises(ValueError, match="leaves the Run Bundle"):
        resolve_artifact_path(dashboard_run.root, "../../pyproject.toml")


def test_stop_requested_before_worker_process_exists_is_preserved() -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor._condition = threading.Condition()
    supervisor._jobs = {"run-1": Job(run_id="run-1", status="running")}

    job = supervisor.stop("run-1")

    assert job is not None
    assert job["status"] == "stopping"


def test_repeated_stop_does_not_signal_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor._condition = threading.Condition()
    process = SimpleNamespace(pid=4321, poll=lambda: None)
    supervisor._jobs = {
        "run-1": Job(run_id="run-1", status="running", process=process)
    }
    calls = []
    watchdog = SimpleNamespace()
    monkeypatch.setattr(
        supervisor,
        "_signal_stop",
        lambda run_id, target: calls.append((run_id, target)) or watchdog,
    )

    first = supervisor.stop("run-1")
    second = supervisor.stop("run-1")

    assert first is not None and first["status"] == "stopping"
    assert second is not None and second["status"] == "stopping"
    assert calls == [("run-1", process)]


def test_failed_stop_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor._condition = threading.Condition()
    process = SimpleNamespace(pid=4321, poll=lambda: None)
    supervisor._jobs = {
        "run-1": Job(
            run_id="run-1",
            status="stopping",
            process=process,
            error="Dashboard could not terminate the worker process tree.",
        )
    }
    calls = []
    watchdog = SimpleNamespace()
    monkeypatch.setattr(
        supervisor,
        "_signal_stop",
        lambda run_id, target: calls.append((run_id, target)) or watchdog,
    )

    result = supervisor.stop("run-1")

    assert result is not None
    assert result["status"] == "stopping"
    assert result["error"] is None
    assert calls == [("run-1", process)]
    assert supervisor._jobs["run-1"].stop_watchdog is watchdog


def test_forced_windows_stop_targets_the_complete_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(*, timeout: int):
            return 1

        @staticmethod
        def kill():
            raise AssertionError("taskkill should handle the complete process tree")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    RunSupervisor._terminate_process_tree(FakeProcess(), platform="nt")

    assert calls == [["taskkill.exe", "/PID", "4321", "/T", "/F"]]


def test_failed_windows_tree_kill_signals_the_process_group_before_parent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = []

    class FakeProcess:
        pid = 4321
        terminated = False

        def poll(self):
            return 1 if self.terminated else None

        def send_signal(self, value):
            signals.append(value)

        def kill(self):
            self.terminated = True

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1),
    )
    process = FakeProcess()

    RunSupervisor._terminate_process_tree(process, platform="nt")

    assert signals == [getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)]
    assert process.terminated is True


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree fallback")
def test_failed_windows_stop_is_contained_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnkillableProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

        @staticmethod
        def send_signal(value):
            raise OSError("signal denied")

        @staticmethod
        def kill():
            raise OSError("kill denied")

        @staticmethod
        def wait(*, timeout: int):
            raise subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1),
    )
    supervisor = object.__new__(RunSupervisor)
    supervisor._condition = threading.Condition()
    supervisor._jobs = {
        "run-1": Job(
            run_id="run-1",
            status="running",
            process=UnkillableProcess(),
        )
    }

    result = supervisor.stop("run-1")

    assert result is not None
    assert result["status"] == "stopping"
    assert result["error"] == "Dashboard could not terminate the worker process tree."


def test_force_stop_checks_descendants_even_after_parent_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class ExitedParent:
        @staticmethod
        def wait(*, timeout: int):
            return 0

    monkeypatch.setattr(
        RunSupervisor,
        "_terminate_process_tree",
        lambda process: calls.append(process),
    )
    process = ExitedParent()

    RunSupervisor._force_stop_after_grace(process)

    assert calls == [process]


def test_failed_async_force_stop_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(poll=lambda: None)
    supervisor = object.__new__(RunSupervisor)
    supervisor._condition = threading.Condition()
    supervisor._jobs = {
        "run-1": Job(run_id="run-1", status="stopping", process=process)
    }
    monkeypatch.setattr(
        supervisor,
        "_force_stop_after_grace",
        lambda target: None,
    )

    supervisor._force_stop_and_record("run-1", process)

    assert supervisor._jobs["run-1"].error == (
        "Dashboard could not terminate the worker process tree."
    )


def test_dashboard_shutdown_marks_queued_runs_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor.project_root = tmp_path
    supervisor._condition = threading.Condition()
    supervisor._queue = deque(["run-1"])
    supervisor._jobs = {"run-1": Job(run_id="run-1", status="queued")}
    supervisor._closed = False
    joined = []
    supervisor._thread = SimpleNamespace(join=lambda *, timeout: joined.append(timeout))
    monkeypatch.setattr(
        supervisor,
        "_reconcile_manifest",
        lambda *args, **kwargs: ("stopped", None),
    )

    supervisor.close()

    assert supervisor._jobs["run-1"].status == "stopped"
    assert supervisor._jobs["run-1"].completed_at is not None
    assert list(supervisor._queue) == []
    assert joined == [30]


def test_dashboard_shutdown_joins_active_force_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor.project_root = tmp_path
    supervisor._condition = threading.Condition()
    supervisor._queue = deque()
    process = SimpleNamespace(alive=True)
    process.poll = lambda: None if process.alive else 0
    supervisor._jobs = {
        "run-1": Job(run_id="run-1", status="running", process=process)
    }
    supervisor._closed = False
    worker_joins = []
    watchdog_joins = []
    supervisor._thread = SimpleNamespace(join=lambda *, timeout: worker_joins.append(timeout))
    def join_watchdog(*, timeout: int) -> None:
        watchdog_joins.append(timeout)
        process.alive = False

    watchdog = SimpleNamespace(join=join_watchdog)
    monkeypatch.setattr(supervisor, "_signal_stop", lambda *args: watchdog)

    supervisor.close()

    assert supervisor._jobs["run-1"].status == "stopping"
    assert watchdog_joins == [30]
    assert worker_joins == [30]


def test_dashboard_shutdown_retries_a_failed_stopping_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor.project_root = tmp_path
    supervisor._condition = threading.Condition()
    supervisor._queue = deque()
    process = SimpleNamespace(alive=True)
    process.poll = lambda: None if process.alive else 0
    supervisor._jobs = {
        "run-1": Job(
            run_id="run-1",
            status="stopping",
            process=process,
            error="Dashboard could not terminate the worker process tree.",
        )
    }
    supervisor._closed = False
    supervisor._thread = SimpleNamespace(join=lambda *, timeout: None)
    calls = []

    def retry_stop(run_id, target):
        calls.append((run_id, target))
        process.alive = False
        return None

    monkeypatch.setattr(supervisor, "_signal_stop", retry_stop)

    supervisor.close()

    assert calls == [("run-1", process)]
    assert supervisor._jobs["run-1"].error is None


def test_dashboard_shutdown_retries_after_a_watchdog_cannot_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(RunSupervisor)
    supervisor.project_root = tmp_path
    supervisor._condition = threading.Condition()
    supervisor._queue = deque()
    process = SimpleNamespace(alive=True)
    process.poll = lambda: None if process.alive else 0
    watchdog_joins = []
    watchdog = SimpleNamespace(
        is_alive=lambda: True,
        join=lambda *, timeout: watchdog_joins.append(timeout),
    )
    supervisor._jobs = {
        "run-1": Job(
            run_id="run-1",
            status="stopping",
            process=process,
            stop_watchdog=watchdog,
        )
    }
    supervisor._closed = False
    supervisor._thread = SimpleNamespace(join=lambda *, timeout: None)
    forced = []

    def force_stop(target):
        forced.append(target)
        process.alive = False

    monkeypatch.setattr(supervisor, "_terminate_process_tree", force_stop)

    supervisor.close()

    assert watchdog_joins == [30]
    assert forced == [process]


def test_manifest_reconciliation_write_failure_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenStore:
        manifest = SimpleNamespace(status="running")

        @staticmethod
        def execution_lock():
            return nullcontext()

        @staticmethod
        def set_status(status: str, error=None) -> None:
            raise OSError("disk unavailable")

    monkeypatch.setattr(
        "video_generator.dashboard.jobs.RunStore.open",
        lambda root: BrokenStore(),
    )

    status, error = RunSupervisor._reconcile_manifest(
        tmp_path / "run-1",
        "failed",
        return_code=1,
        error=None,
    )

    assert status == "failed"
    assert error is not None and "disk unavailable" in error


def test_manifest_reconciliation_does_not_overwrite_an_external_run(
    dashboard_run,
) -> None:
    dashboard_run.set_status("running")
    result = []

    def reconcile() -> None:
        result.append(
            RunSupervisor._reconcile_manifest(
                dashboard_run.root,
                "failed",
                return_code=1,
                error="worker could not acquire the Run lock",
            )
        )

    with dashboard_run.execution_lock():
        thread = threading.Thread(target=reconcile)
        thread.start()
        thread.join(timeout=5)

    assert not thread.is_alive()
    status, error = result[0]
    assert status == "running_external"
    assert error is not None and "another executor owns it" in error
    assert RunStore.open(dashboard_run.root).manifest.status == "running"


def test_manifest_reconciliation_open_failure_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_generator.dashboard.jobs.RunStore.open",
        lambda root: (_ for _ in ()).throw(OSError("manifest unavailable")),
    )

    status, error = RunSupervisor._reconcile_manifest(
        tmp_path / "run-1",
        "stopped",
        return_code=0,
        error=None,
    )

    assert status == "stopped"
    assert error is not None and "manifest unavailable" in error


def test_dashboard_rejects_untrusted_hosts(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/bootstrap", headers={"Host": "evil.example"})

    assert response.status_code == 400


def test_dashboard_csp_does_not_depend_on_inline_styles(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        page = client.get("/")
        script = client.get("/static/app.js")

    assert "style-src 'self'" in page.headers["content-security-policy"]
    assert "'unsafe-inline'" not in page.headers["content-security-policy"]
    assert ".style." not in script.text


def test_invalid_model_combination_returns_actionable_422(dashboard_run) -> None:
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/bootstrap").json()["dashboard_token"]
        response = client.post(
            "/api/preflight",
            headers={"X-Dashboard-Token": token},
            json={
                "brief": {"idea_direction": "A lantern crosses the snow"},
                "options": {
                    "profile": "local",
                    "offline": True,
                    "task_overrides": {"script_draft": "openai:gpt-5.6-terra"},
                },
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["kind"] == "unsupported"
    assert "offline Run selects cloud Backends" in response.json()["detail"]["message"]


@pytest.mark.parametrize("suffix", [".xhtml", ".svgz", ".mhtml"])
def test_active_artifact_types_default_to_download(dashboard_run, suffix: str) -> None:
    path = dashboard_run.root / "previews" / f"unsafe{suffix}"
    path.write_text("active content", encoding="utf-8")
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(
            f"/api/runs/{dashboard_run.manifest.run_id}/files/previews/{path.name}"
        )

    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment;")


def test_fanout_item_hashes_are_reflected_in_artifact_inventory(dashboard_run) -> None:
    record_root = dashboard_run.root / "stages" / "140-images" / "item-records"
    record_root.mkdir(parents=True)
    (record_root / "scene-001.json").write_text(
        json.dumps({"output_hashes": {"previews/identity.svg": "abc"}}),
        encoding="utf-8",
    )
    app = create_dashboard_app(dashboard_run.root.parents[1], supervisor_factory=FakeSupervisor)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        detail = client.get(f"/api/runs/{dashboard_run.manifest.run_id}").json()

    identity = next(item for item in detail["files"] if item["path"] == "previews/identity.svg")
    manifest = next(item for item in detail["files"] if item["path"] == "manifest.json")
    assert identity["hash_recorded"] is True
    assert manifest["hash_recorded"] is False


def test_run_detail_uses_one_manifest_snapshot(
    dashboard_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = dashboard_views._read_optional
    manifest_reads = []

    def tracked_read(path: Path):
        value = original_read(path)
        if path.name == "manifest.json":
            manifest_reads.append(path)
        return value

    monkeypatch.setattr(dashboard_views, "_read_optional", tracked_read)
    supervisor = FakeSupervisor(dashboard_run.root.parents[1])

    detail = run_detail(
        dashboard_run.root.parents[1],
        dashboard_run.root,
        supervisor,
    )

    assert detail["summary"]["updated_at"] == detail["manifest"]["updated_at"]
    assert len(manifest_reads) == 1


def test_run_listing_uses_the_canonical_root_guard(
    dashboard_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = dashboard_run.root.parent / "escape"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")
    original_resolve = dashboard_views.resolve_run_root
    checked = []

    def guarded(project_root: Path, run_id: str) -> Path:
        checked.append(run_id)
        if run_id == "escape":
            raise FileNotFoundError(run_id)
        return original_resolve(project_root, run_id)

    monkeypatch.setattr(dashboard_views, "resolve_run_root", guarded)

    runs = list_runs(
        dashboard_run.root.parents[1],
        FakeSupervisor(dashboard_run.root.parents[1]),
    )

    assert "escape" in checked
    assert [item["run_id"] for item in runs] == [dashboard_run.manifest.run_id]
