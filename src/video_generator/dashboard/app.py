from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .jobs import RunSupervisor
from .views import (
    list_runs,
    resolve_artifact_path,
    resolve_run_root,
    run_detail,
    run_summary,
)
from .. import __version__
from ..config import load_environment, load_raw_config, resolve_config
from ..contracts import (
    ContentFormat,
    ContentMode,
    CreativeBrief,
    NarrationPace,
    OutputLanguage,
    PUBLIC_STAGES,
    Quality,
    TASK_IDS,
    TASK_PROTOCOL,
    VisualShotMode,
)
from ..errors import CheckpointError, ConfigurationError, VideoGeneratorError
from ..preflight import run_preflight
from ..profiles import BACKEND_DESCRIPTORS, PROFILES
from ..prompting import build_frozen_assets
from ..provenance import build_runtime_snapshot
from ..run_store import RunStore


TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "Research": ("search", "research"),
    "Story": (
        "ideate",
        "select",
        "outline",
        "script_draft",
        "review_story",
        "review_spoken",
        "review_constraints",
        "script_revision",
        "claim_inventory",
        "factual_review",
        "duration_repair",
    ),
    "Voice": ("narration_synthesis", "caption_alignment"),
    "Visuals": ("visual_plan", "image_prompt_compile", "image_generate", "visual_review"),
    "Music": ("music_brief", "music_generate"),
}

SAFE_INLINE_ARTIFACT_SUFFIXES = {
    ".aac",
    ".ass",
    ".avif",
    ".csv",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".log",
    ".m4a",
    ".md",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".srt",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
}


class DashboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunOptions(DashboardModel):
    profile: Literal[
        "local",
        "cloud-openai",
        "cloud-gemini",
        "cloud-openai-gemini",
        "hybrid-local-first",
    ] = "local"
    output_language: OutputLanguage = OutputLanguage.FINNISH
    duration_seconds: Annotated[FiniteFloat, Field(ge=10, le=3600)] = 90
    quality: Quality = Quality.DRAFT
    content_mode: ContentMode = ContentMode.FICTION
    content_format: ContentFormat = ContentFormat.NARRATIVE
    narration_pace: NarrationPace = NarrationPace.STANDARD
    narration_delivery: Annotated[str, Field(max_length=500)] = ""
    style: Annotated[str, Field(min_length=1, max_length=120)] = "ms_paint_stick"
    style_description: Annotated[str, Field(max_length=1000)] = ""
    offline: bool = False
    cost_ceiling_usd: Annotated[FiniteFloat, Field(ge=0, le=10000)] = 10
    idea_candidates: Annotated[int, Field(ge=1, le=10)] = 5
    research_query_limit: Annotated[int, Field(ge=0, le=20)] = 5
    research_source_limit: Annotated[int, Field(ge=0, le=50)] = 10
    visual_target_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 15
    visual_min_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 8
    visual_max_seconds: Annotated[FiniteFloat, Field(gt=0, le=180)] = 25
    visual_shot_mode: VisualShotMode = VisualShotMode.SCENE_LOCKED
    shot_target_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 3
    shot_min_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 2
    shot_max_seconds: Annotated[FiniteFloat, Field(gt=0, le=180)] = 5
    music_enabled: bool = False
    captions_enabled: bool = True
    animated_captions: bool = False
    task_overrides: dict[str, str] = Field(default_factory=dict)


class RunRequest(DashboardModel):
    brief: CreativeBrief
    options: RunOptions


def _sanitized_defaults(project_root: Path) -> dict[str, Any]:
    raw = load_raw_config(project_root / "config.toml")
    values = raw.model_dump(mode="json", exclude={"voice", "audience", "usage_purpose", "failure_policy"})
    values.pop("schema_version", None)
    return values


def _backend_catalog(environment: dict[str, str]) -> dict[str, Any]:
    result = {}
    for backend_id, descriptor in sorted(BACKEND_DESCRIPTORS.items()):
        if descriptor.provider == "deterministic":
            continue
        result[backend_id] = {
            "backend_id": backend_id,
            "provider": descriptor.provider,
            "model_id": descriptor.model_id,
            "revision": descriptor.revision,
            "protocols": sorted(item.value for item in descriptor.protocols),
            "cloud": descriptor.cloud,
            "runner": descriptor.runner,
            "supports_vision": descriptor.supports_vision,
            "supports_reference_images": descriptor.supports_reference_images,
            "reservation_usd": descriptor.reservation_usd,
            "configured": all(environment.get(name) for name in descriptor.required_env),
            "notes": descriptor.notes,
        }
    return result


def _task_catalog(backends: dict[str, Any]) -> list[dict[str, Any]]:
    group_by_task = {
        task_id: group for group, task_ids in TASK_GROUPS.items() for task_id in task_ids
    }
    return [
        {
            "task_id": task_id,
            "group": group_by_task.get(task_id, "Other"),
            "protocol": TASK_PROTOCOL[task_id].value,
            "backend_options": [
                backend_id
                for backend_id, descriptor in backends.items()
                if TASK_PROTOCOL[task_id].value in descriptor["protocols"]
            ],
        }
        for task_id in TASK_IDS
    ]


def _resolve_request(project_root: Path, payload: RunRequest):
    config_path = project_root / "config.toml"
    environment = load_environment(config_path)
    config = resolve_config(
        config_path,
        overrides=payload.options.model_dump(mode="json"),
        environment=environment,
    )
    return config, environment


def create_dashboard_app(
    project_root: Path,
    *,
    supervisor_factory: Callable[[Path], RunSupervisor] = RunSupervisor,
) -> FastAPI:
    project_root = project_root.resolve()
    dashboard_token = secrets.token_urlsafe(32)
    static_root = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.supervisor = supervisor_factory(project_root)
        try:
            yield
        finally:
            app.state.supervisor.close()

    app = FastAPI(
        title="Video Generator Dashboard",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost"],
        www_redirect=False,
    )
    app.state.project_root = project_root
    app.state.dashboard_token = dashboard_token

    def supervisor(request: Request) -> RunSupervisor:
        return request.app.state.supervisor

    def api_run_root(run_id: str) -> Path:
        try:
            return resolve_run_root(project_root, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_run_root(run_id: str, request: Request) -> Path:
        run_root = api_run_root(run_id)
        try:
            run_summary(project_root, run_root, supervisor(request))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Run manifest is invalid.") from exc
        return run_root

    async def require_dashboard_token(
        request: Request,
        x_dashboard_token: Annotated[str | None, Header()] = None,
    ) -> None:
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(status_code=415, detail="Mutations require application/json.")
        if not secrets.compare_digest(x_dashboard_token or "", dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard token.")
        origin = request.headers.get("origin")
        if origin:
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin.rstrip("/") != expected.rstrip("/"):
                raise HTTPException(status_code=403, detail="Cross-origin mutations are not allowed.")

    mutation_guard = Depends(require_dashboard_token)

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request,
        error: ConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "kind": error.kind.value,
                    "message": error.message,
                    "action": error.action,
                }
            },
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; "
            "script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html", media_type="text/html")

    @app.get("/api/bootstrap")
    def bootstrap(request: Request) -> dict[str, Any]:
        environment = load_environment(project_root / "config.toml")
        backends = _backend_catalog(environment)
        defaults = _sanitized_defaults(project_root)
        profiles = {
            name: bindings
            for name, bindings in PROFILES.items()
            if name != "deterministic-test"
        }
        default_task_bindings = dict(profiles.get(str(defaults["profile"]), {}))
        default_task_bindings.update(defaults.get("task_overrides") or {})
        return {
            "version": __version__,
            "dashboard_token": dashboard_token,
            "defaults": defaults,
            "default_task_bindings": default_task_bindings,
            "profiles": profiles,
            "backends": backends,
            "tasks": _task_catalog(backends),
            "task_groups": TASK_GROUPS,
            "stages": PUBLIC_STAGES,
            "runs": list_runs(project_root, supervisor(request)),
        }

    @app.get("/api/runs")
    def runs(request: Request) -> list[dict[str, Any]]:
        return list_runs(project_root, supervisor(request))

    @app.post("/api/preflight", dependencies=[mutation_guard])
    def preflight(payload: RunRequest) -> dict[str, Any]:
        config, environment = _resolve_request(project_root, payload)
        report = run_preflight(config=config, environment=environment, live=False)
        return report.model_dump(mode="json")

    @app.post("/api/runs", status_code=201, dependencies=[mutation_guard])
    def create_run(payload: RunRequest, request: Request) -> dict[str, Any]:
        config, environment = _resolve_request(project_root, payload)
        report = run_preflight(config=config, environment=environment, live=False)
        if not report.ready:
            raise HTTPException(status_code=422, detail=report.model_dump(mode="json"))
        assets = build_frozen_assets(config)
        assets["runtime_snapshot"] = build_runtime_snapshot(config)
        assets["creation_preflight"] = report.model_dump(mode="json")
        store = RunStore.create(
            project_root=project_root,
            config=config,
            brief=payload.brief,
            frozen_assets=assets,
        )
        job = supervisor(request).enqueue(store.manifest.run_id)
        return {
            "run_id": store.manifest.run_id,
            "job": job,
            "summary": run_summary(project_root, store.root, supervisor(request)),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> dict[str, Any]:
        run_root = api_run_root(run_id)
        try:
            return run_detail(project_root, run_root, supervisor(request))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Run manifest is invalid.") from exc

    @app.post("/api/runs/{run_id}/resume", dependencies=[mutation_guard])
    def resume_run(run_id: str, request: Request) -> dict[str, Any]:
        run_root = api_run_root(run_id)
        try:
            detail = run_detail(project_root, run_root, supervisor(request))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Run manifest is invalid.") from exc
        effective_status = detail["summary"]["status"]
        if effective_status == "complete":
            raise HTTPException(status_code=409, detail="A completed Run does not need resuming.")
        if effective_status == "running_external":
            raise HTTPException(
                status_code=409,
                detail="This Run is already executing outside this dashboard.",
            )
        return {"run_id": run_id, "job": supervisor(request).enqueue(run_id)}

    @app.post("/api/runs/{run_id}/stop", dependencies=[mutation_guard])
    def stop_run(run_id: str, request: Request) -> dict[str, Any]:
        run_root = api_run_root(run_id)
        job = supervisor(request).stop(run_id)
        if job is None:
            raise HTTPException(status_code=409, detail="This dashboard is not supervising that Run.")
        if job["status"] == "stopped":
            lock_acquired = False
            try:
                store = RunStore.open(run_root)
                with store.execution_lock():
                    lock_acquired = True
                    store = RunStore.open(run_root)
                    if store.manifest.status in {"created", "running"}:
                        store.set_status("stopped")
            except CheckpointError:
                if not lock_acquired:
                    return {
                        "run_id": run_id,
                        "job": job,
                        "warning": (
                            "The dashboard queue stopped, but another executor owns this Run. "
                            "Its status was left unchanged."
                        ),
                    }
                return {
                    "run_id": run_id,
                    "job": job,
                    "warning": (
                        "The dashboard queue stopped, but the Run status could not be saved. "
                        "You can resume this Run manually."
                    ),
                }
            except (OSError, ValueError, VideoGeneratorError):
                return {
                    "run_id": run_id,
                    "job": job,
                    "warning": (
                        "The dashboard queue stopped, but the Run status could not be saved. "
                        "You can resume this Run manually."
                    ),
                }
        return {"run_id": run_id, "job": job, "warning": None}

    @app.get("/api/runs/{run_id}/events", response_class=EventSourceResponse)
    async def run_events(
        run_id: str,
        request: Request,
        run_root: Path = Depends(event_run_root),
    ) -> AsyncIterable[ServerSentEvent]:
        prior = ""
        sequence = 0
        while not await request.is_disconnected():
            snapshot = run_summary(project_root, run_root, supervisor(request))
            encoded = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if encoded != prior:
                sequence += 1
                prior = encoded
                yield ServerSentEvent(
                    data=encoded,
                    event="run",
                    id=str(sequence),
                    retry=1000,
                )
            await asyncio.sleep(0.75)

    @app.get("/api/runs/{run_id}/files/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str) -> FileResponse:
        run_root = api_run_root(run_id)
        try:
            path = resolve_artifact_path(run_root, artifact_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        media_type = None
        safe_inline = path.suffix.lower() in SAFE_INLINE_ARTIFACT_SUFFIXES
        response = FileResponse(
            path,
            media_type=media_type,
            filename=None if safe_inline else path.name,
            content_disposition_type="inline" if safe_inline else "attachment",
        )
        return response

    return app


def run_dashboard(project_root: Path, *, port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(
        create_dashboard_app(project_root),
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )
