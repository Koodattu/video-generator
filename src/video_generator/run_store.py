from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Iterable, TypeVar

from pydantic import BaseModel

from .contracts import (
    CloudCallRecord,
    CreativeBrief,
    PUBLIC_STAGES,
    ResolvedRunConfig,
    RunManifest,
    StageRecord,
    UsageRecord,
    utc_now,
)
from .errors import CheckpointError, ErrorKind, VideoGeneratorError
from .util import (
    atomic_write_json,
    hash_value,
    read_json,
    relative_path,
    replace_path,
    sha256_file,
)


T = TypeVar("T", bound=BaseModel)


STAGE_NUMBER = {stage: (index + 1) * 10 for index, stage in enumerate(PUBLIC_STAGES)}


@dataclass(frozen=True)
class Workspace:
    stage: str
    attempt: int
    work_dir: Path
    final_dir: Path
    item_id: str | None = None


class RunExecutionLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None

    def __enter__(self) -> "RunExecutionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: IO[bytes] | None = None
        try:
            handle = self.path.open("a+b")
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if handle is not None:
                handle.close()
            raise CheckpointError(
                "this Run Bundle is already executing in another process",
                kind=ErrorKind.NOT_READY,
                action="Wait for the other generate/resume process to finish.",
            ) from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _dump(value: BaseModel | dict[str, Any] | list[Any]) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _rewrite_paths(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and (value == old_prefix or value.startswith(old_prefix + "/")):
        return new_prefix + value[len(old_prefix) :]
    return value


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.config_path = self.root / "inputs" / "config.resolved.json"
        self.brief_path = self.root / "inputs" / "brief.json"
        self.frozen_root = self.root / "inputs" / "frozen-assets"
        self.manifest = RunManifest.model_validate(read_json(self.manifest_path))
        expected_hashes = {
            "config": hash_value(read_json(self.config_path)),
            "brief": hash_value(read_json(self.brief_path)),
            "frozen assets": hash_value(read_json(self.frozen_root / "assets.json")),
        }
        recorded_hashes = {
            "config": self.manifest.config_hash,
            "brief": self.manifest.brief_hash,
            "frozen assets": self.manifest.frozen_assets_hash,
        }
        mismatches = [name for name in expected_hashes if expected_hashes[name] != recorded_hashes[name]]
        if mismatches:
            raise CheckpointError(
                f"Run Bundle frozen inputs were modified: {', '.join(mismatches)}",
                kind=ErrorKind.INVALID_OUTPUT,
                action="Restore the original Run Bundle or create an explicit rerun from the affected stage.",
            )
        ledger_total = sum(float(item.reserved_usd) for item in self.manifest.cost_reservations)
        if abs(ledger_total - float(self.manifest.reserved_cost_usd)) > 0.0001:
            raise CheckpointError(
                "Run cost reservation ledger does not match the recorded total",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        call_ids = [record.call_id for record in self.manifest.cloud_calls]
        if len(call_ids) != len(set(call_ids)):
            raise CheckpointError(
                "Run cloud-call ledger contains duplicate call IDs",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        if self.manifest.cloud_cost_ledger_version:
            direct_calls = {
                record.call_id: record
                for record in self.manifest.cloud_calls
                if not record.inherited and not record.legacy
            }
            reservations = {
                usage.call_id: usage
                for usage in self.manifest.cost_reservations
                if usage.call_id
            }
            if set(direct_calls) != set(reservations):
                raise CheckpointError(
                    "Run cloud-call ledger does not match direct cost reservations",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            if any(
                abs(float(record.reserved_usd) - float(reservations[call_id].reserved_usd))
                > 0.000001
                for call_id, record in direct_calls.items()
            ):
                raise CheckpointError(
                    "Run cloud-call reservation amounts do not match the reservation ledger",
                    kind=ErrorKind.INVALID_OUTPUT,
                )

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        config: ResolvedRunConfig,
        brief: CreativeBrief,
        frozen_assets: dict[str, Any],
        parent_run_id: str | None = None,
        fork_stage: str | None = None,
        run_id: str | None = None,
    ) -> "RunStore":
        run_id = run_id or _new_run_id()
        root = (project_root / "runs" / run_id).resolve()
        if root.exists():
            raise FileExistsError(root)
        for directory in (
            root / "inputs" / "frozen-assets",
            root / "stages",
            root / "outputs",
            root / "logs",
            root / "work",
        ):
            directory.mkdir(parents=True, exist_ok=False if directory == root / "inputs" / "frozen-assets" else True)
        config_data = config.model_dump(mode="json")
        brief_data = brief.model_dump(mode="json")
        atomic_write_json(root / "inputs" / "config.resolved.json", config_data)
        atomic_write_json(root / "inputs" / "brief.json", brief_data)
        atomic_write_json(root / "inputs" / "frozen-assets" / "assets.json", frozen_assets)
        manifest = RunManifest(
            run_id=run_id,
            parent_run_id=parent_run_id,
            fork_stage=fork_stage,
            config_hash=hash_value(config_data),
            brief_hash=hash_value(brief_data),
            frozen_assets_hash=hash_value(frozen_assets),
            cloud_cost_ledger_version=1,
        )
        atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> "RunStore":
        return cls(root)

    @property
    def config(self) -> ResolvedRunConfig:
        return ResolvedRunConfig.model_validate(read_json(self.config_path))

    @property
    def brief(self) -> CreativeBrief:
        return CreativeBrief.model_validate(read_json(self.brief_path))

    @property
    def frozen_assets(self) -> dict[str, Any]:
        return read_json(self.frozen_root / "assets.json")

    def _save_manifest(self) -> None:
        self.manifest.updated_at = utc_now()
        atomic_write_json(self.manifest_path, self.manifest.model_dump(mode="json"))

    def set_status(self, status: str, error: VideoGeneratorError | None = None) -> None:
        self.manifest.status = status  # type: ignore[assignment]
        if error:
            self.manifest.warnings.append(f"{error.kind.value}: {error.message}")
        self._save_manifest()

    def add_warning(self, warning: str) -> None:
        if warning not in self.manifest.warnings:
            self.manifest.warnings.append(warning)
            self._save_manifest()

    def execution_lock(self) -> RunExecutionLock:
        return RunExecutionLock(self.root.parent / ".locks" / f"{self.manifest.run_id}.lock")

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_NUMBER:
            raise ValueError(f"unknown public stage: {stage}")
        return self.root / "stages" / f"{STAGE_NUMBER[stage]:03d}-{stage}"

    def stage_record(self, stage: str) -> StageRecord | None:
        return self.manifest.stages.get(stage)

    def _validate_outputs(self, record: StageRecord) -> None:
        if not record.output_paths or set(record.output_paths) != set(record.output_hashes):
            raise CheckpointError(
                f"checkpoint {record.stage!r} has incomplete output hash coverage",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        for path_value, expected_hash in record.output_hashes.items():
            path = (self.root / path_value).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise CheckpointError(
                    f"checkpoint path escaped Run Bundle: {path_value}", kind=ErrorKind.INVALID_OUTPUT
                ) from exc
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise CheckpointError(
                    f"completed checkpoint is missing or corrupt: {path_value}",
                    kind=ErrorKind.INVALID_OUTPUT,
                    action=f"Use rerun {self.root} --from {record.stage}; resume will not repeat it silently.",
                )
        for item_id in record.item_ids:
            item_path = self.item_record_path(record.stage, item_id)
            item_relative = relative_path(item_path, self.root)
            if item_relative not in record.output_hashes:
                raise CheckpointError(
                    f"checkpoint {record.stage!r} does not cover item record {item_id!r}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            try:
                item_record = StageRecord.model_validate(read_json(item_path))
            except (FileNotFoundError, ValueError) as exc:
                raise CheckpointError(
                    f"checkpoint item record is missing or invalid: {record.stage}/{item_id}",
                    kind=ErrorKind.INVALID_OUTPUT,
                ) from exc
            self._validate_outputs(item_record)

    def validate_completed_outputs(self) -> None:
        for record in self.manifest.stages.values():
            if record.status == "complete":
                self._validate_outputs(record)

    def reusable_record(
        self,
        stage: str,
        *,
        input_hash: str,
        config_hash: str,
        backend_id: str,
        backend_revision: str,
        prompt_version: str,
        schema_hash: str,
    ) -> StageRecord | None:
        record = self.stage_record(stage)
        if record is None or record.status != "complete":
            return None
        expected = (
            input_hash,
            config_hash,
            backend_id,
            backend_revision,
            prompt_version,
            schema_hash,
        )
        actual = (
            record.input_hash,
            record.config_hash,
            record.backend_id,
            record.backend_revision,
            record.prompt_version,
            record.schema_hash,
        )
        if actual != expected:
            raise CheckpointError(
                f"completed stage {stage!r} no longer matches its frozen inputs",
                kind=ErrorKind.INVALID_OUTPUT,
                action=f"Use rerun {self.root} --from {stage}; resume will not regenerate it silently.",
            )
        self._validate_outputs(record)
        return record

    def load_artifact(self, record: StageRecord, model: type[T]) -> T:
        if not record.output_paths:
            raise CheckpointError(f"stage {record.stage!r} has no artifact path", kind=ErrorKind.INVALID_OUTPUT)
        path_value = record.output_paths[0]
        if path_value not in record.output_hashes:
            raise CheckpointError(
                f"stage {record.stage!r} artifact is not hash-covered", kind=ErrorKind.INVALID_OUTPUT
            )
        path = (self.root / path_value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise CheckpointError("artifact path escaped the Run Bundle", kind=ErrorKind.INVALID_OUTPUT) from exc
        return model.model_validate(read_json(path))

    def workspace(self, stage: str, *, item_id: str | None = None) -> Workspace:
        stage_record = self.stage_record(stage)
        attempt = (
            stage_record.attempt
            if item_id is not None and stage_record and stage_record.status == "running"
            else self.next_attempt(stage)
        )
        if item_id is not None:
            self._validate_item_id(item_id)
        suffix = item_id or "stage"
        work_dir = self.root / "work" / stage / suffix / f"attempt-{attempt:03d}-{uuid.uuid4().hex[:8]}"
        if item_id:
            final_dir = self.stage_dir(stage) / "items" / item_id / f"attempt-{attempt:03d}"
        else:
            final_dir = self.stage_dir(stage) / f"attempt-{attempt:03d}"
        work_dir.mkdir(parents=True, exist_ok=False)
        return Workspace(stage, attempt, work_dir, final_dir, item_id)

    def next_attempt(self, stage: str) -> int:
        stage_record = self.stage_record(stage)
        return (
            stage_record.attempt + 1
            if stage_record and stage_record.status in {"failed", "running"}
            else 1
        )

    def begin_stage(
        self,
        stage: str,
        *,
        input_hash: str,
        config_hash: str,
        backend_id: str,
        backend_revision: str,
        prompt_version: str,
        schema_hash: str,
        attempt: int,
    ) -> StageRecord:
        record = StageRecord(
            stage=stage,
            status="running",
            attempt=attempt,
            input_hash=input_hash,
            config_hash=config_hash,
            backend_id=backend_id,
            backend_revision=backend_revision,
            prompt_version=prompt_version,
            schema_hash=schema_hash,
            started_at=utc_now(),
        )
        self.manifest.stages[stage] = record
        self.manifest.status = "running"
        self._save_manifest()
        return record

    def _workspace_prefixes(self, workspace: Workspace) -> tuple[str, str]:
        project_root = Path(self.config.project_root)
        return (
            relative_path(workspace.work_dir, project_root),
            relative_path(workspace.final_dir, project_root),
        )

    def _move_workspace(self, workspace: Workspace) -> None:
        if workspace.final_dir.exists():
            raise FileExistsError(workspace.final_dir)
        workspace.final_dir.parent.mkdir(parents=True, exist_ok=True)
        replace_path(workspace.work_dir, workspace.final_dir)

    def _item_record_files(self, stage: str) -> list[Path]:
        root = self.stage_dir(stage) / "item-records"
        return sorted(path for path in root.glob("*.json") if path.is_file())

    def promote_stage(
        self,
        workspace: Workspace,
        artifact: BaseModel | dict[str, Any],
        *,
        usage: Iterable[UsageRecord] = (),
        warnings: Iterable[str] = (),
        extra_files: Iterable[Path] = (),
    ) -> dict[str, Any]:
        resolved_extra_files = [path.resolve() for path in extra_files]
        for path in resolved_extra_files:
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise CheckpointError(
                    f"extra checkpoint output escaped the Run Bundle: {path}",
                    kind=ErrorKind.INVALID_OUTPUT,
                ) from exc
            if not path.is_file():
                raise CheckpointError(
                    f"extra checkpoint output does not exist: {path}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
        old_prefix, new_prefix = self._workspace_prefixes(workspace)
        artifact_data = _rewrite_paths(_dump(artifact), old_prefix, new_prefix)
        atomic_write_json(workspace.work_dir / "artifact.json", artifact_data)
        self._move_workspace(workspace)
        artifact_path = workspace.final_dir / "artifact.json"
        artifact_relative = relative_path(artifact_path, self.root)
        all_files = [
            artifact_path,
            *sorted(path for path in workspace.final_dir.rglob("*") if path.is_file() and path != artifact_path),
            *self._item_record_files(workspace.stage),
            *resolved_extra_files,
        ]
        all_relative = [relative_path(path, self.root) for path in all_files]
        record = self.manifest.stages[workspace.stage]
        record.status = "complete"
        record.output_paths = all_relative
        record.output_hashes = {relative_path(path, self.root): sha256_file(path) for path in all_files}
        record.usage = list(usage)
        record.warnings = list(warnings)
        record.item_ids = [path.stem for path in self._item_record_files(workspace.stage)]
        record.completed_at = utc_now()
        record.error = None
        self._save_manifest()
        return artifact_data

    def item_record_path(self, stage: str, item_id: str) -> Path:
        self._validate_item_id(item_id)
        return self.stage_dir(stage) / "item-records" / f"{item_id}.json"

    @staticmethod
    def _validate_item_id(item_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", item_id):
            raise ValueError(f"invalid checkpoint item ID: {item_id!r}")

    def reusable_item(
        self,
        stage: str,
        item_id: str,
        *,
        input_hash: str,
        config_hash: str,
        backend_id: str,
        backend_revision: str,
        prompt_version: str = "",
        schema_hash: str = "",
    ) -> StageRecord | None:
        path = self.item_record_path(stage, item_id)
        if not path.exists():
            return None
        record = StageRecord.model_validate(read_json(path))
        if record.status != "complete":
            return None
        expected = (input_hash, config_hash, backend_id, backend_revision, prompt_version, schema_hash)
        actual = (
            record.input_hash,
            record.config_hash,
            record.backend_id,
            record.backend_revision,
            record.prompt_version,
            record.schema_hash,
        )
        if actual != expected:
            raise CheckpointError(
                f"completed item {stage}/{item_id} no longer matches its frozen inputs",
                kind=ErrorKind.INVALID_OUTPUT,
                action=f"Use rerun {self.root} --from {stage}; resume will not regenerate it silently.",
            )
        self._validate_outputs(record)
        return record

    def promote_item(
        self,
        workspace: Workspace,
        artifact: BaseModel | dict[str, Any],
        *,
        input_hash: str,
        config_hash: str,
        backend_id: str,
        backend_revision: str,
        prompt_version: str = "",
        schema_hash: str = "",
        usage: Iterable[UsageRecord] = (),
        warnings: Iterable[str] = (),
    ) -> dict[str, Any]:
        if workspace.item_id is None:
            raise ValueError("item promotion requires an item workspace")
        old_prefix, new_prefix = self._workspace_prefixes(workspace)
        artifact_data = _rewrite_paths(_dump(artifact), old_prefix, new_prefix)
        atomic_write_json(workspace.work_dir / "artifact.json", artifact_data)
        self._move_workspace(workspace)
        artifact_path = workspace.final_dir / "artifact.json"
        artifact_relative = relative_path(artifact_path, self.root)
        all_files = [artifact_path, *sorted(path for path in workspace.final_dir.rglob("*") if path.is_file() and path != artifact_path)]
        all_relative = [relative_path(path, self.root) for path in all_files]
        record = StageRecord(
            stage=workspace.stage,
            status="complete",
            attempt=workspace.attempt,
            input_hash=input_hash,
            config_hash=config_hash,
            backend_id=backend_id,
            backend_revision=backend_revision,
            prompt_version=prompt_version,
            schema_hash=schema_hash,
            output_paths=all_relative,
            output_hashes={relative_path(path, self.root): sha256_file(path) for path in all_files},
            usage=list(usage),
            warnings=list(warnings),
            started_at=utc_now(),
            completed_at=utc_now(),
        )
        atomic_write_json(self.item_record_path(workspace.stage, workspace.item_id), record.model_dump(mode="json"))
        stage_record = self.manifest.stages.get(workspace.stage)
        if stage_record and workspace.item_id not in stage_record.item_ids:
            stage_record.item_ids.append(workspace.item_id)
        self._save_manifest()
        return artifact_data

    def load_item_artifact(self, record: StageRecord, model: type[T]) -> T:
        return self.load_artifact(record, model)

    def completed_item_ids(self, stage: str) -> list[str]:
        records_root = self.stage_dir(stage) / "item-records"
        result: list[str] = []
        for path in sorted(records_root.glob("*.json")):
            try:
                record = StageRecord.model_validate(read_json(path))
                if record.status == "complete":
                    self._validate_outputs(record)
                    result.append(path.stem)
            except (CheckpointError, ValueError):
                continue
        return result

    def complete_fanout_stage(
        self,
        stage: str,
        artifact: BaseModel | dict[str, Any],
        *,
        usage: Iterable[UsageRecord] = (),
        warnings: Iterable[str] = (),
    ) -> dict[str, Any]:
        stage_dir = self.stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = stage_dir / "aggregate.json"
        artifact_data = _dump(artifact)
        atomic_write_json(artifact_path, artifact_data)
        artifact_relative = relative_path(artifact_path, self.root)
        record = self.manifest.stages[stage]
        record.status = "complete"
        all_files = [artifact_path, *self._item_record_files(stage)]
        record.output_paths = [relative_path(path, self.root) for path in all_files]
        record.output_hashes = {
            relative_path(path, self.root): sha256_file(path) for path in all_files
        }
        record.item_ids = [path.stem for path in self._item_record_files(stage)]
        record.usage = list(usage)
        record.warnings = list(warnings)
        record.completed_at = utc_now()
        self._save_manifest()
        return artifact_data

    def fail_stage(self, stage: str, error: VideoGeneratorError) -> None:
        record = self.manifest.stages.get(stage)
        if record:
            record.status = "failed"
            record.error = {
                "kind": error.kind.value,
                "message": error.message,
                "action": error.action,
                "details": error.details,
            }
            record.completed_at = utc_now()
        self.manifest.status = "failed"
        self._save_manifest()

    def stop_after(self, stage: str) -> None:
        self.manifest.status = "stopped"
        self.manifest.warnings.append(f"intentionally stopped after {stage}")
        self._save_manifest()

    @staticmethod
    def _legacy_cloud_calls_from_stages(
        stages: dict[str, StageRecord],
        *,
        incurred_in_run_id: str,
        inherited: bool,
        pricing_snapshot: str,
    ) -> list[CloudCallRecord]:
        calls: list[CloudCallRecord] = []
        for stage in PUBLIC_STAGES:
            stage_record = stages.get(stage)
            if stage_record is None:
                continue
            for index, usage in enumerate(stage_record.usage, start=1):
                if (
                    float(usage.reserved_usd) <= 0
                    and usage.estimated_usd is None
                    and usage.actual_usd is None
                ):
                    continue
                cost_known = usage.estimated_usd is not None or usage.actual_usd is not None
                calls.append(
                    CloudCallRecord(
                        call_id=(
                            f"legacy-{incurred_in_run_id}-{stage}-{index:03d}"
                        ),
                        task_id=usage.task_id,
                        backend_id=usage.backend_id,
                        stage=stage,
                        status="settled" if cost_known else "unresolved",
                        provider_request_id=usage.provider_request_id,
                        reserved_usd=usage.reserved_usd,
                        estimated_usd=usage.estimated_usd,
                        actual_usd=usage.actual_usd,
                        billable_units=dict(usage.billable_units),
                        pricing_snapshot=usage.pricing_snapshot or pricing_snapshot,
                        cost_basis=(
                            usage.cost_basis
                            or "legacy stage usage predates the per-call cost ledger"
                        ),
                        incurred_in_run_id=incurred_in_run_id,
                        inherited=inherited,
                        legacy=True,
                        started_at=stage_record.started_at or utc_now(),
                        completed_at=stage_record.completed_at or utc_now(),
                        elapsed_seconds=usage.elapsed_seconds,
                        warnings=list(
                            dict.fromkeys(
                                [
                                    *usage.warnings,
                                    "legacy usage could not be reconstructed as an exact provider call",
                                ]
                            )
                        ),
                    )
                )
        return calls

    def _bootstrap_legacy_cloud_ledger(self) -> None:
        if self.manifest.cloud_cost_ledger_version:
            return
        self.manifest.cloud_calls.extend(
            self._legacy_cloud_calls_from_stages(
                self.manifest.stages,
                incurred_in_run_id=self.manifest.run_id,
                inherited=False,
                pricing_snapshot=self.config.pricing_snapshot,
            )
        )
        self.manifest.cloud_cost_ledger_version = 1

    def reserve_cost(self, amount: float, *, task_id: str, backend_id: str) -> str:
        self._bootstrap_legacy_cloud_ledger()
        ceiling = float(self.config.cost_ceiling_usd)
        prospective = float(self.manifest.reserved_cost_usd) + amount
        if prospective > ceiling + 1e-9:
            raise VideoGeneratorError(
                (
                    f"cloud reservation for {task_id} on {backend_id} would raise the Run total "
                    f"to ${prospective:.2f}, above the ${ceiling:.2f} Cost Ceiling"
                ),
                kind=ErrorKind.BUDGET_EXCEEDED,
                action="Increase cost_ceiling_usd explicitly or choose a lower-cost Run Profile.",
            )
        call_id = uuid.uuid4().hex
        self.manifest.reserved_cost_usd = prospective
        self.manifest.cost_reservations.append(
            UsageRecord(
                task_id=task_id,
                backend_id=backend_id,
                call_id=call_id,
                reserved_usd=amount,
                warnings=["pre-call conservative reservation"],
            )
        )
        running_stage = next(
            (
                stage
                for stage in reversed(PUBLIC_STAGES)
                if self.manifest.stages.get(stage)
                and self.manifest.stages[stage].status == "running"
            ),
            "",
        )
        self.manifest.cloud_calls.append(
            CloudCallRecord(
                call_id=call_id,
                task_id=task_id,
                backend_id=backend_id,
                stage=running_stage,
                reserved_usd=amount,
                pricing_snapshot=self.config.pricing_snapshot,
                incurred_in_run_id=self.manifest.run_id,
                warnings=["pre-call conservative reservation"],
            )
        )
        self._save_manifest()
        return call_id

    def settle_cost(self, call_id: str, usage: UsageRecord) -> None:
        record = next((item for item in self.manifest.cloud_calls if item.call_id == call_id), None)
        if record is None:
            raise CheckpointError(
                f"cloud call ledger entry does not exist: {call_id}",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        cost_known = usage.estimated_usd is not None or usage.actual_usd is not None
        record.status = "settled" if cost_known else "unresolved"
        record.provider_request_id = usage.provider_request_id
        record.estimated_usd = usage.estimated_usd
        record.actual_usd = usage.actual_usd
        record.billable_units = dict(usage.billable_units)
        record.pricing_snapshot = usage.pricing_snapshot or self.config.pricing_snapshot
        record.cost_basis = usage.cost_basis
        record.elapsed_seconds = usage.elapsed_seconds
        record.completed_at = utc_now()
        record.warnings = list(dict.fromkeys([*record.warnings, *usage.warnings]))
        if not cost_known:
            record.warnings = list(
                dict.fromkeys(
                    [*record.warnings, "provider call completed but billable cost is unresolved"]
                )
            )
        self._save_manifest()

    def mark_cost_unresolved(
        self,
        call_id: str,
        *,
        elapsed_seconds: float,
        error: BaseException,
    ) -> None:
        record = next((item for item in self.manifest.cloud_calls if item.call_id == call_id), None)
        if record is None:
            return
        record.status = "unresolved"
        record.elapsed_seconds = elapsed_seconds
        record.completed_at = utc_now()
        if isinstance(error, VideoGeneratorError):
            record.error = {
                "kind": error.kind.value,
                "message": error.message,
                "action": error.action,
            }
        else:
            record.error = {
                "kind": ErrorKind.INTERNAL.value,
                "message": str(error) or type(error).__name__,
                "action": None,
            }
        record.warnings = list(
            dict.fromkeys(
                [*record.warnings, "provider call failed; billing outcome could not be confirmed"]
            )
        )
        self._save_manifest()

    @classmethod
    def fork(
        cls,
        *,
        parent: "RunStore",
        config: ResolvedRunConfig,
        brief: CreativeBrief,
        frozen_assets: dict[str, Any],
        fork_stage: str,
    ) -> "RunStore":
        if fork_stage not in PUBLIC_STAGES:
            raise ValueError(f"unknown fork stage: {fork_stage}")
        if Path(parent.config.project_root).resolve() != Path(config.project_root).resolve():
            raise CheckpointError(
                "a rerun cannot move a Run Bundle to a different project root",
                kind=ErrorKind.UNSUPPORTED,
            )
        fork_index = PUBLIC_STAGES.index(fork_stage)
        upstream: list[tuple[str, StageRecord]] = []
        for stage in PUBLIC_STAGES[:fork_index]:
            record = parent.manifest.stages.get(stage)
            if not record or record.status != "complete":
                raise CheckpointError(
                    f"parent does not have a valid completed upstream stage: {stage}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            parent._validate_outputs(record)
            upstream.append((stage, record))
        child = cls.create(
            project_root=Path(config.project_root),
            config=config,
            brief=brief,
            frozen_assets=frozen_assets,
            parent_run_id=parent.manifest.run_id,
            fork_stage=fork_stage,
        )
        old_run_prefix = relative_path(parent.root, Path(config.project_root))
        new_run_prefix = relative_path(child.root, Path(config.project_root))
        for stage, record in upstream:
            source = parent.stage_dir(stage)
            destination = child.stage_dir(stage)
            if source.exists():
                shutil.copytree(source, destination)
                for json_path in destination.rglob("*.json"):
                    data = read_json(json_path)
                    rewritten = _rewrite_paths(data, old_run_prefix, new_run_prefix)
                    if rewritten != data:
                        atomic_write_json(json_path, rewritten)
                for item_record_path in (destination / "item-records").glob("*.json"):
                    item_record = StageRecord.model_validate(read_json(item_record_path))
                    item_record.output_hashes = {
                        path_value: sha256_file(child.root / path_value)
                        for path_value in item_record.output_paths
                    }
                    atomic_write_json(item_record_path, item_record.model_dump(mode="json"))
            copied_record = record.model_copy(deep=True)
            copied_record.output_hashes = {
                path_value: sha256_file(child.root / path_value)
                for path_value in copied_record.output_paths
            }
            child.manifest.stages[stage] = copied_record
        upstream_stages = {stage for stage, _ in upstream}
        inherited_calls = [
            call.model_copy(
                deep=True,
                update={
                    "inherited": True,
                    "incurred_in_run_id": call.incurred_in_run_id or parent.manifest.run_id,
                },
            )
            for call in parent.manifest.cloud_calls
            if call.stage in upstream_stages
        ]
        if not inherited_calls:
            inherited_calls = child._legacy_cloud_calls_from_stages(
                child.manifest.stages,
                incurred_in_run_id=parent.manifest.run_id,
                inherited=True,
                pricing_snapshot=parent.config.pricing_snapshot,
            )
        child.manifest.cloud_calls = inherited_calls
        child.manifest.cloud_cost_ledger_version = 1
        child._save_manifest()
        return child


CONFIG_IMPACT: dict[str, str] = {
    "output_language": "research",
    "offline": "research",
    "duration_seconds": "ideate",
    "content_mode": "research",
    "audience": "script-draft",
    "idea_candidates": "ideate",
    "research_query_limit": "research",
    "research_source_limit": "research",
    "voice": "narration",
    "style": "visual-plan",
    "style_description": "visual-plan",
    "visual_target_seconds": "outline",
    "visual_min_seconds": "outline",
    "visual_max_seconds": "outline",
    "quality": "captions",
    "delivery_width": "captions",
    "delivery_height": "captions",
    "fps": "narration",
    "captions_enabled": "captions",
    "animated_captions": "captions",
    "music_enabled": "music-brief",
    "failure_policy": "music",
    "motion_style": "render",
}


TASK_STAGE_IMPACT: dict[str, str] = {
    "search": "research",
    "research": "research",
    "ideate": "ideate",
    "select": "select",
    "outline": "outline",
    "script_draft": "script-draft",
    "review_story": "review-story",
    "review_spoken": "review-spoken",
    "review_constraints": "review-constraints",
    "script_revision": "script-revision",
    "factual_review": "script-revision",
    "narration_synthesis": "narration",
    "duration_repair": "narration",
    "caption_alignment": "captions",
    "visual_plan": "visual-plan",
    "image_prompt_compile": "image-prompt-compile",
    "image_generate": "image-prompt-compile",
    "visual_review": "visual-review",
    "music_brief": "music-brief",
    "music_generate": "music-brief",
}


def earliest_config_impact(old: ResolvedRunConfig, new: ResolvedRunConfig) -> str | None:
    old_data = old.model_dump(mode="json", exclude={"created_at", "cost_ceiling_usd", "pricing_snapshot"})
    new_data = new.model_dump(mode="json", exclude={"created_at", "cost_ceiling_usd", "pricing_snapshot"})
    stages = []
    for field, stage in CONFIG_IMPACT.items():
        if old_data.get(field) != new_data.get(field):
            stages.append(stage)
    old_bindings = old_data.get("task_bindings", {})
    new_bindings = new_data.get("task_bindings", {})
    for task_id, stage in TASK_STAGE_IMPACT.items():
        if old_bindings.get(task_id) != new_bindings.get(task_id):
            stages.append(stage)
    if not stages:
        return None
    return min(stages, key=PUBLIC_STAGES.index)
