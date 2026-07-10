from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .backends.base import Backend
from .backends.brave import BraveSearchBackend
from .backends.deterministic import (
    DeterministicAlignmentBackend,
    DeterministicImageBackend,
    DeterministicMusicBackend,
    DeterministicSearchBackend,
    DeterministicSpeechBackend,
    DeterministicStructuredTextBackend,
)
from .backends.elevenlabs import (
    ElevenLabsAlignmentBackend,
    ElevenLabsMusicBackend,
    ElevenLabsSpeechBackend,
)
from .backends.gemini import GeminiImageBackend, GeminiSearchBackend, GeminiStructuredTextBackend
from .backends.local import (
    LocalAlignmentBackend,
    LocalImageBackend,
    LocalMusicBackend,
    LocalSpeechBackend,
    LocalStructuredTextBackend,
)
from .backends.openai import OpenAIImageBackend, OpenAIStructuredTextBackend, OpenAIWebSearchBackend
from .contracts import BackendDescriptor, ProbeReport, ResolvedRunConfig
from .errors import BackendError, ErrorKind
from .profiles import BACKEND_DESCRIPTORS
from .runners import RunnerManager


class BackendRegistry:
    """Static Backend registry. It intentionally has no dynamic imports or entry points."""

    def __init__(
        self,
        *,
        config: ResolvedRunConfig,
        environment: Mapping[str, str],
        run_root: Path,
        descriptors: Mapping[str, BackendDescriptor] | None = None,
    ) -> None:
        self.config = config
        self.environment = dict(environment)
        self.project_root = Path(config.project_root).resolve()
        self.run_root = run_root.resolve()
        self.descriptors = dict(descriptors or BACKEND_DESCRIPTORS)
        self.runners = RunnerManager(project_root=self.project_root, run_root=self.run_root)
        self._instances: dict[str, Any] = {}

    def descriptor(self, backend_id: str) -> BackendDescriptor:
        try:
            return self.descriptors[backend_id]
        except KeyError as exc:
            raise BackendError(f"unknown Backend: {backend_id}", kind=ErrorKind.UNSUPPORTED) from exc

    def for_task(self, task_id: str) -> Any:
        return self.get(self.config.task_bindings[task_id])

    def get(self, backend_id: str) -> Any:
        if backend_id in self._instances:
            return self._instances[backend_id]
        descriptor = self.descriptor(backend_id)
        if self.config.offline and descriptor.cloud:
            raise BackendError(
                f"offline Run cannot use cloud Backend {backend_id}", kind=ErrorKind.UNSUPPORTED
            )
        missing = [name for name in descriptor.required_env if not self.environment.get(name)]
        if missing:
            raise BackendError(
                f"Backend {backend_id} is missing credentials: {', '.join(missing)}",
                kind=ErrorKind.NOT_READY,
                action=f"Add {', '.join(missing)} to .env, then run Preflight again.",
            )
        instance = self._build(backend_id)
        if hasattr(instance, "descriptor"):
            instance.descriptor = descriptor
        self._instances[backend_id] = instance
        return instance

    def _build(self, backend_id: str) -> Any:
        if backend_id == "brave:web":
            return BraveSearchBackend(self.environment["BRAVE_SEARCH_API_KEY"])
        if backend_id == "openai:web":
            return OpenAIWebSearchBackend(self.environment["OPENAI_API_KEY"])
        if backend_id in {"openai:gpt-5.5", "openai:gpt-5.6-terra"}:
            return OpenAIStructuredTextBackend(
                self.environment["OPENAI_API_KEY"], backend_id=backend_id, workspace_root=self.run_root
            )
        if backend_id == "openai:gpt-image-2":
            return OpenAIImageBackend(
                self.environment["OPENAI_API_KEY"], workspace_root=self.project_root, run_root=self.run_root
            )
        if backend_id == "gemini:search":
            return GeminiSearchBackend(self.environment["GEMINI_API_KEY"])
        if backend_id == "gemini:gemini-3.5-flash":
            return GeminiStructuredTextBackend(
                self.environment["GEMINI_API_KEY"], workspace_root=self.run_root
            )
        if backend_id == "gemini:gemini-3.1-flash-image":
            return GeminiImageBackend(
                self.environment["GEMINI_API_KEY"], workspace_root=self.project_root, run_root=self.run_root
            )
        if backend_id == "elevenlabs:eleven_multilingual_v2":
            return ElevenLabsSpeechBackend(
                self.environment["ELEVENLABS_API_KEY"],
                workspace_root=self.project_root,
                run_root=self.run_root,
                configured_voice_id=self.config.voice.elevenlabs_voice_id,
            )
        if backend_id == "elevenlabs:forced-alignment":
            return ElevenLabsAlignmentBackend(
                self.environment["ELEVENLABS_API_KEY"], workspace_root=self.project_root, run_root=self.run_root
            )
        if backend_id == "elevenlabs:music_v2":
            return ElevenLabsMusicBackend(
                self.environment["ELEVENLABS_API_KEY"], workspace_root=self.project_root, run_root=self.run_root
            )
        if backend_id in {"local:llama-server", "local:qwen3.6-27b-q4-vision"}:
            return LocalStructuredTextBackend(backend_id, self.runners)
        if backend_id == "local:voxcpm2":
            return LocalSpeechBackend(backend_id, self.runners)
        if backend_id == "local:parakeet-tdt-0.6b-v3":
            return LocalAlignmentBackend(backend_id, self.runners)
        if backend_id == "local:flux.2-klein-4b":
            return LocalImageBackend(backend_id, self.runners)
        if backend_id == "local:ace-step-1.5-xl-turbo":
            return LocalMusicBackend(backend_id, self.runners)
        if backend_id == "deterministic:search":
            return DeterministicSearchBackend()
        if backend_id == "deterministic:structured":
            return DeterministicStructuredTextBackend()
        if backend_id == "deterministic:speech":
            return DeterministicSpeechBackend(self.project_root)
        if backend_id == "deterministic:alignment":
            return DeterministicAlignmentBackend()
        if backend_id == "deterministic:stick":
            return DeterministicImageBackend(self.project_root)
        if backend_id == "deterministic:music":
            return DeterministicMusicBackend(self.project_root)
        raise BackendError(f"Backend has no registered adapter: {backend_id}", kind=ErrorKind.UNSUPPORTED)

    def probe(self, backend_id: str, *, live: bool = False) -> ProbeReport:
        return self.get(backend_id).probe(live=live)

    def close(self) -> None:
        for instance in self._instances.values():
            if isinstance(instance, Backend):
                instance.close()
            elif hasattr(instance, "close"):
                instance.close()
        self.runners.close()

    def release_local_workers(self) -> None:
        """Release the exclusive GPU before deterministic media rendering."""

        self.runners.close()

    def __enter__(self) -> "BackendRegistry":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
