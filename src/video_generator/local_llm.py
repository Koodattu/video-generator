from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigurationError


_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
CONTEXT_TIERS = (8192, 16384, 32768, 65536, 131072, 262144)


class LocalLlmProfile(BaseModel):
    """Auditable inputs and launch settings for one local llama.cpp evaluation variant."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profile_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,79}$")]
    model_id: Annotated[str, Field(min_length=1, max_length=200)]
    model_repo: Annotated[
        str, Field(min_length=3, max_length=200, pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    ]
    model_revision: str
    model_path: Annotated[str, Field(min_length=1, max_length=1000)]
    model_sha256: str
    license_name: Annotated[str, Field(min_length=1, max_length=200)]

    llama_cpp_revision: str
    llama_server_path: Annotated[str, Field(min_length=1, max_length=1000)]
    llama_server_sha256: str
    llama_runtime_files: dict[str, str]

    context_size: Literal[8192, 16384, 32768, 65536, 131072, 262144] = 32768
    batch_size: Annotated[int, Field(ge=32, le=4096)] = 512
    micro_batch_size: Annotated[int, Field(ge=32, le=4096)] = 512
    gpu_layers: Annotated[int, Field(ge=0, le=999)] = 999
    flash_attention: bool = True
    speculation: Literal["none", "draft-mtp"] = "none"
    speculative_tokens: Annotated[int, Field(ge=1, le=8)] = 2

    draft_model_id: Annotated[str, Field(max_length=200)] = ""
    draft_model_repo: Annotated[str, Field(max_length=200)] = ""
    draft_model_revision: str = ""
    draft_model_path: Annotated[str, Field(max_length=1000)] = ""
    draft_model_sha256: str = ""

    @model_validator(mode="after")
    def validate_provenance(self) -> "LocalLlmProfile":
        for name, value in (
            ("model_revision", self.model_revision),
            ("llama_cpp_revision", self.llama_cpp_revision),
        ):
            if not _FULL_COMMIT.fullmatch(value) or set(value.casefold()) == {"0"}:
                raise ValueError(f"{name} must be a nonzero full 40-character commit hash")
        for name, value in (
            ("model_sha256", self.model_sha256),
            ("llama_server_sha256", self.llama_server_sha256),
        ):
            if not _SHA256.fullmatch(value) or set(value.casefold()) == {"0"}:
                raise ValueError(f"{name} must be a nonzero SHA-256")
        server_name = Path(self.llama_server_path).name
        if server_name.casefold() != "llama-server.exe":
            raise ValueError("llama_server_path must name llama-server.exe")
        if not self.llama_runtime_files:
            raise ValueError("llama_runtime_files must declare llama-server.exe and every sibling DLL")
        normalized_runtime_names: set[str] = set()
        for name, expected_hash in self.llama_runtime_files.items():
            path = Path(name)
            if path.name != name or path.suffix.casefold() not in {".exe", ".dll"}:
                raise ValueError("llama_runtime_files keys must be plain .exe or .dll filenames")
            folded = name.casefold()
            if folded in normalized_runtime_names:
                raise ValueError("llama_runtime_files contains case-insensitive duplicate names")
            normalized_runtime_names.add(folded)
            if not _SHA256.fullmatch(expected_hash) or set(expected_hash.casefold()) == {"0"}:
                raise ValueError(f"llama_runtime_files[{name!r}] must be a nonzero SHA-256")
        declared_server_hash = next(
            (
                value
                for name, value in self.llama_runtime_files.items()
                if name.casefold() == "llama-server.exe"
            ),
            "",
        )
        if declared_server_hash.casefold() != self.llama_server_sha256.casefold():
            raise ValueError(
                "llama_runtime_files must declare llama-server.exe with llama_server_sha256"
            )
        if self.micro_batch_size > self.batch_size:
            raise ValueError("micro_batch_size must not exceed batch_size")

        draft_values = (
            self.draft_model_id,
            self.draft_model_repo,
            self.draft_model_revision,
            self.draft_model_path,
            self.draft_model_sha256,
        )
        if any(draft_values) and not all(draft_values):
            raise ValueError("all draft_model fields are required when a separate drafter is used")
        if any(draft_values) and self.speculation != "draft-mtp":
            raise ValueError("a separate drafter requires speculation = 'draft-mtp'")
        if self.draft_model_revision:
            if not re.fullmatch(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", self.draft_model_repo):
                raise ValueError("draft_model_repo must be a Hugging Face owner/repository ID")
            if not _FULL_COMMIT.fullmatch(self.draft_model_revision) or set(
                self.draft_model_revision.casefold()
            ) == {"0"}:
                raise ValueError("draft_model_revision must be a nonzero full 40-character commit hash")
            if not _SHA256.fullmatch(self.draft_model_sha256) or set(
                self.draft_model_sha256.casefold()
            ) == {"0"}:
                raise ValueError("draft_model_sha256 must be a nonzero SHA-256")
        return self

    def resolve_path(self, value: str, profile_path: Path) -> Path:
        raw = Path(value)
        return (raw if raw.is_absolute() else profile_path.parent / raw).resolve()


def load_local_llm_profile(path: Path) -> LocalLlmProfile:
    path = path.resolve()
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        return LocalLlmProfile.model_validate(payload)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"local LLM profile does not exist: {path}") from exc
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise ConfigurationError(f"invalid local LLM profile {path}: {exc}") from exc
