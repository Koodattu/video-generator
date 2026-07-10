from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..contracts import (
    AlignmentRequest,
    AlignmentResult,
    BackendDescriptor,
    ImageRequest,
    ImageResult,
    MusicRequest,
    MusicResult,
    ProbeReport,
    SearchRequest,
    SearchResult,
    SourceDocument,
    SourceFetchRequest,
    SpeechRequest,
    SpeechResult,
    StructuredTextRequest,
    StructuredTextResult,
)


class Backend:
    descriptor: BackendDescriptor

    def probe(self, *, live: bool = False) -> ProbeReport:
        raise NotImplementedError

    def close(self) -> None:
        return None


@runtime_checkable
class SearchBackend(Protocol):
    descriptor: BackendDescriptor

    def search(self, request: SearchRequest) -> SearchResult: ...

    def fetch(self, request: SourceFetchRequest) -> SourceDocument: ...


@runtime_checkable
class StructuredTextBackend(Protocol):
    descriptor: BackendDescriptor

    def complete(self, request: StructuredTextRequest) -> StructuredTextResult: ...


@runtime_checkable
class SpeechBackend(Protocol):
    descriptor: BackendDescriptor

    def synthesize(self, request: SpeechRequest) -> SpeechResult: ...


@runtime_checkable
class AlignmentBackend(Protocol):
    descriptor: BackendDescriptor

    def align(self, request: AlignmentRequest) -> AlignmentResult: ...


@runtime_checkable
class ImageBackend(Protocol):
    descriptor: BackendDescriptor

    def generate(self, request: ImageRequest, output_path: Path) -> ImageResult: ...


@runtime_checkable
class MusicBackend(Protocol):
    descriptor: BackendDescriptor

    def generate(self, request: MusicRequest) -> MusicResult: ...

