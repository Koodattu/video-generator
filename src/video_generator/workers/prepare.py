from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ..util import replace_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_asset_manifest(destination: Path, *, repo: str, revision: str) -> Path:
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        relative = path.relative_to(destination)
        if path.name == "asset-manifest.json" or ".cache" in relative.parts:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "source": f"https://huggingface.co/{repo}",
        "revision": revision,
        "files": files,
    }
    temporary = destination / ".asset-manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = destination / "asset-manifest.json"
    replace_path(temporary, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--allow", action="append", default=[])
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=destination,
        allow_patterns=args.allow or None,
        token=os.environ.get("HF_TOKEN") or None,
    )
    write_asset_manifest(destination, repo=args.repo, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
