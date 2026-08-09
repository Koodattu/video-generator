from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_generator.backends.local import LocalImageBackend
from video_generator.contracts import ImageGenerationSettings, ImageRequest
from video_generator.profiles import BACKEND_DESCRIPTORS, image_generation_dimensions
from video_generator.runners import RunnerManager
from video_generator.util import atomic_write_json


DEFAULT_BACKENDS = (
    "local:flux.2-klein-4b",
    "local:z-image-turbo",
    "local:qwen-image-2512-nf4",
)

PROMPTS = (
    "A tiny red sailboat crossing a glassy arctic bay at blue hour, wide cinematic composition, crisp reflections, no text or logos.",
    "A cozy reading nook inside a futuristic lunar habitat, warm amber lamps, Earth visible through a round window, editorial photography, no text.",
    "An old green tram moving through a rainy Nordic city street, pedestrians with colorful umbrellas, cinematic realism, no signs or lettering.",
    "A curious orange fox examining a small brass telescope in a snowy pine forest, storybook illustration, expressive pose, no text.",
    "A precise exploded-view illustration of a mechanical pocket watch floating above a dark workbench, clean parts separation, dramatic studio light, no labels.",
    "A brutalist concrete library softened by hanging gardens and shallow reflecting pools, sunny morning, architectural visualization, people for scale, no text.",
    "A bowl of ramen photographed from a low three-quarter angle, rising steam, handmade ceramic bowl, moody restaurant lighting, food photography, no text.",
    "A lone astronaut tending bright wildflowers inside a transparent greenhouse on Mars, red dunes beyond, hopeful science-fiction concept art, no text.",
    "Three friendly robots playing jazz in a compact subway station, dynamic silhouettes, limited teal and coral palette, screen-print poster style, no lettering.",
    "A macro photograph of morning dew suspended across a spider web, soft green bokeh, delicate rainbow refraction, natural light, no text or watermark.",
)

NEGATIVE_PROMPT = (
    "text, letters, captions, logos, watermarks, signatures, duplicate subjects, malformed anatomy, "
    "extra limbs, low detail, blurry focal subject"
)


@dataclass(frozen=True)
class ModelSettings:
    steps: int
    guidance: float | None
    cpu_offload: bool


MODEL_SETTINGS = {
    "local:flux.2-klein-4b": ModelSettings(steps=4, guidance=1.0, cpu_offload=True),
    "local:z-image-turbo": ModelSettings(steps=9, guidance=0.0, cpu_offload=False),
    "local:qwen-image-2512-nf4": ModelSettings(steps=50, guidance=4.0, cpu_offload=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prepared local image models sequentially through their pinned runners."
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=tuple(MODEL_SETTINGS),
        dest="backends",
        help="Backend to run; repeat to select several. Defaults to all usable prepared backends.",
    )
    parser.add_argument(
        "--images-per-model",
        type=int,
        default=len(PROMPTS),
        choices=range(1, len(PROMPTS) + 1),
        metavar=f"1-{len(PROMPTS)}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs") / "image-benchmarks",
        help="Parent directory for generated images and benchmark reports.",
    )
    return parser.parse_args()


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def command_output(command: list[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def summarize_model(model: dict[str, Any], expected_count: int) -> None:
    successful = [item for item in model["images"] if item["status"] == "success"]
    generation_times = [float(item["generation_seconds"]) for item in successful]
    wall_times = [float(item["wall_seconds"]) for item in successful]
    peak_values = [
        float(item["peak_vram_mb"])
        for item in successful
        if item.get("peak_vram_mb") is not None
    ]
    model["success_count"] = len(successful)
    model["failure_count"] = len(model["images"]) - len(successful)
    model["unattempted_count"] = max(0, expected_count - len(model["images"]))
    model["generation_seconds_total"] = sum(generation_times)
    model["generation_seconds_mean"] = statistics.mean(generation_times) if generation_times else None
    model["generation_seconds_median"] = statistics.median(generation_times) if generation_times else None
    model["generation_seconds_p95"] = percentile_95(generation_times)
    model["image_wall_seconds_total"] = sum(wall_times)
    model["peak_vram_mb_max"] = max(peak_values) if peak_values else None


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = (
        "backend_id",
        "model_id",
        "prompt_index",
        "seed",
        "status",
        "width",
        "height",
        "load_seconds",
        "wall_seconds",
        "generation_seconds",
        "peak_vram_mb",
        "output_path",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in report["models"]:
            for image in model["images"]:
                writer.writerow(
                    {
                        "backend_id": model["backend_id"],
                        "model_id": model["model_id"],
                        "prompt_index": image["prompt_index"],
                        "seed": image["seed"],
                        "status": image["status"],
                        "width": model["width"],
                        "height": model["height"],
                        "load_seconds": model.get("load_wall_seconds"),
                        "wall_seconds": image.get("wall_seconds"),
                        "generation_seconds": image.get("generation_seconds"),
                        "peak_vram_mb": image.get("peak_vram_mb"),
                        "output_path": image.get("output_path"),
                        "error": image.get("error"),
                    }
                )


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    backends = tuple(args.backends or DEFAULT_BACKENDS)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (project_root / args.output_root / timestamp).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    manager = RunnerManager(project_root=project_root, run_root=run_root)
    report: dict[str, Any] = {
        "benchmark_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "project_python": sys.executable,
        "project_revision": command_output(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=project_root,
        ),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            cwd=project_root,
        ),
        "output_root": str(run_root),
        "images_per_model": args.images_per_model,
        "prompt_count": args.images_per_model,
        "prompts": list(PROMPTS[: args.images_per_model]),
        "negative_prompt": NEGATIVE_PROMPT,
        "metric_definitions": {
            "load_wall_seconds": "Runner process startup through a successful health response.",
            "generation_seconds": "Timed pipeline inference inside the worker; image encoding is excluded.",
            "wall_seconds": "Host-side image request including inference, encoding, protocol, and validation.",
            "peak_vram_mb": (
                "Peak CUDA memory reported by PyTorch for the managed worker, not total GPU memory in use."
            ),
            "total_wall_seconds": "Readiness checks, runner startup, all requests, and cleanup.",
        },
        "excluded_prepared_backends": {
            "local:ideogram-4-nf4": (
                "Excluded from the usable set: repository metadata records that its corrected smoke "
                "returned a rejected safety placeholder and no usable comparison image."
            )
        },
        "models": [],
    }
    benchmark_started = time.monotonic()

    print(f"Benchmark output: {run_root}", flush=True)
    try:
        for backend_id in backends:
            descriptor = BACKEND_DESCRIPTORS[backend_id]
            settings = MODEL_SETTINGS[backend_id]
            width, height = image_generation_dimensions(
                backend_id, delivery_width=1280, delivery_height=720
            )
            model_dir = run_root / backend_id.replace(":", "--")
            model_dir.mkdir(parents=True)
            model: dict[str, Any] = {
                "backend_id": backend_id,
                "model_id": descriptor.model_id,
                "width": width,
                "height": height,
                "settings": {
                    "inference_steps": settings.steps,
                    "guidance_scale": settings.guidance,
                    "cpu_offload": settings.cpu_offload,
                },
                "images": [],
            }
            report["models"].append(model)
            model_started = time.monotonic()
            print(f"\n[{backend_id}] static readiness check", flush=True)

            probe_started = time.monotonic()
            probe = manager.probe(backend_id, live=False)
            model["probe_seconds"] = time.monotonic() - probe_started
            model["probe"] = probe.model_dump(mode="json")
            if not probe.ready:
                model["status"] = "not_ready"
                model["error"] = "static readiness check failed"
                model["success_count"] = 0
                model["failure_count"] = 0
                model["unattempted_count"] = args.images_per_model
                model["total_wall_seconds"] = time.monotonic() - model_started
                print("  skipped: static readiness check failed", flush=True)
                continue

            spec = manager.load_spec(backend_id)
            model["model_revision"] = spec.model_revision
            model["runtime_revision"] = spec.runtime_revision
            model["setup_source_revision"] = spec.setup_source_revision

            try:
                print(f"[{backend_id}] loading model", flush=True)
                load_started = time.monotonic()
                health = manager.invoke(backend_id, "health", {})
                model["load_wall_seconds"] = time.monotonic() - load_started
                model["health"] = health
                print(f"  loaded in {model['load_wall_seconds']:.2f}s", flush=True)

                backend = LocalImageBackend(backend_id, manager)
                for index, prompt in enumerate(PROMPTS[: args.images_per_model], start=1):
                    seed = 2026072100 + index
                    output_path = model_dir / f"{index:02d}.png"
                    request = ImageRequest(
                        scene_id=f"benchmark-{index:03d}",
                        target_backend_id=backend_id,
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        quality="low",
                        seed=seed,
                        settings=ImageGenerationSettings(
                            inference_steps=settings.steps,
                            guidance_scale=settings.guidance,
                            output_format="png",
                            aspect_ratio="16:9",
                            cpu_offload=settings.cpu_offload,
                        ),
                    )
                    image_record: dict[str, Any] = {
                        "prompt_index": index,
                        "prompt": prompt,
                        "seed": seed,
                    }
                    image_started = time.monotonic()
                    try:
                        result = backend.generate(request, output_path)
                        image_record.update(
                            {
                                "status": "success",
                                "wall_seconds": time.monotonic() - image_started,
                                "generation_seconds": result.usage.elapsed_seconds,
                                "peak_vram_mb": result.usage.peak_vram_mb,
                                "output_path": result.asset.image.path,
                                "sha256": result.asset.image.sha256,
                                "generation_settings": result.asset.generation_settings,
                            }
                        )
                        print(
                            f"  {index:02d}/{args.images_per_model}: "
                            f"{image_record['generation_seconds']:.2f}s generation, "
                            f"{image_record['wall_seconds']:.2f}s wall",
                            flush=True,
                        )
                    except Exception as exc:
                        image_record.update(
                            {
                                "status": "failed",
                                "wall_seconds": time.monotonic() - image_started,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"  {index:02d}/{args.images_per_model}: FAILED: {exc}", flush=True)
                    model["images"].append(image_record)
            except Exception as exc:
                model["status"] = "failed"
                model["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  model failed: {exc}", flush=True)
            finally:
                manager.stop_current()
                model["cleanup"] = manager.last_cleanup.get(backend_id)
                model["total_wall_seconds"] = time.monotonic() - model_started
                summarize_model(model, args.images_per_model)
                if model.get("status") is None:
                    model["status"] = (
                        "success" if model["success_count"] == args.images_per_model else "partial"
                    )
                atomic_write_json(run_root / "benchmark.json", report)
    finally:
        manager.close()

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["total_wall_seconds"] = time.monotonic() - benchmark_started
    report["total_success_count"] = sum(model.get("success_count", 0) for model in report["models"])
    report["total_failure_count"] = sum(
        model.get("failure_count", 0) + model.get("unattempted_count", 0)
        for model in report["models"]
    )
    atomic_write_json(run_root / "benchmark.json", report)
    write_csv(report, run_root / "images.csv")
    print(
        f"\nDone: {report['total_success_count']} images succeeded, "
        f"{report['total_failure_count']} failed in {report['total_wall_seconds']:.2f}s.",
        flush=True,
    )
    print(f"JSON: {run_root / 'benchmark.json'}", flush=True)
    print(f"CSV:  {run_root / 'images.csv'}", flush=True)
    return 0 if report["total_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
