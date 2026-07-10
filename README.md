# Video Generator

A planned Python CLI for producing narrated, still-image videos from a creative brief. One command will coordinate research, story development, narration, scene images, captions, optional music, and local FFmpeg rendering.

The project is currently in the architecture phase. No working generator has been implemented yet. The planning documents define the contracts first so local, cloud, and hybrid Backends can be selected independently without duplicating the workflow.

## Design goals

- Run end to end by default, with durable checkpoints and resumable failures.
- Support English and Finnish as one Output Language per Run.
- Run every expensive local model in an isolated process so a 24 GB GPU can reuse VRAM between stages.
- Mix local and cloud Backends per task through curated Run Profiles.
- Produce simple 16:9 videos with spoken narration, static generated images, hard cuts, and captions.
- Keep the normal interface to `config.toml`, `brief.toml`, and `.env`.

## Planning documents

- [Domain language](CONTEXT.md)
- [Contract design](docs/contracts.md)
- [Architecture](docs/architecture.md)
- [Prompt system](docs/prompt-system.md)
- [Initial model matrix](docs/model-matrix.md)
- [Implementation plan](docs/implementation-plan.md)
- [Architecture decisions](docs/adr/)

Example inputs are provided in [config.example.toml](config.example.toml), [brief.example.toml](brief.example.toml), and [.env.example](.env.example). These files describe the intended CLI and are not executable yet.

## Intended CLI

```powershell
Copy-Item config.example.toml config.toml
Copy-Item brief.example.toml brief.toml
Copy-Item .env.example .env

video-generator setup --profile local
video-generator preflight --config config.toml
video-generator generate --config config.toml --brief brief.toml
video-generator resume runs/<run-id>
video-generator rerun runs/<run-id> --from <stage>
```

Setup may download pinned model assets. Preflight is read-only. Generate performs no surprise model downloads.

## Scope

The initial Usage Purpose is private, personal, noncommercial research, education, and entertainment. Voice cloning is limited to the user's own voice or a voice used with explicit permission. Model and asset licenses still need to be recorded per Run.
