# Video Generator

A local-first Python CLI that turns one creative brief into a narrated still-image video. The typed
workflow supports fiction and evidence-gated factual content, narrative/explainer/myth-buster formats,
measured narration delivery, and either one image per editorial Scene or a faster timed Shot sequence.
It optionally creates instrumental music and renders MP4 delivery files with FFmpeg.

The default `generate` command runs end to end. Every public stage and expensive per-visual item is
checkpointed in an immutable Run Bundle, so `resume` does not silently repeat valid paid or local
model work. A task can select a local, OpenAI, Gemini, ElevenLabs, Brave, or mixed Backend without
changing the workflow.

## Current v0 scope

- One Output Language per Run: English (`en`) or Finnish (`fi`).
- Fiction inspired by bounded research, or live factual research with Evidence Records, a complete
  Claim Inventory, and an independent pre-TTS Factual Review. Offline factual Runs are rejected.
- Narrative, explainer, and factual myth-buster editorial formats; slow, standard, and fast measured
  narration presets with an optional custom delivery direction.
- Static 16:9 images with hard cuts; default output is 1280x720 draft or 1920x1080 final at 30 fps.
- Selectable SRT captions, plus optional ASS captions with the active spoken word highlighted in a
  second, burned-in MP4.
- Optional ambient instrumental music mixed below narration.
- Built-in `ms_paint_stick` image style and arbitrary additional style IDs described in config.
  Production profiles render every style through their configured generative image model; the
  programmatic stick renderer is an explicit deterministic test Backend only.
- Personal, noncommercial use; voice cloning is limited to your own voice or explicit permission.
- Final-quality local Visual Review is intentionally readiness-gated because the planned Qwen vision
  runner has not yet been proven reliable in 24 GB VRAM. Local draft Runs work without it.

## Requirements

- Windows 10/11 with Python 3.11 (3.12 is also accepted by the orchestrator).
- [`uv`](https://docs.astral.sh/uv/) for locked environments.
- FFmpeg and ffprobe on `PATH`, with libx264 and AAC; libass is additionally required for animated
  captions.
- For local CUDA Backends: a current NVIDIA driver and substantial free disk space. Setup selects
  CUDA-compatible PyTorch wheels for Torch workers; faster-whisper uses its isolated CTranslate2
  runtime instead. Only one model worker owns the GPU at a time.
- WSL2 is optional and needed only when explicitly selecting the retained Parakeet/NeMo Backend.
  The default local profile runs natively on Windows.

## Install the orchestrator

Always run project Python commands from the repository virtual environment:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade uv
uv sync --active --all-extras
```

Copy the safe examples. `.env`, `config.toml`, `brief.toml`, `local-llm.toml`, `private/`, model
caches, downloads, and Run Bundles are ignored by Git.

```powershell
Copy-Item config.example.toml config.toml
Copy-Item brief.example.toml brief.toml
Copy-Item .env.example .env
Copy-Item local-llm.example.toml local-llm.toml
New-Item -ItemType Directory -Force private\voice
```

Edit `config.toml` and `brief.toml`. Put only required API keys in `.env`; never put secrets in TOML.
For a local clone, copy your authorized reference WAV (and preferably its exact UTF-8 transcript) to
`private/voice/` and keep `voice.authorization = "self"`. For ElevenLabs, set
`ELEVENLABS_VOICE_ID` in `.env` (or `voice.elevenlabs_voice_id` in TOML) to your authorized voice ID;
no private reference recording is uploaded by this program.

The 90-second example is the intended first useful target. For the first mechanical check, set
`duration_seconds = 30`. The configured duration is both the goal and hard ceiling; accepted measured
narration must occupy 85-100% of it.

## Fastest first Run: cloud

Choose `cloud-openai`, `cloud-gemini`, or `cloud-openai-gemini` in `config.toml`, set the corresponding
API keys plus `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`. The mixed profile uses pinned GPT-5.4
mini for text, Gemini 3.1 Flash Image, ElevenLabs narration, and keyless DDGS/DuckDuckGo research.
Increase `cost_ceiling_usd` only after reading the Preflight estimate.

```powershell
video-generator setup --config config.toml --no-download
video-generator preflight --config config.toml --live
video-generator generate --config config.toml --brief brief.toml
```

`--live` performs explicit cloud access probes and, for local Backends, sequential worker health/model-load
checks without generating media. Preflight never downloads models or writes a Run. Generate repeats
non-live readiness checks, freezes the report/config/prompts/schemas/profile,
then creates `runs/<run-id>/`. Runtime cost reservations enforce the hard ceiling before each cloud
request; no provider call is silently rerouted.

## Run the local dashboard

The FastAPI dashboard is the control plane for creating, supervising, and inspecting Runs. Start it
from the project root, then open `http://127.0.0.1:8765/`:

```powershell
video-generator dashboard
# Or choose another loopback port:
video-generator dashboard --port 8877
```

The New Run dialog selects a profile and an explicit Backend for every workflow task, runs the same
read-only non-live Preflight as the CLI, and creates a frozen Run Bundle before queueing execution.
The dashboard deliberately serializes Runs through one child-process worker so local GPU memory and
cloud budgets are not oversubscribed. Progress is streamed from the authoritative atomic manifest;
closing the browser does not stop the worker. Stop interrupts the complete worker process tree and
leaves the Run resumable.

Each Run view joins the 19-stage timeline, script Scenes, timed visual Shots, narration clips, Visual Briefs, compiled
image prompts, generated images, Visual Reviews, delivery media, costs, logs, and every file below
the Run directory. Files with recorded stage hashes are distinguished from internal or untracked files.
The server binds only to `127.0.0.1`; it is not an authenticated multi-user service and should not be
published through a reverse proxy.

Cloud accounting is append-only per provider attempt. The dashboard separates:

- the conservative ceiling reservation used to prevent accidental overspend;
- calculated public list-price usage from provider-reported billable units;
- provider-reported cost when a provider supplies it; and
- unresolved maximum exposure for calls whose billing outcome is unknown.

The complete dated pricing table is frozen into every new Run. Calculated list price is an estimate,
not an invoice: subscriptions, free quotas, cached-token rules, regional pricing, discounts, taxes,
and later provider reconciliation can differ.

## Prepare the local profile

The local profile uses:

| Task | Initial Backend |
|---|---|
| Research search | Keyless DDGS/DuckDuckGo, or no search when `offline = true` |
| Text/reviews/prompt compilation | Manifest-selected GGUF through stock `llama-server.exe` |
| English/Finnish voice clone | VoxCPM2 |
| Local timestamps | faster-whisper large-v3-turbo through CTranslate2 on native Windows |
| Images | FLUX.2 Klein 4B through Diffusers |
| Optional music | ACE-Step 1.5 XL Turbo |

Setup pins runtime/model revisions, stores assets in `.cache/`, writes hashes and runner manifests,
and may take a long time. Generate is offline with respect to model repositories and never downloads
missing weights.

The placement policy is Windows first. The default LLM, VoxCPM, faster-whisper, FLUX, ACE-Step,
FFmpeg, and orchestrator paths run natively. Parakeet/NeMo remains an explicit WSL2 override for
matched English/Finnish comparison; it is not required for an ordinary local Run.

### 1. Prepare one auditable local LLM profile

`local-llm.toml` describes one benchmark variant: exact target GGUF, optional compatible drafter,
full repository commit(s), SHA-256 hashes, license, exact stock llama.cpp commit/build, context tier,
and MTP setting. Setup rejects branch names, abbreviated commits, zero placeholders, altered files,
and arbitrary server launch overrides.

Download one candidate first instead of the entire candidate matrix. Pin the full Hugging Face commit;
do not use `main`:

```powershell
# Inspect the two curated 24 GB benchmark candidates without downloading anything.
video-generator models list
video-generator models download qwen3.6-27b-q4-mtp --dry-run
video-generator models download gemma-4-26b-a4b-q4-mtp --dry-run

# Download one exact, commit-pinned candidate into .cache/models/llm and verify its SHA-256.
video-generator models download gemma-4-26b-a4b-q4-mtp
```

The curated Qwen artifact has embedded MTP. The curated Gemma artifact includes its separate Q4 MTP
drafter. Both quantizations come from the Unsloth repositories linked in the model matrix, are public,
and require no Hugging Face login. The command downloads only the listed GGUF file(s), never the whole
repository, and records the source revision plus independent hashes in `asset-manifest.json`. These are
benchmark candidates rather than automatic defaults; begin with MTP disabled and promote a variant only
after it passes the English/Finnish fixtures. The stock llama.cpp runtime remains a separate pinned input.

For an arbitrary candidate not in the curated list, use the manual path below:

```powershell
New-Item -ItemType Directory -Force .cache\models\llm\CANDIDATE-ID, downloads\llama.cpp

uvx --from huggingface-hub hf download ORGANIZATION/REPOSITORY MODEL.gguf `
  --revision FULL_40_CHARACTER_COMMIT `
  --local-dir .cache\models\llm\CANDIDATE-ID

Get-FileHash .cache\models\llm\CANDIDATE-ID\MODEL.gguf -Algorithm SHA256
Get-FileHash downloads\llama.cpp\llama-server.exe -Algorithm SHA256
Get-ChildItem downloads\llama.cpp\*.dll | Get-FileHash -Algorithm SHA256
```

Download a pinned Windows x64 CUDA build from the stock
[`llama.cpp` releases](https://github.com/ggml-org/llama.cpp/releases), merge its required CUDA DLL
bundle into the same `downloads\llama.cpp\` directory, and fill `local-llm.toml`. Setup copies or adopts only
`llama-server.exe`, its sibling DLLs, the selected GGUF, and optional drafter into `.cache/`, then
records every copied hash. Unsloth may supply a quantized GGUF, but it is not the inference runtime.
When `model_path` points anywhere under `.cache\models\llm\`, Setup verifies and adopts the GGUF in
place rather than creating another multi-gigabyte copy. MTP-off/on profiles may therefore share the
same target artifact while retaining different `profile_id` and launch metadata.

Start with `context_size = 32768`, `speculation = "none"`, and one server slot. Treat the same model
with `speculation = "draft-mtp"` as a separate benchmark profile. Embedded MTP needs no drafter path;
models that ship a separate MTP assistant require every `draft_model_*` field. Larger context tiers
are separate launches and must prove they fit; the program does not silently allocate 256K.

### 2. Optional: install WSL2 only for a Parakeet override

Skip this section for the default faster-whisper Backend. If you explicitly override alignment to
Parakeet, Setup will not install or choose a Linux distribution for you. One example:

```powershell
wsl --install -d Ubuntu
wsl -d Ubuntu -- python3.12 --version
wsl -d Ubuntu -- python3.12 -m venv --help
wsl -d Ubuntu -- ffmpeg -version
wsl -d Ubuntu -- nvidia-smi -L
```

If that distribution does not provide Python 3.12, its `venv` module, FFmpeg, or CUDA access, install
them inside the distribution or pass the name of another prepared distribution with
`--wsl-distro`. Setup deliberately does not install operating-system packages or choose a distro.

### 3. Prepare all Backends active in config

With music disabled and draft quality, this prepares the selected LLM, VoxCPM, faster-whisper Turbo,
FLUX, and optionally Brave. Set `offline = true` if you want no web search and no Brave key. Each
Backend receives an isolated, locked venv. Setup asks `uv` to select a CUDA-compatible PyTorch wheel
only for Torch workers; faster-whisper pins CTranslate2 separately. Live Preflight then loads each
worker sequentially. For llama.cpp it additionally requires the child server PID to exit and records
before/load/after aggregate VRAM observations; aggregate drift is advisory on Windows because
unrelated WDDM applications can change it.

```powershell
video-generator setup --config config.toml --llm-profile local-llm.toml

video-generator preflight --config config.toml --live
video-generator generate --config config.toml --brief brief.toml
```

For model comparison, give every target/context/MTP combination a unique `profile_id`, prepare one at
a time, then run the same English and Finnish fixtures. A new Setup replaces the active
`local:llama-server` runner manifest; existing Run Bundles will refuse to resume until their original
profile is restored, which prevents accidental cross-model continuation.

```powershell
video-generator evaluate --suite smoke --profile local --language en --config config.toml --live-preflight
video-generator evaluate --suite smoke --profile local --language fi --config config.toml --live-preflight
```

When `music_enabled = true`, Setup also prepares ACE-Step. You can prepare or repair one Backend
without touching the others:

```powershell
video-generator setup --backend local:ace-step-1.5-xl-turbo
video-generator setup --backend local:faster-whisper-large-v3-turbo
# Optional retained WSL2 alternative:
video-generator setup --backend local:parakeet-tdt-0.6b-v3 --wsl-distro Ubuntu
```

Use `--no-download` to verify existing assets/environments. A missing runner, exact model revision,
WSL distribution, key, voice file, FFmpeg capability, disk allowance, or Cost Ceiling makes Preflight
fail with an explicit action. It never substitutes another model.

## Mix Backends per task

Curated profiles are `local`, `cloud-openai`, `cloud-gemini`, `cloud-openai-gemini`, and
`hybrid-local-first`. Advanced overrides live under `[task_overrides]` in config. For example:

```toml
[task_overrides]
script_draft = "openai:gpt-5.6-terra"
image_prompt_compile = "local:llama-server"
image_generate = "gemini:gemini-3.1-flash-image"
narration_synthesis = "elevenlabs:eleven_multilingual_v2"
```

Every override is validated against its protocol, language, usage purpose, Offline setting, and
capabilities before a Run is created. English and Finnish use the same workflow contracts; separate
orchestration code is not duplicated by language.

## Editorial format, visual cadence, and pacing

The default remains the original behavior: `content_mode = "fiction"`,
`content_format = "narrative"`, `narration_pace = "standard"`, and
`visual_shot_mode = "scene_locked"`. These axes are independent, so a factual narrative can remain
measured while a fictional explainer can use a fast timed-image sequence.

`scene_locked` produces one generated image for each editorial Scene. `cadenced` builds a deterministic,
frame-aligned Shot schedule after narration and word alignment, then asks the configured image model for
one still per Shot. `shot_target_seconds`, `shot_min_seconds`, and `shot_max_seconds` control that visual
rhythm without forcing the outline or TTS checkpoint boundaries to become equally short.

New Runs plan visuals as one storyboard rather than as isolated prompts. The visual plan freezes each
recurring character's body form, proportions, face and markings, wardrobe, and non-negotiable identity
constraints. Every Scene also records the incoming state, outgoing state, persistent elements, and
the exact current event. Prompt compilation receives adjacent Scenes only as continuity context and is
instructed never to depict a later event early.

When the selected image Backend supports references, the first accepted image containing a recurring
character becomes an identity/style reference for later Scenes. References are restricted to the
characters present in that Scene and are hash-covered as part of the effective image request. They are
guidance rather than a mathematical identity guarantee, so inspect the Visuals and Visual Review artifacts
before delivery.

Narration presets resolve to explicit word-rate, pause, and pitch-preserving tempo targets. Fast delivery
uses tighter pauses and a denser script; slow delivery uses fewer words and more room for emphasis. The
final measured rate is validated, and short narration is repaired with useful spoken content instead of
padding the timeline with silence.

The current single-plan cadence limit is 72 generated Shots. Longer videos can use `scene_locked`, a
larger `shot_target_seconds`, or multiple Runs; configuration rejects an oversized cadenced plan before
paid generation begins.

## Resume, inspect, and intentionally rerun

```powershell
# Stop deliberately after a checkpoint for inspection.
video-generator generate --config config.toml --brief brief.toml --stop-after script-revision

# Continue with the Run's frozen config/prompts/schemas.
video-generator resume runs\<run-id>

# Preview invalidation, remaining readiness, and cost without creating a child.
video-generator rerun runs\<run-id> --from images --dry-run

# Create a parent-linked child Run and carry forward verified upstream artifacts.
video-generator rerun runs\<run-id> --from images

# Supply intentional new config/brief; the command refuses a fork later than their earliest impact.
video-generator rerun runs\<run-id> --from research --config config.toml --brief brief.toml
```

Run Bundles contain resolved non-secret inputs, frozen production assets, stage/item manifests,
hashes, usage/cost reservations, normalized intermediates, logs, and delivery outputs. They do not
copy private voice recordings or credentials.

Old Runs are preserved by default. Pruning is a dry run unless `--yes` is explicit, preserves parents
needed by surviving children, and never touches `.cache/models` or `private/`:

```powershell
video-generator runs prune --older-than 30
video-generator runs prune --older-than 30 --yes
```

## Evaluation harness

The fixed smoke suite creates 30-second draft Runs, `draft-quality` creates 90-second draft Runs, and
quality creates 90-second final Runs. Use `draft-quality` for the local profile until its optional
final-quality vision reviewer passes evaluation. Omitting `--language` runs both English and Finnish.
These commands perform real generation and can incur cost:

```powershell
video-generator evaluate --suite smoke --language fi --config config.toml
video-generator evaluate --suite draft-quality --profile local --config config.toml --live-preflight
video-generator evaluate --suite quality --config config.toml
```

## CLI exit codes

| Code | Meaning |
|---:|---|
| 0 | success or intentional dry run/stop |
| 1 | internal/transient failure |
| 2 | invalid or unsupported configuration |
| 3 | environment/Backend not ready |
| 4 | Cost Ceiling would be exceeded |
| 5 | provider/media/contract output invalid or policy refusal |
| 130 | interrupted by the user |

## Design references

- [Domain language](CONTEXT.md)
- [Contracts](docs/contracts.md)
- [Architecture](docs/architecture.md)
- [Prompt system](docs/prompt-system.md)
- [Model matrix and pins](docs/model-matrix.md)
- [Architecture decisions](docs/adr/)

The implementation favors a fixed inspectable workflow over open-ended agent loops: one structured
repair per LLM result, one measured narration Duration Repair, one final-quality image regeneration
batch, and no hidden fallbacks. That keeps cost, provenance, resumption, and failure behavior legible.
