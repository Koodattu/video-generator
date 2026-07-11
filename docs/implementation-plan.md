# Implementation plan

The code milestones below are implemented. Their live acceptance criteria remain deliberately open until the pinned cloud/local Backends are prepared and the explicit smoke and quality suites are run.

## Delivery strategy

All four curated profiles and both languages are v0 scope. “Support them from the start” means their contracts, capability descriptors, configuration shape, fixtures, and acceptance matrix are designed before provider code. It should not mean implementing every unstable SDK and CUDA stack simultaneously before any end-to-end path works.

The safe order is contract-first, then vertical slices. Each milestone leaves a runnable, testable system and narrows the source of failure.

## v0 success criteria

The v0 release is complete only when:

- `setup`, read-only `preflight`, end-to-end `generate`, `resume`, and `evaluate` work from the CLI;
- `local`, `cloud-openai`, `cloud-gemini`, and `hybrid-local-first` have explicit, inspectable mappings and pass their applicable conformance tests;
- 30-second smoke Runs and 60–90-second English and Finnish Runs complete for every profile;
- a successful output uses 90–100% of its Duration Budget and never exceeds the configured hard limit after frame-grid quantization;
- every Scene has one generated image, measured narration timing, and stable provenance;
- captions are enabled by default and produce SRT plus an embedded selectable track;
- optional burned ASS captions produce a separate file from the same Caption Track;
- optional instrumental music works in each profile or fails according to its declared Failure Policy;
- cloud cost guarding prevents a call that would cross the frozen ceiling;
- local models run one at a time and VRAM returns near the measured baseline after runner termination;
- a killed Run resumes without silently repeating valid paid or expensive work;
- neither secrets nor copies of private voice references appear in source control, ordinary logs, or Run Bundles;
- FFmpeg/ffprobe QC validates codecs, streams, resolution, frame rate, captions, duration, and audio health;
- model IDs, asset hashes, prompt/schema/profile versions, usage, license/terms metadata, and warnings are preserved.

Factual mode may be advertised as supported only when claim/evidence capture and factual review also pass. Until then, configuration validation must reject it with a clear planned-capability message rather than running a fiction workflow under a factual label.

## Foundation choices to validate

The orchestrator should target Python 3.11, matching the current machine and the local-model compatibility range. Use `uv` with a committed lockfile. The minimum likely orchestrator dependencies are Pydantic v2 for contracts and a small `.env` reader; TOML parsing, CLI parsing, subprocess execution, hashing, and filesystem state can use the standard library initially. Do not adopt a DAG framework, ORM, database, distributed queue, or dependency-injection container.

Provider SDKs should be optional dependency groups. Each incompatible local runtime gets a pinned runner environment rather than sharing the orchestrator environment. Setup owns creation of those environments and model downloads; Generate only launches already-prepared runners.

The final dependency list is an implementation decision to confirm against current SDK docs when each adapter is built. No dependency should be added merely because it might be useful later.

## Milestone 1 — Contracts and Run store

Deliver:

- package skeleton and the static Backend registry;
- Pydantic models for every artifact in [Contract design](contracts.md);
- configuration resolution for `config.toml`, `brief.toml`, `.env`, built-in profiles, and task overrides;
- the canonical task-ID registry, with exactly one primary Backend binding per task and a separately bound Search capability available only to `research`;
- secret redaction and validation;
- stage/item manifests, hashing, per-Scene atomic promotion, frozen prompt/schema/profile assets, resume rules, and parent-linked `rerun --from` semantics;
- bounded task executor with deterministic fake Search/Text/Speech/Alignment/Image/Music Backends;
- `setup`, `preflight`, `generate --stop-after`, `resume`, `rerun --from`, and `evaluate` command shells;
- capability descriptors, probe reports, typed errors, Cost Ceiling reservation, and usage records.

Exit checks:

- schemas reject provider objects, unknown IDs, illegal language/profile combinations, and secret-bearing resolved config;
- a fake complete Run creates every expected normalized artifact;
- interruption after every stage resumes at the correct next stage;
- package prompt/schema/profile upgrades do not alter an existing Run's frozen resume behavior;
- `rerun --from` explains affected stages and cost, creates a new parent-linked Run, and leaves the parent unchanged;
- a supplied configuration change that invalidates upstream work makes `rerun` report the earliest valid fork instead of carrying inconsistent artifacts;
- a completed paid stage is not repeated by ordinary resume;
- `preflight` produces no filesystem or network mutation beyond explicitly permitted live probes and no model downloads.

## Milestone 2 — Deterministic media vertical slice

Build the media pipeline before involving generative models:

- use fixture Scene audio and images to create a Narration Timeline;
- normalize and concatenate audio, including declared pauses;
- reconcile fixture word timings into a Caption Track;
- derive SRT, embedded `mov_text`, and optional ASS output;
- build a deterministic Render Plan;
- render H.264/AAC MP4 with static cuts and run ffprobe QC;
- add a synthetic/fixture Music Bed and verify exact trimming/fade/mix behavior;
- implement the explicitly selected deterministic stick/image test Backend.

Exit checks:

- a 10–30-second fixture renders on the currently installed Windows FFmpeg;
- Scene cuts, captions, narration, and music all derive from the same Timeline;
- output is 16:9, 30 fps, `yuv420p`, fast-start MP4, matches the Narration Timeline rounded to the frame grid, and never exceeds the hard budget;
- caption text and time spans are identical across SRT, selectable MP4, and burned ASS variants;
- narration remains intelligible and the mix does not clip.

This milestone proves the product can make a valid video independently of model quality.

## Milestone 3 — Reference cloud path

Implement Gemini 3.5 Flash, Gemini Search, Gemini 3.1 Flash Image, ElevenLabs TTS/timestamps, and local FFmpeg as the first real reference path. Gemini is suggested first because its selected model IDs and capabilities are generally available and explicit; this is implementation order, not a quality conclusion.

Deliver:

- versioned prompt assets and JSON Schema output for all bounded text tasks;
- actual research call accounting plus safe bounded source retrieval and normalized excerpt capture;
- five-candidate selection, outline, draft, three reviews, revision, and validation;
- per-Scene ElevenLabs synthesis with character-to-word timing normalization;
- measured Duration Repair of selected Scenes;
- visual planning, separately assigned Gemini-target prompt compilation, image generation, optional review, one regeneration batch, and one re-review of replacements;
- optional ElevenLabs Music v2;
- pricing snapshot and cost reservation for every call.

Exit checks:

- 30-second English and Finnish smoke Runs complete;
- one 60–90-second fiction fixture per language meets duration, safety, captions, media, and provenance criteria;
- a forced invalid JSON result, refused image, rate limit, and Cost Ceiling exhaustion each produce the intended bounded outcome;
- source queries and retained URLs are present without dumping full pages into downstream prompts;
- redirect/private-address/MIME/size/time protections pass adversarial source-fetch fixtures.

## Milestone 4 — OpenAI-led cloud path

Implement Responses API text/search, GPT Image 2, vision review, and the same ElevenLabs/media adapters.

Deliver:

- GPT-5.6 Terra as the curated default, with model access verified by live Preflight;
- Responses structured-output and bounded web-search adapters;
- separately assigned GPT Image 2 prompt compiler with legal 16:9 generation dimensions and reference inputs;
- cloud profile cost and usage accounting;
- conformance fixtures shared with the Gemini path.

Exit checks mirror Milestone 3. The same Creative Brief should be runnable through both cloud profiles so quality, cost, latency, image consistency, and failure rates can be compared rather than judged from unrelated outputs.

## Milestone 5 — Local runtime foundation

Implement the exclusive GPU lease and platform-aware runner manager before integrating all models.

Deliver:

- model manifest format with source, revision, files, SHA-256, license, disk/memory expectations, and runtime revision;
- local cache under `./.cache/models`;
- native-Windows and WSL2 process launchers plus one workspace path-mapping utility;
- runner start/health/stop lifecycle, structured logs, timeout/cancellation, and crash cleanup;
- declared VRAM reservations plus live sequential load/process-exit probes; llama-server additionally records before/load/peak/post-exit GPU observations and GPU-PID cleanup;
- exact Setup actions and read-only Preflight probes;
- no-download enforcement during Generate.

Exit checks:

- a dummy native worker and dummy WSL worker pass the same lifecycle contract;
- termination releases file handles and exits cleanly; llama-server's GPU PID disappears and aggregate memory is compared with its Windows pre-launch baseline;
- missing WSL distribution, model, runtime, or disk space yields an exact setup command;
- invalid Windows/WSL path translation is caught before model launch;
- a runner cannot write outside its Run work directory or model cache.

The current machine has WSL2 enabled but no distribution. Setup should report this and install nothing without an explicit setup action.

## Milestone 6 — Local Backends

Integrate one model at a time against existing conformance tests:

1. a manifest-selected GGUF through pinned stock Windows `llama-server.exe`, beginning with one Qwen and one Gemma candidate and separate MTP-off/on variants;
2. VoxCPM2 with native compatibility mode;
3. faster-whisper large-v3-turbo on native Windows plus exact-script timing reconciliation, retaining Parakeet v3 as an explicit WSL2 comparison Backend;
4. FLUX.2 klein 4B with a native Windows benchmark;
5. Qwen vision path for Visual Review, separately memory-tested;
6. ACE-Step XL Turbo and standard Turbo comparison.

For each Backend:

- first pass its protocol fixtures without the full workflow;
- then run its stage inside a fake remainder of the workflow;
- then add it to a 30-second local end-to-end Run;
- record cold-start, warm batch, peak VRAM/RAM, disk, output properties, and cleanup;
- evaluate English and Finnish where language matters.

Exit checks:

- the full local profile completes in both languages without another generative cloud service;
- live research is independently switchable from local inference and Offline blocks all network calls;
- local captions preserve the canonical transcript above the required reconciliation coverage;
- the selected LLM, VoxCPM, FLUX, and ACE-Step never coexist in VRAM;
- failures after any model family resume without repeating prior families.

If Qwen vision is not reliable in 24 GB, final-quality local Visual Review remains unavailable until an explicit smaller local vision Backend is evaluated. It must not be silently skipped.

## Milestone 7 — Hybrid, optional features, and factual mode

Freeze the first `hybrid-local-first` mapping only after local and cloud measurements exist. The planned initial mapping is the promoted local GGUF/FLUX/ACE-Step plus ElevenLabs narration/timestamps and independent search.

Complete:

- music on/off and strict/omit-with-warning behavior across all profiles;
- burned animated-caption template based on the canonical Caption Track;
- final-quality Visual Review/regeneration behavior;
- character-reference handling for capable image Backends;
- claim/evidence artifacts and factual review, or keep factual mode rejected;
- Usage Purpose/license and voice-authorization reporting;
- Run pruning that cannot remove model assets or private source recordings.

Exit checks cover every feature combination exposed by the example config. “Optional” means selectable, not allowed to disappear silently once enabled.

## Milestone 8 — Evaluation and profile promotion

Create fixed fixtures rather than tuning on whichever output looked interesting that day.

The suite should contain:

- short and 60–90-second briefs in English and Finnish;
- at least one dialogue-like passage read by the single narrator;
- Finnish names, numbers, compounds, and English loanwords;
- recurring-character visual continuity;
- an intentionally hard Duration Repair case;
- unsafe/constraint-violating inputs;
- local caption cases with hesitations or pronunciation mismatch;
- research injection and source-quality traps;
- cloud-budget and transient-failure simulations;
- optional music and silent/no-music controls.

Measure:

- schema/contract pass rate and retry rate;
- story, originality, spoken-naturalness, and safety rubrics;
- predicted versus measured Scene and total duration;
- voice similarity, intelligibility, and inter-Scene continuity;
- caption coverage and timing error;
- visual brief/style/identity success and regeneration rate;
- music unobtrusiveness and narration intelligibility;
- cold/warm runtime, peak VRAM/RAM, disk, provider usage, and actual cost;
- for local GGUF variants: model load/prompt/generation throughput, MTP acceptance where exposed, context tier, schema validity, and shutdown time;
- end-to-end completion and resume correctness.

Automated model judges support comparisons; deterministic checks and human listening/visual review anchor them. Profile defaults change only with a versioned evaluation report. English and Finnish may select different models only when this evidence justifies the operational complexity.

## Milestone 9 — Scale to ten minutes

Do not begin with ten-minute Runs. After 60–90 seconds is reliable, test 3 minutes, 5 minutes, then 10 minutes.

At each step evaluate:

- story coherence and prompt-context growth;
- 30–50 Scene planning and batch resume behavior;
- voice consistency across many Scene clips;
- cumulative caption drift;
- model reload count and total runtime;
- image and music disk usage;
- music provider duration limits/loop audibility;
- Run Bundle size and pruning;
- cloud ceiling estimates versus actual cost.

The same contracts should scale. A new workflow abstraction is justified only by a measured failure at scale.

## Verification pyramid

| Level | What it proves | Runs when |
| --- | --- | --- |
| Unit | schemas, config, hashes, budgets, timeline math, reconciliation, command building | every change |
| Contract | every Backend obeys its protocol and typed failures | adapter changes |
| Mocked integration | stage order, retries, frozen resume, parent-linked rerun, failure policy | every change |
| Deterministic media E2E | actual FFmpeg output and ffprobe QC | media changes and CI where available |
| Live cloud smoke | access, current API shape, small paid request | opt-in with credentials |
| Live local smoke | runtime/model/platform/VRAM lifecycle | GPU machine |
| Quality suite | English/Finnish output and profile comparisons | explicit evaluation runs |

Live tests are clearly marked and budget-capped. Unit tests never require credentials, network, or model downloads.

## Risk register

| Risk | Test early | Planned response |
| --- | --- | --- |
| Local GGUF provenance or 24 GB fit | full commit/hash review and 32K stock llama-server smoke | evaluate one Qwen and one Gemma candidate first; no untracked download or automatic 256K |
| Finnish creative quality | matched EN/FI story fixtures | language-specific prompt guidance first; separate model only with evidence |
| Scene TTS prosody discontinuity | listen to adjacent emotional Scenes | pass text context, preserve voice settings, adjust Scene boundaries; do not merge architecture prematurely |
| Finnish cloned-voice quality | VoxCPM and Eleven A/B with authorized recordings | language-matched references; model split only if needed |
| Local caption mismatch | exact-script reconciliation fixture | coverage threshold and visible failure; never substitute ASR text |
| Native Windows CUDA incompatibility | per-Backend contract smoke | WSL2 runner for that Backend only |
| VRAM not released | post-process `nvidia-smi` checks | kill process tree, fail lifecycle test, avoid in-process CUDA imports |
| Weak character continuity | recurring-character fixture and review | stronger identity anchors/reference inputs, accept benign inconsistency |
| Visual self-review unavailable locally | Qwen vision memory/quality fixture | explicit smaller VLM evaluation or declared draft behavior |
| Duration drift | calibrated voice rates plus measured repair fixture | one targeted repair; resumable failure rather than speed/truncation |
| Cloud model/access churn | Setup model probes and pinned profile versions | fail with exact supported override; never silently reroute |
| Cost estimates wrong | compare reservation to actual usage | dated price snapshot, bounded outputs, conservative reservation |
| Music duration/account capability conflict | account capability probe and long fixture | cap one generation at 600 seconds; deterministic loop only when explicit |
| Old FFmpeg edge cases | actual render/caption/mux smoke | capability-based warning and documented upgrade if test fails |
| Research copying or prompt injection | adversarial grounded-search excerpts | no direct page fetch in v0; treat excerpts as data, paraphrase pack, bounded calls, provenance |
| Factual claims unsupported | claim/evidence fixture | block factual mode until review contract passes |
| Disk growth | repeated Run and model setup test | Run Bundle accounting and explicit `runs prune`; separate model cache |

## Recommended first implementation slice

Start with Milestones 1 and 2 only. The first visible artifact should be a valid short fixture video produced through the real Run store and Timeline, not a notebook that calls one model. Once that base passes, the cloud reference path gives fast feedback on prompt and artifact design before local runtime debugging begins.

This ordering does not reduce the committed v0 scope. It protects the contract design from being accidentally shaped around whichever provider happened to be integrated first.
