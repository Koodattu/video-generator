# Contract design

This is the provider boundary for v0. It is intentionally a small static system, not a general plugin framework or workflow engine.

## Contract rules

1. Workflow Tasks exchange validated, provider-neutral artifacts.
2. Provider SDK objects never enter artifacts. Provider-specific prompt syntax appears only in explicitly Backend-bound request artifacts, never in upstream provider-neutral domain artifacts.
3. Every Scene keeps the same stable ID, such as `scene-001`, from outline through delivery; cadenced
   visuals add immutable `shot-001` IDs without replacing their parent Scene IDs.
4. All timing derives from one final Narration Timeline.
5. Large media travels by workspace-relative file reference, never as base64 in JSON or logs.
6. Every artifact and Backend descriptor has an explicit schema version.
7. A completed artifact is reusable only when its inputs, relevant configuration, task instructions, schema, and Backend revision still match.
8. Backends are explicitly selected. No adapter silently changes model, provider, platform, or optional-feature behavior.

Python should model these contracts with Pydantic and persist their normalized JSON representation. Pydantic is an implementation choice, not a wire-format dependency: local runners consume ordinary JSON and media files.

## Canonical artifacts

The durable artifact set is deliberately finite.

| Artifact | Required content | Produced by |
| --- | --- | --- |
| `ResolvedRunConfig` | Secret-free effective settings, frozen Run Profile, Backend assignments, limits, pricing snapshot | configuration resolver |
| `CreativeBrief` | Idea direction, tone, themes, constraints, research focus, and optional anchor/question/misconception/landing fields | input normalization |
| `ResearchPack` | Queries, retained sources, compact findings, cautions, clichés; evidence records in factual mode | research |
| `CandidateSet` | Bounded Story Candidates | ideation |
| `ExplainerCandidateSet` | Bounded concepts with modern anchor, question, thesis, evidence ladder, and landing | ideation |
| `SelectionReport` / `ExplainerSelectionReport` | Format-specific rubric scores, concise rationale, chosen candidate ID | selection |
| `StoryOutline` | Ordered Scenes with purpose, emotional beat, visual opportunity, provisional duration share | outlining |
| `ExplainerOutline` | Ordered Scenes with arc role, key point, Evidence IDs, visual opportunity, and provisional duration share | outlining |
| `NarrationScript` | Spoken text per stable Scene ID | script workflow |
| `ReviewReport` | Findings by rubric, severity, Scene ID, and actionable recommendation | review tasks |
| `ClaimInventory` / `FactualReviewReport` | Exact spoken claims, Evidence links, independent verdicts, and uncovered-claim list | factual gate |
| `NarrationTimeline` | Final narration asset, exact total duration, Scene start/end times, optional word timings | narration assembly |
| `VisualPlan` | Character Identities, resolved Style Profile, one Visual Brief per Scene | visual planning |
| `TimedVisualPlan` | Resolved Style Profile and canonical, frame-aligned generated-image Shots linked to parent Scenes | visual planning |
| `ImageRequest` / `TimedImageRequest` | Backend-specific compiled prompt and generation settings linked to one Scene or Shot | prompt compiler |
| `VisualReviewReport` | Brief/style/identity scores and one targeted regeneration instruction | visual review |
| `RemotionEditPlan` | Host-timed Shots, bounded rhythm, deterministic quality report, one of eight fixed templates, visible copy, asset intent, motion, transition, and SFX presets | host allocator plus one rhythm call and one small direction call per Shot |
| `RemotionAssetRequestSet` | Host-owned asset IDs, kinds, English search queries or generated prompts, and evidence-linked source IDs | deterministic Remotion asset compiler |
| `RemotionAssetBundle` / `AssetRights` | Resolved local media references, hashes, transformations, source/creator/license/attribution data, and review warnings | Remotion asset resolver |
| `CaptionTrack` | Canonical words and monotonic time spans | timing normalization |
| `MusicBrief` | Instrumental intent, mood arc, exclusions, requested duration | music planning |
| `MusicAsset` | Normalized media reference, actual duration, generation provenance | music generation/post-processing |
| `RenderPlan` | Exact media references, time ranges, codecs, caption modes, audio mix settings | render planning |
| `RemotionRenderPlan` | Fixed composition, edit plan, resolved assets, word-caption frames, renderer/runtime settings, and exact output paths | deterministic Remotion render planning |
| `DeliveryManifest` | Output files, hashes, media properties, QC result, warnings | delivery verification |

Prompts, raw Backend responses needed for debugging, review reports, and usage records are also persisted, but they do not replace normalized artifacts. Hidden model reasoning is never requested or stored.

## Narration Timeline

The Narration Timeline is the master clock. Its invariant is:

```text
scene.start = previous_scene.end
scene.end = scene.start + normalized_audio_duration + declared_pause_after
timeline.duration = final_scene.end
```

Scene-locked visuals use these boundaries; cadenced Shots use frame-aligned intervals derived from them.
Music is fitted to this duration. Caption words must stay inside their Scene and the Timeline. Before
narration acceptance, the media layer computes `delivery_ceiling = floor(duration_budget × fps) / fps`.
The Narration Timeline must end at or before that ceiling. The final video stream rounds the Timeline
duration up to a frame boundary, so it may outlast narration by less than one frame, but it remains at or
below the delivery ceiling and never exceeds the configured Duration Budget.

Duration Repair may change and resynthesize selected Scene text once. It may not add, remove, or reorder Scenes in v0. The final Narration Timeline is created only after that repair opportunity has completed. Visual planning and music generation therefore cannot become stale because of a later script-length change.

## Backend protocols

There is no universal `Backend.run(anything)` method. Adapters implement one or more narrow capability protocols:

```python
class SearchBackend(Protocol):
    def search(self, request: SearchRequest) -> SearchResult: ...
    def fetch(self, request: SourceFetchRequest) -> SourceDocument: ...

class StructuredTextBackend(Protocol):
    def complete(self, request: StructuredTextRequest) -> StructuredTextResult: ...

class SpeechBackend(Protocol):
    def synthesize(self, request: SpeechRequest) -> SpeechResult: ...

class AlignmentBackend(Protocol):
    def align(self, request: AlignmentRequest) -> AlignmentResult: ...

class ImageBackend(Protocol):
    def generate(self, request: ImageRequest) -> ImageResult: ...

class MusicBackend(Protocol):
    def generate(self, request: MusicRequest) -> MusicResult: ...
```

These sketches show contract shape, not final Python signatures. Visual review reuses a vision-capable Structured Text Backend. FFmpeg rendering is a deterministic internal service, not an AI Backend.

The Structured Text request carries the task name, task-instruction version, language, validated input-artifact references, desired JSON Schema, bounded output size, and permitted tools. Only `research` may receive the configured Search Backend as a bounded supporting tool. The request does not expose the whole Run Bundle by default.

Search returns normalized source IDs, URLs, titles, provider excerpts, and citation metadata. The `fetch` method is reserved for a future factual-mode capability; v0 adapters reject direct page fetching and use provider-grounded excerpts only. Any later independent HTTP implementation must pin the validated connection address, re-check redirects and peers, reject loopback/private/link-local destinations, send no ambient credentials or cookies, enforce MIME/byte/time/redirect limits, and parse content without executing scripts. All source content remains untrusted prompt data.

The Speech request carries one Scene's exact spoken text, Voice Profile reference, Output Language, delivery audio specification, optional preceding/following text context, and a deterministic seed only if supported. Provider-specific voice controls stay in adapter settings. The result declares the actual sample rate, channels, duration, and timing precision rather than letting the orchestrator assume them.

Alignment receives the exact canonical transcript plus audio. ASR text can help establish timing but can never replace the Narration Script. Reconciliation reports unmatched spans and fails below a configured coverage threshold.

## Backend descriptor and live probe

Static declarations and live readiness are separate. A `BackendDescriptor` contains:

- stable Backend ID and protocol capabilities;
- provider, model ID, pinned revision or API alias policy;
- cloud/local execution and native-Windows/WSL runner support;
- supported languages and input modalities;
- features such as structured output, tool calls, reference images, voice cloning, timing precision, and seed support;
- hard limits such as context, image dimensions, and maximum audio/music duration;
- required environment-variable names and prepared model assets;
- license/terms metadata and compatible Usage Purposes;
- approximate VRAM/RAM/disk requirements and whether the GPU must be exclusive.

A `ProbeReport` records what is true now: executable and model presence, credential availability, a lightweight API/model-access check when allowed, GPU/runtime compatibility, FFmpeg capabilities, and observed limits. Setup may change readiness. Preflight only reads and probes it.

The Backend registry is ordinary static Python data. Dynamic discovery, entry points, arbitrary third-party plugins, and config-driven imports are out of scope for v0.

## Workflow Task assignment

A Run Profile maps each fixed Workflow Task ID to exactly one compatible Backend ID. Several text tasks can point to one Structured Text Backend, but their instructions, schemas, limits, and usage records remain separate. `research` alone may also call the separately bound `search` support capability; that binding is counted and persisted like every other Backend call.

| Task ID | Protocol | Required when |
| --- | --- | --- |
| `search` | Search | live research is enabled; supporting binding for `research` |
| `research` | Structured Text | always |
| `ideate` | Structured Text | always |
| `select` | Structured Text | always |
| `outline` | Structured Text | always |
| `script_draft` | Structured Text | always |
| `review_story` | Structured Text | always |
| `review_spoken` | Structured Text | always |
| `review_constraints` | Structured Text | always |
| `script_revision` | Structured Text | always |
| `claim_inventory` | Structured Text | factual mode; repeated after any Duration Repair |
| `factual_review` | Structured Text | factual mode only |
| `narration_synthesis` | Speech | always |
| `duration_repair` | Structured Text | measured narration misses its accepted band |
| `caption_alignment` | Alignment | captions or cadenced Shots need timing and TTS timing is insufficient |
| `visual_plan` | Structured Text | `still_image` Runs |
| `remotion_rhythm` | Structured Text | `remotion_explainer` Runs; called once for the canonical Shot schedule |
| `remotion_direction` | Structured Text | `remotion_explainer` Runs; called separately for each Shot |
| `remotion_asset_select` | Structured Text | binding required for `remotion_explainer`; called only when more than one eligible candidate exists |
| `image_prompt_compile` | Structured Text | `still_image` Runs; output is bound to the selected Image Backend |
| `image_generate` | Image | all still-image Runs; Remotion fallback or targeted repair when resolved media is unavailable or unsuitable |
| `visual_review` | vision-capable Structured Text | selected quality/profile requires it; individual stills or composed Remotion proxy frames |
| `music_brief` | Structured Text | music is enabled |
| `music_generate` | Music | music is enabled |

`image_prompt_compile` is a real, separately selectable Workflow Task rather than hidden string concatenation. It receives provider-neutral visual artifacts plus a bounded descriptor for the target Image Backend and emits a validated Image Request for that Backend. Its own Backend/model, instructions, and output hash are persisted. A deterministic compiler Backend may implement the same task for simple/test renderers.

The Remotion task boundary is deliberately smaller. The host first creates canonical IDs, word anchors,
times, frame ranges, and bounded narration excerpts, preferring sentence and clause boundaries that fit
the configured Shot bounds. `remotion_rhythm` receives that immutable schedule once and returns only an
ordered Beat function, attention level, evidence flag, and eligible section-start flag per supplied
Shot. It cannot change IDs, timing, copy, templates, assets, or rights. The contract limits high
attention, rejects four repeated functions and consecutive evidence requirements, and reserves
readable source evidence for eligible interior Shots. `remotion_direction` then
receives one Shot and its assigned Beat at a time
and returns only `template`, `headline`, `supporting_text`, `body_lines`, `asset_kind`, `asset_query`,
and `sfx`. The host derives purpose, motion, transition, generated-image prompt, asset/source IDs, URLs,
paths, licenses, and renderer settings. The raw seven-field response is preserved, then a deterministic
host canonicalizer resolves position-owned templates, removes stale fields that the fixed component
cannot consume, bounds layout copy, and downgrades templates whose body or media prerequisites are
missing. Unknown enums, malformed types, and extra fields remain hard validation failures. In factual
mode, deterministic validation requires every visible copy field to be a contiguous phrase from the
already reviewed narration excerpt. When stock search
returns multiple eligible records, `remotion_asset_select` receives those records and may return only
one supplied `candidate_id`; no model may create a URL or rights claim.
Visual-only Brief requirements are then assessed individually against the complete assembled plan;
promotion fails when a `must_include` or `avoid` rule remains unresolved. The same aggregate gate runs
for immutable dashboard edit overrides.
Deterministic plan-quality validation also enforces readable code/source dwell, nonconsecutive source
screenshots, bounded repetition and rapid-cut density, scene-grounded evidence when a Beat requires it,
an intentional low-attention or breathing Beat in longer plans, and section transitions that match the
frozen rhythm plan.
Evidence sources are offered as screenshot options only when their exact host or parent domain appears
in the frozen `remotion_source_screenshot_hosts` trust allowlist. The resolver checks that policy again
before launching the restricted capture browser, which applies the same policy to every redirect,
frame, and subresource request. Offline configuration rejects that allowlist and the resolver has an
independent last-mile offline guard. Each host-built edit Shot persists only the eligible source IDs for
its own Scene; the Dashboard cannot create a source-screenshot child without one of those IDs.

Deterministic validation, shot allocation, audio normalization, rendering, and media QC are internal
stages rather than model tasks. The still-image production path always uses the configured Image
Backend. Remotion may instead use an owned file, allowlisted stock asset, or evidence-linked screenshot,
but the configured Image Backend remains the explicit fallback and repair path. The deterministic stick
Backend exists only as an explicit test/manual backend and is never a fallback.

## Local runner boundary

The orchestrator must not import every CUDA stack. The Structured Text adapter owns one stock `llama-server.exe` through a stdlib control worker, while Diffusers, VoxCPM, Parakeet, or ACE-Step use dedicated Python workers. The common lifecycle is what matters:

1. acquire the single exclusive GPU lease;
2. stop an incompatible resident runner;
3. start and probe the selected native-Windows or WSL runner;
4. execute a bounded request using files under the workspace;
5. validate JSON and media before promoting outputs;
6. reuse the runner for adjacent requests using the same model;
7. terminate it at the next model-family boundary or after failure.

There is no VRAM bin-packing. One GPU model family owns the card at a time. A single path utility maps workspace paths into WSL; runners may not write outside the Run's work directory and model cache.

`local-llm.toml` is a typed Setup input, not a workflow plugin. It declares a stable profile ID, model/repository IDs, full commit and SHA-256 values, license, target GGUF, optional compatible drafter GGUF, stock llama.cpp commit and executable hash, context/batch settings, and `none` or `draft-mtp`. Setup copies and hashes only those assets. The runtime hard-forces loopback, a generated API key, one slot, and controlled model/host/port arguments. Provider output remains subject to the ordinary JSON Schema plus domain-model validation.

Remotion is a deterministic renderer, not a model Backend. Setup copies the wheel-bundled scaffold when
needed, installs the exact npm lock, validates TypeScript, and prepares a pinned Chrome Headless Shell.
Preflight hashes the Node executable, complete installed Node dependency tree, complete browser runtime,
composition sources, and local media-library inputs. Remotion subprocesses receive only an allowlisted
operating-system/tool environment, never provider or model API keys. Generated bundles are built in a
private staging directory, atomically published, and verified from an exact per-file attestation before
every reuse. The frozen runtime snapshot is verified again immediately before every proxy or final
Remotion render. Generate refuses an unprepared or changed runtime and never downloads Node packages or
a browser.

## Item checkpoints

Stages that fan out do not wait for the whole batch before preserving valid work. TTS items key by Scene;
image generation, visual review, and regeneration key by Scene or Shot according to visual mode. Every
item writes to its own work directory, validates, and atomically promotes a manifest and media hash. The
aggregate stage becomes complete only when every required item is promoted. Resume reuses valid items
and schedules only missing or invalid items; stage-level hashes still determine downstream completion.
Remotion asset resolution also checkpoints by Shot. Its composed proxy review evaluates three frames per
Shot and can replace at most one generated asset per failed Shot before the second, terminal review.

## Run immutability and rerun

Run creation freezes the resolved profile, task instructions, prompts, JSON Schemas, and pricing metadata into the Run Bundle. `resume` uses those frozen assets and can retry failed/unstarted work or reuse valid item/stage checkpoints; it does not reinterpret the Run through whatever defaults happen to be installed later.

The only operation that intentionally invalidates completed work is `rerun RUN_DIR --from STAGE [--config config.toml]`. It creates a new Run with `parent_run_id` and `fork_stage`, copies or content-addresses independently validated upstream artifacts without changing their provenance, and resolves current or supplied assets for the chosen stage onward. If supplied configuration would invalidate an earlier artifact, the command refuses the requested fork point and reports the earliest valid stage rather than silently moving it. It displays the new plan and cost reservation before any Backend call and never mutates the parent Run.

Dashboard Remotion edits follow the same rule: an edit creates a validated child Run at `visual-plan`
with a complete frozen override rather than patching the parent artifact. Optional manual asset approval
creates a child at `visual-review` whose approval payload hashes the complete approved asset records.
The workflow checks those hashes again after any visual-review regeneration; changed assets require a
new approval child before rendering.

## Errors, retries, and omission

Adapters return typed failures: `not_ready`, `unsupported`, `transient`, `invalid_output`, `policy_refusal`, `budget_exceeded`, or `internal`. The orchestrator decides retries; adapters do not hide them.

- Transient cloud failures receive bounded backoff.
- Invalid structured text receives bounded validation repair: at most one retry for ordinary cloud
  tasks and at most two for local or length-sensitive text tasks. A leaked host label inside a
  single-field `spoken_text` response is repaired from only that invalid text plus the deterministic
  error; the complete Script input is not resent.
- Script revision and Duration Repair are explicit Workflow Tasks, not generic retries.
- Final-quality visual review may cause one targeted image regeneration batch. Regenerated images are reviewed once more, cannot trigger another regeneration, and must pass or fail according to the declared policy.
- OOM and unsupported-capability errors fail with the exact setup or override action; they do not silently select a smaller model.
- An enabled optional feature either succeeds or follows the declared Failure Policy. Any omission is recorded in the Delivery Manifest and shown to the user.

## Provenance and cost

Every stage records the Backend/model revision, prompt/task/schema versions, input and output hashes, attempts, elapsed time, warnings, and usage units. Cloud-cost guarding uses a dated pricing snapshot and bounded request maxima. If a provider cannot supply a defensible upper bound, the request needs an explicit configured reservation before it starts.

Local stages record runtime, runner lifecycle data, optional worker-reported peak VRAM, and model asset hashes. The llama-server worker records baseline/load/peak/post-exit aggregate GPU observations and requires its managed PID to disappear; aggregate memory tolerance is advisory on Windows WDDM. Model licenses and provider terms are checked against the declared Usage Purpose during Setup and Preflight; they are not inferred from a filename. Remotion media rights are independently parsed, filtered, and recorded for each resolved asset at execution time, then emitted as delivery credits and warnings.
