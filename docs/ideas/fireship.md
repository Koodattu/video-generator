# Recommended architecture

Build this as an **AI-directed video renderer**, not as a model that directly generates an MP4.

The LLM should make editorial decisions:

* what to say,
* where each cut occurs,
* what visual function each shot serves,
* which asset or animation appears,
* what text is emphasized,
* and when jokes, screenshots, diagrams, or code appear.

A deterministic engine should then convert those decisions into frames.

For this kind of fast technical explainer, my default stack would be:

> **LLM director → structured timeline JSON → Remotion scene components → FFmpeg post-processing → MP4**

Remotion is particularly suitable because it treats videos as parameterized React applications. You can create reusable components, pass them JSON props, preview them in a browser, and render them programmatically. Its documentation also covers multi-track timelines, captions, transitions, server-side rendering, and cloud rendering. ([Remotion][1])

The overall pipeline would look like this:

```text
Topic or source material
        ↓
Research agent
        ↓
Claims, sources and quotations
        ↓
Scriptwriter agent
        ↓
Narration script with stable beat IDs
        ↓
TTS engine + word timestamps
        ↓
Director agent
        ↓
Structured edit plan / timeline JSON
        ↓
Asset retrieval and generation
        ↓
Remotion composition
        ↓
Low-resolution preview
        ↓
Visual/audio QC agent + human review
        ↓
Final Remotion render + FFmpeg encode
```

## 1. The LLM should produce an edit plan, not video frames

The most important architectural decision is to constrain the LLM.

Do not ask it:

> “Create a two-minute video about PostgreSQL.”

Ask it to produce something closer to an edit decision list:

```json
{
  "video": {
    "fps": 30,
    "width": 1920,
    "height": 1080
  },
  "scenes": [
    {
      "id": "scene-001",
      "startAnchor": {
        "wordId": "word-0001",
        "offsetMs": -100
      },
      "endAnchor": {
        "wordId": "word-0024",
        "offsetMs": 180
      },
      "purpose": "hook",
      "template": "HeadlineZoom",
      "assetId": "asset-postgres-logo",
      "headline": "Your database is lying to you",
      "motionPreset": "rapid-punch-in",
      "transitionOut": "hard-cut",
      "claimIds": ["claim-001"]
    },
    {
      "id": "scene-002",
      "startAnchor": {
        "wordId": "word-0025"
      },
      "endAnchor": {
        "wordId": "word-0059"
      },
      "purpose": "explanation",
      "template": "DiagramFlow",
      "props": {
        "nodes": ["Application", "Connection pool", "PostgreSQL"],
        "activeNodeAtWord": {
          "Connection pool": "word-0038",
          "PostgreSQL": "word-0049"
        }
      },
      "transitionOut": "hard-cut",
      "claimIds": ["claim-002", "claim-003"]
    },
    {
      "id": "scene-003",
      "startAnchor": {
        "wordId": "word-0060"
      },
      "endAnchor": {
        "wordId": "word-0071"
      },
      "purpose": "comic-relief",
      "template": "MemeCutaway",
      "assetId": "asset-panicking-server",
      "maxDurationMs": 1300
    }
  ]
}
```

The schema is the contract between the AI and renderer.

Use strict JSON Schema or Zod validation. Major cloud model APIs now support schema-constrained structured output, which is substantially safer than extracting JSON from ordinary prose. ([OpenAI Developers][2])

Validation should reject:

* unknown scene templates,
* nonexistent assets,
* unsupported transitions,
* invalid word anchors,
* negative durations,
* unreadably short text scenes,
* conflicting full-screen layers,
* and unlicensed external assets.

The LLM should never be the final authority on whether its output is renderable.

## 2. Anchor edits to spoken words, not hard-coded seconds

Voice synthesis will not produce the exact same duration every time. Changing one sentence can shift every later cut.

Therefore, the timeline should be anchored to the narration:

```json
{
  "wordId": "word-0087",
  "text": "transaction",
  "startMs": 23154,
  "endMs": 23561
}
```

The timeline compiler converts timestamps into frames:

```text
frame = round(timestampMs / 1000 × fps)
```

A visual can then enter on a particular word:

```json
{
  "type": "label",
  "text": "ACID",
  "enterAtWordId": "word-0087",
  "exitAtWordId": "word-0103"
}
```

This has several advantages:

* Re-recording one narration section does not require the LLM to recalculate every timestamp.
* Code highlights can appear precisely when the relevant function is mentioned.
* Captions use the same timing data.
* Memes and sound effects can land directly on punchline words.
* The editor can change TTS providers without rewriting the edit plan.

For TTS, synthesize narration in chunks of approximately one to three sentences. Each chunk should have a stable ID. This lets you regenerate a weak line without reproducing the entire voiceover, while avoiding the disconnected prosody that can result from generating every sentence completely independently.

## 3. Remotion as the main composition engine

A Remotion project is effectively a React application whose state changes by frame number.

Your renderer would contain a library of controlled scene components such as:

```text
ColdOpen
HeadlineZoom
SourceScreenshot
SourceQuote
LogoMontage
CodeReveal
CodeDiff
TerminalPlayback
DiagramFlow
ArchitectureLayers
MetricCounter
ComparisonSplit
TimelineHistory
KineticKeyword
BrowserDemo
MemeCutaway
StockVideoCutaway
ChapterBumper
EndCard
```

The director LLM selects components and fills their props.

For example:

```tsx
<CodeReveal
  codeAssetId="code-example-17"
  language="typescript"
  reveal={[
    {line: 1, atWordId: "word-113"},
    {line: 2, atWordId: "word-120"},
    {line: 5, atWordId: "word-134"}
  ]}
  focusLineAtWord={{
    "word-127": 2,
    "word-139": 5
  }}
/>
```

The renderer resolves the word IDs into frames, loads the code asset, calculates animations, and produces the shot.

This is preferable to having the LLM invent arbitrary React and CSS for every shot. A fixed component library gives you:

* visual consistency,
* predictable rendering,
* known text limits,
* controlled motion,
* easier debugging,
* and fewer catastrophic layout failures.

Remotion can render video or individual frames programmatically, making it practical to generate preview stills before committing to a full render. It also supports parameterized compositions, so the same composition can render many different videos from different input JSON. ([Remotion][3])

### Controlled code generation as an escape hatch

The component library will not cover every technical idea. For genuinely novel diagrams, the LLM can be permitted to create a new scene component, but only through a sandboxed process:

1. The LLM generates one isolated TypeScript/React component.
2. Imports are restricted to an allowlist.
3. Network and filesystem access are disabled.
4. TypeScript and schema checks run.
5. A single still frame is rendered.
6. A multimodal model checks the result.
7. A short proxy clip is rendered.
8. The component is either accepted, repaired, or rejected.

The agent should not rewrite the central renderer or dependency configuration for each video.

A useful ratio would be:

* 80–90% predefined components,
* 10–20% generated or specially implemented scenes.

That still permits variety without making every render an unpredictable software-development task.

## 4. What FFmpeg should do

FFmpeg should be the media-processing layer underneath or after Remotion.

Use it for:

* probing media duration, resolution and codecs,
* trimming and looping stock footage,
* converting GIFs to MP4 or WebM,
* normalizing frame rates,
* resizing and cropping,
* extracting audio,
* joining narration segments,
* loudness normalization,
* music ducking,
* generating waveform information,
* final encoding,
* and creating proxy versions.

FFmpeg supports complex filtergraphs, overlays, concatenation and broad media conversion. ([FFmpeg][4])

I would not use raw FFmpeg commands as the primary motion-graphics language. It can technically compose almost everything, but complicated filtergraphs become difficult to generate, inspect, preview and maintain. Remotion should determine the visual composition; FFmpeg should handle media plumbing and final processing.

Also, convert downloaded GIFs into an ordinary video format before composition. Animated GIF is inefficient and frequently has inconsistent timing or transparency behavior.

## 5. Motion Canvas for custom technical animations

Motion Canvas is another TypeScript-based animation system designed around programmatic, voiceover-synchronized explanatory animations. It is particularly useful for:

* animated graphs,
* architecture diagrams,
* node-and-arrow systems,
* data flows,
* algorithm visualizations,
* and custom vector sequences.

It is more specialized toward informative animation than general non-linear video editing, so I would use it as an optional scene generator rather than as the complete editing environment. A Motion Canvas sequence could render to an intermediate transparent or full-frame clip and then be placed in the Remotion timeline. ([motioncanvas.io][5])

A reasonable division is:

* **Remotion:** overall timeline, text, code, screenshots, stock clips, audio and transitions.
* **Motion Canvas:** unusually complex technical diagrams.
* **FFmpeg:** conversion, audio treatment and final encoding.

## 6. The editorial agent system

Do not use one giant prompt that researches, writes, sources assets and edits simultaneously. Split the process into explicit passes.

### Researcher

Input:

```text
Topic: Why database connection pools fail under serverless workloads
Audience: intermediate web developers
Maximum duration: 150 seconds
```

Output:

* verified claims,
* source URLs,
* relevant quotations,
* publication dates,
* uncertainty notes,
* and potential visual evidence.

Each claim receives a stable ID:

```json
{
  "claimId": "claim-012",
  "statement": "A connection can outlive the serverless invocation that created it.",
  "sourceId": "source-04",
  "confidence": "high",
  "visualEvidence": {
    "type": "documentation-screenshot",
    "selectorHint": "#connection-management"
  }
}
```

This creates traceability between narration and evidence.

### Scriptwriter

The writer receives only accepted research and the show’s style guide.

It returns narration divided into beats:

```json
{
  "beatId": "beat-009",
  "narration": "The function disappears. The database connection does not.",
  "function": "reversal",
  "claimIds": ["claim-012"],
  "estimatedSeconds": 4.1,
  "visualSuggestions": [
    "function box vanishes while connection line remains",
    "brief abandoned-process reaction cutaway"
  ]
}
```

The script should include intentional functions such as:

* hook,
* setup,
* definition,
* historical context,
* mechanism,
* evidence,
* example,
* reversal,
* joke,
* qualification,
* conclusion.

### TTS and alignment

The accepted script is synthesized. The system stores:

* audio per segment,
* word timestamps,
* phoneme or character timestamps when available,
* actual duration,
* pronunciation overrides,
* and pauses between segments.

Cartesia exposes word- and phoneme-level timestamps. ElevenLabs offers timestamped synthesis with character timing. OpenAI’s TTS supports instruction-based control over delivery and requires disclosure to end users that the voice is AI-generated. ([Cartesia Docs][6])

For a local TTS engine that does not return reliable timing, run forced alignment afterward. WhisperX provides word-level alignment, while Montreal Forced Aligner is designed to align known transcripts with recorded audio. ([GitHub][7])

### Director

The director receives:

* the final script,
* word timings,
* source records,
* available scene templates,
* available assets,
* brand rules,
* pacing constraints,
* and the previous few scenes.

It assigns a visual purpose to each beat:

```text
Evidence
Explanation
Technical proof
Entity identification
Comparison
Comic punctuation
Transition
Breathing room
```

Then it selects a scene template.

A useful set of rules might be:

```text
Never use two full-screen source screenshots consecutively.
Do not show a meme for more than 1.5 seconds unless it contains dialogue.
Keep important code visible for at least 2.5 seconds.
Do not use more than three rapid cutaways in ten seconds.
Show evidence when a concrete factual claim is introduced.
Do not place body text over visually busy footage.
Do not repeat the same animation preset within three scenes.
Use hard cuts by default.
Reserve animated transitions for section changes.
```

This is how you reproduce the underlying editing grammar without copying another channel’s exact identity.

### Asset selector

The asset selector translates abstract requests into actual assets:

```json
{
  "requestId": "asset-request-41",
  "query": "overloaded server rack warning lights",
  "purpose": "comic exaggeration",
  "assetTypes": ["stock-video", "gif"],
  "maximumDurationMs": 1800,
  "requiredLicense": "commercial-reuse",
  "preferredFraming": "center subject",
  "avoid": ["visible logos", "watermarks", "text"]
}
```

Candidate assets are downloaded, normalized, assigned immutable IDs and stored with their rights information.

### Critic

The critic receives a proxy render or representative frames and checks:

* whether visuals correspond to narration,
* whether text is clipped,
* whether code is readable,
* whether sources are visible at appropriate moments,
* whether a joke interrupts an important explanation,
* whether the same visual pattern is being overused,
* whether there are black or missing frames,
* and whether the pace is excessively dense.

The critic should propose a bounded patch to the edit plan rather than regenerate the entire video.

## 7. Browser screenshots and source evidence

Playwright is useful for automatically producing visual evidence.

Your capture tool could accept:

```json
{
  "url": "https://example.com/documentation",
  "selector": "#connection-pooling",
  "viewport": {
    "width": 1440,
    "height": 900
  },
  "highlightText": "maximum connections",
  "hideSelectors": [
    ".cookie-banner",
    ".navigation",
    ".chat-widget"
  ]
}
```

The tool would:

1. Load the page.
2. Wait for it to stabilize.
3. Hide irrelevant interface elements.
4. Scroll the relevant section into view.
5. Add an outline or highlight in the DOM.
6. Capture either the element or page region.
7. Store the URL, title and capture date.
8. Return an internal asset ID.

Playwright supports full-page and clipped screenshots. It also has MCP-based browser-control options, although I would still expose a narrower capture-specific tool in a production system. ([Playwright][8])

For browser demonstrations, the system can record a deterministic scripted session:

```text
Open application
Click “Create project”
Type example input
Run command
Wait for output
Move pointer to result
```

That is significantly more reliable than asking an agent to improvise browser actions while recording the final take.

## 8. GIFs, stock footage and asset rights

Possible sources include:

* GIPHY for GIF search,
* Pexels for stock images and footage,
* Wikimedia Commons for licensed historical or technical media,
* owned meme and reaction packs,
* generated illustrations,
* and direct screenshots from cited sources.

GIPHY’s API requires attribution in applications using its content. Pexels provides an API for photos and video and permits broad free use under its license, subject to restrictions. Wikimedia Commons exposes media programmatically, but individual assets can have different attribution and share-alike requirements. ([GIPHY Developers][9])

Do not assume that an accessible GIF API automatically grants unrestricted commercial reuse. Store an asset record such as:

```json
{
  "assetId": "asset-178",
  "provider": "wikimedia-commons",
  "originalUrl": "...",
  "creator": "...",
  "license": "CC BY-SA 4.0",
  "requiredAttribution": "...",
  "downloadedAt": "2026-07-13T12:44:00Z",
  "contentHash": "sha256:...",
  "approvedForCommercialUse": true
}
```

One current implementation detail: the Tenor API was discontinued on June 30, 2026, so it should not be selected as the foundation for a new system. ([The Verge][10])

For a robust commercial workflow, the safest hierarchy is:

1. Owned or commissioned assets.
2. Procedurally generated graphics.
3. Properly licensed stock.
4. Clearly licensed Commons material.
5. Provider-specific GIF content only where the intended usage is permitted.

## 9. TTS choices

### Cloud-first

The most important capabilities are not just voice quality. You also need:

* stable pronunciation,
* style control,
* low variation between retakes,
* timestamps,
* predictable latency,
* and commercial terms suitable for publishing.

A practical comparison:

| Option     | Strongest use                                              | Timing                                      |
| ---------- | ---------------------------------------------------------- | ------------------------------------------- |
| Cartesia   | Automated editing and low-latency synthesis                | Word and phoneme timestamps                 |
| ElevenLabs | Expressive narration and broad voice tooling               | Timestamped output available                |
| OpenAI TTS | Instruction-controlled delivery and simple API integration | Align afterward if exact timing is required |

For this format, timing quality may matter as much as marginal differences in voice realism.

Create an original voice identity. The voice prompt might specify:

```text
Controlled technical narrator.
Moderately fast but clearly articulated.
Low emotional variance.
Dry treatment of humorous lines.
No imitation of a known individual.
Avoid announcer energy.
Use brief pauses before reversals and conclusions.
```

### Local-first

Current local candidates include Qwen3-TTS and Kokoro. Qwen3-TTS provides instruction-controlled and multilingual variants, including voice-design and custom-voice configurations. Kokoro is a much smaller, lightweight model useful for fast local drafts or simpler production needs. Use cloning only with a voice you own or have permission to reproduce. ([Hugging Face][11])

A local pipeline could be:

```text
Script segment
  → local TTS
  → WAV
  → WhisperX alignment
  → word timing JSON
  → audio cleanup
  → timeline compiler
```

You could use local TTS for rough previews and switch to a higher-quality cloud voice for approved final narration. Because the timeline is word-anchored, the system can recalculate the edit automatically.

## 10. LLM choices

The workflow does not fundamentally depend on a specific model. It depends on:

* reliable schema adherence,
* tool use,
* long-context script handling,
* coding ability for custom scenes,
* visual understanding for render review,
* and the ability to follow a detailed editorial specification.

### Cloud model

A strong cloud model is likely to perform best for:

* final scriptwriting,
* deciding what information to omit,
* generating coherent comedy,
* directing complex visual sequences,
* and reviewing a proxy render.

Use structured output for the script, claims and timeline. Do not rely on a prose response that is later parsed with regular expressions.

### Local model

On a 24 GB GPU, a quantized model in the 27–30B class is a plausible director and code-generation option. Qwen3-Coder-30B-A3B-Instruct supports tool calling and documents local deployment through systems including llama.cpp, Ollama and LM Studio. A 4-bit quantized build is a reasonable configuration to test, although actual memory headroom depends on the quantization implementation, context size and KV cache. ([Hugging Face][12])

A practical local configuration would be:

```text
Model: Qwen3-Coder-30B-A3B-Instruct, 4-bit
Runtime: llama.cpp or Ollama
Context: initially 16K–32K
Temperature:
  research extraction: 0.1–0.3
  timeline generation: 0.1–0.3
  script and humor: 0.6–0.9
  repair passes: 0.1–0.2
```

The same model can perform several roles through separate prompts, but do not place all roles into one context. Persist the intermediate artifacts to the database instead.

A hybrid system is probably the strongest arrangement:

* local model for routine shot planning, asset queries, metadata and repairs,
* cloud model for difficult scriptwriting and final editorial judgment,
* local TTS for previews,
* chosen cloud or local voice for final production.

## 11. MCP: useful, but not the renderer

MCP does not itself generate or edit video. It is a protocol through which the model can call tools.

You could expose tools such as:

```text
research.search
research.fetch_source

browser.capture
browser.record_demo

assets.search_stock
assets.search_gif
assets.search_commons
assets.generate_illustration
assets.get_metadata
assets.approve_license
assets.transcode

tts.synthesize
audio.align_words
audio.normalize
audio.mix

timeline.validate
timeline.calculate_frames
timeline.patch

render.still
render.scene_preview
render.proxy
render.final

qc.inspect_frames
qc.inspect_audio
qc.inspect_proxy

export.otio
export.project_archive
```

The preferred abstraction is high-level:

```json
{
  "tool": "render.scene_preview",
  "arguments": {
    "projectId": "project-12",
    "sceneId": "scene-08",
    "resolution": "960x540"
  }
}
```

Avoid giving the LLM unrestricted access to:

```text
shell.execute("ffmpeg ...")
filesystem.write("/anything")
npm.install("arbitrary-package")
```

That increases security risk and makes jobs difficult to reproduce.

Remotion has agent-oriented documentation and skills, and there are experimental Remotion MCP projects where a model writes React/Remotion code that is compiled and shown in an embedded player. The official Remotion MCP offering is currently more about exposing documentation and context than functioning as a complete autonomous production backend. ([Remotion][13])

For your own application, ordinary typed API functions may be simpler than MCP. MCP becomes valuable when you want the same toolset available to multiple external agents such as Codex, Claude Code, desktop assistants or IDE integrations.

A sensible architecture is:

```text
Internal TypeScript services
        ↑
Ordinary function/API calls
        ↑
Optional MCP adapter
        ↑
External AI agents
```

The internal services remain the source of truth. MCP is only another interface.

## 12. Alternative composition approaches

| Approach                   | Best use                                        | Main limitation                                                              |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| Remotion                   | Custom, branded, programmatic explainer videos  | Requires building and maintaining scene components                           |
| Motion Canvas              | Complex technical diagrams and vector animation | Not a complete NLE replacement                                               |
| FFmpeg only                | Conversion, compositing and audio processing    | Difficult as a high-level animation language                                 |
| Shotstack                  | Fast cloud MVP using JSON timelines             | Less control than a custom React renderer                                    |
| Creatomate                 | Template-driven cloud video generation          | Custom behavior remains platform-dependent                                   |
| JSON2Video                 | Straightforward API-based scene composition     | Better for conventional template videos than highly custom editorial grammar |
| Premiere/editor automation | Human finishing in an established NLE           | GUI/application state makes full automation more fragile                     |
| Text-to-video generation   | Short atmospheric or surreal cutaways           | Poor fit for exact code, typography, screenshots and deterministic timing    |

Shotstack, Creatomate and JSON2Video all expose JSON- or template-based cloud rendering APIs. They are viable when the goal is to reach a working MVP quickly without operating rendering infrastructure. ([Shotstack][14])

For a genuinely distinctive technical-explainer format, Remotion provides more control and avoids becoming constrained by a vendor’s template vocabulary.

## 13. Human editing and NLE export

The canonical project should remain JSON plus referenced assets. However, you can also export an editable timeline for finishing.

OpenTimelineIO represents tracks, clips, timing, transitions and markers while referring to external media. It is an interchange format, not a video renderer. ([opentimelineio.readthedocs.io][15])

A possible workflow is:

```text
AI-generated project JSON
        ↓
Remotion first cut
        ↓
OpenTimelineIO or editor-compatible interchange export
        ↓
Human polish in an NLE
```

This is useful when an editor needs to:

* replace a joke,
* fine-tune timing,
* change music,
* adjust audio automation,
* add manual masking,
* or perform complex color work.

Premiere also exposes plugin and document APIs through UXP, but I would not make a desktop NLE the central autonomous rendering service. ([Adobe Developer][16])

The renderer should be able to produce a complete video without opening a desktop application. NLE export should be an escape hatch for high-value videos.

## 14. Infrastructure

A TypeScript-oriented monorepo could be structured like this:

```text
apps/
  studio/              # Web interface and timeline preview
  api/                 # Projects, assets, scripts and jobs

workers/
  research/            # Web research and source extraction
  director/            # LLM calls and edit-plan generation
  browser/             # Playwright capture and demos
  render/              # Remotion + Chromium rendering
  media/               # FFmpeg transcode and audio jobs
  qc/                  # Visual and audio checks

services/
  local-ai/            # Local LLM, TTS and WhisperX endpoints

packages/
  video-schema/        # Zod/JSON Schema definitions
  scene-library/       # Remotion components
  timeline-compiler/   # Word anchors → frames
  asset-clients/       # Pexels, GIPHY, Commons, etc.
  brand-system/        # Typography, spacing, transitions
  prompt-library/      # Researcher, writer, director, critic
```

Data services:

```text
PostgreSQL
  Projects
  Sources
  Claims
  ScriptBeats
  VoiceSegments
  WordTimings
  Assets
  Scenes
  RenderJobs
  ReviewNotes

Object storage
  Source screenshots
  Voice audio
  Stock video
  Generated images
  Proxy renders
  Final renders

Queue
  Research jobs
  TTS jobs
  Capture jobs
  Render jobs
  QC jobs
```

BullMQ is a practical Redis-backed queue for an MVP, supporting retries and job flows. Temporal is better suited when workflows become long-running, involve many external APIs and must reliably resume after process failures. ([bullmq.io][17])

For the renderer, isolate Remotion and Chromium in a dedicated Node container. The remainder of the application can use whichever TypeScript runtime you prefer.

For cloud rendering, Remotion supports server-side rendering and deployment approaches including Lambda and Cloud Run. Remotion recommends Lambda for highly parallel rendering workloads. ([Remotion][18])

Check Remotion’s commercial license before building a paid service. Its free-use conditions depend partly on organization type and size; larger for-profit organizations require a company license. ([Remotion][19])

## 15. A practical review interface

The web application should not attempt to recreate all of Premiere.

The useful interface is narrower:

```text
Left:
  Script beats and claims

Center:
  Video preview

Right:
  Current scene template and properties

Bottom:
  Simplified scene timeline and voice waveform
```

For each scene, the user should be able to:

* replace the selected asset,
* change the template,
* adjust the word-anchored in/out points,
* modify on-screen text,
* disable a joke,
* choose another animation preset,
* approve source usage,
* and regenerate only that scene.

Remotion provides a browser player, and its documentation includes building multi-track timeline interfaces. It also offers a customizable timeline component as a separate commercial product. ([Remotion][20])

The edit plan remains ordinary structured data. Manual changes should modify the same JSON that the agent produces, rather than creating an unrelated manual-edit format.

## 16. Automated quality control

Before final rendering, generate:

* one still from the start, midpoint and end of each scene,
* a low-resolution proxy,
* an audio waveform,
* caption timing data,
* and a scene-by-scene asset manifest.

Run deterministic checks first:

```text
All assets exist.
Every scene has positive duration.
No unsupported fonts or components.
Text fits within safe bounds.
No scene is unintentionally blank.
No source screenshot is below minimum resolution.
Narration does not clip.
Music exists beneath intended sections.
Every externally sourced asset has rights metadata.
```

Then give the proxy and narration to a multimodal critic:

```text
Does each shot illustrate the narration?
Is the important text legible?
Are any shots visually repetitive?
Does the humor obscure a factual claim?
Are screenshots shown long enough to understand?
Do cuts correspond to semantic changes?
Is there an excessive run of rapid shots?
Does the visual emphasis match the spoken emphasis?
```

Limit automated repair loops. For example:

```text
Maximum two automatic revisions per scene.
After two failures, flag for human review.
```

Otherwise, agents can endlessly oscillate between competing stylistic changes.

## 17. Local-first configuration

For a 24 GB VRAM machine, I would begin with:

```text
Director LLM:
  Quantized Qwen3-Coder-30B-A3B
  llama.cpp or Ollama
  16K–32K context initially

TTS:
  Qwen3-TTS 1.7B for production experiments
  Kokoro for extremely fast previews

Alignment:
  WhisperX

Composition:
  Remotion
  Motion Canvas for selected diagrams
  FFmpeg

Browser:
  Playwright

Storage:
  PostgreSQL
  Local object directory or S3-compatible storage

Queue:
  Redis + BullMQ

Review:
  React web interface with Remotion Player
```

The GPU-heavy jobs can run sequentially:

```text
LLM planning
    ↓
TTS
    ↓
optional image generation
```

Remotion and media processing can run in separate worker processes. This prevents the render environment from depending directly on the local model runtime.

## 18. Cloud-first configuration

A straightforward cloud implementation would be:

```text
LLM:
  Strong cloud model with structured output and tool use

TTS:
  Cartesia or ElevenLabs
  OpenAI TTS where instruction-controlled delivery is preferred

Browser:
  Containerized Playwright worker

Composition:
  Remotion

Rendering:
  Remotion Lambda or Cloud Run

Storage:
  S3 or R2

Database:
  Managed PostgreSQL

Queue/workflow:
  BullMQ initially
  Temporal for more durable orchestration

QC:
  Multimodal cloud model inspecting proxy frames and video
```

For the quickest possible proof of concept, replace Remotion with Shotstack, Creatomate or JSON2Video. Once the editorial grammar becomes more custom, migrate the timeline schema to your own Remotion renderer.

## 19. What I would build first

The first useful version should be intentionally constrained:

### Version 1

* 60–150-second videos.
* One narration voice.
* Eight scene templates.
* Hard cuts and one section transition.
* Code, screenshots, diagrams, stock video and meme cutaways.
* Manual approval of retrieved assets.
* Word-timed captions.
* Local Remotion rendering.
* Human approval before publication.

The initial templates:

```text
KineticHook
HeadlineZoom
SourceScreenshot
CodeReveal
DiagramFlow
ComparisonSplit
MemeCutaway
Conclusion
```

### Version 2

* Automatic source screenshot capture.
* Pexels, GIPHY and Commons search.
* Asset-license records.
* Better animated diagrams.
* Automated proxy inspection.
* Music and sound-effect planning.
* Regeneration of individual scenes.
* Simplified visual timeline editor.

### Version 3

* LLM-generated custom Remotion components.
* Reusable learned shot patterns.
* Multiple aspect-ratio variants.
* Automatic horizontal-to-vertical reframing.
* OpenTimelineIO export.
* Cloud rendering.
* Multiple show formats and brand systems.

## What not to build first

Avoid these initial approaches:

* An agent moving a mouse around Premiere.
* An LLM generating arbitrary FFmpeg commands for the entire video.
* A single prompt that performs research, writing, editing and rendering.
* Pure text-to-video generation for technical UI or code scenes.
* Fully automatic downloading and publishing of random GIFs.
* Letting the model install arbitrary npm packages during rendering.
* Rebuilding the scene system for every episode.

The difficult part is not encoding an MP4. It is translating narration into good editorial decisions and consistently finding appropriate visual evidence.

The strongest implementation is therefore:

> **A structured, word-anchored edit plan; a curated library of reusable Remotion scenes; FFmpeg for media operations; optional Motion Canvas for complex diagrams; and narrow agent tools for research, capture, assets, TTS, rendering and QC.**

That provides the speed and visual density of an internet-native technical explainer while still being deterministic, editable and capable of developing its own visual identity.

[1]: https://www.remotion.dev/?utm_source=chatgpt.com "Remotion | Make videos programmatically"
[2]: https://developers.openai.com/api/docs/guides/structured-outputs?utm_source=chatgpt.com "Structured model outputs | OpenAI API"
[3]: https://www.remotion.dev/docs/api?utm_source=chatgpt.com "API overview | Remotion | Make videos programmatically"
[4]: https://ffmpeg.org/ffmpeg-filters.html?utm_source=chatgpt.com "FFmpeg Filters Documentation"
[5]: https://motioncanvas.io/docs/?utm_source=chatgpt.com "Introduction | Motion Canvas"
[6]: https://docs.cartesia.ai/api-reference/tts/sse?utm_source=chatgpt.com "Text-to-Speech (SSE) - Cartesia Docs"
[7]: https://github.com/m-bain/whisperx?utm_source=chatgpt.com "WhisperX: Automatic Speech Recognition with Word- ..."
[8]: https://playwright.dev/docs/screenshots?utm_source=chatgpt.com "Screenshots"
[9]: https://developers.giphy.com/docs/api/?utm_source=chatgpt.com "GIPHY API docs"
[10]: https://www.theverge.com/tech/959658/google-tenor-api-shutdown-gif-picker?utm_source=chatgpt.com "Google's killing off Tenor GIF searches in other apps"
[11]: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice · Hugging Face"
[12]: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct "Qwen/Qwen3-Coder-30B-A3B-Instruct · Hugging Face"
[13]: https://www.remotion.dev/docs/ai/skills "Agent Skills | Remotion | Make videos programmatically"
[14]: https://shotstack.io/docs/api/?utm_source=chatgpt.com "Shotstack v1 API Reference Documentation"
[15]: https://opentimelineio.readthedocs.io/en/latest/?utm_source=chatgpt.com "Welcome to OpenTimelineIO's documentation! - Read the Docs"
[16]: https://developer.adobe.com/premiere-pro/uxp/?utm_source=chatgpt.com "The Premiere UXP API"
[17]: https://bullmq.io/?utm_source=chatgpt.com "BullMQ - Background Jobs and Message Queue for Node.js ..."
[18]: https://www.remotion.dev/docs/lambda?utm_source=chatgpt.com "@remotion/lambda | Remotion | Make videos ..."
[19]: https://www.remotion.dev/license?utm_source=chatgpt.com "remotion/LICENSE.md at main"
[20]: https://www.remotion.dev/docs/building-a-timeline?utm_source=chatgpt.com "Build a timeline-based video editor"
