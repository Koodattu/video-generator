# Prompt system

## Objective

Prompts are production assets. Each one has a narrow job, a typed output, a version, fixed fixtures, and an evaluation rubric. The project should improve them through evidence from English and Finnish Runs, not by accumulating generic instructions in a giant system prompt.

The same task prompt can run on different conforming Structured Text Backends. Provider-specific API parameters and image-prompt syntax belong to adapters and compilers, not story artifacts.

## Prompt asset layout

The logical built-in prompt layout is shown below. In v0 these assets are compiled from `prompting.py` and frozen into each Run Bundle rather than stored as separate directories:

```text
prompts/
  research/v1/
  ideate/v1/
  select/v1/
  outline/v1/
  script-draft/v1/
  review-story/v1/
  review-spoken/v1/
  review-constraints/v1/
  revise-script/v1/
  claim-inventory/v1/
  duration-repair/v1/
  visual-plan/v1/
  remotion-direction/v1/
  remotion-asset-select/v1/
  image-prompt-compile/
    gpt-image-2/v1/
    gemini-3.1-flash-image/v1/
    flux.2-klein-4b/v1/
  visual-review/v1/
  music-brief/v1/
  factual-review/v1/
```

Each directory contains concise task instructions, its output JSON Schema, and only examples that encode a real product requirement or fix a measured failure. The task record persists the rendered prompt hash, prompt version, schema version, model response, and validation result.

Every Structured Text request should contain four clearly separated blocks:

1. stable role and objective;
2. hard rules, limits, and completion definition;
3. validated input artifacts serialized as data;
4. the exact output schema and rubric.

Research excerpts and web pages are untrusted data. Prompts state that instructions found inside sources must be ignored. Only the research task receives search tools, with actual query calls counted by the orchestrator.

## Shared generation rules

All creative tasks follow these rules unless their narrower task overrides one:

- Work directly in the selected Output Language; do not draft in English and translate into Finnish.
- Treat research as inspiration in fiction mode. Do not reuse distinctive source phrasing, characters, or plot structures.
- Return only the requested structured fields. Do not include hidden reasoning or a chain-of-thought narrative.
- Preserve supplied stable IDs only when the task schema contains an ID field. Never invent operational
  IDs, timings, frame numbers, URLs, paths, licenses, or renderer settings.
- Make uncertainty explicit in structured fields rather than inventing evidence.
- Stay inside the Audience Profile and Creative Brief.
- Prefer specific sensory or behavioral detail over abstract claims about emotion.
- Do not explain the story's lesson to the audience when the events can imply it.

Random temperature alone is not a variance strategy. Candidate generation deliberately varies conflict type, setting use, protagonist relationship, narrative shape, emotional color, and visual opportunities while respecting the same brief.

For 90-second and longer fiction, the outline opens inside concrete action or pressure, establishes
an event-based unanswered question, renews or complicates it once around the middle, and then pays it
off, recontextualizes an earlier moment, or leaves deliberate ending residue. This is a flexible
narrative design target, not a mandate for clickbait, a twist, a subplot, or nonlinear chronology.

## Editorial workflow

### Research

The research task begins from explicit questions derived from the Creative Brief. It may use at most the configured query limit and may retain at most the configured source limit. v0 uses only provider-grounded search results and bounded returned excerpts; it does not fetch arbitrary result pages. Its output is a Research Pack with:

- source title, URL, publisher/domain, retrieval time, and relevant language;
- a short paraphrase of the useful detail;
- motifs, setting details, vocabulary, cultural cautions, and clichés to avoid;
- in factual mode, atomic evidence records with source linkage and confidence.

It stops when it has sufficient diversity or reaches the hard limit. It does not browse recursively merely because another related fact might exist.

### Ideation

The ideation task returns a bounded format-specific Candidate Set. Five is the default and 1–10 is
accepted. Narrative candidates describe premise, pressure, turn, ending direction, emotional promise,
research inspirations, and visual opportunities. Explainer candidates instead describe a modern anchor,
central question, thesis, evidence ladder, human angle, landing, and accuracy risks.

Candidates must be meaningfully different. Renaming the protagonist or changing only the location does not count as another candidate.

### Selection

For v3 formats the selector receives the Creative Brief and bounded Research Pack as well as the
Candidate Set, so evidence strength and research responsibility are evaluated against the source
artifact rather than a candidate's self-description. The selector scores every candidate on a fixed
format-specific scale. Shared criteria include:

- fit to the Duration Budget;
- originality or hook strength without randomness for its own sake;
- complete narrative or explanatory progression;
- strength and variety of simple visuals;
- suitability for spoken narration;
- family-safe general-audience fit;
- responsible use of research.

It returns scores and a concise evidence-based rationale, then chooses one supplied concept. It cannot
generate a replacement candidate during selection.

### Outline

The outline task builds the whole causal or explanatory arc before prose. Each stable Scene includes its
purpose, what changes, emotional or logical beat, visual opportunity, provisional seconds, and continuity
obligations. Scene boundaries follow natural editorial structure; cadenced visual timing is planned later
without fragmenting the outline.

The sum of provisional Scene seconds equals the Duration Budget. That allocation guides writing; it is not accepted as measured timing.

### Narration draft

The writer receives the approved outline, a voice-rate estimate calibrated for the selected language/Voice Profile, and a word envelope for each Scene. It writes only words intended to be spoken. Visual directions, citations, headings, bracketed acting notes, markdown, and review comments live in other fields or artifacts.

Good spoken narration is operationalized as:

- sentences that remain understandable on first hearing;
- varied but controlled sentence and clause length;
- natural contractions and colloquial rhythm where appropriate to the language and tone;
- concrete verbs and images rather than adjective stacks;
- names, numbers, abbreviations, and foreign words written or annotated for reliable pronunciation;
- transitions motivated by events, not repeated summary phrases;
- enough breathing space for suspense and emotion without filler.

Common failure patterns to score include generic throat-clearing, false profundity, constant rhetorical questions, repetitive three-part lists, overexplaining motives, announcing a moral, interchangeable characters, convenient coincidences, repeated recap, and stock phrases such as “little did they know.” These are evaluation signals, not a global blacklist: an otherwise fitting phrase is not rejected mechanically.

For Finnish, fixtures should specifically test natural case selection, clitic use, spoken rhythm, compound words, number pronunciation, anglicisms, and avoidance of translated-English syntax. English and Finnish use the same artifact schema but may use language-specific guidance and examples when evaluation demonstrates a need.

### Reviews and revision

Three review tasks inspect the same draft independently:

| Review | Looks for | Must not do |
| --- | --- | --- |
| Story/originality | causal gaps, weak turns, generic beats, emotional payoff, research-copy risk | rewrite the script |
| Spoken language | listenability, sentence load, rhythm, pronunciation, repetition, language naturalness | change story facts without flagging them |
| Constraints/safety/continuity | duration risk, brief/audience violations, Scene identity drift, factual claims, missing setup/payoff | silently waive a hard rule |

Each finding has severity, Scene ID, evidence from the draft, and a concrete recommendation. The host
groups findings by Scene and asks for one complete replacement `spoken_text` only where an edit is
needed. It preserves the title, Scene ID, order, pause, unchanged Scenes, and finding disposition
bookkeeping itself. The model is never asked to rebuild the complete Script JSON for a local edit.
Each changed Scene is independently rechecked against its original findings. Ordinary story/spoken
findings receive the current Scene plus bounded adjacent context. A brief-constraint finding instead
receives the original trusted brief constraint, exact finding, and complete revised Script, with no
adjacent-context field, because the constraint may span Scenes and the judge must not invent an
unrelated continuity requirement. When that finding includes one explicit full-Scene replacement, the
host uses its word count only to aim at the largest feasible compressed repair inside the residual
Script envelope; the quoted model text is never promoted directly. If a
check rejects the first bounded repair, one final micro-call receives only the current Scene, unresolved
recommendations, and the rechecker's concrete explanation; a second recheck is terminal.

Story and spoken-language reviewers receive a spoken-script-only view of the Brief and Outline. The host
removes visual-only `must_include`/`avoid` entries and `visual_opportunity` fields before those calls;
renderer requirements remain available to visual planning and Remotion direction instead of becoming
unnecessary narration edits.

Initial drafting follows the same boundary: one strict call per Outline Scene returns only
`spoken_text`, and the host assembles the Script. Deterministic checks validate IDs/order, nonempty
spoken text, forbidden markup, configured limits, and required fields. They also reject a trailing
serialization-like host fragment such as `pause_after_seconds 0.35` when a model accidentally appends
it to narration, while allowing identifiers inside a genuine technical explanation. That specific
failure receives a small correction call containing only the invalid text and validation error; it
does not resend or ask the model to reconstruct the Script contract.

### Duration Repair

After TTS, a deterministic allocator selects only the Scenes that can close the measured duration
delta. Each repair call receives one Scene, bounded adjacent context, a feasible host-calculated word
range, and allowed factual evidence when applicable. It returns only one replacement `spoken_text`;
the host retains IDs, pauses, timing measurements, counts, and estimates. Only changed clips are
resynthesized.

## Visual prompt architecture

In the `still_image` branch, the image model never receives the Narration Script and a vague request to
“make a matching image.” The visual pipeline has four layers:

1. `VisualBrief`: what this Scene or timed Shot means and depicts;
2. `StyleProfile`: how every image should look;
3. `CharacterIdentity`: which signature traits must persist;
4. `ImageRequest`: the Backend-specific compiled prompt, references, size, seed, and settings.

An illustrative Visual Brief shape is:

```json
{
  "scene_id": "scene-003",
  "story_moment": "Aino notices that the frozen lantern is blinking in reply",
  "subjects": ["Aino", "blue tin lantern"],
  "action": "Aino crouches and shields the tiny light from falling snow",
  "emotion": "surprised, cautious hope",
  "environment": "nearly empty snowy path at dusk",
  "composition": "medium-wide view, Aino left, lantern lower-right, clear silhouette",
  "must_show": ["red triangular scarf", "three blue light marks"],
  "must_avoid": ["written words", "crowd", "photorealism"],
  "character_ids": ["character-aino"]
}
```

The structured brief remains provider-neutral and may be authored from Finnish narration. The
compiler always expresses both the final prompt and negative prompt in English because the selected
image Backends document or empirically show stronger English prompting; the semantic source and
evaluation remain language-independent. Image-prompt compilation therefore uses English as its
explicit task output language even when the Run narration language is Finnish.

### Built-in `ms_paint_stick` Style Profile

The initial profile should encode these visual properties as structured fields rather than one decorative paragraph:

- white or nearly white raster canvas;
- very simple stick characters with round heads and thin, slightly uneven black lines;
- crude flat shapes and a deliberately limited color palette;
- sparse, naive background marks sufficient to establish place;
- basic-paint-program feel: sincere amateur drawing, small natural inconsistencies, no polished vector geometry;
- one clear focal action and readable silhouettes at video size;
- generous empty space and safe framing for 16:9 delivery;
- no letters, labels, captions, speech bubbles, signatures, logos, UI, or watermarks;
- no photorealism, 3D rendering, elaborate shading, gradients, or glossy concept-art finish.

A free-text `style_description` augments the profile but cannot override safety, aspect ratio, or no-text constraints. Future styles remain data profiles until there is evidence that a plugin system is needed.

### Backend compilers

`image_prompt_compile` is a separate Structured Text Workflow Task with its own selected Backend, instructions, schema, limits, and usage record. It receives a Visual Brief, Style Profile, Character Identities, available reference assets, and a bounded descriptor for the target Image Backend. It returns the target-bound Image Request; the orchestrator validates that request before image generation. This allows a local LLM to compile prompts for a cloud image model, or the reverse, without contaminating the Visual Plan.

The compiler instructions are versioned per target Image Backend. A deterministic compiler may satisfy the same task for the stick renderer or a provider whose mapping needs no model judgment. Either way, the task persists the compiler Backend/model as well as the compiled prompt and target Image Backend.

Each compiler assembles the same information in the form its target Backend handles best:

- GPT Image 2: concise natural-language prompt, exact 16:9 legal generation size, high-fidelity references where useful;
- Gemini 3.1 Flash Image: explicit aspect ratio/image size plus supported character/object references;
- FLUX.2 klein: direct descriptive prompt and only settings the installed runner actually supports;
- deterministic stick Backend: maps the Visual Brief to known shapes and positions without generative prompting.

Compilers must not invent story content. If a Visual Brief lacks a required detail, compilation fails back to visual planning. Adapter-specific defaults are persisted in the Image Request, and all outputs are normalized by the media layer rather than relying on a model to return exactly 1920×1080.

For important recurring characters, final-quality profiles may create or select a reference image before the Scene batch when the Backend supports references. This is an optimization for recognizable traits, not a promise of pixel-perfect continuity.

### Remotion direction and asset selection

The `remotion_explainer` branch does not ask a model for an edit-decision list, timeline JSON, React,
download instructions, or rights metadata. The host first allocates a frame-aligned Shot from canonical
word timings and supplies one narration excerpt plus a bounded menu of templates, asset kinds, and SFX.
One `remotion_direction` request returns exactly:

```text
template
headline
supporting_text
body_lines
asset_kind
asset_query
sfx
```

Every field has a small enum or length bound. The portable structured-output schema intentionally stays
flat because the common local/cloud schema subset cannot express every template-dependent relationship.
Before strict contract validation, the host therefore resolves the opening/final template, discards
fields the selected fixed component cannot render, caps layout text at word boundaries, and downgrades a
body/media-dependent template when its required creative input is absent. It never invents missing body
copy or an asset query. The raw provider response remains recorded beside the canonical aggregate, so
the transformation is auditable and does not require another LLM repair call.

The host also derives the Shot's purpose, motion preset, hard-cut/section-wipe placement,
generated-image prompt, IDs, timing, frames, and paths. Visible factual copy must come verbatim from the
current narration excerpt; a non-contiguous headline is replaced with a bounded exact excerpt, optional
copy is cleared, and an incompatible body template is downgraded. This ensures factual on-screen claims
cannot bypass the completed evidence gate. The model may express an asset search query in English
because the configured media services and image models perform better with English retrieval/prompting;
Finnish narration and captions remain canonical.

Explicit visual-only `must_include` and `avoid` rules are reviewed once more against the complete
assembled edit plan before it is promoted, including dashboard-authored overrides. A trailing explicit
production clause such as “and show a reaction GIF” is split from `idea_direction` only for the Remotion
renderer: its narrative prefix remains in research, ideation, selection, outline, drafting, and review,
while the visual suffix enters only Remotion direction and this plan-level gate. Ambiguous wording stays
in the narrative constraint rather than being silently dropped.

Composed-frame review distinguishes replaceable asset-content failures from hard copy, layout,
template, and source-presentation failures. Only the former may trigger one image regeneration; hard
failures stop with an edit-plan correction instead of replacing unrelated art.

The asset resolver owns provider calls and filters results by source, media type, dimensions, license,
attribution completeness, and local policy. Only when multiple eligible records remain does
`remotion_asset_select` run. Its schema contains one `candidate_id`, and validation requires that exact ID
to exist in the supplied candidate list. It cannot return a URL, filename, license, score, explanation,
or alternate candidate. A zero-candidate result falls through programmatically to the next provider or
the configured Image Backend.

Dashboard text/template/asset-intent edits do not invoke an LLM. The server applies the requested fields
to the complete stored plan, reruns all domain validation, and freezes that full override in an immutable
child Run. Operational fields remain host-owned.

## Visual Review

Still-image Visual Review receives the generated image, its Visual Brief, resolved Style Profile, and
relevant Character Identities. It emits bounded scores and explicit failures for:

- subject/action fulfillment;
- style match;
- recurring identity traits;
- composition and legibility at delivery size;
- forbidden text, logos, or watermarks;
- audience safety and obvious malformed content.

The review cannot request aesthetic churn merely because another image might be prettier. A regeneration is justified by a failed requirement or a configured score threshold. Its targeted instruction states what to preserve and what to correct. All failures are regenerated in one batch, once. Every replacement is reviewed again; that second result cannot request another regeneration. A remaining hard failure stops a strict final-quality Run or follows an explicitly declared non-strict policy.

Draft profiles may skip Visual Review. Final profiles require a conforming vision Backend; they may not silently claim review when the local LLM is configured text-only.

For a final-quality Remotion Run, review instead receives three extracted frames—start, middle, and end—
from each low-resolution composed Shot. Each Shot is a separate bounded call, so the reviewer sees actual
text layout, animation state, and source presentation without returning a large multi-Shot JSON report.
It may request one generated-image correction. The host performs the replacement, rerenders the complete
proxy, and runs one terminal re-review; source-screenshot or layout failures stop rather than being
papered over with unrelated generated media.

## Music prompt

The Music Brief is generated after narration timing and contains only instrumental direction: overall mood, a simple emotional arc with time ranges, tempo/energy range, instrumentation suggestions, texture, exclusions, seamless-loop preference when relevant, and exact requested duration. It prohibits lyrics, speech, recognizable copyrighted melodies, audio logos, and abrupt endings.

The music model does not receive private voice audio. The rendered result is listened to beneath narration in evaluation; standalone musical impressiveness is secondary to unobtrusive support and speech intelligibility.

## Factual review

Factual mode adds a separate claim inventory. Each externally verifiable assertion in the Narration
Script preserves exact spoken wording and references Evidence IDs from the Research Pack. The independent
reviewer returns `supported`, `needs_qualification`, or `unsupported`, and also reports claims missed by
the inventory. Anything except complete supported coverage blocks TTS. A Script changed by measured
Duration Repair is inventoried and reviewed again before resynthesis.

## Evaluation and versioning

A prompt change is promoted only after fixed English and Finnish fixtures compare the previous and candidate versions. Measures include schema success, hard-constraint compliance, duration prediction error, story rubric scores, spoken naturalness, visual-brief fidelity, retries, latency, and cost.

Automated judges are useful for repeated comparisons but are not the sole authority. Deterministic validators cover objective rules, and human listening/visual review calibrates subjective scores. The same model should not draft, self-score, and automatically approve a material prompt change without independent fixtures.

Prompt versions are immutable once used in a Run. Fixes create a new version, and ordinary resume always uses the frozen original. `rerun --from` creates a parent-linked Run when the user intentionally wants current prompt assets from a chosen stage onward.
