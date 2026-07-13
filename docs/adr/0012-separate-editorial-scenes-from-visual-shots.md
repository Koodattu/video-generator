# ADR 0012: Separate editorial Scenes from visual Shots

Status: accepted

## Context

The original workflow used `scene_id` as the outline, TTS, image, review, and render key. That is a good
default for short fiction, but it forces a poor tradeoff for fast explainers: either TTS is fragmented
into two-second clips or one image remains on screen for an entire editorial passage.

## Decision

Scenes remain the stable editorial, script, TTS, and duration-repair unit. A Run independently selects
`scene_locked` or `cadenced` visual mode.

In cadenced mode, the orchestrator creates a frame-aligned Shot schedule only after narration timing is
final. Each Shot has an immutable `shot_id`, parent `scene_id`, narration excerpt, and start/end time. The
visual model supplies content for that canonical schedule; it cannot change timing or identity. Image
prompt compilation, generation, review, regeneration, and rendering fan out by `shot_id`.

The current single-plan implementation accepts at most 72 Shots and raises a configuration error for an
estimated larger plan. Timed visual planning receives a larger bounded output allowance, while rendering
uses an FFconcat manifest so image paths do not expand the Windows command line. Supporting longer
cadenced Runs requires a future chunked visual-plan contract rather than silently exceeding model limits.

Production visuals continue through the configured generative Image Backend. The deterministic stick
renderer remains an explicitly selected test/manual Backend and is never a production fallback.

## Consequences

- Existing Runs and the default scene-locked narrative path retain their original contracts.
- Fast visual cadence no longer dictates outline or TTS checkpoint granularity.
- Costs report editorial Scene count and generated-image Shot count separately.
- Word timing is required for cadenced Runs even when caption files are disabled.
- Dashboard and artifact joins must prefer `shot_id` when present and retain `scene_id` as parent context.
- Oversized cadenced Runs must increase their Shot target, use scene-locked visuals, or be split explicitly.
