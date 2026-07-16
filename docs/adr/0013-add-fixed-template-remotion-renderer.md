# ADR 0013: Add a fixed-template Remotion renderer

Status: accepted

## Context

The original renderer produces narrated still-image videos well, but fast internet-native explainers
need kinetic typography, code, diagrams, source screenshots, short motion assets, and denser cuts. Asking
an LLM to emit a complete timeline, asset metadata, or arbitrary React would make timing, rights,
resumption, and validation depend on one large brittle response.

The project must preserve the existing renderer, work with local English and Finnish models, keep model
calls bounded, and allow stock/owned media without allowing a model to invent URLs or license claims.

## Decision

Add `video_style = "remotion_explainer"` beside the default `still_image` branch. Both branches share
research, evidence gates, narration, alignment, music, Run Bundles, and delivery QC. Only their visual
planning, asset resolution, review, and visual rendering stages differ.

Remotion uses eight fixed, repository-owned templates. The host derives Shot IDs, word anchors, times,
frames, purpose, motion and transition presets, asset IDs, source linkage, generated-image requests,
paths, and renderer settings. One strict `remotion_direction` call per Shot returns bounded visible copy,
a template, an asset kind/query, and an SFX preset. A second call is permitted only when multiple
eligible media records remain and may return one supplied `candidate_id`. The model never emits React,
URLs, file paths, rights metadata, or a whole-video edit plan.

Asset resolution is policy-driven: owned/authorized `media-library/` files, eligible Wikimedia Commons
media, optional Pexels media, then the configured Image Backend. Offline mode permits only local files
and local model generation. Source screenshots must map through evidence attached to the current Scene,
and page capture is disabled unless the source host is present in the Run's explicit trust allowlist.
The same allowlist is enforced for redirects, frames, and subresources before DNS lookup.
The host validates and records source, creator, license, attribution, retrieval time, transformations,
and content hashes. NC and ND assets are rejected; ShareAlike is opt-in. GIPHY and the discontinued
Tenor API are excluded.

Final quality review inspects three frames per Shot from a rendered low-resolution composition and
allows one targeted generated-image replacement followed by one terminal re-review. Dashboard edits and
optional manual asset approval create immutable child Runs. Approval binds to complete asset-record
hashes and must be repeated after a changed asset.

Setup installs the exact npm lock and pinned Chrome Headless Shell. Generate verifies that frozen runtime
and never installs dependencies or a browser. Remotion produces a local visual stream; FFmpeg owns final
encoding/muxing and existing media QC.

## Consequences

- The original still-image behavior and public stage sequence remain compatible.
- Local Gemma-class models handle several small schema-constrained decisions instead of one large JSON
  document; deterministic code handles fields that models are poor at maintaining.
- New visual styles should normally be fixed components plus validated parameters, not generated code.
- Stock-preferred Runs may make media-search/download requests even though all AI inference and rendering
  are local. `offline = true` is required for a network-isolated Run.
- Source screenshot host allowlisting and request filtering mitigate SSRF but are not a complete browser
  sandbox; sensitive network environments should keep the allowlist empty or use Offline mode.
- Current screenshots are viewport captures without DOM-selector/highlight authoring, and the Dashboard
  is a constrained Shot editor rather than a general nonlinear timeline editor.
