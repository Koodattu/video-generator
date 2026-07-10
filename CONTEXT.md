# Video Generation

This context defines the shared language for producing narrated, still-image videos from an initial creative brief.

## Language

**Scene**:
An ordered narrated portion of a video associated with one primary visual. Scene lengths intentionally vary with the content.
_Avoid_: Segment, slide

**Duration Budget**:
The maximum running time allotted to a video, which the finished video should approach without exceeding. The accepted proximity band is a production rule rather than part of the term.
_Avoid_: Target length, maximum duration, goal length

**Backend**:
A configured cloud service or local model runtime that performs one workflow task behind a shared capability contract.
_Avoid_: Engine, module

**Run Profile**:
A named assignment of Backends and generation settings to Workflow Tasks for one Run.
_Avoid_: Mode, stack, preset

**Content Mode**:
The declared relationship between a video's narrative and real-world evidence. `fiction` is an invented story for which research is creative input only; `factual` contains claims intended to be supported by captured sources.
_Avoid_: Genre, hybrid

**Visual Brief**:
A provider-neutral description of what a Scene should depict, including its subject, action, mood, composition, and continuity requirements.
_Avoid_: Image prompt, generation prompt

**Style Profile**:
A reusable specification for the appearance shared by generated visuals, kept separate from Scene content and provider-specific prompt syntax.
_Avoid_: Prompt, theme

**Character Identity**:
The small set of signature visual traits that keeps a recurring character recognizable across Scenes. Pose, proportions, line quality, and incidental details may vary without breaking the identity.
_Avoid_: Exact likeness, character prompt

**Visual Review**:
A structured assessment of a generated visual against its Visual Brief, Style Profile, and relevant Character Identities. Final-quality runs may use one review-driven regeneration attempt.
_Avoid_: Image approval, vibe check

**Offline**:
A run constraint that prohibits all network access. It is independent of whether generative Backends execute locally, and its results must not be presented as current research.
_Avoid_: Local

**Run**:
One durable attempt to produce a video from a creative brief and a frozen Run Profile. It owns its intermediate artifacts, status, and final output.
_Avoid_: Session, job

**Checkpoint**:
A persisted boundary after a completed workflow stage from which a Run can resume or intentionally pause.
_Avoid_: Save, snapshot

**Usage Purpose**:
The declared intended use against which Backend and model license compatibility is evaluated.
_Avoid_: License mode

**Voice Profile**:
The authorized narrator identity and synthesis settings used for a Run. Any cloning reference is a private input owned by, or used with permission from, the person it represents.
_Avoid_: Voice model, speaker preset

**Setup**:
An explicit operation that prepares and validates one or more Run Profiles. It may create local caches, download pinned model assets, and verify configured Backend credentials.
_Avoid_: Install, preflight

**Preflight**:
A read-only readiness check automatically performed before a Run. It validates only the selected Run Profile and never downloads models or modifies credentials.
_Avoid_: Setup, initialization

**Output Language**:
The language in which a Run's story is authored, reviewed, narrated, and optionally captioned. Research sources may use other languages.
_Avoid_: Primary language, narration language

**Workflow Task**:
A named, bounded production operation with defined inputs, outputs, and completion criteria. Different Workflow Tasks may use different Backends.
_Avoid_: Agent, role

**Creative Brief**:
The required creative direction that bounds research and ideation for a Run. It contains at least a topic or idea direction and may add tone, themes, required elements, and exclusions, but it is not a selected story or script.
_Avoid_: Prompt, concept, script

**Story Candidate**:
One proposed story direction produced from the Creative Brief and available research. It competes with other candidates and is not yet an outline or script.
_Avoid_: Idea, draft

**Story Concept**:
The Story Candidate chosen by the selector as the basis for outlining and scriptwriting.
_Avoid_: Winner, selected idea

**Research Pack**:
A compact collection of researched inspiration and retained source metadata prepared for ideation. It may include motifs, surprising details, settings, vocabulary, cultural cautions, and clichés to avoid, but it is not a story or evidence that fiction must reproduce literally.
_Avoid_: Research dump, source list, context

**Story Outline**:
The structured story arc and ordered Scene plan derived from the Story Concept before narration prose is written. It assigns each Scene a narrative purpose, emotional beat, visual opportunity, and provisional share of the Duration Budget.
_Avoid_: Script, scene list, storyboard

**Narration Script**:
The approved text organized into ordered Scenes and intended to be spoken verbatim by the Narrator Voice. It excludes visual directions, research citations, review commentary, and formatting that should not be spoken aloud.
_Avoid_: Copy, screenplay, transcript

**Duration Repair**:
A targeted revision and resynthesis used when measured narration falls outside the acceptable Duration Budget range. It changes selected Scenes and is not another general editorial pass.
_Avoid_: Rewrite, time stretching, trimming

**Audience Profile**:
The content-suitability boundary applied to a Run's story and visuals.
_Avoid_: Age rating, safety prompt

**Music Bed**:
An optional instrumental background track shaped to the story and mixed beneath narration for the full video. It is not a vocal song and must conform to the finished narration duration.
_Avoid_: Song, soundtrack, score

**Caption Track**:
A word-timed text representation of the narration that is independent of its delivery format. It may produce a sidecar file, a selectable player track, or captions rendered into the video.
_Avoid_: Transcript, subtitles file

**Delivery Format**:
The aspect ratio, resolution, and frame rate of rendered video output.
_Avoid_: Canvas, video size

**Motion Style**:
The presentation rule for movement and transitions between Scene visuals.
_Avoid_: Transition preset, animation

**Visual Cadence**:
The soft planning target for how often a new Scene visual appears while allowing narrative boundaries to justify variation.
_Avoid_: Fixed interval, image duration

**Cost Ceiling**:
The maximum estimated cloud spend permitted for a Run. The harness must not knowingly begin a Backend call whose configured worst-case cost would exceed the remaining ceiling.
_Avoid_: Cost estimate, spending target

**Failure Policy**:
The declared outcome when an enabled optional capability cannot complete.
_Avoid_: Fallback, best effort

**Evaluation Suite**:
A fixed collection of English and Finnish fixtures and quality rubrics used to compare conforming Backends and Run Profiles. It measures quality, reliability, duration accuracy, runtime, resource use, and cloud cost outside ordinary generation.
_Avoid_: Test suite, benchmark claim

**Run Bundle**:
The durable directory containing a Run's resolved non-secret configuration, normalized intermediate artifacts, generated media, status, usage, hashes, and provenance. It excludes credentials, hidden model reasoning, and copies of private voice-reference recordings.
_Avoid_: Output folder, cache

**Narration Timeline**:
The authoritative timing artifact that places the final narration audio, ordered Scenes, declared pauses, and optional words on one clock. Visuals, captions, music, and rendering derive their timing from it.
_Avoid_: Audio timeline, timestamps, edit decision list
