# Initial model matrix

Verified against first-party documentation on 2026-07-10. These are v0 candidates, not timeless claims or quality winners. A model enters a built-in profile only after capability conformance, an English/Finnish fixture, platform smoke tests, and license/terms checks.

## Selection policy

The matrix separates four questions:

1. Does the exact model/service and required capability exist?
2. Can this account and machine actually use it?
3. Does it conform to the provider-neutral contract?
4. Does it beat alternatives on this project's fixed quality/cost fixtures?

Setup answers the first two with pinned assets and live probes. Contract tests answer the third. `evaluate` answers the fourth. A model's benchmark reputation alone is insufficient.

## Recommended profile assignments

| Workflow capability | `local` | `cloud-openai` | `cloud-gemini` | `hybrid-local-first` |
| --- | --- | --- | --- | --- |
| Live search | Brave Search; none when Offline | OpenAI web search | Gemini Google Search | Brave Search |
| Research reduction and creative text | Manifest-selected GGUF through stock llama.cpp | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Script reviews | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Visual planning | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Image-prompt compilation | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Narration | VoxCPM2 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 |
| Word timing | Parakeet v3 plus exact-script reconciliation | ElevenLabs returned timestamps | ElevenLabs returned timestamps | ElevenLabs returned timestamps |
| Image generation | FLUX.2 klein 4B | GPT Image 2 | Gemini 3.1 Flash Image | FLUX.2 klein 4B |
| Visual review | Qwen3.6 vision path, evaluation-gated | GPT-5.6 Terra | Gemini 3.5 Flash | Qwen3.6 vision path, evaluation-gated |
| Music brief | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Music when enabled | ACE-Step 1.5 XL Turbo | ElevenLabs Music v2 | ElevenLabs Music v2 | ACE-Step 1.5 XL Turbo |
| Render | local FFmpeg | local FFmpeg | local FFmpeg | local FFmpeg |

The cloud profile names identify the leading text/image provider, not every service. ElevenLabs remains the initial cloud voice and music provider. `hybrid-local-first` spends cloud budget only on narration/timing by default, where voice-clone quality and exact timestamps remove substantial local complexity.

Profile mappings are versioned and inspectable. They never switch dynamically after an error. Every listed alternative below is an explicit override or a future profile revision, not a silent fallback.

## OpenAI

### Text and research

`gpt-5.6-terra` is the balanced GPT-5.6 variant the original idea identified and is now the curated `cloud-openai` default. OpenAI's current model guide recommends Terra when intelligence and cost both matter. Live Preflight still verifies access for the configured project before a Run spends credits. The Responses API is the integration surface for structured generation and bounded web search. [latest-model guide](https://developers.openai.com/api/docs/guides/latest-model), [web search](https://developers.openai.com/api/docs/guides/tools-web-search)

Model aliases can change behavior. Reproducible evaluations should prefer dated snapshots when offered; otherwise the Run records the alias and request date, and profile changes require a new profile version.

### Images

The correct model is `gpt-image-2`, with dated snapshot `gpt-image-2-2026-04-21` currently documented. It supports generation, editing, and reference-image input. It is the OpenAI image primary. Image edges must be valid for the API; 1920×1080 is not directly legal because 1080 is not divisible by 16. Generate a 16:9 image such as 2048×1152 and normalize it deterministically to delivery resolution. Character references help but do not guarantee perfect recurrence. [GPT Image 2](https://developers.openai.com/api/docs/models/gpt-image-2), [image generation guide](https://developers.openai.com/api/docs/guides/image-generation)

OpenAI currently has no documented dedicated music-generation endpoint in the API guide set. The architecture must not invent one; the OpenAI-led profile uses ElevenLabs Music.

## Google Gemini

`gemini-3.5-flash` is the stable Gemini text primary, with structured output, function calling, and Google Search support. It should be the first cloud reference profile implemented because its exact generally available capability set is explicit. [Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash), [Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)

The correct current “Nano Banana 2” image ID is `gemini-3.1-flash-image`, not `nano-banana-2-flash`. Request `aspect_ratio = "16:9"` and a 2K image, then normalize the returned dimensions. Its supported reference-image features are useful for recurring characters. `gemini-3.1-flash-lite-image` is a cost-oriented candidate and `gemini-3-pro-image` a quality-oriented candidate, but neither becomes a default without the same fixtures. [Gemini 3.1 Flash Image](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image), [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)

Google's image-language guidance does not list Finnish among its best-performance languages. Visual Briefs remain faithful to Finnish narration, but the image prompt compiler should emit English by default and verify that policy through fixtures.

Gemini Lyria 3 is an optional short-form music experiment. `lyria-3-clip-preview` produces 30 seconds; `lyria-3-pro-preview` produces roughly a couple of minutes. It is preview, single-turn, and not suitable as the default for the later ten-minute target. [Lyria 3](https://ai.google.dev/gemini-api/docs/music-generation)

## ElevenLabs

### Narration and timing

The final-quality cloud narrator should initially use a Professional Voice Clone with `eleven_multilingual_v2` for both English and Finnish. ElevenLabs recommends it for high-quality content creation and long-form stability. `eleven_v3` is more expressive, but current guidance warns that Professional Voice Clones are not yet as optimized for it; it remains an A/B candidate. `eleven_flash_v2_5` is the draft-speed/cost option. [models](https://elevenlabs.io/docs/overview/models), [voice cloning](https://elevenlabs.io/docs/eleven-api/concepts/voice-cloning)

The with-timestamps endpoint returns character timing alongside audio. The adapter deterministically groups it into canonical words. No cloud STT is required on this path. ElevenLabs Forced Alignment is a possible explicit timing override when audio and exact transcript already exist, but Finnish quality still needs a fixture. [TTS with timestamps](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps), [Forced Alignment](https://elevenlabs.io/docs/api-reference/forced-alignment/create)

Voice cloning requires the user's own authorized recordings, kept under `private/`. A profile must not upload a reference merely because a path is configured; Setup and Preflight show which service will receive it.

### Music

`music_v2` is the cloud music primary because it supports instrumental generation and structured composition plans and reuses the ElevenLabs credential. The v0 Backend contract caps one generation at 600 seconds. A longer Narration Timeline therefore requests a shorter explicitly seamless instrumental loop and fits it deterministically; live Preflight still confirms account/model access. [composition plans](https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans), [compose API](https://elevenlabs.io/docs/api-reference/music/compose)

## Local text and vision

### Manifest-selected GGUF through stock llama.cpp

The workflow no longer bakes in one local text model. One typed `local-llm.toml` selects an exact target GGUF, optional compatible drafter, full repository and stock llama.cpp commits, independent file hashes, reviewed license, context tier, and MTP mode. A stdlib control worker owns one native-Windows `llama-server.exe`, reuses it for the adjacent text batch, and keeps JSON Schema plus domain validation outside the model runtime. [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

Qwen3.6 27B/35B-A3B and Gemma 4 31B/26B-A4B GGUF variants are reasonable first candidates, not defaults. Qwen MTP can be embedded in the target GGUF; current Gemma 4 MTP artifacts use a separate assistant GGUF. The same target must be evaluated with MTP off and on because speculative decoding can improve generation while worsening prompt processing, memory, or runtime stability. [Qwen 27B MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF), [Qwen 35B-A3B MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), [Gemma 4 31B QAT GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF), [Gemma 4 26B-A4B QAT GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)

Start with one Qwen and one Gemma candidate rather than downloading the whole matrix. Use one slot, 32K as the first application benchmark tier, and MTP disabled. Context is allocated at server startup; 64K/128K/256K are separate relaunch profiles and become usable only after real 24 GB fit, structured-output, and quality evidence. A native model context limit does not prove that its quantized runtime plus KV cache and work buffers fit this GPU.

Visual Review remains a separate capability. A text GGUF cannot claim it merely because the original architecture family is multimodal. The existing Qwen vision path stays evaluation-gated until its projector/runtime passes memory and image fixtures; another explicit VLM may win independently of the text benchmark.

Do not create separate Finnish orchestration or assume a separate Finnish LLM. Matched English/Finnish fixtures decide whether a language-specific Backend becomes justified.

## Local narration and alignment

### VoxCPM2

`openbmb/VoxCPM2` is the local TTS primary. It is Apache-2.0, supports English and Finnish among its documented languages, supports voice cloning, and is reported around 8 GB VRAM on an RTX 4090-class path. The implemented v0 runner uses native Windows with `optimize = false`; another platform is considered only if this path fails its live fixtures. [VoxCPM repository](https://github.com/OpenBMB/VoxCPM), [VoxCPM2 weights](https://huggingface.co/openbmb/VoxCPM2), [Windows/Triton FAQ](https://voxcpm.readthedocs.io/en/latest/faq.html)

Use language-matched recordings of the same authorized voice rather than separate TTS models unless evaluation says otherwise. The adapter probes the actual output sample rate and duration instead of hardcoding a documentation claim. VoxCPM does not provide the authoritative word timing required for captions.

VoxCPM's optional `voxcpm[timestamps]`/stable-ts post-processing is worth benchmarking as an Alignment implementation. It is not native authoritative TTS timing and must pass the same exact-script coverage checks as Parakeet.

### Parakeet TDT 0.6B v3

`nvidia/parakeet-tdt-0.6b-v3` is the local timing primary because Finnish is explicitly supported and it returns word, segment, and character timestamps. It is CC BY 4.0 and Linux is the preferred platform, so the initial runner should use WSL2. It is ASR rather than forced alignment: caption text stays identical to the Narration Script, while a reconciliation algorithm maps recognized times onto it and fails visibly on poor coverage. [Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

Parakeet is loaded only when captions are enabled. Scene boundaries use probed TTS clip durations, so local video generation remains possible without STT when captions are explicitly disabled.

`Qwen/Qwen3-ASR-1.7B` remains an accuracy candidate, but its companion Qwen forced aligner does not currently list Finnish. It is not the v0 default for this requirement. [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

`faster-whisper` with Whisper large-v3-turbo is the mature ecosystem fallback to evaluate if Parakeet's Finnish reconciliation or WSL runtime is unreliable. It remains an explicit Backend choice and must preserve the canonical script just like Parakeet.

## Local images

`black-forest-labs/FLUX.2-klein-4B` is the local image primary. It is Apache-2.0, supports text-to-image and reference editing, and first-party memory figures range from roughly 8.4 to 13 GB; the descriptor reserves the conservative 13 GB figure. The implemented v0 runner uses native Windows with Diffusers. A WSL2 runner would require a separate implementation and acceptance run. [FLUX.2 klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL comparison](https://bfl.ai/models/flux-2-klein), [official inference repository](https://github.com/black-forest-labs/flux2)

The 4B model is the simpler default even though the declared personal noncommercial Usage Purpose may allow experiments with other licenses. Larger variants are not implied fallbacks and need their own memory/license evaluation.

## Local music

`ACE-Step/acestep-v15-xl-turbo` is the initial local Music Backend. ACE-Step 1.5 is MIT-licensed, documents native Windows paths, supports instrumental output and requested durations from 10 to 600 seconds, and recommends at least 20 GB for XL without offload. On 24 GB it must run alone, batch 1. The standard Turbo checkpoint and smaller planning model are explicit lower-memory alternatives if evaluation favors them. [ACE-Step repository](https://github.com/ace-step/ACE-Step-1.5), [XL Turbo checkpoint](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [GPU compatibility](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md), [inference parameters](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)

Upstream guidance favors shorter instrumental generations even though ten minutes is accepted. Ten-minute coherence, exact duration, and whether XL materially beats standard Turbo are Evaluation Suite questions. Pin the code revision because the loader uses remote model code.

## Search for local and hybrid Runs

“Local” describes inference, not research connectivity. Brave Search is a reasonable independent Search Backend because its API has a simple credential and query interface. It lets research stay provider-neutral when the selected local LLM performs the reduction. Pricing and quotas are unstable and therefore live in dated pricing metadata, not this architecture. [Brave Search API](https://brave.com/search/api/), [authentication](https://api-dashboard.search.brave.com/documentation/guides/authentication)

When `offline = true`, the Search Backend is disabled. Fiction can proceed from supplied material or clearly non-current model knowledge; factual mode requires supplied sources and cannot claim currentness.

## Hardware and platform implications

The target machine has 24 GB VRAM and 64 GB system RAM. These figures are planning reservations, not benchmark results:

| Local Backend | Conservative implication |
| --- | --- |
| 14–19 GB GGUF candidate | Near the GPU limit once context/MTP/work buffers and Windows display use are included; start at 32K, one slot, MTP off |
| VoxCPM2 | About 8 GB reported; native compatibility path |
| Parakeet 0.6B | Small relative to other stages; WSL2 for the supported platform |
| FLUX.2 klein 4B | Reserve 13 GB; batch images while resident |
| ACE-Step XL Turbo | Reserve the full card; no concurrent model |

Only one local model family is resident. Process termination is the v0 VRAM-release boundary. For llama-server, live Preflight now requires process exit/GPU-PID disappearance when observable and records baseline/load/peak/post-exit aggregate VRAM. Exact aggregate return is advisory on Windows WDDM. Other model workers retain sequential load/process-exit probes until they add equivalent telemetry.

The model cache is `./.cache/models`, not a global surprise cache. Download manifests pin repositories, revisions, exact files, hashes, licenses, expected disk size, and required runtime. Hugging Face credentials are used only for gated downloads and never copied into a Run Bundle.

Several candidate stacks currently depend on source revisions rather than sufficiently current released packages. VoxCPM, Parakeet/NeMo, FLUX/Diffusers, Qwen ASR, and ACE-Step therefore need isolated environment manifests with pinned Python, CUDA/PyTorch compatibility, package versions, and source commits. Setup must never install a floating Git branch.

## Evaluation-only alternatives

These candidates from the supplied local-model research are useful contingencies, not additional v0 implementation commitments:

| Candidate | Evaluate when | Caution |
| --- | --- | --- |
| Additional Qwen3.6/Gemma 4 GGUF variants | the first one-per-family pair misses speed, fit, or quality | do not spend roughly another 35 GB before the first pair provides evidence |
| [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | VoxCPM is too heavy or unreliable on Windows/WSL | Finnish is listed and the model is MIT-licensed, but multilingual latency/quality needs testing and output is watermarked |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3-turbo | Parakeet alignment/reconciliation is unreliable | mature but larger/slower than needed if Parakeet works |
| MTP-on variant of the same target | generation throughput is a bottleneck | stock llama.cpp support is recent; it must beat MTP-off without schema, quality, or cleanup regressions |

## License snapshot

| Asset | Documented license |
| --- | --- |
| Selected local GGUF | model-specific; frozen in `local-llm.toml` and the runner manifest |
| VoxCPM2 | Apache 2.0 |
| Parakeet TDT 0.6B v3 | CC BY 4.0 |
| FLUX.2 klein 4B | Apache 2.0 |
| ACE-Step 1.5 | MIT |

This table is not legal advice and does not cover training inputs, hosted-service terms, voice rights, or generated-output restrictions. Setup stores the exact license text/revision associated with every pulled asset and compares it with `personal_noncommercial`; later commercial use requires a fresh review.

## Promotion gates

A candidate becomes a profile default only after:

1. exact model access or asset hashes are verified;
2. contract conformance passes on its actual Windows/WSL runner;
3. a 30-second English and Finnish smoke Run completes;
4. duration, schema, media, and VRAM cleanup checks pass;
5. fixed 60–90-second quality fixtures are reviewed;
6. cost/runtime and license/terms are recorded;
7. any profile change receives a new profile version.

This is especially important for local GGUF provenance/fit/MTP behavior, Qwen vision memory, Finnish VoxCPM voice similarity, Parakeet reconciliation coverage, FLUX Windows support, ACE-Step long-form coherence, Terra account access, and ElevenLabs music-duration limit.
