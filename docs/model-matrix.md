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
| Research reduction and creative text | Qwen3.6-27B Q4 candidate | GPT-5.5; Terra after access probe | Gemini 3.5 Flash | Qwen3.6-27B Q4 candidate |
| Script reviews | same Qwen text runner | GPT-5.5 | Gemini 3.5 Flash | same Qwen text runner |
| Visual planning | Qwen3.6-27B Q4 candidate | GPT-5.5 | Gemini 3.5 Flash | Qwen3.6-27B Q4 candidate |
| Image-prompt compilation | Qwen3.6-27B Q4 candidate | GPT-5.5 | Gemini 3.5 Flash | Qwen3.6-27B Q4 candidate |
| Narration | VoxCPM2 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 |
| Word timing | Parakeet v3 plus exact-script reconciliation | ElevenLabs returned timestamps | ElevenLabs returned timestamps | ElevenLabs returned timestamps |
| Image generation | FLUX.2 klein 4B | GPT Image 2 | Gemini 3.1 Flash Image | FLUX.2 klein 4B |
| Visual review | Qwen3.6 vision path, evaluation-gated | vision-capable GPT-5.5 | Gemini 3.5 Flash | Qwen3.6 vision path, evaluation-gated |
| Music brief | Qwen3.6-27B Q4 candidate | GPT-5.5 | Gemini 3.5 Flash | Qwen3.6-27B Q4 candidate |
| Music when enabled | ACE-Step 1.5 XL Turbo | ElevenLabs Music v2 | ElevenLabs Music v2 | ACE-Step 1.5 XL Turbo |
| Render | local FFmpeg | local FFmpeg | local FFmpeg | local FFmpeg |

The cloud profile names identify the leading text/image provider, not every service. ElevenLabs remains the initial cloud voice and music provider. `hybrid-local-first` spends cloud budget only on narration/timing by default, where voice-clone quality and exact timestamps remove substantial local complexity.

Profile mappings are versioned and inspectable. They never switch dynamically after an error. Every listed alternative below is an explicit override or a future profile revision, not a silent fallback.

## OpenAI

### Text and research

`gpt-5.6-terra` is the attractive balanced GPT-5.6 variant the original idea identified. However, OpenAI currently describes the GPT-5.6 family as a limited preview for selected organizations. The portable `cloud-openai` profile should therefore start with generally available `gpt-5.5`; Setup may expose Terra as a conditional override only after a real model-access and structured-output probe succeeds. The Responses API is the appropriate integration surface for structured generation and bounded web search. [GPT-5.6 preview/access](https://help.openai.com/en/articles/20001325-a-preview-of-gpt-5-6-sol-terra-and-luna), [latest-model guide](https://developers.openai.com/api/docs/guides/latest-model), [web search](https://developers.openai.com/api/docs/guides/tools-web-search)

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

`music_v2` is the cloud music primary because it supports instrumental generation and structured composition plans and reuses the ElevenLabs credential. First-party pages currently conflict between a five-minute overview limit and API/composition-plan schemas accepting ten minutes. The Backend descriptor should conservatively advertise five minutes until a live capability probe proves more for the configured account. This does not affect 60–90-second v0 Runs. [composition plans](https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans), [compose API](https://elevenlabs.io/docs/api-reference/music/compose)

## Local text and vision

### Qwen3.6-27B

`Qwen/Qwen3.6-27B` is the initial local LLM candidate for both English and Finnish. It is an Apache-2.0 dense multimodal model with official llama.cpp support. The official BF16 and currently published Q8 MTP GGUF weights do not fit fully in 24 GB VRAM. The practical v0 path is a pinned Q4/IQ4 quantization, text-only at first, approximately 32K context, batch/parallelism 1, and MTP disabled. [model card](https://huggingface.co/Qwen/Qwen3.6-27B), [official repository](https://github.com/QwenLM/Qwen3.6), [official ggml-org MTP GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-MTP-GGUF)

There is no upstream Qwen Q4 artifact to bless automatically. Before `setup` can expose this Backend, the project must choose one of two auditable options:

- self-quantize the official weights with a pinned llama.cpp revision; or
- pin a reviewed third-party quantization by repository revision and file hash.

The exact artifact, quantization recipe, SHA-256, source license, and llama.cpp revision belong in the model manifest. MTP and long context remain disabled until evaluation proves a net gain without structured-output regressions.

The same model's vision path is a candidate for local Visual Review, but its projector/runtime memory on this machine must be benchmarked separately. A text-only runner cannot claim Visual Review. If the Qwen vision path does not fit reliably, the final local profile needs an explicitly selected smaller vision Backend; draft local Runs may skip review as declared.

Do not create separate Finnish orchestration or assume a separate Finnish LLM. The Evaluation Suite decides whether a language-specific Backend becomes justified.

## Local narration and alignment

### VoxCPM2

`openbmb/VoxCPM2` is the local TTS primary. It is Apache-2.0, supports English and Finnish among its documented languages, supports voice cloning, and is reported around 8 GB VRAM on an RTX 4090-class path. Native Windows has documented Triton limitations; `optimize = false` is the compatibility path and WSL2 is the preferred accelerated fallback. [VoxCPM repository](https://github.com/OpenBMB/VoxCPM), [VoxCPM2 weights](https://huggingface.co/openbmb/VoxCPM2), [Windows/Triton FAQ](https://voxcpm.readthedocs.io/en/latest/faq.html)

Use language-matched recordings of the same authorized voice rather than separate TTS models unless evaluation says otherwise. The adapter probes the actual output sample rate and duration instead of hardcoding a documentation claim. VoxCPM does not provide the authoritative word timing required for captions.

VoxCPM's optional `voxcpm[timestamps]`/stable-ts post-processing is worth benchmarking as an Alignment implementation. It is not native authoritative TTS timing and must pass the same exact-script coverage checks as Parakeet.

### Parakeet TDT 0.6B v3

`nvidia/parakeet-tdt-0.6b-v3` is the local timing primary because Finnish is explicitly supported and it returns word, segment, and character timestamps. It is CC BY 4.0 and Linux is the preferred platform, so the initial runner should use WSL2. It is ASR rather than forced alignment: caption text stays identical to the Narration Script, while a reconciliation algorithm maps recognized times onto it and fails visibly on poor coverage. [Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

Parakeet is loaded only when captions are enabled. Scene boundaries use probed TTS clip durations, so local video generation remains possible without STT when captions are explicitly disabled.

`Qwen/Qwen3-ASR-1.7B` remains an accuracy candidate, but its companion Qwen forced aligner does not currently list Finnish. It is not the v0 default for this requirement. [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

`faster-whisper` with Whisper large-v3-turbo is the mature ecosystem fallback to evaluate if Parakeet's Finnish reconciliation or WSL runtime is unreliable. It remains an explicit Backend choice and must preserve the canonical script just like Parakeet.

## Local images

`black-forest-labs/FLUX.2-klein-4B` is the local image primary. It is Apache-2.0, supports text-to-image and reference editing, and first-party memory figures range from roughly 8.4 to 13 GB; the descriptor should reserve the conservative 13 GB figure. Native Windows with Diffusers is benchmark-first, with WSL2 fallback. [FLUX.2 klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL comparison](https://bfl.ai/models/flux-2-klein), [official inference repository](https://github.com/black-forest-labs/flux2)

The 4B model is the simpler default even though the declared personal noncommercial Usage Purpose may allow experiments with other licenses. Larger variants are not implied fallbacks and need their own memory/license evaluation.

## Local music

`ACE-Step/acestep-v15-xl-turbo` is the initial local Music Backend. ACE-Step 1.5 is MIT-licensed, documents native Windows paths, supports instrumental output and requested durations from 10 to 600 seconds, and recommends at least 20 GB for XL without offload. On 24 GB it must run alone, batch 1. The standard Turbo checkpoint and smaller planning model are explicit lower-memory alternatives if evaluation favors them. [ACE-Step repository](https://github.com/ace-step/ACE-Step-1.5), [XL Turbo checkpoint](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [GPU compatibility](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md), [inference parameters](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)

Upstream guidance favors shorter instrumental generations even though ten minutes is accepted. Ten-minute coherence, exact duration, and whether XL materially beats standard Turbo are Evaluation Suite questions. Pin the code revision because the loader uses remote model code.

## Search for local and hybrid Runs

“Local” describes inference, not research connectivity. Brave Search is a reasonable independent Search Backend because its API has a simple credential and query interface. It lets research stay provider-neutral when Qwen performs the reduction. Pricing and quotas are unstable and therefore live in dated pricing metadata, not this architecture. [Brave Search API](https://brave.com/search/api/), [authentication](https://api-dashboard.search.brave.com/documentation/guides/authentication)

When `offline = true`, the Search Backend is disabled. Fiction can proceed from supplied material or clearly non-current model knowledge; factual mode requires supplied sources and cannot claim currentness.

## Hardware and platform implications

The target machine has 24 GB VRAM and 64 GB system RAM. These figures are planning reservations, not benchmark results:

| Local Backend | Conservative implication |
| --- | --- |
| Qwen3.6-27B Q4 | Near the GPU limit once runtime/context overhead is included; start at 32K, one request, probe partial/full offload |
| VoxCPM2 | About 8 GB reported; native compatibility path or WSL2 |
| Parakeet 0.6B | Small relative to other stages; WSL2 for the supported platform |
| FLUX.2 klein 4B | Reserve 13 GB; batch images while resident |
| ACE-Step XL Turbo | Reserve the full card; no concurrent model |

Only one local model family is resident. Process termination is the VRAM-release boundary. Setup records peak observed VRAM from smoke tests and refuses a profile whose configured runner exceeds the machine.

The model cache is `./.cache/models`, not a global surprise cache. Download manifests pin repositories, revisions, exact files, hashes, licenses, expected disk size, and required runtime. Hugging Face credentials are used only for gated downloads and never copied into a Run Bundle.

Several candidate stacks currently depend on source revisions rather than sufficiently current released packages. VoxCPM, Parakeet/NeMo, FLUX/Diffusers, Qwen ASR, and ACE-Step therefore need isolated environment manifests with pinned Python, CUDA/PyTorch compatibility, package versions, and source commits. Setup must never install a floating Git branch.

## Evaluation-only alternatives

These candidates from the supplied local-model research are useful contingencies, not additional v0 implementation commitments:

| Candidate | Evaluate when | Caution |
| --- | --- | --- |
| [Gemma 4 QAT](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) | Qwen Q4 provenance, Finnish quality, or runtime support is unacceptable | validate current exact checkpoint and Finnish quality; do not assume multilingual breadth equals quality |
| [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | VoxCPM is too heavy or unreliable on Windows/WSL | Finnish is listed and the model is MIT-licensed, but multilingual latency/quality needs testing and output is watermarked |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3-turbo | Parakeet alignment/reconciliation is unreliable | mature but larger/slower than needed if Parakeet works |
| `ik_llama.cpp` plus Qwen MTP | plain llama.cpp is correct but too slow | requires a separately pinned runtime and must beat the no-MTP path without schema regressions |

## License snapshot

| Asset | Documented license |
| --- | --- |
| Qwen3.6-27B | Apache 2.0 |
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

This is especially important for the Qwen Q4 provenance, Qwen vision memory, Finnish VoxCPM voice similarity, Parakeet reconciliation coverage, FLUX Windows support, ACE-Step long-form coherence, Terra account access, and ElevenLabs music-duration limit.
