# Model matrix

Verified against first-party documentation through 2026-07-17. These are dated decisions, not timeless claims. A model enters a built-in profile only after capability conformance, English/Finnish fixtures, platform smoke tests, license/terms checks, and human quality evaluation.

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
| Live search | DDGS/DuckDuckGo; none when Offline | OpenAI web search | Gemini Google Search | DDGS/DuckDuckGo |
| Research reduction and creative text | Manifest-selected GGUF through stock llama.cpp | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Script reviews | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Visual planning | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Image-prompt compilation | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Narration | Higgs TTS 3 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 | ElevenLabs Multilingual v2 |
| Word timing | faster-whisper large-v3-turbo plus exact-script reconciliation | ElevenLabs returned timestamps | ElevenLabs returned timestamps | ElevenLabs returned timestamps |
| Image generation | FLUX.2 klein 4B | GPT Image 2 | Gemini 3.1 Flash Image | FLUX.2 klein 4B |
| Visual review | Qwen3.6 vision path, evaluation-gated | GPT-5.6 Terra | Gemini 3.5 Flash | Qwen3.6 vision path, evaluation-gated |
| Music brief | same local GGUF runner | GPT-5.6 Terra | Gemini 3.5 Flash | same local GGUF runner |
| Music when enabled | ACE-Step 1.5 XL Turbo | ElevenLabs Music v2 | ElevenLabs Music v2 | ACE-Step 1.5 XL Turbo |
| Render | local FFmpeg | local FFmpeg | local FFmpeg | local FFmpeg |

The cloud profile names identify the leading text/image provider, not every service. ElevenLabs remains the initial cloud voice and music provider. `hybrid-local-first` spends cloud budget only on narration/timing by default, where voice-clone quality and exact timestamps remove substantial local complexity.

`cloud-openai-gemini` is the mixed benchmark profile: pinned `gpt-5.4-mini-2026-03-17`
for structured text, Gemini 3.1 Flash Image, ElevenLabs Multilingual v2, and keyless bounded
DDGS/DuckDuckGo research.

Profile mappings are versioned and inspectable. They never switch dynamically after an error. Every listed alternative below is an explicit override or a future profile revision, not a silent fallback.

## OpenAI

### Text and research

`gpt-5.6-terra` is the balanced GPT-5.6 variant the original idea identified and is now the curated `cloud-openai` default. OpenAI's current model guide recommends Terra when intelligence and cost both matter. Live Preflight still verifies access for the configured project before a Run spends credits. The Responses API is the integration surface for structured generation and bounded web search. [latest-model guide](https://developers.openai.com/api/docs/guides/latest-model), [web search](https://developers.openai.com/api/docs/guides/tools-web-search)

The mixed cloud benchmark instead pins `gpt-5.4-mini-2026-03-17`, the cost-efficient GPT-5.4
variant with Responses API and Structured Outputs support. The dated snapshot keeps comparisons
reproducible instead of silently following an alias.

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

The workflow no longer bakes in one local text model. One typed `local-llm.toml` selects an exact target GGUF, optional compatible drafter, full repository and stock llama.cpp commits, independent file hashes, reviewed license, context tier, and MTP mode. A stdlib control worker owns one native-Windows `llama-server.exe` and reuses it for the adjacent text batch. Python supplies the task-specific schema, owns IDs and aggregation, and performs semantic validation; llama.cpp grammar constrains the model response to the requested JSON shape. [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

Qwen3.6, Gemma 4, and EuroLLM-22B are explicit candidates, not defaults. Qwen MTP can be embedded in the target GGUF; current Gemma 4 MTP artifacts use a separate assistant GGUF. EuroLLM has no official GGUF, so the curated entry pins Bartowski's exact Q4_K_M file and independent SHA-256; it advertises Finnish and uses the existing llama.cpp JSON-Schema grammar rather than claiming native tool-call training. [EuroLLM model](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512), [pinned EuroLLM GGUF](https://huggingface.co/bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF), [llama-server structured output](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

Use one slot and MTP disabled. Start Qwen/Gemma at 32K and EuroLLM at 16K; EuroLLM's 32K context is a later fit variant. Context is allocated at server startup, and a native context limit does not prove that the quantized runtime plus KV cache and work buffers fit this GPU.

The optional EuroLLM profile passed direct English and Finnish strict one-field schema smokes through
llama.cpp `/completion` at about 53 generated tokens/second. Finnish was not clearly better than the
current Gemma profile, which reached about 91 tokens/second in the matched smoke. Full workflow-schema
and script-quality validation remain open.

Visual Review remains a separate capability. A text GGUF cannot claim it merely because the original architecture family is multimodal. The existing Qwen vision path stays evaluation-gated until its projector/runtime passes memory and image fixtures; another explicit VLM may win independently of the text benchmark.

Do not create separate Finnish orchestration or assume a separate Finnish LLM. Matched English/Finnish fixtures decide whether a language-specific Backend becomes justified.

## Local narration and alignment

### Higgs TTS 3 primary; VoxCPM2 and OmniVoice alternatives

`bosonai/higgs-tts-3-4b` is the preferred EN/FI local narrator. Windows runs the exact pinned
SGLang-Omni Linux image through Docker Desktop's WSL2 engine; no separate user distro, host-network
mode, privileged container, or cloud inference is used. Setup downloads the exact model revision,
hashes its files, records the image digest and image ID, and Generate starts one short-lived offline
container behind the project's exclusive GPU lease. Canonical narration remains plain text for
captions and factual review; Python may prepend only allowlisted Higgs delivery tokens and verifies
that removing them reproduces the canonical text exactly. The first RTX 4090 health load reached
about 22.8 GB and returned to the host VRAM baseline after shutdown. Creator-use output requires the
supplied Boson AI attribution; hosting or product embedding requires a separate commercial license.
[Higgs model](https://huggingface.co/bosonai/higgs-tts-3-4b),
[SGLang-Omni Higgs cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)

`openbmb/VoxCPM2` and `k2-fsa/OmniVoice` are the two retained alternatives. VoxCPM2 is Apache-2.0,
supports English/Finnish voice cloning, and uses its native-Windows compatibility path. OmniVoice
supports native Windows CUDA, Finnish and English, transcript-assisted voice cloning, and 24 kHz
output; its weights are treated conservatively as CC-BY-NC. Both leave authoritative word timing to
the host/faster-whisper path. [VoxCPM repository](https://github.com/OpenBMB/VoxCPM),
[VoxCPM2 weights](https://huggingface.co/openbmb/VoxCPM2),
[OmniVoice model](https://huggingface.co/k2-fsa/OmniVoice),
[OmniVoice repository](https://github.com/k2-fsa/OmniVoice)

`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` and `XRXRX/X-Voice` Stage 1 are retained as
legacy/lower-quality comparison Backends. They remain addressable by explicit task override, and
their setup, worker, provenance, and tests remain in the project, but they are not recommended
choices. MOSS uses its pinned local codec and SDPA path. X-Voice requires the exact transcript and
actual reference language and has CC-BY-NC weights.
[MOSS-TTS v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5),
[MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v2),
[X-Voice](https://github.com/sunnyxrxrx/X-Voice)

OmniVoice also passed short English and Finnish synthesis plus ASR smokes at about 6.1 GB peak VRAM.
These are component smokes, not the 30-second bilingual end-to-end or 60–90-second promotion gates.

### faster-whisper large-v3-turbo

`faster-whisper` 1.2.1 with CTranslate2 4.8.1 and the commit-pinned `dropbox-dash/faster-whisper-large-v3-turbo` conversion is the current local timing primary. It runs natively on Windows, supports explicit English/Finnish language selection, and returns word timestamps. It is ASR rather than forced alignment: the exact Narration Script remains canonical, and deterministic reconciliation maps recognized times onto it while rejecting poor coverage. The runtime and model are MIT-licensed. [faster-whisper v1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1), [Turbo conversion](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo)

### Parakeet TDT 0.6B v3

`nvidia/parakeet-tdt-0.6b-v3` remains an explicit comparison Backend because Finnish is supported and it returns word, segment, and character timestamps. It is CC BY 4.0 and its retained NeMo runner uses WSL2. It follows the same canonical-script reconciliation contract as faster-whisper and is never selected as a silent fallback. [Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

The selected Alignment Backend is loaded when captions are enabled or cadenced Shots require word-grounded
visual boundaries and TTS lacks trustworthy timing. Scene boundaries use probed TTS clip durations, so
scene-locked local generation remains possible without STT when captions are explicitly disabled.

`Qwen/Qwen3-ASR-1.7B` remains an accuracy candidate, but its companion Qwen forced aligner does not currently list Finnish. It is not the v0 default for this requirement. [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## Local images

`black-forest-labs/FLUX.2-klein-4B` is the local image primary. It is Apache-2.0, supports text-to-image and reference editing, and first-party memory figures range from roughly 8.4 to 13 GB; the descriptor reserves the conservative 13 GB figure. The implemented v0 runner uses native Windows with Diffusers. A WSL2 runner would require a separate implementation and acceptance run. [FLUX.2 klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL comparison](https://bfl.ai/models/flux-2-klein), [official inference repository](https://github.com/black-forest-labs/flux2)

The 4B model is the simpler default even though the declared personal noncommercial Usage Purpose may allow experiments with other licenses. Larger variants are not implied fallbacks and need their own memory/license evaluation.

`Tongyi-MAI/Z-Image-Turbo` is the first image challenger: Apache-2.0, native Diffusers/SDPA, 9 steps, and an official sub-16 GB VRAM target. Turbo guidance is zero, so its negative input would be ignored; Python instead folds exclusions into the approved positive prompt and records that policy. [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [Diffusers Z-Image](https://huggingface.co/docs/diffusers/api/pipelines/z_image)

The implemented 1024×576 doodle smoke was valid and peaked near 22.6 GB on this RTX 4090, materially
above the official planning claim. Do not run it concurrently with another GPU model.

`ideogram-ai/ideogram-4-nf4-diffusers` is gated and noncommercial. Its adapter never invokes Ideogram's hosted magic-prompt service and never asks the orchestration LLM for another large JSON object. Python deterministically wraps the existing approved image prompt in the strict native caption schema, uses a per-step guidance schedule, and begins with model CPU offload. [Ideogram 4 repository](https://github.com/ideogram-oss/ideogram4), [prompt schema](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md), [Diffusers pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/ideogram4)

`Qwen/Qwen-Image-2512` is loaded through selective on-the-fly NF4 quantization of the transformer and
text encoder plus CPU offload. The small `time_text_embed`, `img_in`, `txt_in`, `norm_out`, and
`proj_out` boundary modules remain in their original precision. The adapter uses the model's
documented 50-step path at 1664×928, passes native negative conditioning with
`true_cfg_scale = 4.0`, and does not advertise reference-image editing. Full BF16 is not a 24 GB
configuration. Python removes negative-prompt clauses that conflict with approved positive visual or
style terms before the request reaches the model.
[Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512),
[Qwen-Image repository](https://github.com/QwenLM/Qwen-Image),
[Diffusers Qwen pipeline](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)

The earlier Qwen smoke used 20 steps at 1024×576 and produced a noisy image. Because the 2512 base
checkpoint is not a distilled 20-step model, that result is not a representative quality comparison.
Qwen remains experimental until the corrected 50-step path is run and judged. Ideogram was not
selected by any comparison-video configuration; its separate component smoke loaded the model but
returned its gray safety placeholder, which the adapter rejected. No usable Ideogram image exists.

### Native-Windows exclusions

| Candidate | Decision under the native-Windows-only rule |
|---|---|
| HiDream-O1 Dev | Not integrated: the official custom runtime hard-requires or source-edits around FlashAttention, whose Windows path is not supported/tested sufficiently for this project |

Neither exclusion is a failed quality or VRAM benchmark; the project did not run an unsupported
platform route.

## Local music

`ACE-Step/acestep-v15-xl-turbo` is the initial local Music Backend. ACE-Step 1.5 is MIT-licensed, documents native Windows paths, supports instrumental output and requested durations from 10 to 600 seconds, and recommends at least 20 GB for XL without offload. On 24 GB it must run alone, batch 1. The standard Turbo checkpoint and smaller planning model are explicit lower-memory alternatives if evaluation favors them. [ACE-Step repository](https://github.com/ace-step/ACE-Step-1.5), [XL Turbo checkpoint](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [GPU compatibility](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md), [inference parameters](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)

Upstream guidance favors shorter instrumental generations even though ten minutes is accepted. Ten-minute coherence, exact duration, and whether XL materially beats standard Turbo are Evaluation Suite questions. Pin the code revision because the loader uses remote model code.

## Search for local and hybrid Runs

“Local” describes inference, not research connectivity. Curated local and mixed profiles use the
pinned DDGS package with the DuckDuckGo backend, explicit `us-en`/`fi-fi` regions, moderate safe
search, bounded result counts, and returned snippets only. Arbitrary page extraction remains
disabled. This path is keyless but has no provider SLA, so `offline = true` or a zero query limit
remains the deterministic no-network option. Brave is retained only as an explicit legacy override.

When `offline = true`, the Search Backend is disabled and fiction can proceed without live sources.
Factual mode currently requires live bounded search with nonzero query and source limits; supplied offline
evidence ingestion is not implemented, so Offline factual configuration is rejected.

## Hardware and platform implications

The target machine has 24 GB VRAM and 64 GB system RAM. These figures are planning reservations:

| Local Backend | Conservative implication |
| --- | --- |
| 14–19 GB GGUF candidate | Near the GPU limit once context/MTP/work buffers and Windows display use are included; start at 32K, one slot, MTP off |
| Higgs TTS 3 | Preferred; managed Docker Desktop/WSL2 runtime, about 22.8 GB measured at startup, run alone |
| VoxCPM2 | Alternative; about 8 GB reported, native compatibility path |
| OmniVoice | Alternative native EN/FI narrator |
| MOSS-TTS v1.5 + codec | Legacy/lower quality; retained for comparisons, run alone |
| X-Voice Stage 1 | Legacy/lower quality; low-memory, noncommercial weights |
| faster-whisper Turbo | Native CTranslate2 CUDA worker; live probe must confirm the exact Windows wheel and GPU |
| Parakeet 0.6B | Optional WSL2 comparison Backend |
| FLUX.2 klein 4B | Reserve 13 GB; batch images while resident |
| Z-Image-Turbo | Reserve the full card despite the lower official claim |
| Ideogram 4 NF4 | Experimental; no usable smoke, budget the full card |
| Qwen-Image-2512 NF4 | Experimental selective-NF4/offload path; 50 steps at 1664×928, run alone |
| ACE-Step XL Turbo | Reserve the full card; no concurrent model |

Measured component smokes on 2026-07-15 reached approximately 6.1 GB peak for OmniVoice, 12.9 GB
for MOSS-TTS, 2.0 GB for X-Voice, 22.6 GB for Z-Image Turbo, and 17.5 GB for Qwen-Image. Measurements
are implementation- and fixture-specific, not general vendor specifications.

Only one local model family is resident. Process termination is the v0 VRAM-release boundary. Live
Preflight requires fresh process-exit and observable GPU-PID cleanup evidence for every CUDA runner.
llama-server additionally records baseline/load/peak/post-exit aggregate VRAM; exact aggregate return
is advisory on Windows WDDM.

The model cache is `./.cache/models`, not a global surprise cache. Download manifests pin repositories, revisions, exact files, hashes, licenses, expected disk size, and required runtime. Hugging Face credentials are used only for gated downloads and never copied into a Run Bundle.

Every local stack uses an isolated environment manifest. faster-whisper/CTranslate2 pins exact releases; VoxCPM, Parakeet/NeMo, FLUX/Diffusers, Qwen ASR, and ACE-Step additionally pin model or source revisions and CUDA/PyTorch compatibility. Setup must never install a floating Git branch.

## Evaluation-only alternatives

These candidates from the supplied local-model research are useful contingencies, not additional v0 implementation commitments:

| Candidate | Evaluate when | Caution |
| --- | --- | --- |
| Additional Qwen3.6/Gemma 4 GGUF variants | the first one-per-family pair misses speed, fit, or quality | do not spend roughly another 35 GB before the first pair provides evidence |
| [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | VoxCPM is too heavy or unreliable on Windows/WSL | Finnish is listed and the model is MIT-licensed, but multilingual latency/quality needs testing and output is watermarked |
| Parakeet TDT 0.6B v3 | matched faster-whisper comparison is needed | retained WSL2/NeMo environment is heavier to operate |
| MTP-on variant of the same target | generation throughput is a bottleneck | stock llama.cpp support is recent; it must beat MTP-off without schema, quality, or cleanup regressions |

## License snapshot

| Asset | Documented license |
| --- | --- |
| Selected local GGUF | model-specific; frozen in `local-llm.toml` and the runner manifest |
| VoxCPM2 | Apache 2.0 |
| OmniVoice weights | CC-BY-NC |
| MOSS-TTS v1.5 / Audio Tokenizer v2 | Apache 2.0 |
| X-Voice Stage 1 weights | CC-BY-NC |
| faster-whisper Turbo runtime/model | MIT |
| Parakeet TDT 0.6B v3 | CC BY 4.0 |
| FLUX.2 klein 4B | Apache 2.0 |
| Z-Image-Turbo | Apache 2.0 |
| Ideogram 4 NF4 | Ideogram 4 Non-Commercial |
| Qwen-Image-2512 | Apache 2.0 |
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

This is especially important for local GGUF provenance/fit/MTP behavior, Qwen vision memory, Finnish VoxCPM voice similarity, faster-whisper/Parakeet reconciliation coverage, FLUX Windows support, ACE-Step long-form coherence, Terra account access, and ElevenLabs music-duration limit.
