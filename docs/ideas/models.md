# Model research report — July 11, 2026

> Historical research note. The July 17 human comparison supersedes this shortlist: Higgs TTS 3 is
> the preferred local narrator; VoxCPM2 and OmniVoice are alternatives; X-Voice and MOSS-TTS are
> retained legacy/lower-quality options. The original Qwen-Image sample used an unsupported
> quality shortcut (20 steps at 1024×576) and must not be treated as representative. No usable
> Ideogram image was produced: no comparison video selected it, and its component smoke returned the
> model's safety placeholder.

I read the repository, current model profile, completed run artifacts, and the attached report. The attachment’s general direction is sensible, but its shortlist is incomplete—especially for TTS and current image generation.

## Bottom line

For this machine and project:

- **LLM:** benchmark **EuroLLM-22B-Instruct-2512** first. Use **Poro 2 8B Instruct** as the Finnish-quality control.
- **TTS:** keep VoxCPM2 as the baseline. Test **OmniVoice** for overall fit, **Higgs TTS 3** for quality/expressiveness, and **X-Voice** for strong published Finnish/RTX 4090 evidence. **MOSS-TTS Local v1.5** is the best permissively licensed alternative.
- **Images:** test **Z-Image-Turbo** first. It is the clearest practical replacement candidate for FLUX on 24 GB. Test **Ideogram 4 NF4** as the higher-quality noncommercial option, then **HiDream-O1 Dev**. **Qwen-Image-2512** is viable but not my first choice for this workload.

If downloading only one challenger per category: **EuroLLM-22B, OmniVoice, and Z-Image-Turbo**. Add **Higgs TTS 3** and **Ideogram 4 NF4** when testing the quality ceiling.

## What the repository changes

The current local stack is a manifest-selected GGUF, VoxCPM2, faster-whisper, and FLUX.2 Klein ([README](C:/Users/Juha/Desktop/Projektit/video-generator/README.md:93)). The active profile is Gemma 4 26B-A4B Q4 with 16K context and an MTP drafter ([local-llm.toml](C:/Users/Juha/Desktop/Projektit/video-generator/local-llm.toml:2)).

Important implications:

- LLM replacement is comparatively easy: an exact GGUF can use the existing stock `llama-server` path.
- TTS and images have family-specific workers—currently VoxCPM and `Flux2KleinPipeline`—so each new family needs a small isolated adapter, not merely a different model path ([workers](C:/Users/Juha/Desktop/Projektit/video-generator/src/video_generator/workers/main.py:192)).
- Models own the GPU sequentially. There is no need to choose tiny models merely to keep multiple models resident.
- Image prompts are deliberately emitted in English even for Finnish videos ([prompt policy](C:/Users/Juha/Desktop/Projektit/video-generator/docs/prompt-system.md:167)). Native Finnish image prompting is therefore irrelevant.
- The image contract requires exact dimensions and accepts negative prompts and references ([contracts](C:/Users/Juha/Desktop/Projektit/video-generator/src/video_generator/contracts.py:572)), but initial `reference_paths` are currently empty ([workflow](C:/Users/Juha/Desktop/Projektit/video-generator/src/video_generator/workflow.py:1904)). Editing should be secondary to plain text-to-image quality, composition and diversity.
- This is not a first proof-of-life benchmark: completed English and Finnish end-to-end runs already exist ([Finnish run manifest](C:/Users/Juha/Desktop/Projektit/video-generator/runs/20260711T164943Z-49e8ddb6/manifest.json:1157)).

ComfyUI is not currently the integration surface. The attachment’s ComfyUI recommendation would introduce an additional managed server rather than simply enabling more checkpoints.

## LLM alternatives beyond Qwen and Gemma

| Priority | Model | Assessment |
|---|---|---|
| 1 | [EuroLLM-22B-Instruct-2512](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512) | Best first candidate. 22.6B dense, explicit English and Finnish among 35 languages, 32K context, Apache 2.0. Community GGUFs exist. A Q4 build should fit 24 GB, although the exact GGUF, chat template and llama.cpp revision must be tested. |
| 2 | [Llama-Poro-2-8B-Instruct](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-Instruct) | Best Finnish specialist/control. Trained specifically for English and Finnish; its published Finnish scores are strong for 8B. Small and easy to run, but its 8,192-token context may be too restrictive for some repository stages. Llama 3.3 Community License. |
| 3 | [Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509) | Interesting fully open coverage model: Apache 2.0, 65K context and 1,811 claimed languages. Community GGUFs are available. However, the card does not isolate Finnish quality, so treat it as a multilingual coverage experiment rather than a likely winner. |

EuroLLM is the strongest direct answer to “something other than Qwen/Gemma that genuinely targets Finnish.” It was trained on four trillion multilingual tokens and explicitly covers every EU language.

Poro 2 is especially valuable as a judge of Finnish naturalness. Its model card reports Finnish IFEval 66.54, MTBench 6.75 and a Finnish writing score of 8.05. Its general reasoning ceiling will be below the current 26B model, but if a larger model loses to Poro on spoken Finnish, that is useful evidence.

Two additional controls do not currently qualify as verified Finnish recommendations:

- [Ministral 3 14B Instruct](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) officially fits 24 GB in FP8, supports 256K context and advertises JSON output. Its published language list does not name Finnish, so Finnish must be proven empirically.
- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) runs within 16 GB in MXFP4 and supports Structured Outputs, but requires the Harmony response format and does not document Finnish quality. That creates unnecessary compatibility risk for the current llama.cpp worker.

I would not spend time on 70B+ or current 100B-class MoE models. A low active-parameter count does not eliminate their total-weight and memory-bandwidth cost when most weights must be CPU-offloaded.

## TTS alternatives

| Role | Model | Project assessment |
|---|---|---|
| Best overall fit | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | About 0.6B, English and Finnish among 646 languages, short-reference voice cloning, optional reference transcript, and speed/duration controls. Easily within 24 GB by model size. Native Windows is unverified. Code is Apache 2.0, but weights are CC-BY-NC. |
| Best expressive quality candidate | [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | Real current release—not a rumour. Finnish is in Boson’s highest “WER/CER under 5” tier. Supports transcript-assisted cloning plus emotion, style, speed, pitch, pause and sound-effect controls. Likely fits 24 GB, but official speed data is H100-based. Linux/SGLang/Docker-oriented, so WSL2 is the realistic route. |
| Best Finnish/4090 published evidence | [X-Voice](https://github.com/sunnyxrxrx/X-Voice) | 0.4B, 30 languages including Finnish, zero-shot cross-language cloning. Its [paper](https://arxiv.org/abs/2605.05611) reports RTX 4090 RTF 0.073 and Finnish WER 4.41 for Stage 1. Very new and requires eSpeak-NG/bash-oriented setup. Code MIT; weights CC-BY-NC. |
| Best permissive full-featured candidate | [MOSS-TTS Local Transformer v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | English and Finnish among 31 languages, zero-shot cloning, 48 kHz stereo, explicit pause and fixed-duration control. Apache 2.0. Exact 24 GB peak is unpublished and the recommended stack is Linux-oriented. |
| Lightweight fallback | [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | 500M, Finnish and English, zero-shot cloning, MIT, built-in PerTh watermark. Easy model size, but not a top Finnish recommendation: [Resemble’s own report](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages) gives Finnish 17.52% CER versus English 0.65%. |

### Direct answers

**Higgs V3:** Yes. Its canonical current name is Higgs TTS 3. It is probably the most interesting quality/expressiveness experiment here. The license permits personal research/noncommercial work and includes a creator grant for monetized videos or podcasts with prominent attribution. Embedding it in a product or hosted service requires a separate commercial license.

**Chatterbox V3:** Worth testing, but as a lightweight fallback—not as the presumed Finnish winner. Chatterbox Turbo is English-only and should not be used for the shared EN/FI backend.

**Coqui/XTTS-v2:** Skip for this project. It has good cloning and mature tooling, but [its official 16-language list](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md) does not include Finnish. It also uses the Coqui Public Model License rather than a permissive model license.

**MOSS-TTS:** Strong recommendation. Its fixed-duration and pause controls are unusually relevant because recent repository failures are narration-duration misses rather than model crashes. Exploiting fixed duration would eventually require the adapter to pass a target duration.

**Fish Audio S2 Pro:** Watchlist only. Expressive and capable, but Finnish evidence is weaker, the license is noncommercial and the recommended configuration consumes most of the available 24 GB.

None of these should replace faster-whisper for authoritative caption timing without a separate timing evaluation.

## Image-generation alternatives beyond FLUX

| Priority | Model | Assessment |
|---|---|---|
| 1 | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Best practical challenger. 6B, Apache 2.0, native Diffusers support, eight function evaluations, optional CPU offload, and officially fits comfortably within 16 GB VRAM. |
| 2 | [Ideogram 4 NF4](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers) | Best high-quality candidate under the repository’s current personal/noncommercial scope. Ideogram says NF4 fits one 24 GB GPU. Native Diffusers, exact project dimensions supported, excellent layout and composition. Gated, noncommercial weights. |
| 3 | [HiDream-O1-Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604) | Strongest permissive frontier experiment. 8B, MIT, 28 steps, strong reported composition and dense-prompt performance. However, 24 GB fit is unproven and it uses a custom Transformers/FlashAttention runtime, making native Windows materially riskier. |
| 4 | [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | Viable but lower priority. Apache 2.0 and Diffusers support, improved realism and detail, but 50 steps plus quantization/offload make it slower. Its typography/editing strengths are mostly irrelevant here. |
| 5 | [Z-Image base](https://huggingface.co/Tongyi-MAI/Z-Image) | Use if Turbo lacks diversity or style range. Supports negative prompting and greater diversity, but requires 28–50 steps and has no official 24 GB result. |

### Why Z-Image-Turbo is first

It is the cleanest match for the current managed Diffusers architecture. It is fast enough for scene batches and has direct consumer-VRAM evidence. The main risks are:

- The official table labels its diversity “low.” Check whether multiple scenes collapse toward similar compositions or faces.
- Turbo uses guidance zero and does not offer the base model’s normal CFG/negative-prompt behaviour. “No text, logo or watermark” constraints should therefore be included in the positive prompt.
- Exact `1280×720` and `2048×1152` outputs still need a smoke test.

### Why Ideogram 4 is unusually suitable

Ideogram 4 was released June 3, 2026. It supports any 256–2048 dimensions in multiples of 16, so both project resolutions are valid. It was trained on structured JSON captions for composition, object placement, palette and style. That maps unusually well to the repository’s existing structured visual briefs.

Its hard limitations are the noncommercial license, gated download, CUDA/NF4 Windows uncertainty and required safety controls. Also, although it is not Qwen-Image, its text encoder is Qwen3-VL.

### Qwen-Image caveat

Qwen-Image-2512’s official presets are `1664×928` for landscape and `928×1664` for portrait. They
are approximate rather than mathematically exact ratios, so the Image Request contract treats those
two documented Qwen sizes as explicit native exceptions before normalizing to delivery dimensions.

Qwen-Image 2.0 has been announced, but as of this report the official project does not provide a corresponding downloadable local open-weight checkpoint. For local work, 2512 remains the relevant Qwen candidate.

### Other current models

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) is very current and aesthetically interesting, but its 12B transformer and roughly 62 GB repository have no official 24 GB deployment claim. I would not start with community quantizations before Z-Image, Ideogram and HiDream are tested.

SANA and SD3.5 remain useful low-memory or ecosystem controls, but neither is the best likely quality upgrade from FLUX.2 Klein.

## Recommended bake-off order

1. **LLM:** current Gemma vs EuroLLM-22B Q4; add Poro 2 as the Finnish control.
2. **TTS:** current VoxCPM2 vs OmniVoice and Higgs TTS 3; then X-Voice and MOSS. Chatterbox last.
3. **Images:** current FLUX.2 Klein vs Z-Image-Turbo and Ideogram 4 NF4; try HiDream next, Qwen-Image as a later control.

Measure:

- LLM: JSON-schema success, retry rate, Finnish spoken naturalness, script-length accuracy and 16K/32K throughput.
- TTS: same authorized speaker in EN/FI, names, numbers, compounds, loanwords, omissions/repetitions, speaker similarity, duration error, RTF and VRAM cleanup.
- Images: exact dimensions, subject/action/spatial fidelity, style range, unwanted text/logos, diversity across scenes, recurring-character consistency, warm latency and peak VRAM/RAM.

The attachment was right to retain the current stack as the baseline, but its next-step shortlist should be updated to include **EuroLLM/Poro, OmniVoice/Higgs/X-Voice/MOSS, and Z-Image/Ideogram 4**.

This was a source and repository audit; I did not download or benchmark new models. No files were changed.
