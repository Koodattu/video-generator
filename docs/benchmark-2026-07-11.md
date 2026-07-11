# Six-video benchmark — 2026-07-11

This benchmark used a 120-second target, draft delivery at 1280×720/30 fps, and eight generated
images per video. Wall time was measured outside the CLI and includes preflight, bounded DDGS
research, text generation and review, speech synthesis, alignment, image generation, captioning,
and both video encodes.

| Text / media stack | Language | Run ID | Wall time | Actual video | Wall / video |
|---|---|---|---:|---:|---:|
| Qwen 3.6 / local media | English | `20260711T161046Z-80c6d09d` | 613.199 s (10:13.2) | 114.888 s | 5.337× |
| Qwen 3.6 / local media | Finnish | `20260711T164943Z-49e8ddb6` | 571.785 s (9:31.8) | 113.988 s | 5.016× |
| Gemma 4 / local media | English | `20260711T170422Z-ffe1ea7f` | 361.305 s (6:01.3) | 107.822 s | 3.351× |
| Gemma 4 / local media | Finnish | `20260711T182949Z-b1116dc4` | 487.199 s (8:07.2) | 103.922 s | 4.688× |
| GPT-5.4 mini / Gemini / ElevenLabs | English | `20260711T184336Z-f38c5e76` | 239.210 s (3:59.2) | 109.622 s | 2.182× |
| GPT-5.4 mini / Gemini / ElevenLabs | Finnish | `20260711T185743Z-f1bf53be` | 467.998 s (7:48.0) | 113.122 s | 4.137× |

All six successful Run Bundles are complete. Each has eight image prompts, and both the positive and
negative prompts were detected as English. Each delivery has a primary H.264/AAC MP4 with selectable
`mov_text` captions, a separate MP4 with animated current-word highlighting burned in, and an SRT
sidecar. Every delivery manifest passed all 18 checks with zero warnings, and both MP4 variants were
decoded end to end with FFmpeg.

The local timings exclude one-time setup. Observed separately, the Qwen model download took about
269.8 seconds, Qwen runner setup took 102.4 seconds, and switching the local runner to Gemma took
85.1 seconds. Failed development attempts are also excluded from the benchmark table. In particular,
strict draft word-count failures stopped before speech/image generation, and one Finnish cloud attempt
exposed a 22 ms AAC/container rounding edge at the 120-second boundary. The final implementation keeps
the content timeline within budget and permits no more than one 30 fps frame of muxing tolerance.

Verification after the implementation: `104 passed` in the full pytest suite.
