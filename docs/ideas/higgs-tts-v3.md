Supported Languages
The model reaches single-digit WER/CER on 102 languages, which split into two tiers.

WER/CER under 5 — polished, production-quality (85)
🇿🇦 Afrikaans · 🇸🇦🇪🇬 Arabic · 🇦🇲 Armenian · 🇮🇳 Assamese · 🇪🇸 Asturian · 🇦🇿 Azerbaijani · 🇷🇺 Bashkir · 🇪🇸 Basque · 🇧🇾 Belarusian · 🇧🇩🇮🇳 Bengali · 🇧🇦 Bosnian · 🇧🇬 Bulgarian · 🇪🇸 Catalan · 🇵🇭 Cebuano · 🇮🇶 Central Kurdish · 🇨🇳 Chinese · 🇭🇷 Croatian · 🇨🇿 Czech · 🇩🇰 Danish · 🇳🇱🇧🇪 Dutch · 🇷🇺 Eastern Mari · 🇺🇸🇬🇧🇦🇺 English · 🌐 Esperanto · 🇪🇪 Estonian · 🇫🇮 Finnish · 🇫🇷🇨🇦 French · 🇪🇸 Galician · 🇬🇪 Georgian · 🇩🇪🇦🇹 German · 🇬🇷 Greek · 🇮🇳 Gujarati · 🇭🇹 Haitian Creole · 🇳🇬 Hausa · 🇮🇱 Hebrew · 🇮🇳 Hindi · 🇭🇺 Hungarian · 🇮🇩 Indonesian · 🇮🇹 Italian · 🇯🇵 Japanese · 🇮🇩 Javanese · 🇮🇳 Kannada · 🇰🇿 Kazakh · 🇰🇷 Korean · 🇷🇼 Kinyarwanda · 🇰🇬 Kyrgyz · 🇱🇻 Latvian · 🇨🇩 Lingala · 🇱🇹 Lithuanian · 🇰🇪 Luo · 🇲🇰 Macedonian · 🇲🇾🇮🇩 Malay · 🇮🇳 Malayalam · 🇲🇹 Maltese · 🇳🇿 Māori · 🇮🇳 Marathi · 🇲🇳 Mongolian · 🇳🇵 Nepali · 🇳🇴 Norwegian · 🇫🇷 Occitan · 🇮🇷🇦🇫 Persian · 🇵🇱 Polish · 🇵🇹🇧🇷 Portuguese · 🇷🇴 Romanian · 🇷🇺 Russian · 🇿🇦 Sepedi · 🇷🇸 Serbian · 🇿🇼 Shona · 🇸🇰 Slovak · 🇸🇮 Slovene · 🇪🇸🇲🇽 Spanish · 🇹🇿🇰🇪 Swahili · 🇸🇪 Swedish · 🇵🇭 Tagalog · 🇹🇯 Tajik · 🇮🇳🇱🇰 Tamil · 🇮🇳 Telugu · 🇹🇭 Thai · 🇹🇷 Turkish · 🇺🇦 Ukrainian · 🇵🇰🇮🇳 Urdu · 🇨🇳 Uyghur · 🇺🇿 Uzbek · 🇻🇳 Vietnamese · 🇿🇦 Xhosa · 🇿🇦 Zulu

WER/CER between 5 and 10 — usable, but less polished (17)
🇦🇱 Albanian · 🇲🇼🇿🇲 Chichewa/Nyanja · 🇮🇳🇵🇰 Eastern Punjabi · 🇺🇬 Ganda · 🇮🇸 Icelandic · 🇮🇪 Irish · 🇩🇿 Kabyle · 🇨🇻 Kabuverdianu · 🇰🇪 Kamba · 🇻🇦 Latin · 🇱🇺 Luxembourgish · 🇪🇹🇰🇪 Oromo · 🇦🇫🇵🇰 Pashto · 🇵🇰🇮🇳 Sindhi · 🇸🇴 Somali · 🇦🇴 Umbundu · 🇬🇧 Welsh

Control Tokens
All tags follow <|category:value|> syntax and can be inserted mid-utterance.

For how to place these tags when writing the target text (sentence-level vs. inline, sfx formatting, stacking, worked examples), see PROMPTING.md.

Emotion — elation, amusement, enthusiasm, determination, pride, contentment, affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear, disgust, bitterness, sadness, shame, helplessness
Token	Description
<|emotion:elation|>	Elation / joy
<|emotion:amusement|>	Amusement / playful laughter
<|emotion:enthusiasm|>	Enthusiasm / excitement
<|emotion:determination|>	Determination / firmness
<|emotion:pride|>	Pride / confidence
<|emotion:contentment|>	Calm satisfaction
<|emotion:affection|>	Warmth / affection
<|emotion:relief|>	Relief
<|emotion:contemplation|>	Thoughtful / reflective
<|emotion:confusion|>	Confused
<|emotion:surprise|>	Surprised
<|emotion:awe|>	Awe / wonder
<|emotion:longing|>	Longing / yearning
<|emotion:arousal|>	Heightened desire
<|emotion:anger|>	Anger
<|emotion:fear|>	Fear
<|emotion:disgust|>	Disgust
<|emotion:bitterness|>	Bitterness
<|emotion:sadness|>	Sadness
<|emotion:shame|>	Shame
<|emotion:helplessness|>	Helplessness
 
Style — singing, shouting, whispering
Token	Description
<|style:singing|>	Singing
<|style:shouting|>	Shouting / projected voice
<|style:whispering|>	Whisper
 
Sound effects — cough, laughter, crying, screaming, burping, humming, sigh, sniff, sneeze
Pair each token with the matching onomatopoeia immediately after it.

Token	Description	Suggested onomatopoeia
<|sfx:cough|>	Cough	Ahem
<|sfx:laughter|>	Laughter	Haha / Hehe
<|sfx:crying|>	Crying	Boohoo / Sob
<|sfx:screaming|>	Screaming	Ahh / Aaah
<|sfx:burping|>	Burping	Burp
<|sfx:humming|>	Humming	Hmm / Mmm
<|sfx:sigh|>	Sigh	Uh / Ahh
<|sfx:sniff|>	Sniff	Sff
<|sfx:sneeze|>	Sneeze	Achoo
 
Prosody
Speed — speed_very_slow, speed_slow, speed_fast, speed_very_fast
Pauses — pause, long_pause
Pitch — pitch_low, pitch_high
Delivery — expressive_high, expressive_low
Token	Effect
<|prosody:speed_very_slow|>	≈0.65× speed
<|prosody:speed_slow|>	≈0.85× speed
<|prosody:speed_fast|>	≈1.2× speed
<|prosody:speed_very_fast|>	≈1.4× speed
<|prosody:pitch_low|>	≈−3 semitones
<|prosody:pitch_high|>	≈+2.5 semitones
<|prosody:pause|>	≈400–700 ms pause
<|prosody:long_pause|>	≈700–1500 ms pause
<|prosody:expressive_high|>	More expressive delivery
<|prosody:expressive_low|>	Flatter delivery
 
Evaluation Benchmarks
Multilingual Voice Clone
We evaluate Higgs TTS 3 on public multilingual TTS suites and our internal 111-language Higgs-Multilingual set, covering both common and lower-resource languages.

WER / CER (↓, ×100) macro-averaged across each benchmark's language set. Lower is better; bold marks the best per row. All numbers are reproducible end-to-end with original metrics and normalization.

Benchmark	Higgs TTS v2	Higgs TTS 3	Fish Audio S2 Pro	Qwen3-TTS-1.7B	VibeVoice-7B	IndexTTS-2	MiMo-Audio-7B-Instruct	MOSS-TTS-v1.5	OmniVoice	ChatterBox	FireRedTTS-2
SeedTTS	2.10	1.11	1.31	1.30	3.59	1.63	3.70	1.73	1.21	17.00	1.72
CV3	21.19	4.41	4.60	7.73	11.66	129.26	71.55	6.11	4.92	32.62	19.20
MiniMax-Multilingual	49.86	2.74	5.15	27.41	8.21	112.91	85.67	3.78	2.98	49.30	12.52
Higgs-Multilingual	52.24	3.61	8.68	97.09	13.74	57.71	59.61	21.28	3.63	57.52	33.69
Emergent TTS
Win-rate (↑) per category — judge preference vs the BASELINE row; bold marks the highest win-rate per column. For a fair comparison, every model shares the same reference audio per prompt, and we run the benchmark text verbatim — no inline control tags inserted.

Model	Overall ↑	Emotions ↑	Foreign Words ↑	Paralinguistics ↑	Complex Pronunciation ↑	Questions ↑	Syntactic Complexity ↑
Higgs TTS 3	53.65%	53.75%	48.75%	68.57%	25.10%	61.43%	60.71%
Fish Audio S2 Pro	43.80%	53.04%	33.93%	53.75%	18.16%	55.00%	45.71%
Qwen3-TTS-1.7B	38.84%	45.54%	24.64%	44.29%	30.00%	53.39%	34.11%
IndexTTS-2	31.12%	39.29%	5.36%	42.50%	12.45%	45.89%	38.93%
MOSS-TTS-v1.5	43.89%	60.54%	35.18%	51.43%	11.63%	53.21%	47.32%
OmniVoice	40.82%	61.07%	28.75%	52.68%	13.67%	45.00%	40.36%
Usage
SGLang Usage
Pair the weights in this repo with SGLang-Omni — a production serving stack with continuous batching for multi-codebook decoding and the same inline tag controls. The Higgs TTS cookbook walks you through installation, server launch, request examples, and the full API reference.

See the Higgs TTS cookbook for the full details.

Install and Serve
docker pull lmsysorg/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  lmsysorg/sglang-omni:dev /bin/zsh

git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v -e .

export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
hf download bosonai/higgs-tts-3-4b

sgl-omni serve \
  --model-path bosonai/higgs-tts-3-4b \
  --port 8000

Zero-shot synthesis
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav

Voice cloning
Supplying the reference transcript (text) materially improves cloning fidelity.

import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "ref.wav",
            "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
        }],
        "temperature": 0.8, "top_k": 50, "max_new_tokens": 1024,
    },
)
with open("output.wav", "wb") as f:
    f.write(resp.content)

Streaming (Server-Sent Events)
Set "stream": true to receive base64-encoded WAV chunks as the vocoder emits them — sub-second time-to-first-audio. Each event carries audio.data (base64 WAV bytes); the terminal event has finish_reason: "stop" plus usage metadata.

import requests, base64, json

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Get the trust fund to the bank early.", "stream": True},
    stream=True,
) as resp, open("output.wav", "wb") as f:
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        event = json.loads(line[6:])
        if event.get("finish_reason") == "stop":
            break
        audio = event.get("audio") or {}
        if audio.get("data"):
            f.write(base64.b64decode(audio["data"]))

Inline control tokens
Embed <|emotion:…|>, <|style:…|>, <|prosody:…|>, and <|sfx:…|> tokens directly in input. Two rules:

Delivery tokens first. Emotion, style, and the prosody speed / pitch / expressive tokens shape the whole turn — put them at the start of input. Positional tokens (<|prosody:pause|>, <|prosody:long_pause|>, <|sfx:…|>) go inline exactly where they fire.
Pair every <|sfx:…|> with its onomatopoeia. E.g. <|sfx:laughter|>Haha, <|sfx:sigh|>Uh, <|sfx:sneeze|>Achoo. The written sound gives the model the acoustic cue to realize the effect.
Example — amusement + laughter:

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, I was not ready for that."}' \
  --output output.wav