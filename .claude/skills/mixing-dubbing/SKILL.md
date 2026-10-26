---
name: mixing-dubbing
description: >
  Use when re-voicing or translating audio/video with text-to-speech via the
  `mixing` package: turning text into spoken audio, synthesizing narration,
  translating subtitles, or producing a foreign-language dub of a video from
  its SRT. Trigger on requests like "dub this video", "re-voice this in
  another language", "text to speech / TTS this", "synthesize narration",
  "translate these subtitles", "make a French version of this clip", or
  "read this script in voice X". For transcribing speech → SRT in the first
  place, see mixing-transcript; for the `output` protocol and ffmpeg, see the
  mixing router skill.
---

# mixing-dubbing — TTS re-voicing & SRT translation

Synthesize speech with ElevenLabs, translate SRT cues, and dub a video from its
SRT — all from `mixing.dubbing`. TTS + SRT primitives are stdlib-light;
`dub_video_from_srt` is the only piece that pulls `moviepy`.

## Prereqs (read once)

- **`ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`)** must be set (or pass `api_key=`) for any TTS / voice
  call. Results are **cached on disk** (keyed by text + voice + model +
  settings), so re-running the same line is free and offline.
- **ffmpeg on PATH** for `dub_video_from_srt` (`mixing.has_ffmpeg()`).
- Translation's *default* path needs `aix`: `pip install 'mixing[llm]'`. Avoid
  it entirely by passing your own `translate_fn=`.

```python
from mixing.dubbing import text_to_speech, synthesize_to_file, list_voices, find_voice
from mixing.dubbing import translate_srt, parse_srt, dump_srt, Cue
from mixing.dubbing import dub_video_from_srt  # this import pulls moviepy
```

## Text → speech

`text_to_speech(...)` returns raw **bytes**; `synthesize_to_file(...)` writes a
file and returns the `Path`. (TTS predates the egress protocol — `synthesize_to_file`
takes a positional `path`, not `output=`; `dub_video_from_srt` below uses `output=`.)

```python
# bytes in memory
audio: bytes = text_to_speech("Hello world", voice_id, language_code="en")

# write to a file → returns Path
synthesize_to_file("Bonjour le monde", voice_id, "hello.mp3", language_code="fr")
```

`text_to_speech(text, voice_id, *, api_key=None, model_id="eleven_multilingual_v2",
output_format="mp3_44100_128", voice_settings=None, language_code=None,
timeout=600.0, cache=True, refresh=False) -> bytes`

- `model_id` default `eleven_multilingual_v2` — one voice speaks ~30 languages
  (the right default for dubbing).
- `voice_settings` is **merged over** `DFLT_VOICE_SETTINGS`
  (`stability=0.5, similarity_boost=0.8, style=0.0, use_speaker_boost=True`);
  pass only the keys you change.
- `output_format` e.g. `mp3_44100_128` (default), `mp3_44100_192` (paid tier),
  `pcm_44100`.
- `cache=True` uses the disk cache; `cache=False` disables it; a path uses that
  dir. `refresh=True` forces a re-call and overwrites the cached entry.
- `synthesize_to_file(text, voice_id, path, **kwargs)` forwards `**kwargs` to
  `text_to_speech`.

## Picking a voice

```python
voices = list_voices()  # account voices: voice_id, name, category, labels
v = find_voice("Brian")  # first whose name/labels match (case-insensitive)
v = find_voice("french")  # match on a label keyword
voice_id = v["voice_id"]
```

Need a voice your account doesn't have yet (e.g. a native French ad voice)? Pull
one from the public **shared** library, add it, then synthesize:

```python
hits = search_shared_voices(language="fr", use_cases="advertisement", gender="female")
top = hits[0]
voice_id = add_shared_voice(top["public_owner_id"], top["voice_id"], name="FR Ad")
synthesize_to_file("Découvrez notre offre.", voice_id, "ad_fr.mp3", language_code="fr")
```

- `search_shared_voices(*, language=, use_cases=, gender=, category=, sort=,
  page_size=40, api_key=, **extra_params)` → dicts with `voice_id` **and**
  `public_owner_id` (both needed to add).
- `add_shared_voice(public_owner_id, voice_id, name, *, api_key=)` → account-local
  `voice_id`. Idempotent-ish: if a voice with that name already exists it returns
  the existing id rather than erroring.

## SRT primitives (re-exported from `mixing.srt`)

```python
cues = parse_srt(open("promo.srt").read())  # -> list[Cue]
cues[0].start, cues[0].end, cues[0].text  # times are seconds (float)
srt_text = dump_srt(cues)  # serialize, renumbered from 1
from mixing.dubbing import srt_time_to_seconds  # "00:00:01,500" -> 1.5
from mixing.srt import seconds_to_srt_time  # 1.5 -> "00:00:01,500"
```

`Cue(index: int, start: float, end: float, text: str)` — `.duration` is
`end - start` (never negative). Note: `seconds_to_srt_time` lives in
`mixing.srt` (not re-exported by `mixing.dubbing`).

## Translate an SRT (timings preserved)

```python
fr_srt = translate_srt(open("promo.srt").read(), "French")  # LLM via aix
fr_srt = translate_srt(cues, "Spanish", source_language="English")  # cues also accepted
```

`translate_srt(srt, target_language, *, source_language=None, translate_fn=None)
-> str` — translates only cue *text*, keeps every timing. The default
`translate_fn` (`default_translate_fn`) calls an LLM through `aix` in a single
batched call and **must preserve segment count** (raises `ValueError` otherwise).

Avoid the `aix` dependency by injecting any `(texts, target_language,
source_language) -> list[str]` callable:

```python
def my_translate(texts, target_language, source_language=None):
    return [translate_one(t, target_language) for t in texts]  # your engine


fr_srt = translate_srt(srt_text, "French", translate_fn=my_translate)
```

## End-to-end: dub a video from its SRT

```python
dub_video_from_srt(
    "promo.mp4",
    "promo.srt",  # video + (path | raw SRT text | list[Cue])
    voice_id=voice_id,
    output="promo.en.mp4",  # the ONE egress param (None → <stem>.<lang|dub>.mp4 beside input)
    language_code="en",
)
```

`dub_video_from_srt(video, srt, *, voice_id, output=None, api_key=None,
model_id="eleven_multilingual_v2", output_format="mp3_44100_128",
voice_settings=None, language_code=None, fit="speed", max_speedup=1.5,
keep_original_audio=0.0, work_dir=None, keep_work=False, cache=True,
synth_fn=None) -> Path`

It synthesizes each cue, lays the clips on a silent track at their start times,
and muxes back over the video.

- `fit="speed"` (default): overlong lines are gently time-compressed (capped at
  `max_speedup`, hard-clamped to ffmpeg's 2.0 ceiling) so later cues hold sync;
  short lines pad with silence. `fit="natural"` never alters speed (lines may
  drift). No speech is ever trimmed.
- `keep_original_audio` ∈ [0,1]: `0.0` (default) fully replaces the audio;
  `0.15` keeps a bit of background music/ambience under the narration.
- `synth_fn=(text, out_path) -> Path` swaps the TTS backend (or injects a stub
  in tests) without an API key.

### Translate then dub (foreign-language dub)

```python
en_srt = open("promo.srt").read()
fr_srt = translate_srt(en_srt, "French")
dub_video_from_srt(
    "promo.mp4",
    fr_srt,  # pass the translated SRT text directly
    voice_id=find_voice("french")["voice_id"],
    language_code="fr",
    output="promo.fr.mp4",
)
```

## Gotchas

- `text_to_speech` returns **bytes**; only `synthesize_to_file` / `dub_video_from_srt`
  write files. TTS uses positional `path`; `dub_video_from_srt` uses `output=`.
- `voice_settings` is merged, not replaced — `{"stability": 0.7}` keeps the
  other defaults.
- `language_code` is an *ISO-639-1* hint (`"fr"`, `"en"`); some models ignore it.
  It also seeds the default output filename in `dub_video_from_srt`.
- No API key → `RuntimeError`; missing `aix` on the default translate path →
  `ImportError` (install `mixing[llm]` or pass `translate_fn=`).
- The disk cache makes repeat runs free — rely on it; pass `refresh=True` to
  force a fresh synthesis, `cache=False` to bypass entirely.
- `import mixing.dubbing` stays light; only accessing `dub_video_from_srt` pulls
  `moviepy`. Keep TTS/SRT-only scripts off that name to stay fast.
