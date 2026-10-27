---
name: mixing-audio
description: >
  Use for AUDIO editing with the `mixing` package: trim/crop a clip, fade in or
  fade out, join/concatenate clips (with optional crossfade), mix music under a
  voice or overlay one track on another, loop a short bed to a target length,
  duck a bed under dialogue, normalize loudness, convert to mono,
  resample, peek at raw samples, align two recordings of the same performance,
  or split a long recording (concert, DJ mix, radio show, podcast) into
  segments. Trigger on phrasings like "trim audio", "fade out the end", "join
  these audio clips", "add background music under the narration", "make it
  louder / normalize", "convert to mono", "align two recordings", "find the
  offset between these takes", "does this clip contain the song", "is the song in this video at all", "split a concert into songs", "separate speech
  from music", "find the songs in this radio show / tape", "songs vs jingles
  vs DJ talk", "what did the host say before each song", "trim the DJ talk off the start/end of each song", "where does the host stop talking", "loop this ambient clip to N seconds", "duck the music under
  the dialogue", "level the speakers", "one speaker is too quiet", "match loudness per speaker / LUFS", "this voice sounds too low", "measure the pitch / f0 of each speaker", "shift the pitch up two semitones". For video audio (replace/normalize a video's track) use
  mixing-video; this skill is audio-file → audio-file.
---

# mixing-audio — edit audio files

`mixing.audio` gives you a sliceable `Audio` view plus free functions for
fades, crop, concat, overlay, normalize, alignment, and segmentation. Time is
**seconds (float)** by default. Needs ffmpeg + the `[audio]` extra
(`pip install mixing[audio]` → pydub/soundfile).

```python
import mixing

assert mixing.has_ffmpeg(), "install ffmpeg (brew install ffmpeg / apt install ffmpeg)"
from mixing.audio import Audio  # or facade: mixing.Audio
```

## The `output` protocol (applies to every writer here)

`output=None` → object producers return the in-memory `Audio`; `Audio.save`
and the `crop_*`/`save_audio_clip` file producers save **beside the input**.
A file path writes there; a directory auto-names; a callable is a sink. All
return a `Path` (or the sink's value). Never `saveas=`/`output_path=`.

## `Audio` — sliceable, chainable, in-memory

```python
a = Audio("song.mp3")  # time_unit defaults to "seconds"
a.duration
a.full_duration  # this view vs the whole source (s)
a.sample_rate
a.channels
a.sample_count  # 44100, 2, ...

a[10:30]  # 10s–30s → new Audio (lazy view, no copy)
a[-30:]  # last 30s
Audio("song.mp3", time_unit="samples")[0:44100]  # 1s @ 44.1kHz (kw-only)
Audio("song.mp3", time_unit="milliseconds")[500:1500]

# chain effects, then save (each returns a new Audio)
a[5:120].fade_in(2).fade_out(3).save(output="final.mp3")  # -> Path
a[10:30].save()  # output=None → beside input, returns Path
a1 = Audio("p1.mp3")
a2 = Audio("p2.mp3")
(a1 + a2).save(output="joined.mp3")  # __add__ concatenates

# transforms (pure pydub, no extra deps) — all return a new Audio:
a.normalize()  # peak-normalize; normalize(headroom=0.1)
a.to_mono()  # downmix to 1 channel
a.resample(16000)  # change sample rate (Hz)
a.overlay(other, position=5.0, gain_during_overlay=-6)  # mix `other` on top

# context manager (frees the in-memory buffer on exit):
with Audio("song.mp3") as a:
    dur = a.duration

# raw samples (collections.abc.Mapping[int, float], normalized to [-1, 1]):
s = a.samples
s[0]
s[-1]
s[1000:2000]  # negative index + slicing supported
```

`save(output=None, *, format=None, bitrate="192k", **export_kwargs)` —
`format` auto-detected from the extension when `None`.

## Free functions (file-or-object in, `output`-governed out)

```python
from mixing.audio import (
    crop_audio,
    fade_in,
    fade_out,
    concatenate_audio,
    overlay_audio,
    loop_audio,
    duck_audio,
    save_audio_clip,
    find_audio_offset,
)

# trim — first 3 positional, rest keyword-only:
crop_audio("song.mp3", 10, 30, output="clip.mp3")  # -> Path
crop_audio("song.mp3", 44100, 88200, time_unit="samples")  # output=None → beside input

# fades accept a path OR an Audio; output=None returns the Audio object:
fade_in("intro.mp3", 2.0, output="faded.mp3")  # -> Path
faded = fade_out("song.mp3", 3.0)  # -> Audio (not saved yet)

# join any number of clips; crossfade is in seconds:
concatenate_audio("intro.mp3", "main.mp3", "outro.mp3", output="full.mp3")
concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)  # 500ms crossfade → Audio

# music under voice: mix_ratio is the PROMINENCE OF THE OVERLAY
#   1.0 = only overlay, 0.0 = only background, 0.5 = equal blend
overlay_audio("music.mp3", "voice.mp3", position=5.0, mix_ratio=0.35, output="mix.mp3")

# loop a short bed to a target length, crossfading every join (no click):
bed = loop_audio("room_tone.wav", 90.0)  # -> 90s Audio
loop_audio("waves.wav", 240.0, crossfade_s=1.0, output="bed.wav")  # -> Path

# duck a bed under dialogue (sidechain ducking, NOT a constant gain):
quiet_bed = duck_audio("bed.wav", "dialogue.wav", duck_db=-12)  # -> Audio

# extract a clip (audio_src=None pulls a path from the clipboard):
save_audio_clip("song.mp3", 10, 30, output="clip.mp3", format="mp3")

# align two recordings of the same take (cross-correlation) → offset in SECONDS:
offset = find_audio_offset("camera_audio.wav", "studio.mp3")  # float, sample_rate=16000
# positive => query (studio) starts `offset`s into the reference (camera)

# ...and when the clip is NOT one continuous take, ask which PARTS align:
from mixing.audio import aligned_spans

for sp in aligned_spans("song.mp3", "phone_recording.mp4"):
    sp.clip_start_s, sp.clip_end_s  # in the CLIP's timeline
    sp.offset_s, sp.confidence  # reference_time = clip_time + offset_s
    sp.support  # how much of the span's evidence reaches that offset — see below
    sp.margin  # ...and how far that is AHEAD of the runner-up offset — see below
    sp.reference_span  # the same extent on the SONG timeline
```

### One offset, or several?

`find_audio_offset` / `align_clips_to_reference` answer **"where does this clip sit?"**
with a single number. That is the right answer only for one continuous take. A recording
that was **stopped and restarted** has no such number — and the single-offset model does
not say so, it returns the offset of whichever part correlated best and describes the rest
of the clip wrongly, *at a confidence that clears any gate*. Measured on a clip holding two
takes of the same song: `0.494`.

`aligned_spans` is the windowed answer. A clip that IS one take returns exactly one span,
so it is a safe replacement rather than a different tool.

**Boundary resolution is `window_s` (default 20 s), and no better.** A window is evidence
that its whole extent aligns; a boundary falling inside one degrades that window rather
than locating itself within it. Ample for telling two takes apart, **not enough to cut
on**. Shrink `window_s` to buy precision — cost is linear in the window count.

### Repetitive music: gate on `support`, not on `confidence` alone

Both `aligned_spans` and `align_clips_to_reference` decide the offset by **consensus**
across analysis windows rather than by one argmax, because on a reference that repeats
verbatim the correlation's top two peaks are near-tied (0.987–0.993 second-to-first on
real music) and the argmax is close to a coin flip. Before the consensus pass a
verse/chorus reference split one continuous take into 4 spans, three of them at wrong
offsets — the worst scoring *higher* than the correct one.

So read the two numbers as different questions:

- **`confidence`** — how well the clip matches *where this answer puts it*. On repetitive
  material it stays high, and it should: the match really is that good.
- **`support`** (0–1, or `None`) — how much of the windows' own evidence reaches that
  offset. A window whose independent argmax landed there counts a full 1.0; a window that
  only put it on its **ballot** — could not separate it from its own answer — counts half,
  scaled by how close it scored; a window that never considered it counts nothing. 1.0 on
  material with no repeats; ~0.72 on verse/chorus; ~0.56 on an exactly tiling reference,
  where every offset is equally true and no confidence would ever say so. It also drops
  when only *part* of the clip is the reference at all (measured 0.545 on a
  half-song/half-noise clip), which is what localises where a clip stops matching.
- **`margin`** (a difference of two `support` tallies, or `None`) — how far that evidence
  sits **ahead of the best OTHER offset's**. Support cannot see a runner-up; this is the
  field that can. ~0.99 with no repeats; ~0.33 on verse/chorus; ~0.00 on an exactly tiling
  reference, where nine offsets are equally true.

High confidence with low support means *"it fits here beautifully — and it would fit
elsewhere too."*

**`support > 0.5` means at least one independent window found the offset unaided.** That
is the boundary the half-weight buys: ballot mentions alone can never carry the tally past
0.5, so the top half of the range is reserved for unaided agreement — which is exactly
what the whole number used to mean before it was graded. Gate above 0.5 if that is the
question you are asking; gate lower if "the clip's evidence points here" is enough.

**If you carried a threshold over from 0.0.48 or earlier, move it — do not reuse the
number.** Every value in `(0, 0.5]` is now reachable with *zero* windows having found the
offset unaided, so a gate that used to mean "a quarter of the windows located this
themselves" now means "some of the clip's evidence mentions it". A gate at 0.25 is inside
that band. If your threshold was chosen to mean unaided agreement, its equivalent on this
scale is **`> 0.5`**, not `> 0.25`.

That is not a cosmetic re-tune: measured at the default window, an **exactly tiling**
reference (the offset is a free choice among nine) lands just *below* 0.5 and a reference
that is **one half twice** lands just *above* it. A gate at 0.5 separates those two; a
gate at 0.25 passes both.

**What is unchanged by the grading is an undisputed argmax, not a long clip.** A 60 s clip
at the default 20 s window is bit-identical on a non-repeating reference — every window
got there on its own, so there is nothing to add. The same clip against a reference that
repeats moves from 0.00 to about 0.50, because no window resolves the repeat alone and all
of them have the true offset on their ballots. Length was never what made the old number
safe.

**It was a bare argmax headcount until 0.0.49** (issue #45), and the same fixtures read
0.44 / 0.11 under the old definition. The headcount broke on short clips: an argmax is a
real opinion at a 20 s window and close to a coin flip at the ~4 s window a 12 s clip is
fitted to, so the statistic got *less* confident exactly as fitting the window made the
estimator *more* reliable. Measured on real cross-device material, 21 alignments that were
**all correct** scored 0.00–1.00 with six at 0.00, and a gate at 0.25 refused about half
of them. Where the argmax was already decisive — a long clip at the default 20 s window —
the graded tally reports exactly what the headcount did.

**`align_clips_to_reference` fits the window to each clip.** Its `window_s`/`hop_s` default
to `None`, which means `min(20 s, clip_duration / 3)` with a floor of 3 s (300 onset-envelope
frames) and a hop of half whatever window comes out. A clip holding three default windows or
more is measured at 20 s exactly, so long clips are untouched; a short clip that used to be
one window — no vote, `support=None`, and whatever a single correlation said — now gets
several. Measured on real cross-device material: a 10 s clip against a 250 s reference came
back 102 s from the truth at confidence 0.834 as one window, and was right at every window
from 3 s to 6 s. **The threshold is independent LOOKS, not one window**: support needs two
windows separated by half a window, so the default grid needed about `window_s + hop_s` =
30 s before a clip had a second opinion — 22, 24 and 26 s clips against a 90 s reference all
read `None`, and the 22 s one was 64 s wrong at confidence 0.279 where `window_s=10` was
right to 3 ms. Pass a number to fix the window yourself; an explicit value is never
overridden.

**Read `ClipAlignment.window_s` whenever you gate on `support`.** It reports the window that
clip's vote was actually held at — the scale its `support` is on — and it is `None` exactly
when `support` is. Since the window is fitted per clip, a fixed threshold across clips of
different lengths compares numbers that answer different questions: measured on real
cross-device material, **21 alignments that were all correct reported the pre-0.0.49
argmax headcount from 0.00 to 1.00**, largely by clip length, because a 4 s window on a
12 s clip is a weaker opinion than a 20 s window on a 60 s one. Grading the tally softens
that spread but does not remove it. Scale your threshold to `window_s`, or pass an explicit
`window_s` to put every clip on one scale. `aligned_spans` is NOT adapted — there `window_s` is boundary resolution, which
is the caller's to choose.

**An explicit `window_s` a clip cannot hold raises `WindowTooWideForClip`** (0.0.53, issue #43). The bound is `clip_duration / 1.5` — `clip_duration / (1 + MAX_SUPPORT_OVERLAP)` — and **it does not involve `hop_s`**: the analysis grid always appends a tail window one window's length from the clip's end, so a second look exists or not regardless of the hop. (The intuitive `duration - hop_s` is wrong both ways: `window_s=11, hop_s=1` on a 15 s clip satisfies it and still returns `support=None`; `window_s=6, hop_s=12` violates it and measures `support=1.0`.) Above the bound you used to get one whole-clip correlation and a `support=None` that looked exactly like a clip too short to support at any window. The error is importable from `mixing` or `mixing.errors`, subclasses `ValueError`, and carries `clip_index`, `clip_duration_s`, `window_s`, `hop_s` and `max_window_s` — the largest window that still leaves a second look, computed in samples so a retry at exactly that value is guaranteed to measure. This is the explicit path only: `window_s=None` still fits the window to each clip and answers every clip, including one that lands on a single window at the floor.

**Reading `window_s` is necessary and, since 0.0.49, no longer sufficient.** A rule of the
form *"below the default window, treat the value as unmeasured"* was a complete defence
while `support` was an argmax headcount, because the window was the only axis that moved
it. Not any more: what a repeating reference does to the tally, it does at `window_s=20`
too — a 60 s clip there goes from 0.00 to about 0.50 on a bed that tiles. `window_s` still
tells you how strong each opinion was; nothing puts the number back on the scale a
pre-0.0.49 threshold was calibrated against. **Re-measure such a threshold against this
statistic; do not re-point it at it.**

**If you have a window guard, DELETE it — do not keep it alongside a new threshold.** A
rule that declines to trust `support` below the default window has to send those clips
somewhere, and the only thing left is `confidence`. That fallback is weaker than the
statistic the guard was protecting against. Measured on six pure-noise clips at a fitted
6.67 s window: confidences of 0.017–0.173 cleared a consumer's floor of 0.1 for **four of
the six**, while graded `support > 0.5` refused all six. Grading is what makes `support`
usable at a fitted window in the first place, so gate on it directly.

**`support is None` means NOT MEASURED — do not read it as 1.0.** You get it with
`consensus=False`, for a clip shorter than one window *at the window in force*, and when
the windows overlap by more than half (two windows sharing 95% of their samples are one
look read twice, not two opinions). One window cannot disagree with itself, and a unanimous
vote of one would vouch for an offset nothing corroborated. Measured, both at a pinned
`window_s=20`: a 15 s clip truly at offset 30 against a two-identical-halves reference comes
back at offset 75.0 with confidence 0.979 — and at the fitted default it comes back at 30.0,
which is the point of fitting it; and on real footage a 10 s clip at `window_s=9.5, hop_s=0.5`
had its two near-identical windows agree on an offset 102 s wrong. A `support` of 1.0 in
either case carries the wrong answer through any gate. When it is `None`, fall back to
`confidence` and know you are trusting a single opinion.

One consequence worth knowing: the tail window that covers a clip whose length is not a
whole number of hops usually starts less than half a window after its neighbour, so it is
excluded from the tally and the denominator loses one. Measured, that moved one real
clip's support from 0.45 to 0.50 — offsets are unaffected either way.

**`support` is relative to `window_s` — if you change the window, revisit your
threshold.** A shorter window is a weaker opinion, so fewer of them agree. Measured on the
same three correct cross-device alignments, on the pre-0.0.49 headcount: 0.45 / 0.64 / 0.73
at `window_s=20`, and 0.19 / 0.16 / 0.23 at `window_s=5`. It is deliberately not
normalised, so compare clips at
the same window and re-tune any gate you move the window under. With the fitted default,
clips of different lengths are NOT at the same window — pass an explicit `window_s` when you
rank clips by support.

### `support` is a floor; `margin` is a separator — gate on both

`support` answers *"how much of the clip's own evidence reaches this offset"*. It cannot
answer *"and how much reaches somewhere else instead?"* — and on repetitive material that
second question is the one that decides whether the answer is trustworthy. `margin`
(0.0.51, issue #47) answers it: the same graded tally, read at the best offset **outside
`offset_tolerance_s`** of the answer, subtracted. Same scale as `support`, same units,
same `None` rule.

Why a third number rather than a better threshold on the second: three scalars have now
been measured *not* to separate correct from wrong on real cross-device material.
`confidence` — the wrong offset's peak scored 0.987–0.993 of the right one's, sometimes
higher. `support` — an ambiguous tiling lands mid-scale *by construction*, because every
alias genuinely is reached by the same evidence. The window guard — deleted in 0.0.50,
having been measured not to separate them either. A fraction cannot see a runner-up.

Read the pair as a 2×2:

| | wide margin | margin ≈ 0 |
|---|---|---|
| **high support** | the evidence points here and nowhere else | it fits here *and fits somewhere else exactly as well* — an exact tiling |
| **low support** | thin evidence, undisputed — a short clip on a repeating bed. **A support-only gate refuses this and should not** | nothing is known |

Measured on synthetic material at `window_s=10`: no repetition `support 1.00 / margin
0.99`; verse/chorus `0.72 / 0.33`; exactly tiling `0.50 / -0.00`. And at the fitted
window, a **correct** 16 s clip on a bed that tiles every 2 s: `0.50 / +0.12`. That last
pair is the point — the ambiguous tiling and the correct short clip are
**indistinguishable on support** and opposite on margin.

**Measured on real cross-device material, `margin` is not an addition to a `support`
gate — it is a better gate.** 24 correct alignments against 6 pure-noise clips:
`support > 0.5` (the gate this skill recommended above) passed **18/24** and refused
**0/6**; `support > 0.5 and margin > 0` passed the same 18/24, so the conjunction added
nothing; and **`margin > 0` alone passed 24/24 and still refused 0/6**, recovering every
correct short clip the support floor was turning away. Five of the six noise clips came
back **negative** (-0.167 to -0.317) and no correct alignment did — read a negative
margin as a refusal, not just as a failure to vouch.

**Keep a floor under it anyway.** The sixth noise clip scored exactly `+0.000` — refused
by an exact tie, which a different draw does not guarantee — and two correct alignments
passed at `+0.014` and `+0.029` on a support of 0.34–0.35, the "thin evidence,
undisputed" row with nothing beneath it. `support > 0.25 and margin > 0` scored the same
24/6 while giving five of the six noise clips a second, independent reason to fail.
Thirty cases is encouraging and is not proof — so if you gate on one number, gate on
`margin`; if you gate on two, put a low support floor under it rather than the old 0.5.

**A wide margin is not a claim of sub-tolerance precision.** Offsets closer together
than `offset_tolerance_s` are one hypothesis by construction, so a candidate that near is
the answer and never subtracts: at the default tolerance a rival 0.20 s away leaves the
margin at 1.000, and the same rival at 0.30 s takes it to 0.505. Margin says nothing else
has an equal claim *at a different offset*; how sharply the offset itself is located is
`offset_tolerance_s`, which is yours to set.

**It can be slightly negative**, and is not clamped. The offset is chosen by the vote's
headcount over every window; the tally is graded and read over the independent windows
only, so the two need not rank identically. A negative margin says the tally actually
prefers somewhere else — the strongest *do not trust this* the object carries.

**`margin is None` exactly where `support is None`**, for the same reason: one independent
look has no runner-up to be ahead of, and a number invented there would vouch. It is
relative to `window_s` for the same reason `support` is — same tally, same scale.

`near_tie_ratio=0.0` disables the vote and restores the old per-window argmax;
`align_clips_to_reference(..., consensus=False)` restores its single whole-clip
correlation exactly (cheaper: one correlation instead of one per window).

### Cross-device recordings: `feature=` picks the offset, not just the score

`align_clips_to_reference` and `aligned_spans` default to `feature='envelope'`, and under
consensus (the default) that choice **moves the offset**. Two microphones in a room are not
sample-correlated even when the alignment is exact, so a raw-waveform correlation can put
its best peak somewhere the clip never was — measured on real multi-device footage at 15 s
from the truth, and *at `support=1.00`*, because the bias is the same in every window and a
vote ratifies what it cannot vary. The onset envelope reads WHEN energy arrives, which two
devices share, so it nominates the lag the waveform never offers; each nomination is then
scored by the better of the two views, so material with no onsets (a smooth tone, an export
against its own master) still lands on the waveform's answer.

Pass `feature='waveform'` when the clip comes from the SAME source as the reference.
`find_audio_offset` / `find_audio_offset_detailed` are single-shot — no windows, no vote —
so there `feature=` still changes only the confidence.

### Is the song in this clip at all? (`contains_song`, landmark fingerprints)

The aligner answers *where* a clip sits; `contains_song` answers *whether* the song is in it, independently (Shazam / audfprint landmark hashes, numpy + scipy only). Use it as a second opinion beside the aligner's `support`/`margin`, never instead of it.

```python
from mixing.audio import contains_song, landmarks, song_verdict, SongThresholds

index = landmarks("song.wav")  # hash the song ONCE (a Landmarks index)
for clip in clips:  # path, Audio, AudioSegment, or array + sr= (float or int PCM)
    m = contains_song(clip, index)  # SongMatch
    if m.verdict == "contains":  # "contains" | "absent" | "undecided"
        print(clip, m.offset_s)  # song time at the clip's t=0 (within one hop)
    print(
        m.votes, m.ratio, m.coverage, m.reason
    )  # K, R = K/best rival, K/overlapping hashes; why

song_verdict(record)  # re-judge a stored m.to_dict() under ITS thresholds
song_verdict(record, thresholds=SongThresholds(min_ratio=6))  # ...or under new ones
```

- **Rule:** "contains" = K >= 12 and R >= 5; "absent" = K <= 5 on >= 240 clip hashes and no speed probe (+-1-3 %) finds the song; else **"undecided", never a pass**. `SongMatch` has no truth value (`if m:` raises), so compare `m.verdict`.
- **Measured on synthetic media:** 0 of 115 hard negatives (same-key, same-beat-grid songs) "contains"; 87 of 98 degraded true cuts "contains", the rest "undecided". `rho` and `coverage` are reported but do not separate, so they are not judged.
- **"absent" means "not this recording within +-3 % of its speed".** A transposition of a semitone or more, or heavy clipping, can read absent. A looped bar reads "undecided" (R ~ 1: every repetition is an equally good offset).
- **Thresholds are uncalibrated.** They live in `SongThresholds`; matching knobs (`query_shifts`, `speed_probes`, tolerances) in `MatchParams`; hashing in `LandmarkParams`. Every `SongMatch` records all three, plus `off_speed`/`probed_speeds`; `song_verdict` never calls a record "absent" whose configured speed probes did not run. Calibrate on known non-matching clips, aliases and same-key songs included, with `query_shifts` as shipped.

## Segmentation — split a long recording into pieces

```python
from mixing.audio import find_segments, extract_segments, Segment

segs = find_segments("concert.wav", strategy="self_similarity")  # -> list[Segment]
for s in segs:
    s.start, s.end, s.duration, s.label, s.score  # all seconds; label/score optional
    s.as_start_end()  # (start, end)
    s.as_offset_duration()  # (start, duration)

# discover AND export in one call (output is a DIRECTORY, one file per segment):
paths = extract_segments(
    "concert.wav", strategy="self_similarity", output="songs/", format="mp3"
)  # -> list[Path]
# or pass timestamps you already have ((start, end) tuples or Segments):
extract_segments("mix.mp3", segments=[(0, 245), (247, 445)], output="tracks/")
```

Strategies (pick by regime; pass tuning kwargs straight through to `find_segments`):

| `strategy=` | When | Key kwargs |
|---|---|---|
| `"silence"` (default) | clean gaps (DJ mix, audiobook) | `silence_thresh_db=-40`, `min_silence_len=1.0` |
| `"energy_novelty"` | fade-outs / quiet moments, no true silence | `valley_threshold_factor=0.5`, `min_peak_distance_seconds=30` |
| `"self_similarity"` | concert/live: spectral content changes, level steady | `kernel_seconds=12.0`, `min_peak_distance_seconds=30` |
| `"speech_music"` | speech vs *instrumental* music, no extra deps (handcrafted features: cannot tell sung songs from talk) | `low_energy_threshold=0.5`, `min_segment_duration=3.0` |
| `"audioset"` | **radio shows / tapes with sung songs**: learned speech vs music, songs vs jingles (`mixing[ml]`) | `min_song_duration=75`, `snap_radius=4.0`, `music_threshold=0.5`, `classifier=` |

`find_segments(..., strategy=, min_segment_duration=, max_segment_duration=,
merge_gap=, pad_start=, pad_end=, **strategy_kwargs)`. You can also pass your
own `(AudioSegment, **kwargs) -> list[Segment]` callable as `strategy`. The
`segment_by_silence/energy/self_similarity/speech_music/audioset` functions are exported
too if you want to call one directly on a pydub `AudioSegment`.

### Radio shows: songs, jingles and the host's intros (`mixing[ml]` extra)

```python
from mixing.audio import find_segments, extract_segments, text_before_segments
from mixing.srt import parse_vtt_words

segs = find_segments("show.m4a", strategy="audioset", min_song_duration=75)
# back-to-back Segments labelled "speech" | "song" | "jingle" (no min_song_duration -> "music")
songs = [s for s in segs if s.label == "song"]
extract_segments("show.m4a", songs, output="songs/", format="m4a")

# what the host said just before each song (where titles are announced):
words = parse_vtt_words(
    open("show.en.vtt").read()
)  # YouTube captions -> Scribe-shaped words
intros = text_before_segments(
    songs, words
)  # or transcribe(...)["words"], or parse_srt cues
```

A pretrained AudioSet tagger (default: AST, `MIT/ast-finetuned-audioset-10-10-0.4593`) scores P(speech)/P(music) in 5 s windows every 2 s; music windows (P > 0.5 and > P(speech)) are median-filtered, grouped into runs, and every boundary is snapped to the quietest 100 ms within ±4 s. On a 29-min 1970s radio tape it found the same 7 songs and 7 jingles as the reference prototype (all 14 song edges within 0.05 s) in ~50 s on Apple-silicon MPS; two sibling tapes gave 8 and 7 songs.

- **The model is a seam**: `classifier=` takes any `(samples, sr) -> WindowProbabilities` (`starts`, `ends`, `speech`, `music` arrays). Wrap foley's `PannsTagger` or a CLAP tagger in a few lines; `ast_audioset_classifier(window_seconds=, hop_seconds=, device=, model_name=)` tunes the default. Tests inject a stub here, so they need no torch.
- `pip install 'mixing[ml]'` (torch + transformers). Imported lazily: `import mixing.audio` never loads torch; calling the default classifier without the extra raises an `ImportError` naming `mixing[ml]`. The model (~350 MB) downloads from the Hugging Face hub on first use.
- Non-music segments are all labelled `"speech"`: silence and noise included.
- Resampling matters: the strategy hands native-rate samples to the classifier, which resamples with an anti-aliased polyphase filter. pydub's own resampler is not anti-aliased and dropped the tagger's music probabilities from ~0.8 to ~0.5 on the real tape.

### Host talk at the start/end of a carved song (`find_talk_edges`)

Carved songs often still carry the announcer: an intro over the opening bars, a back-announce over the fade. `find_talk_edges` says where (talk over music counts as talk), so you can trim it or annotate it without re-analysing:

```python
from mixing.audio import find_talk_edges
from mixing.srt import parse_vtt_words

from pydub import AudioSegment

tape = AudioSegment.from_file("show.m4a")  # decode once, not once per song
words = parse_vtt_words(open("show.en.vtt").read())  # optional, tape timeline
for song in songs:  # Segments from find_segments(..., strategy="audioset")
    e = find_talk_edges(tape, song, words=words)  # times in the tape timeline
    e.intro, e.outro  # Segment(label="talk") or None
    e.clean_start, e.clean_end  # the music proper; e.clean is a Segment
    e.snapped_to_words  # True if the transcript refined an edge
# a carved song file works too: find_talk_edges("song.m4a") (shift caption times by -song.start)
```

- Same classifier seam as `"audioset"` (`classifier=`), default AST with 3 s windows every 0.5 s (`mixing[ml]`). Only P(speech) and P(music) are used.
- Sung vocals read as speech too (0.4-0.65 inside singing vs 0.5-0.8 for a host over a bed), so a window *seeds* talk only if `P(speech) >= talk_threshold + music_penalty * P(music)` (0.45 + 0.4 x music: singing comes with the full band); talk already under way at the cut needs only `edge_threshold` (0.6). Seeded regions grow down to `talk_floor` (0.4). The song body is the first/last non-talk run of `min_music_seconds` (8): shorter music (a sting) stays inside the intro, talk inside the body is never an edge.
- `words=` (Scribe words, `parse_vtt_words`, Whisper `"word"` dicts, or cues, in `audio`'s timeline) snaps the intro end to the last word starting in the 1.5 s before the audio estimate and the outro start to the first word within 1.5 s of it. It sharpens *where*, not *whether*: lyrics are transcribed too, so the audio decides what is talk.
- Measured against 99 hand-labelled edges of 57 songs from 9 tapes (captions, about ±0.5 s). On the 20 edges with real talk in the test split (labelled after the defaults were frozen, scored once): with captions, median error 0.08 s, 75% within 0.5 s, 90% within 1.8 s, worst 11.6 s (a host over a loud bed that never reached the seed bar); audio only, median 0.5 s, 90% within 1 s. Edges with no talk came out exact. Labels: `mixing/tests/data/talk_edges_labels.json` (times only; the slow test re-scores them with `MIXING_RUN_SLOW=1 MIXING_MWM_TAPES=<tapes dir>`).
- Known misses: a sung line that reads as talk right after the host with no long enough music between them (the intro runs on into the song); a jingle of `min_music_seconds` or more *before* the host's intro (it is taken for the song body, so the intro is missed: raise `min_music_seconds` above the jingle); talk only in the last ~1.5 s (the last 3 s window barely sees it).
- `talk_edges_from_probs(probs, start=, end=, ...)` is the audio half on precomputed `WindowProbabilities`, for re-tuning without re-running the model.

### Remove the host's talk, keep the music (`remove_talk`, `mixing[separation]` extra)

`find_talk_edges` says *where* the host talks; `remove_talk` takes the talk out and keeps the music under it (singing included), so a song can start at its real start instead of at `clean_start`:

```python
from mixing.audio import find_talk_edges, remove_talk, separate_speech

edges = find_talk_edges(audio, song, words=words)
clean = remove_talk(
    audio, edges
)  # TalkEdges, Segments, (start, end) pairs or (k, 2) arrays, in seconds
remove_talk(
    "song.wav", [(0.0, 12.4)], output="out/"
)  # writes out/song_talk_removed.wav
speech, music = separate_speech("clip.wav")  # the seam on its own
```

Install the default separator in two steps. **Never `pip install clearvoice` with its dependencies**: clearvoice 0.1.2 pins `numpy<2`, `librosa==0.10.2.post1`, `opencv-python==4.10.0.84` (a second `cv2` next to media's `opencv-contrib-python`), `soundfile==0.12.1` and scenedetect.

```bash
pip install 'mixing[separation]'                  # torch, torchaudio, huggingface_hub, librosa, soundfile, einops, rotary_embedding_torch, yamlargparse, tqdm, torchinfo, joblib
pip install --no-deps 'clearvoice==0.1.2'          # exact pin: the adapter calls 0.1.2's internals
pip install 'mixing[demucs]'                      # only for the htdemucs strategy
```

- The separator runs only on each span widened by `margin` (0.3 s) plus the strategy's `context_s`, never the whole file. The music estimate fades in linearly over the margin before the span and out over the margin after it, so the span itself is all music estimate; `margin=0` is a hard cut. Overlapping widened spans merge (the fades cross between them). Everything outside the widened spans is sample-identical to the input. Spans are validated alike whatever their form (finite, `end > start`).
- `audio`: an `Audio`, a path, an `AudioSamples` view, or an array `(n,)` / `(n, ch)` with `sr=` (float in [-1, 1], or signed-integer PCM, scaled by full scale and given back in its dtype). Results come back in the input's kind; `output=` follows the egress protocol, and `**save_kwargs` reach `Audio.save` for array results too.
- `separator=`: `"mossformer2_se_48k"` (default; MossFormer2-SE-48K speech enhancer *inverted*: mixture minus the speech estimate, always split with a Wiener mask, so a bad estimate can never make the music louder than the mixture; keeps singing; Apache-2.0), `"htdemucs"` (Demucs' `vocals` stem as speech; **instrumental beds only**, it removes singing too; MIT), a `SeparationStrategy`, or any callable `(samples (n, ch), sr) -> (speech, music)` (this is how to plug in Bandit-v2 or another DnR model; wrap it in `SeparationStrategy(..., mono=True, sample_rate=...)` to get per-channel runs and resampling).
- `SEPARATORS` is the registry; every entry has `license` and `commercial_ok` (filter: `{k: s for k, s in SEPARATORS.items() if s.commercial_ok}`). CC BY-NC checkpoints are deliberately not registered.
- A mono-only strategy runs per channel (radio music is wide, the talk centred: this matters). **A strategy at a lower rate than the input passes the band above its Nyquist to the "music" untouched, talk included** (sibilants above 8 kHz stay in with a 16 kHz enhancer). `invert_enhancer(enhance)` turns any speech enhancer into a separator.
- Long spans work: the adapter trims clearvoice's padded output for inputs over 20 s, and pads the input when clearvoice would otherwise leave the last 0.5 s undecoded.
- Evidence (#63): on real talk over singing, DnR-trained separators reach +9.7 dB SI-SDR, inverted MossFormer2-SE +7.3, vocal/instrumental separators (htdemucs) -3.7; over an instrumental, +3.8 / +0.8 / +5.4. On the synthetic slow-test clip (formant voice at 0 dB over a stereo instrumental, CPU, numpy 2): MossFormer2 +0.1 -> +5.1 dB, htdemucs +0.1 -> +10.2 dB.

## Beat / downbeat / onset analysis (`mixing[beats]` extra)

```python
from mixing.audio import beat_grid, BeatGrid

grid = beat_grid("song.mp3")  # -> BeatGrid; installs librosa lazily on first call
grid.beat_times  # np.ndarray of beat instants, seconds, ascending
grid.downbeat_times  # best-effort; empty array for the default (only) "librosa" backend
grid.onset_env  # onset-strength envelope, one value per STFT hop
grid.onset_hop_s  # seconds between onset_env frames (hop_length / sample_rate)
grid.tempo_bpm  # estimated global tempo
grid.to_dict()  # JSON-round-trippable summary (arrays -> lists; onset_env omitted)
```

`backend="librosa"` is the only shipped backend (ISC, commercial-clean); a
stronger downbeat tracker can be added later behind the same `BeatGrid` shape.
Needs `pip install mixing[beats]` (`librosa`) — `import mixing.audio` stays
light without it; the import happens lazily inside `beat_grid` itself. First
customer is muvid's footage-scoring layer (computes one grid on the master
song, maps clips onto it via their known offset).

## A whole file to a loudness target (one gain, streaming)

To bring a file (a narration take, a clip, an hour-long track) to a target **without** riding its dynamics the way ffmpeg `loudnorm` does, use `mixing.loudness.level_file`: it measures once, applies one gain in float, holds a true-peak ceiling with a limiter, and writes the file, all in bounded memory (two decode passes, nothing held but a chunk and a limiter block). Needs only `mixing[loudness]` and an ffmpeg, not the media stack.

```python
from mixing.loudness import level_file

r = level_file(
    "take.wav", "take_leveled.wav", target_lufs=-16, ceiling_db=-1.5, sample_rate=44100
)
(
    r.gain_db,
    r.measured_lufs,
    r.output_lufs,
    r.output,
)  # output_lufs < target: the limiter took loudness off
```

The output is audio only (in place over a video is refused). `sample_rate` resamples while decoding, so the ceiling holds at the rate written. Silence gets no gain; a gain above `max_gain_db` (24 dB) is capped with a warning. `true_peak=False` limits sample peaks; `encode_args=("-c:a", "pcm_s24le")` picks the codec. The pieces are public: `stream_loudness`, `stream_limit_peaks` (the limiter over chunks), `mixing.decode.decode_float_chunks` / `encode_float_chunks`.

## Speaker turns: loudness, voice diagnosis, pitch

Turns are `[(speaker, start_s, end_s), ...]` in **seconds** (any `time_unit`), or `Segment`s whose `label` is the speaker (e.g. from `find_segments`); non-overlapping, boundaries in pauses.

```python
from mixing.audio import (
    loudness,
    level_turns,
    process_turns,
    voice_profile,
    shift_pitch,
)

loudness("talk.wav")  # BS.1770-4 integrated LUFS; -inf for silence

# one speaker much quieter than the other: bring every turn to -16 LUFS
# (pregain -> compressor -> optional denoise -> gain -> -1.5 dBTP true-peak limiter, per turn)
turns = [("host", 0.0, 12.4), ("guest", 12.9, 30.2), ("host", 30.6, 41.0)]
level_turns(
    "talk.wav",
    turns,
    target_lufs=-16,
    chains={"guest": "highpass=f=90"},
    output="leveled.wav",
)

# any per-turn edit: process(speaker, padded, core) -> padded'; gaps untouched, exact length
process_turns("talk.wav", turns, lambda spk, padded, core: padded)

# "the guest sounds too low": diagnose BEFORE shifting pitch
voice_profile(
    "talk.wav", turns
)  # or {"guest": [(12.9, 30.2)]} -- a LIST of spans per label
# {"host": {"duration_s", "f0_hz", "voiced_fraction", "loudness_lufs",
#           "spectral_centroid_hz", "low_band_db"}, "guest": {...}}

shift_pitch(
    "guest.wav", 2, output="guest_up.wav"
)  # +2 semitones, same duration and loudness
```

- `voice_profile`: `f0_hz` is the median f0 over voiced frames of `f0_tracker` (default `pyin_tracker()`, needs `mixing[beats]`; `f0=False` skips it). Configure pyin through the tracker, `f0_tracker=pyin_tracker(fmin=50, fmax=400, min_voiced_prob=0.2)` (`voice_profile` itself no longer takes them). Default range 65-600 Hz; a voice above `fmax` reads as a subharmonic. Spans shorter than the tracker's `min_samples` (pyin: one 128 ms frame) give `f0_hz=None, voiced_fraction=None`. Any `(mono samples, rate) -> (f0 per frame, voiced flags)` callable is a tracker; it applies its own voicing threshold, e.g. over `denote`:

  ```python
  def denote_tracker(*, threshold=0.5, **kwargs):
      def track(y, fs):
          pitch = denote.get_pitch(y, sr=fs, **kwargs)
          return pitch.frequencies, pitch.confidence >= threshold

      return track


  voice_profile("talk.wav", turns, f0_tracker=denote_tracker(backend="torchcrepe"))
  ```
- `low_band_db` is the 60-150 Hz share of the energy: it overlaps a low male fundamental, so compare it across takes of one speaker rather than across speakers. A voice that *sounds* low is often just quieter and boomier: fix with EQ in `level_turns(chains=...)`, not with pitch.
- `shift_pitch(audio, semitones, *, preserve_formants=True, preserve_loudness=True, method="auto")`: `"rubberband"` (ffmpeg built with librubberband; the only method that keeps formants, so the voice stays the same person), `"resample"` (stock ffmpeg `asetrate`+`aresample`+`atempo`, no extra dependency), `"librosa"` (`mixing[beats]`), or a callable `(samples (n, ch) float, rate, semitones, preserve_formants) -> (n, ch)` (length within 5 ms, else `ValueError`). `"auto"` = rubberband if `has_rubberband()`, else resample with a warning when `preserve_formants=True`. Point mixing at a rubberband ffmpeg with `IMAGEIO_FFMPEG_EXE`.
- The ffmpeg methods' delay depends on the ratio (rubberband at 44.1 kHz: +60 ms at -24 st, +22 at -12, -11 at +12); it is measured per ratio on five seeded probes (median; `shift_latency`, cached per filter, rate and ffmpeg executable; `RuntimeError` if the lag hits the 0.5 s search limit, e.g. a filter that outputs silence) and removed, and the tail is kept. Residual on tone bursts: rubberband under 4 ms on average at 44.1/48 kHz; resample's `atempo` smears each burst by 10-30 ms; at the extremes (+-24 st) single bursts can be off by tens of ms with either method.
- Loudness is restored unless that would clip: if the restored peak would exceed full scale, the result is scaled so its peak lands at `CLIP_CEILING` (~-0.1 dBFS), costing only the overshoot in loudness.
- `functools.partial(rubberband_strategy, pitchq="quality")` as `method` dulls the sound less (smaller delay too).

## Common recipes

```python
# Trim, normalize, fade, save:
Audio("raw.wav")[2:65].normalize().fade_in(1).fade_out(2).save(output="clean.mp3")

# 16kHz mono for an STT/ML pipeline:
Audio("voice.mp3").to_mono().resample(16000).save(output="voice_16k.wav")

# Duck background music under narration (music at ~30% under the voice):
overlay_audio(
    "music.mp3", "narration.mp3", position=0.0, mix_ratio=0.3, output="podcast.mp3"
)

# Ambient bed under a dialogue track: loop to length, duck, mix at 25%:
base = Audio("dialogue.wav")
bed = duck_audio(loop_audio("room_tone.wav", base.duration), base)
overlay_audio(base, bed, mix_ratio=0.25, output="with_room_tone.wav")
# (one call over a file: mixing.video.overlay_ambient_bed — works on video too)

# Align a clean studio take to a camera recording, then crop the camera to match:
off = find_audio_offset("camera.wav", "studio.mp3")
crop_audio("camera.wav", off, off + Audio("studio.mp3").duration, output="aligned.wav")

# Split a radio show into talk, songs and jingles and save each (mixing[ml]):
extract_segments("show.mp3", strategy="audioset", min_song_duration=75, output="parts/")
```

## Gotchas

- `Audio(..., time_unit=...)` is **keyword-only**. Slicing uses that unit;
  `step` in a slice (`a[::2]`) is rejected.
- Slicing with a single index (`a[100]`) returns a 1-sample-long `Audio`, not a
  number. For raw values use `a.samples[100]`.
- `output=None` for `Audio.save`/`crop_audio`/`save_audio_clip` writes **next to
  the input** (auto-named) — pass an explicit `output` to control location.
- `overlay_audio`'s `mix_ratio` is the overlay's prominence (1.0 = only overlay).
  Don't confuse it with `Audio.overlay`'s `gain_during_overlay` (a dB number).
- `loop_audio`'s `crossfade_s` is **clamped to half the source duration** —
  each join must still advance the timeline. The tail is cut wherever the target
  lands (mid-loop is normal) and no fade-out is applied; chain `fade_out` if the
  bed ends exposed.
- `duck_audio` is a level-detector ducker, **not** a compressor and **not** a
  VAD: fixed depth (`duck_db`, no ratio/knee), no lookahead, and *any* loud
  sidechain content (music, a slam) ducks the bed exactly like speech would.
  Raise `threshold_db` if room noise is triggering it. The result always has the
  **bed's** duration; a shorter sidechain leaves the tail un-ducked.
- `find_audio_offset` returns **seconds** (positive = query starts inside the
  reference); the two recordings need only a shared correlated component.
- `aligned_spans` merges neighbours that AGREE on the offset across a gap of up to
  one window, and that is not tidying: a dozen seconds of silence mid-take collapses
  the windows inside it to a confidence of exactly 0, so one continuous recording
  would otherwise come back as two spans carrying the **same** offset — a
  stop/restart that never happened. A stop and a restart cannot resume in sync, so
  agreement across a short gap is positive evidence of continuity. The gap bound is
  what keeps it honest: past it, correlation cannot tell "quiet" from "different
  material", and both spans are reported rather than one claiming a range nothing
  measured. Widen with `merge_gap_s` if your material warrants it.
- `extract_segments(output=...)` treats `output` as a **directory** (it emits
  many files), unlike the single-file writers. Default dir is the source's
  parent (or cwd for non-path input).
- Segmentation is a tuned toolbox, not magic — expect to adjust the per-strategy
  kwargs (thresholds, distances) for your material.
```
