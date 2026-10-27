"""Tests for mixing.audio.separation: separate_speech and the remove_talk splice.

Everything above the ``slow`` section runs with fake separators (no torch): the
splice (identity outside the widened spans, linear crossfades, clamping at the
file edges, merging), the span inputs (TalkEdges, Segments, pairs), the framework
around a strategy (per-channel runs, resampling with high-band pass-through), the
registry metadata, and the ImportError naming the extra. The real strategies run
only in the ``slow`` tests at the bottom (``MIXING_RUN_SLOW=1`` plus the
``separation`` / ``demucs`` extras), each on a semi-synthetic clip whose true
music is known, scored by SI-SDR.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("pydub")

from mixing.audio import Audio
from mixing.audio.segmentation import Segment
from mixing.audio.separation import (
    DEFAULT_SEPARATOR,
    SEPARATORS,
    SeparationStrategy,
    invert_enhancer,
    remove_talk,
    separate_speech,
    wiener_split,
)
from mixing.audio.talk_edges import TalkEdges
from mixing.util import EXTRAS

SR = 8000


def _signal(seconds=3.0, *, sr=SR, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    return 0.3 * rng.uniform(-1, 1, size=(int(seconds * sr), channels))


def _zero_music(x, sr):
    """A fake separator: all of it is speech, the music is silence."""
    return x, np.zeros_like(x)


class _Recorder:
    """A fake separator that records what it was given."""

    def __init__(self, music=lambda x: np.zeros_like(x)):
        self.calls: list[tuple[tuple, int]] = []
        self._music = music

    def __call__(self, x, sr):
        self.calls.append((x.shape, sr))
        m = self._music(x)
        return x - m, m


# ---------------------------------------------------------------------------
# The splice
# ---------------------------------------------------------------------------


def test_outside_the_widened_spans_is_bit_identical(make_tone_audio):
    path = make_tone_audio(3.0, fmt="wav", sample_rate=SR, channels=2)
    audio = Audio(str(path))
    out = remove_talk(
        audio, [(1.0, 1.5)], separator=_zero_music, margin=0.2, crossfade=0.2
    )
    raw_in, raw_out = audio._get_segment().raw_data, out._get_segment().raw_data
    width = audio._get_segment().frame_width
    a, b = int(0.8 * SR) * width, int(1.7 * SR) * width
    assert len(raw_out) == len(raw_in)
    assert raw_out[:a] == raw_in[:a] and raw_out[b:] == raw_in[b:]
    assert raw_out[a:b] != raw_in[a:b]
    inner = out._get_segment().get_array_of_samples()
    core = np.array(inner[int(1.0 * SR) * 2 : int(1.5 * SR) * 2])
    assert np.all(core == 0)  # the span itself is all (silent) music


def test_crossfades_are_linear_and_continuous():
    x = np.full((3 * SR, 1), 0.5)
    margin, xf = 0.3, 0.1
    out = remove_talk(
        x, (1.0, 2.0), separator=_zero_music, sr=SR, margin=margin, crossfade=xf
    )
    a, b, n = int(0.7 * SR), int(2.3 * SR), int(xf * SR)
    w = 1 - out[:, 0] / 0.5  # weight of the (zero) music
    assert np.all(w[:a] == 0) and np.all(w[b:] == 0)
    assert np.allclose(w[a + n : b - n], 1)
    ramp = w[a : a + n]
    assert np.allclose(np.diff(ramp), 1 / n)  # linear
    assert np.allclose(w[b - n : b], ramp[::-1])
    assert np.max(np.abs(np.diff(out[:, 0]))) <= 0.5 / n + 1e-12  # no jump


def test_spans_clamp_at_the_file_edges_without_a_fade():
    x = np.full((2 * SR, 1), 0.5)
    out = remove_talk(x, [(-1.0, 0.2), (1.9, 5.0)], separator=_zero_music, sr=SR)
    assert out[0, 0] == 0 and out[-1, 0] == 0  # fully music right at the edges
    assert out[SR, 0] == 0.5  # untouched in the middle


def test_overlapping_widened_spans_merge_into_one_run():
    x = _signal()
    rec = _Recorder()
    remove_talk(x, [(1.0, 1.4), (1.8, 2.2)], separator=rec, sr=SR, margin=0.3)
    assert len(rec.calls) == 1  # 1.4 + 0.3 >= 1.8 - 0.3: one widened span
    rec2 = _Recorder()
    remove_talk(x, [(0.5, 0.7), (2.0, 2.2)], separator=rec2, sr=SR, margin=0.3)
    assert len(rec2.calls) == 2


def test_the_separator_runs_on_the_widened_span_plus_context_only():
    x = _signal(10.0)
    rec = _Recorder()
    strategy = SeparationStrategy("rec", rec, context_s=0.5)
    remove_talk(x, (4.0, 5.0), separator=strategy, sr=SR, margin=0.25)
    ((shape, sr),) = rec.calls
    assert sr == SR and shape == (int((1.0 + 2 * 0.25 + 2 * 0.5) * SR), 1)


def test_talk_edges_segments_and_pairs_give_the_same_result():
    x = _signal(6.0, seed=3)
    intro, outro = Segment(0.0, 1.5, label="talk"), Segment(4.5, 6.0, label="talk")
    edges = TalkEdges(0.0, 6.0, intro, outro, clean_start=1.5, clean_end=4.5)
    halve = lambda a, r: (a / 2, a / 2)
    by_edges = remove_talk(x, edges, separator=halve, sr=SR)
    by_pairs = remove_talk(x, [(0.0, 1.5), (4.5, 6.0)], separator=halve, sr=SR)
    by_segments = remove_talk(x, [intro, outro], separator=halve, sr=SR)
    assert np.array_equal(by_edges, by_pairs) and np.array_equal(by_pairs, by_segments)
    no_talk = TalkEdges(0.0, 6.0, None, None, clean_start=0.0, clean_end=6.0)
    assert np.array_equal(remove_talk(x, no_talk, separator=halve, sr=SR), x)


def test_no_spans_never_calls_the_separator():
    x = _signal()
    rec = _Recorder()
    assert np.array_equal(remove_talk(x, [], separator=rec, sr=SR), x)
    assert rec.calls == []


def test_one_dimensional_input_keeps_its_shape():
    x = _signal()[:, 0].astype(np.float32)
    out = remove_talk(x, (1.0, 2.0), separator=_zero_music, sr=SR)
    assert out.shape == x.shape and out.dtype == np.float32


def test_bad_inputs_are_informative():
    x = _signal()
    with pytest.raises(ValueError, match="end > start"):
        remove_talk(x, (2.0, 1.0), separator=_zero_music, sr=SR)
    with pytest.raises(ValueError, match="unknown separator"):
        remove_talk(x, (1.0, 2.0), separator="nope", sr=SR)
    with pytest.raises(ValueError, match="sr="):
        remove_talk(x, (1.0, 2.0), separator=_zero_music)
    with pytest.raises(ValueError, match="shape"):
        separate_speech(x, separator=lambda a, r: (a[:, 0], a[:, :0]), sr=SR)
    with pytest.raises(ValueError, match="frames"):
        separate_speech(x, separator=lambda a, r: (a[:100], a[:100]), sr=SR)


def test_output_path_writes_the_audio(make_tone_audio, tmp_path):
    path = make_tone_audio(2.0, fmt="wav", sample_rate=SR)
    out = remove_talk(
        str(path), (0.5, 1.0), separator=_zero_music, output=tmp_path / "x.wav"
    )
    assert out == tmp_path / "x.wav" and Audio(str(out)).duration == pytest.approx(2.0)
    arr = remove_talk(
        _signal(), (0.5, 1.0), separator=_zero_music, sr=SR, output=tmp_path
    )
    assert arr == tmp_path / "audio_talk_removed.wav" and arr.exists()


# ---------------------------------------------------------------------------
# The framework around a strategy
# ---------------------------------------------------------------------------


def test_a_mono_strategy_runs_once_per_channel():
    x = _signal(channels=2)
    x[:, 1] *= -1
    rec = _Recorder(music=lambda a: 0.5 * a)
    strategy = SeparationStrategy("rec", rec, mono=True)
    speech, music = separate_speech(x, separator=strategy, sr=SR)
    assert [shape for shape, _ in rec.calls] == [(len(x), 1), (len(x), 1)]
    assert np.allclose(music, 0.5 * x) and np.allclose(speech + music, x)


def test_resampling_around_a_lower_rate_strategy_passes_the_high_band():
    sr, native = 48000, 16000
    t = np.arange(sr) / sr
    x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 15000 * t)
    rec = _Recorder(music=lambda a: a)  # "all music"
    strategy = SeparationStrategy("rec", rec, sample_rate=native)
    speech, music = separate_speech(x, separator=strategy, sr=sr)
    ((shape, got_sr),) = rec.calls
    assert got_sr == native and shape == (native, 1)
    assert music.shape == x.shape
    assert np.allclose(music, x, atol=1e-6)  # the 15 kHz the model never saw is kept
    assert np.max(np.abs(speech)) < 1e-6


def test_resampling_up_to_a_higher_rate_strategy():
    rec = _Recorder(music=lambda a: a)
    strategy = SeparationStrategy("rec", rec, sample_rate=48000)
    x = _signal(1.0, sr=16000)
    _, music = separate_speech(x, separator=strategy, sr=16000)
    assert rec.calls[0][1] == 48000 and music.shape == x.shape


def test_separate_speech_on_audio_returns_audio_like_it(make_tone_audio):
    path = make_tone_audio(1.0, fmt="wav", sample_rate=SR, channels=2)
    audio = Audio(str(path))
    speech, music = separate_speech(audio, separator=lambda a, r: (a * 0, a))
    seg = audio._get_segment()
    assert music._get_segment().raw_data == seg.raw_data
    assert speech._get_segment().frame_count() == seg.frame_count()


def test_wiener_split_sums_to_the_mixture_and_beats_the_plain_residual():
    sr = 16000
    t = np.arange(2 * sr) / sr
    speech = 0.3 * np.sin(2 * np.pi * 300 * t)[:, None]
    music = 0.3 * np.sin(2 * np.pi * 3000 * t)[:, None]
    noisy_estimate = speech + 0.3 * music  # an enhancer that leaks some music
    s, m = wiener_split(speech + music, noisy_estimate, sr)
    assert np.allclose(s + m, speech + music)
    core = slice(sr // 4, -sr // 4)
    err = lambda est: np.sum((est[core] - music[core]) ** 2) / np.sum(music[core] ** 2)
    residual = speech + music - noisy_estimate  # 0.7 * music: error 0.09
    assert err(m) < err(residual) / 2  # the mask puts most of the leak back


def test_invert_enhancer_with_and_without_wiener():
    x = _signal(1.0, sr=16000)
    for wiener in (True, False):
        s, m = invert_enhancer(lambda a, r: 0 * a, wiener=wiener)(x, 16000)
        assert np.allclose(m, x) and np.allclose(s, 0)


# ---------------------------------------------------------------------------
# Registry and extras
# ---------------------------------------------------------------------------


def test_every_registered_strategy_declares_its_licence_and_extra():
    assert DEFAULT_SEPARATOR in SEPARATORS
    for key, strategy in SEPARATORS.items():
        assert strategy.name == key
        assert isinstance(strategy.license, str) and strategy.license
        assert isinstance(strategy.commercial_ok, bool)
        assert strategy.extra in EXTRAS
        assert strategy.sample_rate and strategy.context_s >= 0
        assert strategy.notes
    assert "instrumental" in SEPARATORS["htdemucs"].notes.lower()
    assert SEPARATORS[DEFAULT_SEPARATOR].mono  # MossFormer2-SE is single-channel


def test_a_bare_callable_has_unknown_licence():
    from mixing.audio.separation import _resolve

    strategy = _resolve(_zero_music)
    assert strategy.license is None and strategy.commercial_ok is None
    assert strategy.name == "_zero_music"


def _run_hidden(body: str, *, hidden) -> dict:
    """Run ``body`` in a fresh interpreter with ``hidden`` modules unimportable."""
    prologue = f"import sys\nfor _n in {list(hidden)!r}:\n    sys.modules[_n] = None\n"
    proc = subprocess.run(
        [sys.executable, "-c", prologue + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "separator, module, hint",
    [
        ("mossformer2_se_48k", "clearvoice", "pip install 'mixing[separation]'"),
        ("htdemucs", "demucs", "pip install 'mixing[demucs]'"),
    ],
)
def test_a_strategy_without_its_extra_names_the_extra(separator, module, hint):
    out = _run_hidden(
        f"""
        import json
        import numpy as np
        import mixing.audio
        from mixing.audio import remove_talk
        loaded = "torch" in sys.modules
        try:
            remove_talk(np.zeros((8000, 1)), (0.2, 0.4), separator={separator!r}, sr=8000)
        except ImportError as e:
            print(json.dumps({{"name": e.name, "message": str(e), "torch": loaded}}))
        else:
            print(json.dumps({{"message": None}}))
        """,
        hidden=(module,),
    )
    assert out["message"] is not None, "expected an ImportError"
    assert out["name"] == module and hint in out["message"]
    assert out["torch"] is False  # importing mixing.audio did not pull torch


@pytest.mark.parametrize(
    "module, hint",
    [("clearvoice", "mixing[separation]"), ("demucs", "mixing[demucs]")],
)
def test_require_package_hints_the_separation_extras(monkeypatch, module, hint):
    import importlib

    from mixing.util import require_package

    def deny(name, *a, **k):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", deny)
    with pytest.raises(ImportError, match=hint.replace("[", r"\[").replace("]", r"\]")):
        require_package(module)


# ---------------------------------------------------------------------------
# Real strategies (slow; MIXING_RUN_SLOW=1 and the extras)
# ---------------------------------------------------------------------------

_slow = pytest.mark.skipif(
    os.environ.get("MIXING_RUN_SLOW") != "1",
    reason="real-model test; set MIXING_RUN_SLOW=1 (needs mixing[separation] / "
    "mixing[demucs] + a model download)",
)

CLIP_SR = 44100
CLIP_SECONDS = 6.0
TALK_SPAN = (1.5, 4.5)


def _speech_like(seconds, sr, *, seed=1):
    """A TTS-free voice: a glottal pulse train (gliding f0) through vowel formants,
    in syllables of ~5 Hz with consonant noise bursts between them."""
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = np.cumsum(f0) / sr
    pulses = np.diff(np.floor(phase), prepend=0.0)  # one impulse per period
    source = signal.lfilter([1.0], [1.0, -0.97], pulses)  # glottal tilt
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480)]
    syllable = int(0.2 * sr)
    out = np.zeros(n)
    for k, start in enumerate(range(0, n, syllable)):
        seg = slice(start, min(n, start + syllable))
        voiced = np.zeros(n)
        for f, bw in zip(vowels[k % len(vowels)], (80, 100, 120)):
            r = np.exp(-np.pi * bw / sr)
            a = [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
            voiced[seg] += signal.lfilter([1 - r], a, source[seg])
        env = np.sin(np.linspace(0, np.pi, seg.stop - seg.start)) ** 0.5
        out[seg] = voiced[seg] * env
        burst = slice(seg.start, min(n, seg.start + int(0.03 * sr)))
        hiss = signal.lfilter(
            *signal.butter(2, 3000 / (sr / 2), "high"), rng.normal(size=n)
        )
        out[burst] += 0.3 * hiss[burst]
    return out / np.max(np.abs(out))


def _music_like(seconds, sr, *, seed=2):
    """A stereo instrumental: wide-panned chords with harmonics and a kick."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    chords = [(220, 277, 330), (196, 247, 294), (175, 220, 262), (196, 247, 294)]
    left, right = np.zeros(n), np.zeros(n)
    beat = int(0.5 * sr)
    for i, start in enumerate(range(0, n, 4 * beat)):
        seg = slice(start, min(n, start + 4 * beat))
        for j, f in enumerate(chords[i % len(chords)]):
            tone = sum(np.sin(2 * np.pi * f * h * t[seg]) / h**1.5 for h in range(1, 6))
            pan = (j - 1) * 0.8  # wide
            left[seg] += tone * (1 - pan) / 2
            right[seg] += tone * (1 + pan) / 2
    kick_t = np.arange(int(0.15 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-kick_t * 30)) * kick_t) * np.exp(
        -kick_t * 25
    )
    for start in range(0, n - len(kick), beat):
        left[start : start + len(kick)] += kick
        right[start : start + len(kick)] += kick
    music = np.stack([left, right], axis=1)
    return music / np.max(np.abs(music))


def _si_sdr(estimate, reference):
    """Scale-invariant SDR (dB) of ``estimate`` against ``reference`` (flattened)."""
    est, ref = estimate.ravel(), reference.ravel()
    target = (est @ ref) / (ref @ ref) * ref
    return 10 * np.log10(np.sum(target**2) / np.sum((est - target) ** 2))


def _talk_over_music(*, talk_db=0.0):
    """``(mixture, music)``: the voice centred over the music in ``TALK_SPAN``."""
    music = 0.3 * _music_like(CLIP_SECONDS, CLIP_SR)
    voice = np.zeros(len(music))
    a, b = (int(s * CLIP_SR) for s in TALK_SPAN)
    voice[a:b] = _speech_like(TALK_SPAN[1] - TALK_SPAN[0], CLIP_SR)[: b - a]
    rms = lambda y: np.sqrt(np.mean(y**2))
    voice *= rms(music[a:b]) / rms(voice[a:b]) * 10 ** (talk_db / 20)
    return music + voice[:, None], music


@pytest.mark.slow
@_slow
@pytest.mark.parametrize(
    "separator, module", [("mossformer2_se_48k", "clearvoice"), ("htdemucs", "demucs")]
)
def test_real_strategy_improves_the_music_si_sdr(separator, module):
    pytest.importorskip("torch")
    pytest.importorskip(module)
    mixture, music = _talk_over_music()
    out = remove_talk(
        mixture, TALK_SPAN, separator=separator, sr=CLIP_SR, margin=0.0, crossfade=0.0
    )
    a, b = (int(s * CLIP_SR) for s in TALK_SPAN)
    before, after = _si_sdr(mixture[a:b], music[a:b]), _si_sdr(out[a:b], music[a:b])
    print(
        f"\n{separator}: music SI-SDR in the talk span {before:+.2f} -> {after:+.2f} dB"
    )
    assert after > before
    assert np.array_equal(out[: a - 1], mixture[: a - 1])  # untouched outside
