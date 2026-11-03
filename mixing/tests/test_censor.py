"""``mixing.audio.censor.bleep`` (#169): beep chosen words, keep everything else.

Also the helpers it shares with ``remove_talk``: ``mixing.audio.spans.as_spans``
(an aligner's word mappings are spans too) and ``mixing.audio.samples.float_input``.
"""

import numpy as np
import pytest

from mixing.audio import bleep
from mixing.audio.censor import (
    BEEP_TONE_HZ,
    MUTE,
    Word,
    as_words,
    beep_track,
    widen_spans,
)
from mixing.audio.separation import SeparationStrategy
from mixing.audio.spans import as_spans

SR = 8000


def _tone(seconds, hz, amp=0.5, sr=SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return amp * np.sin(2 * np.pi * hz * t)


def _word_track(n, words, *, amp=0.5, hz=300.0, sr=SR):
    """A voice: a ``hz`` tone of ``amp`` over each ``(start, end)``, silence elsewhere."""
    voice = np.zeros(n)
    for s, e in words:
        a, b = int(s * sr), int(e * sr)
        voice[a:b] = _tone((b - a) / sr, hz, amp, sr)
    return voice


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_spans_accept_aligner_words():
    words = [{"word": "a", "start": 1, "end": 2}, (3, 4)]
    assert as_spans(words) == [(1.0, 2.0), (3.0, 4.0)]
    assert as_words(words) == [Word(1.0, 2.0), Word(3.0, 4.0)]
    assert as_words({"start": 1, "end": 2, "max_end": 2.5}) == [Word(1.0, 2.0, 2.5)]
    assert as_words((1, 2)) == [Word(1.0, 2.0)]
    with pytest.raises(ValueError, match="end > start"):
        as_words([(2, 1)])


def test_a_stem_and_its_backing_become_the_bleeped_mix():
    n = 2 * SR
    voice = _word_track(n, [(0.5, 0.8), (1.2, 1.5)])
    backing = _tone(2.0, 110.0, amp=0.2)
    out = bleep(voice, [(1.2, 1.5)], backing=backing, sr=SR, widen_floor_db=None)
    a, b = int(1.2 * SR), int(1.5 * SR)
    f = int(0.01 * SR)
    # outside the span and its ramp: exactly voice + backing
    np.testing.assert_allclose(out[: a - f], (voice + backing)[: a - f], atol=1e-12)
    np.testing.assert_allclose(out[b + f :], (voice + backing)[b + f :], atol=1e-12)
    # inside: backing + a tone at the word's RMS, and no trace of the word's 300 Hz
    inner = out[a + f : b - f] - backing[a + f : b - f]
    assert _rms(inner) == pytest.approx(_rms(voice[a:b]), rel=0.02)
    spec = np.abs(np.fft.rfft(inner))
    freqs = np.fft.rfftfreq(len(inner), 1 / SR)
    assert freqs[np.argmax(spec)] == pytest.approx(BEEP_TONE_HZ, abs=10)
    assert spec[np.abs(freqs - 300) < 10].max() < 1e-3 * spec.max()


def test_a_mix_alone_is_replaced_by_the_tone_in_the_span():
    voice = _word_track(SR, [(0.3, 0.6)])
    out = bleep(voice, [(0.3, 0.6)], sr=SR, widen_floor_db=None, beep_db=-6.0)
    a, b, f = int(0.3 * SR), int(0.6 * SR), int(0.01 * SR)
    assert _rms(out[a + f : b - f]) == pytest.approx(
        _rms(voice[a:b]) * 10 ** (-6 / 20), rel=0.02
    )
    assert out.shape == voice.shape and out.dtype == voice.dtype


def test_a_separator_keeps_the_music_of_a_single_mix():
    voice = _word_track(SR, [(0.3, 0.6)])
    music = _tone(1.0, 110.0, amp=0.2)

    def separate(x, sr):  # knows the stems: what a perfect separator would return
        assert len(x) == SR  # given the whole file (context_s below)
        m = np.broadcast_to(music[:, None], x.shape)
        return x - m, m

    oracle = SeparationStrategy(name="oracle", separate=separate, context_s=10.0)
    keep = bleep(
        voice + music, [(0.3, 0.6)], sr=SR, separator=oracle, widen_floor_db=None
    )
    a, b, f = int(0.3 * SR), int(0.6 * SR), int(0.01 * SR)
    beeped = keep[a + f : b - f] - music[a + f : b - f]
    spec = np.abs(np.fft.rfft(beeped))
    freqs = np.fft.rfftfreq(len(beeped), 1 / SR)
    assert spec[np.abs(freqs - 110) < 10].max() < 1e-3 * spec.max()  # music kept
    assert spec[np.abs(freqs - 300) < 10].max() < 1e-3 * spec.max()  # word gone
    # the beep is levelled from the removed speech, not from the mix
    assert np.sqrt(np.mean(beeped**2)) == pytest.approx(_rms(voice[a:b]), rel=0.03)


def test_widening_covers_a_held_tail_but_never_the_next_word():
    # the word is aligned to 0.30-0.40, but it is held (decaying) to 0.55,
    # and the next word starts at 0.58
    n = SR
    t = np.arange(n) / SR
    voice = np.zeros(n)
    held = (t >= 0.3) & (t < 0.55)
    decay = np.where(t < 0.4, 1.0, np.exp(-(t - 0.4) / 0.05))
    voice[held] = 0.5 * np.sin(2 * np.pi * 300 * t[held]) * decay[held]
    gap = (t >= 0.55) & (t < 0.58)  # connected speech: the gap is not silent
    voice[gap] = 0.005 * np.sin(2 * np.pi * 300 * t[gap])
    nxt = (t >= 0.58) & (t < 0.8)
    voice[nxt] = 0.5 * np.sin(2 * np.pi * 300 * t[nxt])
    (span,) = widen_spans(voice[:, None], SR, [Word(0.3, 0.4)])
    assert span[0] == pytest.approx(0.3, abs=0.01) and 0.45 < span[1] <= 0.58
    # an over-generous floor runs into the next word unless max_end stops it
    (loose,) = widen_spans(voice[:, None], SR, [Word(0.3, 0.4)], floor_db=-80.0)
    assert loose[1] > 0.58
    (bounded,) = widen_spans(
        voice[:, None], SR, [Word(0.3, 0.4, max_end=0.58)], floor_db=-80.0
    )
    assert bounded[1] <= 0.58


def test_the_beep_is_levelled_from_the_word_not_its_widened_tail():
    # a loud word with a long quiet tail: the widened span's RMS is much lower
    voice = _word_track(SR, [(0.3, 0.4)], amp=0.5)
    voice[int(0.4 * SR) : int(0.65 * SR)] = _tone(0.25, 300.0, amp=0.1)
    out = bleep(voice, [(0.3, 0.4)], sr=SR, widen_floor_db=-30.0)
    a, b = int(0.32 * SR), int(0.6 * SR)
    assert _rms(out[a:b]) == pytest.approx(
        _rms(voice[int(0.3 * SR) : int(0.4 * SR)]), rel=0.05
    )


def test_overlapping_spans_make_one_tone():
    track = beep_track(SR, SR, [((0.2, 0.5), 0.1), ((0.4, 0.7), 0.3)], fade_s=0.0)
    seg = track[int(0.25 * SR) : int(0.65 * SR)]
    assert _rms(seg) == pytest.approx(0.3, rel=0.01)  # the louder level, no beating
    assert np.abs(track[int(0.75 * SR) :]).max() == 0


def test_stereo_int_pcm_round_trips_and_no_spans_is_a_no_op():
    voice = np.stack([_word_track(SR, [(0.3, 0.6)])] * 2, axis=1)
    pcm = np.round(voice * 32767).astype(np.int16)
    out = bleep(pcm, [(0.3, 0.6)], sr=SR, widen_floor_db=None)
    assert out.dtype == np.int16 and out.shape == pcm.shape
    a, b, f = int(0.3 * SR), int(0.6 * SR), int(0.01 * SR)
    np.testing.assert_array_equal(out[: a - f], pcm[: a - f])  # untouched outside
    inner = out[a + f : b - f, 0].astype(float)
    freqs = np.fft.rfftfreq(len(inner), 1 / SR)
    assert freqs[np.argmax(np.abs(np.fft.rfft(inner)))] == pytest.approx(1000, abs=10)
    np.testing.assert_array_equal(out[:, 0], out[:, 1])  # both channels alike
    np.testing.assert_array_equal(bleep(pcm, [], sr=SR), pcm)


def test_spans_overhanging_the_file_are_clipped():
    voice = _word_track(SR, [(0.0, 0.2), (0.85, 1.0)])
    out = bleep(voice, [(-0.1, 0.2), (0.85, 1.3)], sr=SR, widen_floor_db=None)
    assert out.shape == voice.shape and np.isfinite(out).all()


def test_max_end_stops_the_widening_inside_bleep():
    # connected speech: a held word runs into the next with no silence between
    voice = _word_track(SR, [(0.3, 0.6)], amp=0.5)
    voice[int(0.6 * SR) : int(0.9 * SR)] = _tone(0.3, 300.0, amp=0.4)  # next word
    word = {"start": 0.3, "end": 0.4, "max_end": 0.6}
    out = bleep(voice, [word], sr=SR, fade_s=0.0)
    np.testing.assert_allclose(out[int(0.6 * SR) :], voice[int(0.6 * SR) :])
    loose = bleep(voice, [{"start": 0.3, "end": 0.4}], sr=SR, fade_s=0.0)
    assert not np.allclose(loose[int(0.6 * SR) :], voice[int(0.6 * SR) :])


def test_a_mono_backing_is_added_to_every_channel():
    voice = np.stack([_word_track(SR, [(0.3, 0.6)])] * 2, axis=1)
    backing = _tone(1.0, 110.0, amp=0.2)
    out = bleep(voice, [(0.3, 0.6)], sr=SR, backing=backing, widen_floor_db=None)
    np.testing.assert_allclose(out[:100], voice[:100] + backing[:100, None])


def test_bad_combinations_are_refused():
    voice = np.zeros(SR)
    with pytest.raises(ValueError, match="not both"):
        bleep(voice, [(0.1, 0.2)], sr=SR, backing=voice, separator=MUTE)
    with pytest.raises(ValueError, match="one length"):
        bleep(voice, [(0.1, 0.2)], sr=SR, backing=voice[:-10])
    with pytest.raises(ValueError, match="fade_s"):
        bleep(voice, [(0.1, 0.2)], sr=SR, fade_s=-1)
    with pytest.raises(ValueError, match="tone_hz"):
        bleep(voice, [(0.1, 0.2)], sr=SR, tone_hz=SR / 2)
    with pytest.raises(ValueError, match="sample rate"):
        bleep(voice, [(0.1, 0.2)])


def test_a_path_in_writes_a_bleeped_file(tmp_path):
    from mixing.audio import Audio

    voice = _word_track(SR, [(0.3, 0.6)])
    src = tmp_path / "take.wav"
    pcm = np.round(voice * 32767).astype(np.int16)
    from pydub import AudioSegment

    AudioSegment(pcm.tobytes(), frame_rate=SR, sample_width=2, channels=1).export(
        src, format="wav"
    )
    written = bleep(src, [(0.3, 0.6)], output=tmp_path)
    assert written == tmp_path / "take_bleeped.wav"
    assert isinstance(bleep(Audio(src), [(0.3, 0.6)]), Audio)
    back = AudioSegment.from_file(written)
    full_scale = 2 ** (8 * back.sample_width - 1)  # the source's own width
    read = np.array(back.get_array_of_samples(), dtype=float) / full_scale
    expected = bleep(voice, [(0.3, 0.6)], sr=SR)
    np.testing.assert_allclose(read, expected, atol=1e-3)


def test_a_backing_file_at_another_rate_names_the_rates(tmp_path):
    from pydub import AudioSegment

    back = tmp_path / "back.wav"
    AudioSegment.silent(1000, frame_rate=16000).export(back, format="wav")
    with pytest.raises(ValueError, match="16000 Hz, the voice at 8000 Hz"):
        bleep(np.zeros(SR), [(0.1, 0.2)], sr=SR, backing=back)
