"""Tests for ``mixing.audio.fingerprint``: does this clip contain the song?

Every case is synthetic and in memory: a non-repetitive "song" (random chords of
harmonic notes plus noise hits, never the same bar twice) and clips cut from it
under the degradations a phone at a concert adds (noise, a gain change, a
low-pass "muffling", an off-grid start, a speed change), unrelated songs (the
hard ones share a scale and a beat grid), a looped bar (the alias case), and
clips too short or too quiet to say anything.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

scipy_signal = pytest.importorskip("scipy.signal")

from mixing.audio import (  # noqa: E402
    ABSENT,
    CONTAINS,
    SONG_VERDICTS,
    UNDECIDED,
    LandmarkParams,
    Landmarks,
    MatchParams,
    SongMatch,
    SongThresholds,
    contains_song,
    find_audio_offset,
    landmarks,
    song_verdict,
)
from mixing.audio import fingerprint as fp  # noqa: E402

SR = 22050  # not the analysis rate: every case also exercises the resampling
SONG_S = 24.0
CLIP_S = 6.0
CLIP_START_S = 7.3  # off the song's hop grid on purpose
HOP_S = fp.DEFAULT_HOP_LENGTH / fp.DEFAULT_ANALYSIS_SAMPLE_RATE
CONCERT = {"snr_db": 5.0, "gain_db": -12.0, "lowpass_hz": 1000.0}


def _synth_song(
    seconds: float, seed: int, *, sr: int = SR, vocab: int = 36, grid=None
) -> np.ndarray:
    """A non-repetitive tone sequence: 1-3 harmonic notes per onset, plus a noise
    hit on ~40 % of the onsets. Onsets every 0.15-0.45 s, or on a fixed ``grid``
    (s); notes from ``vocab`` semitones above 110 Hz. Peak-normalised to 0.5."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n)
    t0 = 0.0
    while t0 < seconds:
        dur = grid if grid else rng.uniform(0.15, 0.45)
        i0 = int(t0 * sr)
        for _ in range(rng.integers(1, 4)):
            f0 = 110 * 2 ** (rng.integers(0, vocab) / 12)
            m = min(int((dur + 0.3) * sr), n - i0)
            tt = np.arange(m) / sr
            env = np.exp(-tt / (dur * 0.6)) * np.minimum(1, tt / 0.005)
            note = sum(
                0.5**h
                * np.sin(2 * np.pi * f0 * (h + 1) * tt + rng.uniform(0, 2 * np.pi))
                for h in range(4)
            )
            y[i0 : i0 + m] += 0.2 * env * note
        if rng.random() < 0.4:
            m = min(int(0.08 * sr), n - i0)
            y[i0 : i0 + m] += (
                0.1 * rng.standard_normal(m) * np.exp(-np.arange(m) / (0.02 * sr))
            )
        t0 += dur
    return 0.5 * y / np.abs(y).max()


def _speed(x, change):
    """``x`` played ``1 + change`` times faster (pitch and tempo: a resample)."""
    return scipy_signal.resample_poly(x, 1000, round(1000 * (1 + change)))


def _segment(y, *, sr=SR):
    """``y`` as a 16-bit pydub segment at ``sr`` (find_audio_offset's arrays are 44.1 kHz)."""
    from pydub import AudioSegment

    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)


def _degrade(x, *, seed=0, snr_db=None, gain_db=0.0, lowpass_hz=None, sr=SR):
    y = x * 10 ** (gain_db / 20)
    if lowpass_hz:
        y = scipy_signal.sosfilt(
            scipy_signal.butter(4, lowpass_hz, fs=sr, output="sos"), y
        )
    if snr_db is not None:
        power = np.mean(y**2)
        noise = np.random.default_rng(seed).standard_normal(len(y))
        y = y + noise * np.sqrt(power / 10 ** (snr_db / 10))
    return y


def _cut(y, start_s, seconds, *, sr=SR):
    return y[int(start_s * sr) : int((start_s + seconds) * sr)]


@pytest.fixture(scope="module")
def song():
    return _synth_song(SONG_S, seed=1)


@pytest.fixture(scope="module")
def index(song):
    return landmarks(song, sr=SR)


@pytest.fixture(scope="module")
def clip(song):
    return _cut(song, CLIP_START_S, CLIP_S)


# --- contains, at the right offset -----------------------------------------


@pytest.fixture(scope="module")
def xcorr_offset(song, clip):
    """The clip's offset by an independent instrument: waveform cross-correlation."""
    return find_audio_offset(_segment(song), _segment(clip))


@pytest.mark.parametrize(
    "degradation",
    [{}, {"snr_db": 0.0}, {"gain_db": -20.0}, {"lowpass_hz": 1000.0}, CONCERT],
    ids=["clean", "noise_0dB", "gain_-20dB", "lowpass_1k", "concert"],
)
def test_degraded_cut_contains_song_at_its_offset(
    clip, index, xcorr_offset, degradation
):
    m = contains_song(_degrade(clip, **degradation), index, sr=SR)
    assert m.verdict == CONTAINS, m.reason
    assert m.offset_s == pytest.approx(CLIP_START_S, abs=HOP_S)
    # the cross-correlation aligner, a different method, agrees within one hop
    assert m.offset_s == pytest.approx(xcorr_offset, abs=HOP_S)


@pytest.mark.parametrize(
    ("start_s", "expected"),
    [
        (0.0, CONTAINS),
        (3.21, CONTAINS),
        # R = 4.83: the price of R >= 5, which keeps same-key songs out of
        # "contains" -- a real cut the bar leaves undecided, never absent
        (13.01, UNDECIDED),
        (17.77, CONTAINS),
    ],
)
def test_offset_found_wherever_the_clip_was_cut(song, index, start_s, expected):
    m = contains_song(_degrade(_cut(song, start_s, CLIP_S), snr_db=10.0), index, sr=SR)
    assert m.verdict == expected, m.reason
    assert m.offset_s == pytest.approx(start_s, abs=HOP_S)


def test_clip_that_starts_before_the_song_has_a_negative_offset(song, index):
    lead_s = 1.5
    clip = np.concatenate([_synth_song(lead_s, seed=42), _cut(song, 0.0, CLIP_S)])
    m = contains_song(clip, index, sr=SR)
    assert m.verdict == CONTAINS, m.reason
    assert m.offset_s == pytest.approx(-lead_s, abs=HOP_S)


@pytest.mark.parametrize("change", [-0.005, 0.005])
def test_half_a_percent_off_speed_still_contains(clip, index, change):
    m = contains_song(_speed(clip, change), index, sr=SR)
    assert m.verdict == CONTAINS, m.reason
    assert m.offset_s == pytest.approx(CLIP_START_S, abs=2 * HOP_S)


@pytest.mark.parametrize("change", [-0.02, 0.02])
def test_two_percent_off_speed_is_undecided_not_absent(clip, index, change):
    m = contains_song(_speed(clip, change), index, sr=SR)
    assert m.verdict == UNDECIDED, m.reason
    assert m.off_speed == pytest.approx(change)  # a probe found it
    no_probes = MatchParams(speed_probes=())
    assert (
        contains_song(_speed(clip, change), index, sr=SR, match=no_probes).verdict
        == ABSENT
    )


# --- absent, undecided ------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: _synth_song(CLIP_S, seed=7),  # another song, same note vocabulary
        lambda: 0.1 * np.random.default_rng(3).standard_normal(int(CLIP_S * SR)),
    ],
    ids=["unrelated_song", "white_noise"],
)
def test_unrelated_clip_is_absent(index, make):
    m = contains_song(make(), index, sr=SR)
    assert m.verdict == ABSENT, m.reason
    assert m.off_speed is None


#: Unrelated songs in one key on one beat grid: the hard negatives (seeds fixed).
HARD_NEGATIVE_FAMILIES = [(7, 0.25), (5, 0.25), (7, 0.5)]
HARD_NEGATIVE_SEEDS = range(5000, 5015)


@pytest.mark.parametrize(("vocab", "grid"), HARD_NEGATIVE_FAMILIES)
def test_same_key_same_tempo_songs_are_never_contains(vocab, grid):
    song = _synth_song(SONG_S, seed=1000 + vocab, vocab=vocab, grid=grid)
    index = landmarks(song, sr=SR)
    matches = [
        contains_song(_synth_song(CLIP_S, seed=s, vocab=vocab, grid=grid), index, sr=SR)
        for s in HARD_NEGATIVE_SEEDS
    ]
    assert [m.verdict for m in matches].count(CONTAINS) == 0, [
        m.reason for m in matches if m.verdict == CONTAINS
    ]
    # ...while a real cut of that same grid song is found
    real = contains_song(_cut(song, CLIP_START_S, CLIP_S), index, sr=SR)
    assert real.verdict == CONTAINS, real.reason
    assert real.offset_s == pytest.approx(CLIP_START_S, abs=HOP_S)


@pytest.mark.parametrize(
    "make",
    [
        lambda song: _cut(song, CLIP_START_S, 0.3),  # the song, but 0.3 s of it
        lambda song: _synth_song(1.0, seed=8),  # 1 s of something else
        lambda song: 1e-5 * np.random.default_rng(4).standard_normal(int(CLIP_S * SR)),
        lambda song: _cut(song, CLIP_START_S, CLIP_S) * 10 ** (-90 / 20),  # -90 dB
        lambda song: np.zeros(100),  # shorter than one STFT frame
        # integer PCM is scaled: +-1 LSB of int16 is -90 dBFS, not full-scale hiss
        lambda song: (
            np.random.default_rng(5).integers(-1, 2, int(CLIP_S * SR)).astype(np.int16)
        ),
    ],
    ids=[
        "short_song",
        "short_other",
        "hiss_-100dB",
        "song_at_-90dB",
        "tiny",
        "int16_lsb",
    ],
)
def test_too_little_evidence_is_undecided_never_absent(song, index, make):
    m = contains_song(make(song), index, sr=SR)
    assert m.verdict == UNDECIDED, m.reason
    assert m.query_hashes < fp.DEFAULT_MIN_QUERY_HASHES


@pytest.mark.parametrize(("bar_s", "start_s"), [(2.0, 7.37), (1.7, 5.0)])
def test_looped_bar_shows_alias_ambiguity_not_a_wrong_contains(bar_s, start_s):
    looped = np.tile(_synth_song(bar_s, seed=3), 12)
    m = contains_song(_cut(looped, start_s, CLIP_S), landmarks(looped, sr=SR), sr=SR)
    # every repetition of the bar is an equally good offset: K is huge, R is ~1
    assert m.votes >= fp.DEFAULT_MIN_VOTES
    assert m.ratio < fp.DEFAULT_MIN_RATIO
    assert m.verdict == UNDECIDED, m.reason
    # the offset found is SOME repetition of the true one
    phase = (m.offset_s - start_s) / bar_s
    assert abs(phase - round(phase)) * bar_s < 2 * HOP_S


# --- the verdict -----------------------------------------------------------


def test_thresholds_are_overridable_and_recorded(clip, index):
    concert = _degrade(clip, **CONCERT)
    default = contains_song(concert, index, sr=SR)
    assert default.verdict == CONTAINS
    assert default.thresholds == SongThresholds()
    stricter = SongThresholds(min_votes=default.votes + 1)
    m = contains_song(concert, index, sr=SR, thresholds=stricter)
    assert m.verdict == UNDECIDED and m.thresholds == stricter
    assert m.votes == default.votes  # the measurement is unchanged
    unrelated = _synth_song(CLIP_S, seed=7)
    assert contains_song(unrelated, index, sr=SR).verdict == ABSENT
    for t in (
        SongThresholds(min_query_hashes=10**6),
        SongThresholds(max_absent_votes=-1),
    ):
        assert contains_song(unrelated, index, sr=SR, thresholds=t).verdict == UNDECIDED


def test_a_stored_record_is_rejudged_in_one_call(clip, index):
    m = contains_song(_degrade(clip, **CONCERT), index, sr=SR)
    record = m.to_dict()
    assert song_verdict(record)[0] == song_verdict(m)[0] == m.verdict == CONTAINS
    calibrated = SongThresholds(min_coverage=m.coverage + 0.01)
    assert song_verdict(record, thresholds=calibrated)[0] == UNDECIDED
    assert record["thresholds"] == dataclasses.asdict(SongThresholds())
    assert record["match_params"]["query_shifts"] == fp.DEFAULT_QUERY_SHIFTS


def _stats(votes, coverage, ratio, query_hashes):
    return dict(votes=votes, coverage=coverage, ratio=ratio, query_hashes=query_hashes)


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (_stats(12, 0.0, 5.0, 500), CONTAINS),
        (_stats(11, 0.9, 9.0, 500), UNDECIDED),
        (_stats(50, 0.9, 4.9, 500), UNDECIDED),
        (_stats(5, 0.0, 0.0, 240), ABSENT),
        (_stats(5, 0.0, 0.0, 239), UNDECIDED),
        (_stats(6, 0.5, 1.0, 10**4), UNDECIDED),
    ],
)
def test_song_verdict_boundaries(stats, expected):
    verdict, reason = song_verdict(stats)
    assert verdict == expected
    assert reason


def test_overlapping_bands_and_bad_params_are_rejected():
    with pytest.raises(ValueError, match="max_absent_votes"):
        SongThresholds(min_votes=5, max_absent_votes=5)
    with pytest.raises(ValueError, match="query_shifts"):
        MatchParams(query_shifts=0)


def test_match_has_no_truth_value(clip, index):
    m = contains_song(clip, index, sr=SR)
    assert m.verdict in SONG_VERDICTS
    with pytest.raises(TypeError, match="undecided"):
        bool(m)
    with pytest.raises(TypeError):
        if m:  # noqa: SIM108 -- the point: this must not silently pass
            pass
    d = m.to_dict()
    assert d["verdict"] == CONTAINS and d["votes"] == m.votes


# --- the index -------------------------------------------------------------


def test_precomputed_index_equals_on_the_fly(song, clip, index):
    concert = _degrade(clip, snr_db=5.0, gain_db=-12.0, lowpass_hz=1000.0)
    assert isinstance(index, Landmarks) and len(index) > 0
    assert contains_song(concert, index, sr=SR) == contains_song(concert, song, sr=SR)


def test_index_carries_its_params_to_the_clip(song, clip):
    params = LandmarkParams(fan_out=3, peaks_per_second=20.0)
    custom = landmarks(song, sr=SR, params=params)
    assert custom.params == params
    assert len(custom) < len(landmarks(song, sr=SR))
    on_the_fly = contains_song(clip, song, sr=SR, params=params)
    assert contains_song(clip, custom, sr=SR) == on_the_fly
    assert on_the_fly.verdict == CONTAINS


def test_landmarks_are_deterministic_and_nearly_gain_invariant(clip):
    a, again = landmarks(clip, sr=SR), landmarks(clip, sr=SR)
    assert np.array_equal(a.hashes, again.hashes) and np.array_equal(
        a.frames, again.frames
    )
    assert a.duration_s == pytest.approx(CLIP_S, abs=1e-3)
    assert a.hashes.dtype == np.int64 and len(a.hashes) == len(a.frames)
    # -12 dB moves every level alike; only float rounding of near-tied maxima differs
    quieter = landmarks(clip * 0.25, sr=SR)
    pairs = lambda lm: set(zip(lm.hashes.tolist(), lm.frames.tolist()))  # noqa: E731
    assert len(pairs(a) & pairs(quieter)) >= 0.9 * len(pairs(a))


@pytest.mark.parametrize("pps", [10.0, fp.DEFAULT_PEAKS_PER_SECOND])
def test_constellation_density_follows_the_param(song, pps):
    # the cap is local (fewer than pps * window stronger peaks near each one), so
    # the global density is close to pps, not bounded by it
    p = dataclasses.replace(fp.DEFAULT_LANDMARK_PARAMS, peaks_per_second=pps)
    y = fp._mono_at(song, sr=SR, target=p.sample_rate)
    frames, _ = fp._peaks(fp._spectrogram_db(y, p), p)
    assert len(frames) / SONG_S == pytest.approx(pps, rel=0.2)


def test_density_cap_keeps_peaks_with_few_stronger_neighbours():
    frames = np.array([0, 1, 2, 3, 50])
    strength = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    keep = fp._dense_enough(frames, strength, radius=5, allowed=2)
    # the two strongest of the cluster, and the isolated weak one
    assert keep.tolist() == [True, True, False, False, True]


# --- inputs ----------------------------------------------------------------


def test_file_and_audio_inputs_match_the_array(tmp_path, song, clip, index):
    from scipy.io import wavfile

    from mixing.audio import Audio

    path = tmp_path / "clip.wav"
    wavfile.write(path, SR, (clip * 32767).astype(np.int16))
    from_array = contains_song(clip, index, sr=SR)
    for given in (path, str(path), Audio(path)):
        m = contains_song(given, index)
        assert m.verdict == CONTAINS
        assert m.offset_s == pytest.approx(from_array.offset_s, abs=HOP_S)
    stereo = np.stack([clip, clip], axis=1)
    assert contains_song(stereo, index, sr=SR).verdict == CONTAINS


def test_input_errors_are_immediate(clip, index, tmp_path):
    with pytest.raises(ValueError, match="sr="):
        landmarks(clip)
    with pytest.raises(ValueError, match="only for arrays"):
        landmarks(tmp_path / "x.wav", sr=SR)
    with pytest.raises(TypeError, match="unsupported audio"):
        landmarks(object())
    # an index's params and rate are its own: a different one is an error, not ignored
    with pytest.raises(ValueError, match="params differ"):
        contains_song(clip, index, sr=SR, params=LandmarkParams(fan_out=3))
    with pytest.raises(ValueError, match="song_sr"):
        contains_song(clip, index, sr=SR, song_sr=SR)
    same = contains_song(clip, index, sr=SR, params=index.params)
    assert isinstance(same, SongMatch) and same.verdict == CONTAINS
    one_phase = contains_song(clip, index, sr=SR, match=MatchParams(query_shifts=1))
    assert one_phase.match_params.query_shifts == 1


def test_integer_pcm_is_scaled_like_float(clip, index):
    as_float = contains_song(clip, index, sr=SR)
    as_int16 = contains_song((clip * 32767).astype(np.int16), index, sr=SR)
    assert as_int16.verdict == CONTAINS
    assert as_int16.offset_s == pytest.approx(as_float.offset_s, abs=HOP_S)
    assert np.allclose(
        fp._as_float(np.array([-32768, 0, 16384], np.int16)), [-1, 0, 0.5]
    )
    assert np.allclose(
        fp._as_float(np.array([0, 128, 255], np.uint8)), [-1, 0, 127 / 128]
    )
