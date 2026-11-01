"""``mixing.audio.onset_lag``: the lag of a motion signal behind a song's onsets.

The cases are the three a circular, argmax version measured wrong
(reeleehq/mixing#129): a short window inside a song, periodic music, and motion
sampled far coarser than the onset hop.
"""

import math

import numpy as np
import pytest

from mixing.audio import OnsetLag, onset_lag

HOP = 0.01


def _onsets(seconds=20.0, seed=0):
    return np.random.default_rng(seed).random(int(seconds / HOP))


def _motion_following(onset, t, lag_s):
    """Motion at ``t`` that is the onset envelope ``lag_s`` earlier."""
    return np.interp(t - lag_s, np.arange(len(onset)) * HOP, onset)


@pytest.mark.parametrize("true_lag", [0.3, -0.25, 0.0, 0.07])
def test_a_short_window_inside_a_song_finds_its_lag_without_wrapping(true_lag):
    onset = _onsets()
    t = np.arange(10.0, 12.0, HOP)  # 2 s inside a 20 s song
    lag, r = onset_lag(t, _motion_following(onset, t, true_lag), onset, onset_hop_s=HOP)
    assert lag == pytest.approx(true_lag, abs=1e-9) and r == pytest.approx(1.0)


def test_periodic_music_gives_the_smallest_lag_not_the_most_negative():
    beat = 0.5  # 120 bpm: the lag is defined only modulo 0.5 s
    n = np.arange(int(20 / HOP))
    onset = np.where(n % int(beat / HOP) == 0, 1.0, 0.0)
    onset = np.convolve(onset, np.hanning(9), mode="same")
    t = np.arange(5.0, 15.0, HOP)
    lag, r = onset_lag(t, _motion_following(onset, t, 0.1), onset, onset_hop_s=HOP)
    assert lag == pytest.approx(0.1) and r == pytest.approx(1.0)


@pytest.mark.parametrize("fps", [60, 25, 10])
def test_coarse_motion_is_not_reduced_to_scattered_points(fps):
    # a smooth onset envelope (motion cannot follow detail finer than its frames)
    onset = np.convolve(_onsets(), np.hanning(31), mode="same")
    t = np.arange(4.0, 16.0, 1 / fps)
    lag, r = onset_lag(t, _motion_following(onset, t, 0.2), onset, onset_hop_s=HOP)
    assert lag == pytest.approx(0.2, abs=1 / fps)
    assert r > 0.9


def _bump(lags, centre, height, width=0.05):
    return height * np.exp(-0.5 * ((lags - centre) / width) ** 2)


def test_near_tied_peaks_go_to_the_smallest_lag_and_a_lone_peak_is_not_pulled():
    from mixing.audio.lag import _pick_lag

    lags = np.round(np.arange(-100, 101) * HOP, 10)
    # three beats' worth of peaks: the best at -0.9, the true one at 0.1 just under it
    rs = _bump(lags, -0.9, 1.0) + _bump(lags, 0.1, 0.995) + _bump(lags, 0.6, 0.999)
    assert _pick_lag(lags, rs, 0.01) == (pytest.approx(0.1), pytest.approx(0.995))
    assert _pick_lag(lags, rs, 0.0)[0] == pytest.approx(-0.9)  # argmax: the test bites
    # one broad peak at 0.2: its shoulders near 0 are within 0.01 but not a peak
    broad = _bump(lags, 0.2, 0.9, width=0.3)
    assert _pick_lag(lags, broad, 0.01)[0] == pytest.approx(0.2)
    assert _pick_lag(lags, np.full(len(lags), np.nan), 0.01) is None


def test_a_smooth_envelope_does_not_bias_the_lag_toward_zero():
    onset = np.convolve(_onsets(), np.hanning(101), mode="same")
    t = np.arange(4.0, 16.0, HOP)
    lag, r = onset_lag(t, _motion_following(onset, t, 0.2), onset, onset_hop_s=HOP)
    assert lag == pytest.approx(0.2) and r == pytest.approx(1.0)


def test_a_gap_in_the_motion_is_left_out_not_interpolated_across():
    onset = np.convolve(_onsets(seconds=40), np.hanning(9), mode="same")
    t = np.concatenate([np.arange(2.0, 8.0, 1 / 30), np.arange(23.0, 30.0, 1 / 30)])
    lag, r = onset_lag(t, _motion_following(onset, t, 0.25), onset, onset_hop_s=HOP)
    assert lag == pytest.approx(0.25) and r > 0.98  # a made-up ramp would drag r down


def test_repeated_timestamps_do_not_change_the_answer():
    onset = np.convolve(_onsets(), np.hanning(31), mode="same")
    t = np.arange(4.0, 16.0, 0.1)  # 10 fps
    motion = _motion_following(onset, t, 0.3)
    once = onset_lag(t, motion, onset, onset_hop_s=HOP)
    twice = onset_lag(np.repeat(t, 2), np.repeat(motion, 2), onset, onset_hop_s=HOP)
    assert twice == once and once.lag_s == pytest.approx(0.3, abs=0.05)


def test_motion_finer_than_the_hop_is_averaged_down():
    hop = 512 / 22050  # librosa's default hop, ~23 ms
    rng = np.random.default_rng(3)
    onset = np.convolve(rng.random(int(20 / hop)), np.hanning(7), mode="same")
    t = np.arange(3.0, 17.0, 1 / 60)  # 60 fps: ~1.4 frames per hop
    motion = np.interp(t - 0.2, np.arange(len(onset)) * hop, onset)
    lag, r = onset_lag(t, motion, onset, onset_hop_s=hop)
    assert lag == pytest.approx(0.2, abs=hop) and r > 0.95


def test_unsorted_times_and_missing_samples_are_handled():
    onset = np.convolve(
        _onsets(), np.hanning(9), mode="same"
    )  # a real envelope is smooth
    t = np.arange(3.0, 9.0, HOP)
    motion = _motion_following(onset, t, 0.15)
    motion[::17] = np.nan  # dropped pose frames
    order = np.random.default_rng(1).permutation(len(t))
    lag, r = onset_lag(t[order], motion[order], onset, onset_hop_s=HOP)
    assert lag == pytest.approx(0.15) and r > 0.95


def test_too_little_overlap_is_nan_not_none():
    onset = _onsets(seconds=1.0)
    t = np.arange(5.0, 5.05, HOP)  # motion entirely past the envelope's end
    found = onset_lag(t, np.ones_like(t), onset, onset_hop_s=HOP)
    assert (
        isinstance(found, OnsetLag) and math.isnan(found.lag_s) and math.isnan(found.r)
    )
    flat = onset_lag(np.arange(0, 2, HOP), np.ones(200), onset, onset_hop_s=HOP)
    assert math.isnan(flat.lag_s)  # a constant motion correlates with nothing


def test_the_search_stays_within_max_lag():
    onset = _onsets()
    t = np.arange(5.0, 10.0, HOP)
    lag, _ = onset_lag(
        t, _motion_following(onset, t, 0.8), onset, onset_hop_s=HOP, max_lag_s=0.5
    )
    assert abs(lag) <= 0.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(onset_hop_s=0), "onset_hop_s"),
        (dict(onset_hop_s=HOP, max_lag_s=-1), "max_lag_s"),
        (dict(onset_hop_s=HOP, tie_tolerance=-0.1), "tie_tolerance"),
    ],
)
def test_bad_arguments_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        onset_lag([0, 1], [0, 1], [0, 1, 2], **kwargs)
    with pytest.raises(ValueError, match="differ in length"):
        onset_lag([0, 1, 2], [0, 1], [0, 1, 2], onset_hop_s=HOP)
