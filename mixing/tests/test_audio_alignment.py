"""Tests for mixing.audio alignment — find_audio_offset_detailed + align_clips_to_reference.

The multi-device / multicam primitive: align several clips (different-device recordings
of the same reference, so time-shifted + noisy) to one reference timeline, with a
scale-invariant confidence and coverage clamped to the reference. scipy is a base dep, so
these run in CI (no importorskip).
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import (
    AudioOffset,
    ClipAlignment,
    align_clips_to_reference,
    find_audio_offset,
    find_audio_offset_detailed,
)

SR = 16000


def _reference(seconds: float = 20.0) -> np.ndarray:
    """A non-periodic broadband 'song' — several linear chirps + a rhythmic envelope, so
    its autocorrelation has one sharp peak (no alignment ambiguity)."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in [(180, 520), (440, 130), (700, 900), (110, 250)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    return x / np.max(np.abs(x))


@pytest.fixture
def song(tmp_path):
    p = tmp_path / "song.wav"
    wavfile.write(str(p), SR, (_reference() * 32767).astype(np.int16))
    return p, _reference()


def _clip(tmp_path, ref, name, *, start_s, dur_s, gain=1.0, snr_db=15.0, pre_s=0.0):
    """A degraded 'phone recording' of ref[start_s:start_s+dur_s] written to a wav file."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # hash() varies per process
    seg = ref[int(start_s * SR) : int((start_s + dur_s) * SR)].copy()
    p_sig = np.mean(seg**2)
    seg = seg + rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), len(seg))
    if pre_s > 0:  # clip started filming BEFORE the song -> a negative offset
        seg = np.concatenate([rng.normal(0, 0.05, int(pre_s * SR)), seg])
    seg = gain * seg / (np.max(np.abs(seg)) + 1e-9)
    p = tmp_path / name
    wavfile.write(str(p), SR, (seg * 32767).astype(np.int16))
    return p


def test_detailed_recovers_offset_and_returns_confidence(song, tmp_path):
    song_p, ref = song
    clip = _clip(tmp_path, ref, "c.wav", start_s=7.3, dur_s=8.0)
    out = find_audio_offset_detailed(song_p, clip, sample_rate=SR)
    assert isinstance(out, AudioOffset)
    assert out.offset_s == pytest.approx(7.3, abs=0.05)
    assert 0.0 < out.confidence <= 1.0
    assert out.confidence > 0.3  # a real match scores well clear of noise


def test_find_audio_offset_still_returns_a_bare_float(song, tmp_path):
    song_p, ref = song
    clip = _clip(tmp_path, ref, "c.wav", start_s=5.0, dur_s=6.0)
    off = find_audio_offset(song_p, clip, sample_rate=SR)
    assert isinstance(off, float)
    assert off == pytest.approx(5.0, abs=0.05)


def test_confidence_is_scale_invariant(song, tmp_path):
    # Same content at 6x gain must score nearly the same (normalized cross-correlation).
    song_p, ref = song
    quiet = _clip(tmp_path, ref, "q.wav", start_s=4.0, dur_s=8.0, gain=0.15)
    loud = _clip(tmp_path, ref, "l.wav", start_s=4.0, dur_s=8.0, gain=1.0)
    cq = find_audio_offset_detailed(song_p, quiet, sample_rate=SR).confidence
    cl = find_audio_offset_detailed(song_p, loud, sample_rate=SR).confidence
    assert abs(cq - cl) < 0.1


def test_confidence_separates_a_match_from_noise(song, tmp_path):
    song_p, ref = song
    match = _clip(tmp_path, ref, "m.wav", start_s=6.0, dur_s=6.0, snr_db=15)
    rng = np.random.default_rng(0)
    noise_p = tmp_path / "n.wav"
    wavfile.write(
        str(noise_p), SR, (rng.normal(0, 0.3, 6 * SR) * 32767).astype(np.int16)
    )
    c_match = find_audio_offset_detailed(song_p, match, sample_rate=SR).confidence
    c_noise = find_audio_offset_detailed(song_p, noise_p, sample_rate=SR).confidence
    assert c_match > c_noise
    assert c_noise < 0.3


def test_align_clips_returns_clamped_coverage_in_order(song, tmp_path):
    song_p, ref = song
    clips = [
        _clip(tmp_path, ref, "a.wav", start_s=2.0, dur_s=6.0),
        _clip(tmp_path, ref, "b.wav", start_s=11.0, dur_s=7.0),
    ]
    aligns = align_clips_to_reference(song_p, clips, sample_rate=SR)
    assert [a.index for a in aligns] == [0, 1]
    assert all(isinstance(a, ClipAlignment) for a in aligns)
    a0, a1 = aligns
    assert a0.offset_s == pytest.approx(2.0, abs=0.05)
    assert a1.offset_s == pytest.approx(11.0, abs=0.05)
    for a in aligns:  # coverage within [0, ref_duration]
        assert 0.0 <= a.coverage[0] < a.coverage[1] <= 20.0 + 1e-6


def test_align_clamps_coverage_when_clip_runs_past_song_end(song, tmp_path):
    song_p, ref = song  # ref is 20s
    clip = _clip(tmp_path, ref, "tail.wav", start_s=16.0, dur_s=8.0)  # would run to 24s
    (a,) = align_clips_to_reference(song_p, [clip], sample_rate=SR)
    assert a.offset_s == pytest.approx(16.0, abs=0.05)
    assert a.coverage[1] == pytest.approx(20.0, abs=0.05)  # clamped to song end


def test_align_handles_a_clip_that_started_before_the_song(song, tmp_path):
    song_p, ref = song
    clip = _clip(tmp_path, ref, "pre.wav", start_s=0.0, dur_s=6.0, pre_s=2.0)
    (a,) = align_clips_to_reference(song_p, [clip], sample_rate=SR)
    assert a.offset_s == pytest.approx(-2.0, abs=0.1)  # negative offset supported
    assert a.coverage[0] == 0.0  # coverage starts at the song's t=0


def test_every_clip_gets_a_record_however_badly_it_matches(song, tmp_path):
    """No source may silently leave the addressable set.

    This function's output is what a caller persists as *the* alignment artifact, so a clip
    omitted here becomes material nothing downstream can reference, name, or explain — the
    file is still on disk, but it has effectively vanished from the project. Deciding what
    goes into an edit is a matter of REFERENCING sources and intervals; a source must never
    disappear from what can be referenced as a side effect of being measured.

    So the contract is: one record per input clip, always, carrying its original ``index``.
    Whether it is usable is reported by ``overlaps``, not by absence.
    """
    song_p, ref = song
    rng = np.random.default_rng(2)
    clips = [_clip(tmp_path, ref, "good.wav", start_s=3.0, dur_s=6.0)]
    for i in range(2):
        p = tmp_path / f"noise{i}.wav"
        wavfile.write(str(p), SR, (rng.normal(0, 0.3, 4 * SR) * 32767).astype(np.int16))
        clips.append(p)

    aligns = align_clips_to_reference(song_p, clips, sample_rate=SR)

    assert len(aligns) == len(clips), "a measured clip must not vanish from the result"
    assert sorted(a.index for a in aligns) == [0, 1, 2], "indices must map back to inputs"
    for a in aligns:
        # Coverage stays inside the reference timeline whether or not it overlaps.
        assert 0.0 <= a.coverage[0] <= a.coverage[1] <= 20.0 + 1e-6
        assert a.overlaps == (a.coverage[1] > a.coverage[0])
