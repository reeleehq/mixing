"""Tests for mixing.audio.beat_grid — the librosa-backed beat/onset primitive.

librosa is the ``mixing[beats]`` extra; these tests ``importorskip`` it so the base
test env skips cleanly. ``import mixing.audio`` must NOT pull librosa (lazy import).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_import_mixing_audio_does_not_pull_librosa():
    # A clean subprocess: importing mixing.audio must not import librosa (lazy).
    import subprocess
    import sys

    code = (
        "import sys, mixing.audio; "
        "assert 'librosa' not in sys.modules, 'librosa leaked into import mixing.audio'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


def test_beat_grid_recovers_a_steady_pulse():
    pytest.importorskip("librosa")
    from mixing.audio import beat_grid, BeatGrid

    sr = 22050
    dur = 8.0
    period = 0.5  # 120 BPM
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    # A sharp click every `period` seconds → an unambiguous pulse.
    for onset in np.arange(0.0, dur, period):
        i = int(onset * sr)
        x[i : i + 220] += np.hanning(220) * np.sin(2 * np.pi * 1200 * t[i : i + 220])

    bg = beat_grid(x, sample_rate=sr)
    assert isinstance(bg, BeatGrid)
    # Tempo near 120 BPM (allow the 2x/half-time octave the tracker may pick).
    assert bg.tempo_bpm == pytest.approx(120, abs=8) or bg.tempo_bpm == pytest.approx(
        60, abs=8
    ) or bg.tempo_bpm == pytest.approx(240, abs=16)
    assert len(bg.beat_times) >= 5  # a steady pulse yields several beats
    # onset envelope is a dense per-hop array; frame k is at k*onset_hop_s
    assert bg.onset_env.ndim == 1 and len(bg.onset_env) > 0
    assert bg.onset_hop_s == pytest.approx(512 / sr)
    # beats are ascending and within the clip
    bt = bg.beat_times
    assert np.all(np.diff(bt) > 0)
    assert bt[0] >= 0 and bt[-1] <= dur


def test_beat_grid_rejects_unsupported_backend():
    pytest.importorskip("librosa")
    from mixing.audio import beat_grid

    with pytest.raises(ValueError, match="unsupported beat backend"):
        beat_grid(np.zeros(22050), backend="madmom")


def test_beat_grid_to_dict_is_jsonable():
    pytest.importorskip("librosa")
    import json

    from mixing.audio import beat_grid

    sr = 22050
    x = np.random.default_rng(0).normal(0, 0.1, sr * 3)
    bg = beat_grid(x, sample_rate=sr)
    s = json.dumps(bg.to_dict())  # must not raise (no NaN, arrays → lists)
    assert "beat_times" in json.loads(s)


# -- backends, ranked cuts and sections (muvid#133) ---------------------------------


def _grid(beats, env, hop=0.1, downbeats=()):
    from mixing.audio import BeatGrid

    return BeatGrid(
        beat_times=np.asarray(beats, float),
        downbeat_times=np.asarray(downbeats, float),
        onset_env=np.asarray(env, np.float32),
        onset_hop_s=hop,
        sample_rate=22050,
        tempo_bpm=120.0,
    )


def test_rank_beats_orders_by_onset_strength_and_keeps_the_strongest():
    from mixing.audio import rank_beats, ranked_cut_points

    env = np.zeros(50)
    for t, v in ((5, 1.0), (15, 3.0), (25, 2.0), (35, 0.5)):
        env[t] = v
    g = _grid([0.5, 1.5, 2.5, 3.5], env)
    rank = rank_beats(g)
    assert rank.argmax() == 1 and rank.argmin() == 3 and rank.min() == 0 and rank.max() == 1
    assert list(ranked_cut_points(g, keep=0.5)) == [1.5, 2.5]  # time order, top half
    assert list(ranked_cut_points(g, keep=1)) == [0.5, 1.5, 2.5, 3.5]
    with pytest.raises(ValueError):
        ranked_cut_points(g, keep=0)


def test_rank_beats_downbeat_bonus_and_empty():
    from mixing.audio import rank_beats

    env = np.zeros(50)
    env[[5, 15]] = 1.0
    plain = rank_beats(_grid([0.5, 1.5], env))
    boosted = rank_beats(_grid([0.5, 1.5], env, downbeats=[0.5]))
    assert boosted[0] >= plain[0]
    assert len(rank_beats(_grid([], env))) == 0


def test_beat_backend_registry_is_open_and_refuses_duplicates():
    from mixing.audio import beats as b

    assert {"librosa", "beat_this"} <= set(b.BEAT_BACKENDS)
    b.register_beat_backend("x-test", lambda *a, **k: None)
    try:
        with pytest.raises(ValueError, match="already registered"):
            b.register_beat_backend("x-test", lambda *a, **k: None)
    finally:
        del b.BEAT_BACKENDS["x-test"]


def test_beat_this_backend_gives_downbeats_through_the_seam(monkeypatch, tmp_path):
    pytest.importorskip("librosa")
    import sys
    import types

    from mixing.audio import beat_grid

    class FakeFile2Beats:
        def __init__(self, device="cpu"):
            pass

        def __call__(self, path):
            return np.arange(0.5, 6, 0.5), np.arange(0.5, 6, 2.0)

    mod = types.ModuleType("beat_this.inference")
    mod.File2Beats = FakeFile2Beats
    monkeypatch.setitem(sys.modules, "beat_this", types.ModuleType("beat_this"))
    monkeypatch.setitem(sys.modules, "beat_this.inference", mod)
    wav = tmp_path / "a.wav"
    import wave

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
        w.writeframes(np.zeros(22050 * 6, np.int16).tobytes())
    g = beat_grid(str(wav), backend="beat_this")
    assert len(g.downbeat_times) == 3 and g.tempo_bpm == pytest.approx(120)


def test_snap_to_downbeats():
    from mixing.audio.sections import snap_to_downbeats

    assert snap_to_downbeats([9.7, 10.1, 31.0], [0, 8, 16, 24, 32], duration=40) == [8.0, 32.0]


def test_structural_sections_split_a_two_part_song_and_label_repeats(tmp_path):
    pytest.importorskip("librosa")
    from mixing.audio import structural_sections

    sr = 22050
    t = np.arange(int(sr * 10)) / sr

    # A B A B: 10 s of two alternating pitches in 4 blocks of 10 s
    x = np.concatenate([np.sin(2 * np.pi * f * t) for f in (220, 659, 220, 659)])
    import wave

    wav = tmp_path / "abab.wav"  # a file, so the sample rate is unambiguous
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((x * 12000).astype(np.int16).tobytes())
    secs = structural_sections(str(wav), n_sections=4, downbeats=np.arange(0, 40, 2.0))
    assert secs[0].start == 0 and secs[-1].end == pytest.approx(40, abs=0.1)
    assert all(a.end == b.start for a, b in zip(secs, secs[1:]))  # tiles the song
    assert len({s.label for s in secs}) == 2  # the two parts, repeated
    with pytest.raises(ValueError, match="unsupported section backend"):
        structural_sections(str(wav), backend="nope")
