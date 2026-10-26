"""Loudness meter (checked against ffmpeg's ebur128) and per-turn levelling."""

import re
import subprocess

import numpy as np
import pytest

pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import Audio, level_turns, loudness
from mixing.audio.loudness import _check_turns, noise_floor_db
from mixing.util import ffmpeg_exe

RATE = 44100


def _voice(seconds, amplitude, *, freq=220.0, seed=0):
    """A speech-like signal: harmonics under a slow amplitude envelope, plus hiss."""
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4))
    signal = amplitude * envelope * wave / 2 + 0.0005 * rng.standard_normal(len(t))
    pcm = np.clip(signal, -1, 1) * 32767
    return AudioSegment(
        pcm.astype(np.int16).tobytes(), frame_rate=RATE, sample_width=2, channels=1
    )


def _silence(seconds):
    return AudioSegment.silent(int(seconds * 1000), frame_rate=RATE)


def _ffmpeg_lufs(path):
    proc = subprocess.run(
        [
            ffmpeg_exe(),
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return float(re.findall(r"I:\s+(-?\d+\.\d+) LUFS", proc.stderr)[-1])


@pytest.mark.parametrize("amplitude", [0.05, 0.3])
def test_loudness_matches_ffmpeg_ebur128(tmp_path, amplitude):
    segment = _voice(6, amplitude)
    path = tmp_path / "v.wav"
    segment.export(str(path), format="wav")
    assert loudness(Audio(segment)) == pytest.approx(_ffmpeg_lufs(path), abs=0.3)


def test_loudness_silence_and_gain():
    assert loudness(Audio(_silence(2))) == -np.inf
    quiet = _voice(4, 0.1)
    assert loudness(Audio(quiet + 6)) - loudness(Audio(quiet)) == pytest.approx(
        6, abs=0.05
    )


def test_check_turns_rejects_overlap_and_out_of_range():
    with pytest.raises(ValueError):
        _check_turns([("a", 0, 2), ("b", 1, 3)], 5)
    with pytest.raises(ValueError):
        _check_turns([("a", 0, 9)], 5)
    assert _check_turns([("b", 2, 3), ("a", 0, 1)], 5)[0][0] == "a"


def test_level_turns_equalises_speakers():
    gaps = _silence(0.8)
    audio = Audio(
        _voice(3, 0.5, seed=1) + gaps + _voice(3, 0.04, freq=140, seed=2) + gaps
        + _voice(3, 0.12, freq=310, seed=3)
    )  # fmt: skip
    turns = [("a", 0.0, 3.0), ("b", 3.8, 6.8), ("c", 7.6, 10.6)]
    before = [loudness(audio[s:e]) for _, s, e in turns]
    assert max(before) - min(before) > 12

    out = level_turns(audio, turns, target_lufs=-16, ceiling_db=None)
    after = [loudness(out[s:e]) for _, s, e in turns]
    assert all(abs(x - -16) < 1.0 for x in after), after
    assert out.duration == pytest.approx(audio.duration, abs=0.05)


def test_chains_and_callables_and_egress(tmp_path):
    seen = []

    def spy(audio):
        seen.append(audio.duration)
        return audio

    audio = Audio(_voice(4, 0.2))
    out = level_turns(
        audio,
        [("a", 1.0, 3.0)],
        chains={"a": spy, "z": "highpass=f=80"},
        compressor="acompressor=ratio=2",
        output=tmp_path / "leveled.wav",
    )
    assert out.exists()
    assert seen and seen[0] == pytest.approx(
        3.0, abs=0.05
    )  # turn + 0.5 s padding each side


def test_noise_floor_db_is_low_for_quiet_gaps():
    mixed = Audio(_voice(1, 0.3) + _silence(2))
    assert noise_floor_db(mixed) < -80
