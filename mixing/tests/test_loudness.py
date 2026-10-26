"""``mixing.loudness``: array-only meters, group leveling and the peak limiter."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from mixing.loudness import (
    integrated_lufs,
    level_group,
    limit_peaks,
    loudness_range_lu,
    true_peak_db,
)
from mixing.util import has_ffmpeg

FS = 44100


def _voice(seconds, amplitude, *, seed=0, channels=1):
    """Speech-like: harmonics under a slow envelope, plus hiss."""
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    wave = sum(np.sin(2 * np.pi * 220 * h * t) / h for h in (1, 2, 3, 4)) / 2
    x = amplitude * envelope * wave + 0.0005 * rng.standard_normal(len(t))
    return np.tile(x[:, None], (1, channels))


def test_importing_the_module_is_light():
    code = (
        "import sys, mixing.loudness;"
        "bad = [m for m in ('pydub', 'moviepy', 'cv2', 'PIL') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr


@pytest.mark.parametrize("channels", [1, 2])
def test_meters_agree_with_pyloudnorm(channels):
    pyln = pytest.importorskip("pyloudnorm")
    data = _voice(6, 0.3, channels=channels)
    ours = integrated_lufs(data, FS)
    assert ours == pytest.approx(pyln.Meter(FS).integrated_loudness(data), abs=0.1)
    assert integrated_lufs(data, FS, meter="pyloudnorm") == pytest.approx(ours, abs=0.1)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg needed")
def test_ebur128_meter_agrees_with_numpy():
    data = _voice(6, 0.3, channels=2)
    assert integrated_lufs(data, FS, meter="ffmpeg_ebur128") == pytest.approx(
        integrated_lufs(data, FS), abs=0.1
    )


def test_meter_is_a_callable_or_a_known_name():
    data = _voice(1, 0.3)
    assert integrated_lufs(data, FS, meter=lambda d, fs: -23.0) == -23.0
    with pytest.raises(ValueError, match="unknown meter"):
        integrated_lufs(data, FS, meter="nope")


def test_level_group_reaches_the_target_and_keeps_relative_dynamics():
    quiet, loud = _voice(4, 0.05), _voice(4, 0.4, seed=1)
    gain, measured = level_group([quiet, loud], FS, target_lufs=-20.0)
    assert measured == pytest.approx(integrated_lufs(np.vstack([quiet, loud]), FS))
    joined = np.vstack([quiet, loud]) * 10 ** (gain / 20)
    assert integrated_lufs(joined, FS) == pytest.approx(-20.0, abs=0.05)
    # one gain for the group: the members keep their spread (quiet stays quieter)
    spread = integrated_lufs(loud, FS) - integrated_lufs(quiet, FS)
    assert integrated_lufs(loud * 10 ** (gain / 20), FS) - integrated_lufs(
        quiet * 10 ** (gain / 20), FS
    ) == pytest.approx(spread, abs=1e-6)


def test_level_group_edge_cases():
    # silence: nothing to level
    assert level_group([np.zeros((FS, 1))], FS, target_lufs=-20.0) == (0.0, -np.inf)
    # a group of short lines is looped to be measurable
    short = _voice(0.3, 0.3)
    _, measured = level_group([short], FS, target_lufs=-20.0)
    assert np.isfinite(measured)
    # mixed channel counts are downmixed rather than failing
    level_group([_voice(1, 0.3), _voice(1, 0.3, channels=2)], FS, target_lufs=-20.0)
    # a huge gain is capped, with a warning
    with pytest.warns(UserWarning, match="capped"):
        gain, _ = level_group([_voice(2, 0.0005)], FS, target_lufs=-10.0, max_gain_db=6.0)
    assert gain == 6.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        level_group([_voice(2, 0.3)], FS, target_lufs=-30.0)  # a cut never warns


def test_limiter_never_exceeds_its_ceiling():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((FS, 2)) * 0.8
    out = limit_peaks(data, ceiling=0.5, window=int(0.005 * FS))
    assert np.abs(out).max() <= 0.5 + 1e-12
    assert out.shape == data.shape


def test_limiter_leaves_quiet_signal_untouched_and_does_not_move_in_time():
    quiet = _voice(1, 0.1)
    assert limit_peaks(quiet, ceiling=1.0, window=100) is quiet
    burst = quiet.copy()
    burst[FS // 2] = 3.0
    out = limit_peaks(burst, ceiling=1.0, window=100)
    assert np.abs(out[FS // 2]).max() <= 1.0
    assert np.argmax(np.abs(out[:, 0])) == FS // 2


def test_true_peak_option_holds_intersample_peaks():
    n = np.arange(FS)
    # fs/4 sine at 45 degrees: samples sit ~3 dB under the true peak
    data = (np.sin(2 * np.pi * n / 4 + np.pi / 4) * 0.99)[:, None]
    ceiling = 10 ** (-3.0 / 20)
    sample_limited = limit_peaks(data, ceiling=ceiling, window=64)
    tp_limited = limit_peaks(data, ceiling=ceiling, window=64, true_peak=True, fs=FS)
    assert true_peak_db(sample_limited, FS) > -3.0 + 1.0  # sample peak misses it
    assert true_peak_db(tp_limited, FS) <= -3.0 + 0.1
    with pytest.raises(ValueError, match="needs fs"):
        limit_peaks(data, ceiling=ceiling, true_peak=True)


def test_loudness_range_of_steady_signal_is_small():
    assert loudness_range_lu(_voice(10, 0.3), FS) < 2.0
    assert loudness_range_lu(np.zeros((0, 1)), FS) == 0.0
