"""Tests for mixing.resample: polyphase, the numpy FFT fallback, and auto."""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mixing.resample import fft_resample, resample


@pytest.mark.parametrize(
    "n, fs, target",
    [(1000, 44100, 22050), (1001, 22050, 44100), (999, 48000, 16000),
     (1000, 16000, 48000), (7, 8000, 11025), (1, 8000, 16000)],
)
def test_fft_resample_is_scipy_resample_at_polyphase_length(n, fs, target):
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(0).standard_normal((n, 2))
    y = fft_resample(x, fs, target)
    assert y.shape == resample(x, fs, target, method="polyphase").shape
    np.testing.assert_allclose(y, signal.resample(x, len(y), axis=0), atol=1e-12)


def test_fft_resample_keeps_a_band_limited_tone():
    t = np.arange(800) / 8000
    tone = np.sin(2 * np.pi * 500 * t)
    np.testing.assert_allclose(fft_resample(tone, 8000, 16000)[::2], tone, atol=1e-9)


def test_equal_rates_and_empty_input():
    x = np.ones((10, 1))
    assert resample(x, 8000, 8000, method="fft") is x
    assert fft_resample(np.zeros((0, 2)), 8000, 16000).shape == (0, 2)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        resample(np.zeros(10), 8000, 16000, method="sinc")


def test_group_lufs_per_segment_rates_agree_across_methods(monkeypatch):
    """#172 acceptance: the FFT path agrees with polyphase within 0.2 LU."""
    pytest.importorskip("scipy.signal")
    import mixing.loudness as loudness

    rng = np.random.default_rng(1)
    segments = [rng.standard_normal((44100, 1)) * 0.1, rng.standard_normal((22050, 1)) * 0.05]
    poly = loudness.group_lufs(segments, [44100, 22050])
    monkeypatch.setattr(
        loudness, "resample", lambda d, fs, t: resample(d, fs, t, method="fft")
    )
    assert loudness.group_lufs(segments, [44100, 22050]) == pytest.approx(poly, abs=0.2)


def test_auto_falls_back_to_fft_without_scipy():
    """With scipy unimportable, per-segment rates still work (a scipy-free
    meter: any callable)."""
    code = textwrap.dedent(
        """
        import json, sys
        sys.modules["scipy"] = None
        import numpy as np
        from mixing.resample import resample
        from mixing.loudness import group_lufs
        x = np.random.default_rng(0).standard_normal((441, 2))
        rms = lambda d, fs: float(10 * np.log10(np.mean(d ** 2)))
        lufs = group_lufs([x[:, :1], x[:220, :1]], [44100, 22050], meter=rms)
        print(json.dumps({"shape": list(resample(x, 44100, 16000).shape),
                          "lufs": lufs,
                          "scipy": [m for m in sys.modules
                                    if m.startswith("scipy") and sys.modules[m] is not None]}))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["shape"] == [160, 2] and out["scipy"] == []
    assert np.isfinite(out["lufs"])
