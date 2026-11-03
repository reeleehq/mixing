"""``mixing.treatment`` (#165): diegetic-source presets, band-pass -> compressor -> limiter.

Moved from foley's ``foley/weave/broadcast.py`` (reeleehq/foley#108) with its tests.
"""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from mixing.treatment import (  # noqa: E402
    TREATMENTS,
    TreatmentPreset,
    apply_treatment,
    band_pass,
    compress,
    resolve_treatment,
)

SR = 16000


def _line(seconds=1.5, hz=900.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (
        amp * np.sin(2 * np.pi * hz * t) * (1.2 + np.sin(2 * np.pi * 4 * t))
    ).astype("float32")


def _noise(seconds=3.0, amp=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * SR))).astype("float32")


def _band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], 1 / SR)
    return spec[(freqs >= lo) & (freqs < hi)].sum() / spec.sum()


def test_presets_are_named_and_resolvable():
    assert set(TREATMENTS) == {"am_radio", "tv", "pa", "phone"}
    assert all(name == p.name for name, p in TREATMENTS.items())
    assert resolve_treatment("am_radio") is TREATMENTS["am_radio"]
    custom = lambda x, fs: x * 0  # noqa: E731
    assert resolve_treatment(custom) is custom
    x = np.ones(3)
    assert resolve_treatment(None)(x, SR) is x
    with pytest.raises(ValueError, match="unknown treatment 'gramophone'"):
        resolve_treatment("gramophone")
    with pytest.raises(TypeError, match="name, a preset or a callable"):
        resolve_treatment(3)


def test_a_host_can_add_its_own_named_devices():
    horn = TreatmentPreset("horn", 500.0, 2500.0, threshold_db=-20.0, ratio=2.0)
    presets = {**TREATMENTS, "horn": horn}
    assert resolve_treatment("horn", presets=presets) is horn
    with pytest.raises(ValueError, match="horn"):
        resolve_treatment("horn")  # not in the default table


def test_treatment_band_limits_and_ceilings():
    x = _noise(2.0, amp=0.5)  # white noise: energy everywhere
    preset = TREATMENTS["phone"]
    y = apply_treatment(x, SR, preset)
    assert y.dtype == np.float32 and y.shape == x.shape
    inside = _band_energy(y, preset.highpass_hz, preset.lowpass_hz)
    assert inside > 0.9 > _band_energy(x, preset.highpass_hz, preset.lowpass_hz)
    assert np.abs(y).max() <= 10 ** (preset.ceiling_db / 20) + 1e-6


def test_treatment_compresses_dynamics():
    loud, quiet = _line(1.0, amp=0.8), _line(1.0, amp=0.05)
    x = np.concatenate([quiet, loud])
    preset = TreatmentPreset("t", 300.0, 3400.0, threshold_db=-30.0, ratio=8.0)
    y = preset(x, SR)
    n = quiet.shape[0]

    def spread(z):  # loud half over the settled part of the quiet half, in RMS
        return np.sqrt(np.mean(z[n:] ** 2) / np.mean(z[n // 4 : 3 * n // 4] ** 2))

    assert spread(y) < spread(x) / 2


def test_ratio_one_and_no_makeup_leave_the_band_passed_signal_alone():
    x = _noise(1.0, amp=0.1).astype(np.float64)
    flat = TreatmentPreset("flat", 300.0, 3400.0, threshold_db=-60.0, ratio=1.0)
    y = flat(x, SR)
    np.testing.assert_allclose(y, band_pass(x, SR, low_hz=300.0, high_hz=3400.0))


def test_stereo_keeps_its_shape_and_the_channels_are_linked():
    left = _line(1.0, amp=0.8)
    x = np.stack([left, left * 0.25], axis=1)
    y = apply_treatment(x, SR, TREATMENTS["tv"])
    assert y.shape == x.shape
    # one gain for both channels: their ratio survives the compressor and limiter
    loud = np.abs(y[:, 0]) > 1e-3
    np.testing.assert_allclose(y[loud, 1] / y[loud, 0], 0.25, rtol=1e-4)


def test_out_of_phase_channels_are_still_compressed():
    # the detector reads the channels' mean power, not their mean signal (which is 0 here)
    left = _line(1.0, amp=0.8).astype(np.float64)
    x = np.stack([left, -left], axis=1)
    y = compress(x, SR, threshold_db=-30.0, ratio=8.0)
    mono = compress(left, SR, threshold_db=-30.0, ratio=8.0)
    np.testing.assert_allclose(y[:, 0], mono)


def test_a_float64_input_stays_float64_and_ints_become_floats():
    x = _noise(0.5).astype(np.float64)
    assert apply_treatment(x, SR, TREATMENTS["pa"]).dtype == np.float64
    ints = (x * 1000).astype(np.int16)
    assert apply_treatment(ints / 32768.0, SR, TREATMENTS["pa"]).dtype == np.float64
    assert band_pass(ints, SR, low_hz=300.0, high_hz=3400.0).dtype == np.float64


def test_empty_input_comes_back_empty():
    for shape in [(0,), (0, 2)]:
        y = apply_treatment(np.zeros(shape, np.float32), SR, TREATMENTS["phone"])
        assert y.shape == shape and y.dtype == np.float32


def test_the_upper_edge_is_held_under_nyquist_at_a_low_rate():
    x = _noise(0.5)
    y = apply_treatment(x, 8000, TREATMENTS["tv"])  # 7000 Hz > the 4 kHz Nyquist
    assert np.isfinite(y).all()
    with pytest.raises(ValueError, match="passband .* is empty at 2000 Hz"):
        band_pass(x, 2000, low_hz=1500.0, high_hz=3000.0)


@pytest.mark.parametrize(
    "kw",
    [
        dict(highpass_hz=3000.0, lowpass_hz=300.0),
        dict(ratio=0.5),
        dict(detector_s=0.0),
        dict(filter_order=0),
        dict(ceiling_db=1.0),
    ],
)
def test_preset_validates_its_values(kw):
    base = dict(highpass_hz=300.0, lowpass_hz=3000.0, threshold_db=-20.0, ratio=2.0)
    with pytest.raises(ValueError, match="needs"):
        TreatmentPreset("bad", **{**base, **kw})


def test_a_variant_is_a_replace_away():
    hot = dataclasses.replace(TREATMENTS["am_radio"], name="hot", makeup_db=12.0)
    assert hot.highpass_hz == TREATMENTS["am_radio"].highpass_hz and hot.name == "hot"


def test_import_stays_light():
    code = (
        "import sys, mixing.treatment; "
        "heavy = {'moviepy', 'cv2', 'pydub', 'PIL'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
