"""Diegetic-source treatments: a whole program heard through a device.

"The car radio", "the TV in the next room", "the stadium PA", "the phone": a
band-limited, compressed, limited colouring of a whole program (the voice and the
crowd it carries). An audio *look* with no source-specific knowledge, array-only
like :mod:`mixing.loudness` (numpy + scipy, extra ``loudness``), so a package that
holds samples can use it without the media stack.

A treatment is any ``(samples, fs) -> samples`` callable. The named ones are
:class:`TreatmentPreset` values in :data:`TREATMENTS`, each a chain of the small
stages below, every one usable on its own:

1. :func:`band_pass` -- Butterworth band-pass (causal, as the device is);
2. :func:`compress` -- feed-forward compressor on a one-pole RMS detector, the
   channels linked by their mean power, then make-up gain;
3. :func:`mixing.loudness.limit_peaks` -- the lookahead peak limiter, to
   ``ceiling_db``.

:func:`resolve_treatment` turns a ``treatment=`` argument (a preset name, a preset,
any callable, or ``None`` for dry) into the callable, so a host's seam is one line.

Samples are float, mono ``(n,)`` or ``(n, channels)``, full scale 1.0; the shape and
a float dtype are kept (other dtypes come back float64).

>>> import numpy as np
>>> radio = TREATMENTS["am_radio"]
>>> noise = np.random.default_rng(0).standard_normal(8000) * 0.5
>>> out = radio(noise, 16000)
>>> out.shape, bool(np.abs(out).max() <= 10 ** (radio.ceiling_db / 20) + 1e-9)
((8000,), True)
>>> resolve_treatment("phone").name
'phone'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union

from mixing.util import LOUDNESS_EXTRA, install_hint, require_package

try:  # numpy only: scipy is imported where used, as mixing.loudness does
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.treatment needs numpy and scipy: {install_hint(LOUDNESS_EXTRA)}",
        name=e.name,
    ) from e

from mixing.loudness import limit_peaks

#: Butterworth order of the band-pass (per edge).
DEFAULT_FILTER_ORDER = 4
#: The band-pass's upper edge is held under this fraction of Nyquist (a low ``fs``).
MAX_CUTOFF_NYQUIST_FRACTION = 0.95
#: Time constant of the compressor's RMS level detector (s).
DEFAULT_DETECTOR_S = 0.02
#: Floor of the compressor's level detector, so silence has a finite level (dBFS).
DETECTOR_FLOOR_DB = -120.0
#: A preset's peak ceiling (dBFS) after make-up gain.
DEFAULT_TREATMENT_CEILING_DB = -1.0

#: A treatment: float samples and their rate in, the treated samples out.
TreatmentFunc = Callable[[np.ndarray, int], np.ndarray]


def _float(data) -> np.ndarray:
    x = np.asarray(data)
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)


def band_pass(
    data: np.ndarray,
    fs: int,
    *,
    low_hz: float,
    high_hz: float,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """``data`` through a causal Butterworth band-pass ``low_hz``-``high_hz`` (axis 0).

    ``high_hz`` is held under ``MAX_CUTOFF_NYQUIST_FRACTION`` of Nyquist, so a
    preset made for 44.1 kHz still works at a low rate; a ``ValueError`` when that
    leaves no band.

    >>> import numpy as np
    >>> band_pass(np.zeros(10), 8000, low_hz=300, high_hz=3400).shape
    (10,)
    """
    sig = require_package("scipy.signal", extra=LOUDNESS_EXTRA)
    high = min(high_hz, MAX_CUTOFF_NYQUIST_FRACTION * fs / 2.0)
    if not 0 < low_hz < high:
        raise ValueError(f"the passband {low_hz}-{high} Hz is empty at {fs} Hz")
    x = _float(data)
    sos = sig.butter(order, [low_hz, high], btype="bandpass", fs=fs, output="sos")
    return sig.sosfilt(sos, x, axis=0).astype(x.dtype, copy=False)


def compress(
    data: np.ndarray,
    fs: int,
    *,
    threshold_db: float,
    ratio: float,
    makeup_db: float = 0.0,
    detector_s: float = DEFAULT_DETECTOR_S,
) -> np.ndarray:
    """``data`` through a feed-forward compressor, then ``makeup_db`` of gain.

    The level is a one-pole RMS detector (time constant ``detector_s``) over the
    channels' mean power, so the channels are linked (the stereo image holds) and
    out-of-phase channels do not cancel in the detector. Above ``threshold_db``
    (dBFS of that RMS) the level is reduced by ``1 - 1/ratio`` dB per dB.

    >>> import numpy as np
    >>> x = np.full(2000, 0.5)
    >>> y = compress(x, 1000, threshold_db=-20.0, ratio=4.0)
    >>> bool(y[-1] < x[-1])
    True
    """
    lfilter = require_package("scipy.signal", extra=LOUDNESS_EXTRA).lfilter
    x = _float(data)
    power = np.square(x, dtype=np.float64)
    if x.ndim == 2:
        power = power.mean(axis=1)
    a = float(np.exp(-1.0 / max(1.0, detector_s * fs)))
    power = lfilter([1.0 - a], [1.0, -a], power)
    level_db = 10.0 * np.log10(np.maximum(power, 10.0 ** (DETECTOR_FLOOR_DB / 10.0)))
    over = np.maximum(0.0, level_db - threshold_db)
    gain = 10.0 ** ((makeup_db - over * (1.0 - 1.0 / ratio)) / 20.0)
    return (x * (gain if x.ndim == 1 else gain[:, None])).astype(x.dtype, copy=False)


@dataclass(frozen=True)
class TreatmentPreset:
    """A device's colouring: band-pass -> compressor -> make-up -> peak limiter.

    Callable as ``preset(samples, fs)`` (:func:`apply_treatment`), so a preset is one
    value of a :data:`TreatmentFunc` strategy. Derive a variant with
    :func:`dataclasses.replace`.

    Attributes:
        name: The preset's name (a host may record it, as foley does on a licence).
        highpass_hz: Lower edge of the device's passband (Hz).
        lowpass_hz: Upper edge of the device's passband (Hz).
        threshold_db: Compressor threshold (dBFS of the RMS detector).
        ratio: Compressor ratio above the threshold (``1``: none).
        makeup_db: Gain after the compressor (dB).
        filter_order: Butterworth order of the band-pass.
        detector_s: RMS detector time constant (s).
        ceiling_db: Peak ceiling of the final limiter (dBFS).
    """

    name: str
    highpass_hz: float
    lowpass_hz: float
    threshold_db: float
    ratio: float
    makeup_db: float = 0.0
    filter_order: int = DEFAULT_FILTER_ORDER
    detector_s: float = DEFAULT_DETECTOR_S
    ceiling_db: float = DEFAULT_TREATMENT_CEILING_DB

    def __post_init__(self):
        checks = (
            (0 < self.highpass_hz < self.lowpass_hz, "0 < highpass_hz < lowpass_hz"),
            (self.ratio >= 1, "ratio >= 1"),
            (self.filter_order >= 1, "filter_order >= 1"),
            (self.detector_s > 0, "detector_s > 0"),
            (self.ceiling_db <= 0, "ceiling_db <= 0"),
        )
        if problems := [why for ok, why in checks if not ok]:
            raise ValueError(f"treatment {self.name!r} needs {', '.join(problems)}")

    def __call__(self, samples: np.ndarray, fs: int) -> np.ndarray:
        return apply_treatment(samples, fs, self)


def apply_treatment(
    samples: np.ndarray, fs: int, preset: TreatmentPreset
) -> np.ndarray:
    """``samples`` (mono or ``(n, channels)``) as heard through ``preset``'s device.

    :func:`band_pass` -> :func:`compress` -> :func:`mixing.loudness.limit_peaks`
    (sample peak, at ``preset.ceiling_db``). Same shape out; a float dtype is kept.
    """
    x = _float(samples)
    if x.shape[0] == 0:
        return x.copy()
    x = band_pass(
        x,
        fs,
        low_hz=preset.highpass_hz,
        high_hz=preset.lowpass_hz,
        order=preset.filter_order,
    )
    x = compress(
        x,
        fs,
        threshold_db=preset.threshold_db,
        ratio=preset.ratio,
        makeup_db=preset.makeup_db,
        detector_s=preset.detector_s,
    )
    limited = limit_peaks(
        x if x.ndim == 2 else x[:, None],
        ceiling=10.0 ** (preset.ceiling_db / 20.0),
        fs=fs,
    )
    return limited if x.ndim == 2 else limited[:, 0]


#: The named devices. Passbands: ``am_radio`` is foley's #108 starter chain (450-3200
#: Hz); ``phone`` the narrowband telephone channel (300-3400 Hz, ITU-T G.712); ``tv`` a
#: small TV speaker; ``pa`` a horn loudspeaker, squashed hardest. The compressor
#: settings were set by ear, not calibrated.
TREATMENTS: "dict[str, TreatmentPreset]" = {
    p.name: p
    for p in (
        TreatmentPreset(
            "am_radio", 450.0, 3200.0, threshold_db=-24.0, ratio=4.0, makeup_db=6.0
        ),
        TreatmentPreset(
            "tv", 150.0, 7000.0, threshold_db=-20.0, ratio=3.0, makeup_db=4.0
        ),
        TreatmentPreset(
            "pa", 400.0, 4000.0, threshold_db=-26.0, ratio=6.0, makeup_db=8.0
        ),
        TreatmentPreset(
            "phone", 300.0, 3400.0, threshold_db=-22.0, ratio=4.0, makeup_db=6.0
        ),
    )
}

#: What a ``treatment=`` seam takes: a preset name, a :class:`TreatmentPreset`, any
#: :data:`TreatmentFunc`, or ``None`` (dry, no device).
Treatment = Union[str, TreatmentPreset, TreatmentFunc, None]


def _dry(samples: np.ndarray, fs: int) -> np.ndarray:
    return samples


def resolve_treatment(
    treatment: Treatment, *, presets: Mapping[str, TreatmentFunc] = TREATMENTS
) -> TreatmentFunc:
    """A ``treatment=`` argument as a :data:`TreatmentFunc` (``None``: dry).

    A name is looked up in ``presets`` (default :data:`TREATMENTS`; a host passes
    its own mapping to add devices), a callable is returned as is.

    >>> resolve_treatment(None)(1, 2)
    1
    >>> resolve_treatment("am_radio") is TREATMENTS["am_radio"]
    True
    """
    if treatment is None:
        return _dry
    if isinstance(treatment, str):
        try:
            return presets[treatment]
        except KeyError:
            raise ValueError(
                f"unknown treatment {treatment!r}; one of {sorted(presets)}, "
                "a TreatmentPreset, or a (samples, fs) -> samples callable"
            ) from None
    if callable(treatment):
        return treatment
    raise TypeError(
        f"treatment must be a name, a preset or a callable, not {treatment!r}"
    )
