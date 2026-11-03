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
dtype are kept (integer PCM is refused: scale it first).

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
from typing import Callable, Mapping, Union

from mixing.util import LOUDNESS_EXTRA, install_hint, require_package

try:  # numpy only: scipy is imported where used, as mixing.loudness does
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.treatment needs numpy and scipy: {install_hint(LOUDNESS_EXTRA)}",
        name=e.name,
    ) from e

from mixing.loudness import DEFAULT_CEILING_DB, limit_peaks

#: Butterworth order of the band-pass (per edge).
DEFAULT_FILTER_ORDER = 4
#: The band-pass's upper edge is held under this fraction of Nyquist (a low ``fs``).
MAX_CUTOFF_NYQUIST_FRACTION = 0.95
#: Time constant of the compressor's RMS level detector (s).
DEFAULT_DETECTOR_S = 0.02
#: Floor of the compressor's level detector, so silence has a finite level (dBFS).
DETECTOR_FLOOR_DB = -120.0

#: A treatment: float samples and their rate in, the treated samples out.
TreatmentFunc = Callable[[np.ndarray, int], np.ndarray]


def _samples(data) -> np.ndarray:
    """``data`` as a float ``(n,)`` / ``(n, channels)`` array, or an informative error."""
    x = np.asarray(data)
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(
            f"samples must be float, full scale 1.0, got {x.dtype}: scale integer "
            "PCM by its full scale first"
        )
    if x.ndim not in (1, 2):
        raise ValueError(f"samples must be (n,) or (n, channels), got {x.shape}")
    return x


def _band_problems(low_hz: float, high_hz: float, order: int) -> list[str]:
    checks = (
        (0 < low_hz < high_hz, "0 < low_hz < high_hz"),
        (order >= 1, "order >= 1"),
    )
    return [why for ok, why in checks if not ok]


def _compressor_problems(ratio: float, detector_s: float) -> list[str]:
    checks = ((ratio >= 1, "ratio >= 1"), (detector_s > 0, "detector_s > 0"))
    return [why for ok, why in checks if not ok]


def _refuse(problems: list[str], subject: str) -> None:
    if problems:
        raise ValueError(f"{subject} needs {', '.join(problems)}")


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
    leaves no band. Float samples, ``(n,)`` or ``(n, channels)``; dtype kept.

    >>> import numpy as np
    >>> band_pass(np.zeros(10), 8000, low_hz=300, high_hz=3400).shape
    (10,)
    """
    _refuse(_band_problems(low_hz, high_hz, order), "band_pass")
    x = _samples(data)
    high = min(high_hz, MAX_CUTOFF_NYQUIST_FRACTION * fs / 2.0)
    if not low_hz < high:
        raise ValueError(f"the passband {low_hz}-{high} Hz is empty at {fs} Hz")
    if not len(x):
        return x.copy()
    sig = require_package("scipy.signal", extra=LOUDNESS_EXTRA)
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
    (dBFS of that RMS) the level is reduced by ``1 - 1/ratio`` dB per dB. The
    detector starts settled on the first ``detector_s`` of the signal, so a loud
    start is not let through uncompressed. Float samples, ``(n,)`` or
    ``(n, channels)``; dtype kept.

    >>> import numpy as np
    >>> x = np.full(2000, 0.5)
    >>> y = compress(x, 1000, threshold_db=-20.0, ratio=4.0)
    >>> bool(y[-1] < x[-1])
    True
    """
    _refuse(_compressor_problems(ratio, detector_s), "compress")
    x = _samples(data)
    if not len(x):
        return x.copy()
    sig = require_package("scipy.signal", extra=LOUDNESS_EXTRA)
    power = np.square(x, dtype=np.float64)
    if x.ndim == 2:
        power = power.mean(axis=1)
    settle = max(1, int(round(detector_s * fs)))
    a = float(np.exp(-1.0 / settle))
    b, den = [1.0 - a], [1.0, -a]
    zi = sig.lfilter_zi(b, den) * power[:settle].mean()
    power, _ = sig.lfilter(b, den, power, zi=zi)
    level_db = 10.0 * np.log10(np.maximum(power, 10.0 ** (DETECTOR_FLOOR_DB / 10.0)))
    over = np.maximum(0.0, level_db - threshold_db)
    gain = 10.0 ** ((makeup_db - over * (1.0 - 1.0 / ratio)) / 20.0)
    return (x * (gain if x.ndim == 1 else gain[:, None])).astype(x.dtype, copy=False)


@dataclass(frozen=True)
class TreatmentPreset:
    """A device's colouring: band-pass -> compressor -> make-up -> peak limiter.

    Callable as ``preset(samples, fs)`` (:func:`apply_treatment`), so a preset is one
    value of a :data:`TreatmentFunc` strategy. Derive a variant with
    :func:`dataclasses.replace`. Its values are checked by the same rules as
    :func:`band_pass` and :func:`compress`.

    Attributes:
        name: The preset's name (a host may record it, as foley does on a licence).
        low_hz: Lower edge of the device's passband (Hz).
        high_hz: Upper edge of the device's passband (Hz).
        threshold_db: Compressor threshold (dBFS of the RMS detector).
        ratio: Compressor ratio above the threshold (``1``: none).
        makeup_db: Gain after the compressor (dB).
        filter_order: Butterworth order of the band-pass.
        detector_s: RMS detector time constant (s).
        ceiling_db: Peak ceiling of the final limiter (dBFS).
    """

    name: str
    low_hz: float
    high_hz: float
    threshold_db: float
    ratio: float
    makeup_db: float = 0.0
    filter_order: int = DEFAULT_FILTER_ORDER
    detector_s: float = DEFAULT_DETECTOR_S
    ceiling_db: float = DEFAULT_CEILING_DB

    def __post_init__(self):
        problems = [
            *_band_problems(self.low_hz, self.high_hz, self.filter_order),
            *_compressor_problems(self.ratio, self.detector_s),
            *([] if self.ceiling_db <= 0 else ["ceiling_db <= 0"]),
        ]
        _refuse(problems, f"treatment {self.name!r}")

    def __call__(self, samples: np.ndarray, fs: int) -> np.ndarray:
        return apply_treatment(samples, fs, self)


def apply_treatment(
    samples: np.ndarray, fs: int, preset: TreatmentPreset
) -> np.ndarray:
    """``samples`` (float, mono or ``(n, channels)``) as heard through ``preset``'s device.

    :func:`band_pass` -> :func:`compress` -> :func:`mixing.loudness.limit_peaks`
    (sample peak, at ``preset.ceiling_db``). Same shape and dtype out. A NaN or
    infinite sample is refused: the detector and the limiter would spread it over
    the whole output.
    """
    x = _samples(samples)
    if not np.isfinite(x).all():
        raise ValueError("samples hold NaN or infinite values")
    x = band_pass(
        x, fs, low_hz=preset.low_hz, high_hz=preset.high_hz, order=preset.filter_order
    )
    x = compress(
        x,
        fs,
        threshold_db=preset.threshold_db,
        ratio=preset.ratio,
        makeup_db=preset.makeup_db,
        detector_s=preset.detector_s,
    )
    if not len(x):
        return x
    limited = limit_peaks(x, ceiling=10.0 ** (preset.ceiling_db / 20.0), fs=fs)
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
