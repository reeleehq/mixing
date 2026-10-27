"""Characterization tests pinning the current behavior of ``Audio`` / ``AudioSamples``.

These tests lock in the *observable* behavior of the sliceable-audio facade in
``mixing.audio.audio_ops`` (the :class:`Audio` and :class:`AudioSamples` classes)
so an upcoming refactor cannot silently change semantics. They pin behavior, not
implementation: construction, time-unit-aware slicing, the derived duration/sample
properties, ``.save(output=...)``, chained ``fade_in``/``fade_out``, ``overlay``,
and the ``.samples`` Mapping protocol.

Notable current quirks encoded here (a refactor should preserve or deliberately fix):
- ``Audio`` accepts a file path as either a ``str`` or an ``os.PathLike`` (e.g.
  ``pathlib.Path``); both construct the same object. (This used to be a footgun
  where a ``Path`` was silently treated as an ``AudioSegment``; that bug is now
  fixed — see ``test_path_object_constructs_like_str``.)
- ``Audio.samples[i]`` returns a NumPy scalar (``np.floating``), not a builtin
  ``float`` (though it coerces cleanly via ``float(...)``).
"""

import numpy as np
import pytest

pytest.importorskip("pydub")

from mixing.audio.audio_ops import Audio, AudioSamples

# Encoders aren't sample-exact; use a generous tolerance for durations (seconds).
DUR_TOL = 0.05


# --------------------------------------------------------------------------- #
# Construction & basic properties
# --------------------------------------------------------------------------- #


def test_construct_from_str_path_sets_src_path_and_duration(tone_audio):
    audio = Audio(str(tone_audio))
    assert audio.src_path == str(tone_audio)
    assert abs(audio.full_duration - 1.0) < DUR_TOL
    assert abs(audio.duration - 1.0) < DUR_TOL


def test_default_time_unit_is_seconds(tone_audio):
    audio = Audio(str(tone_audio))
    assert audio.time_unit == "seconds"


def test_basic_properties_on_full_audio(tone_audio):
    audio = Audio(str(tone_audio))
    assert audio.sample_rate == 44100
    assert audio.channels == 1
    assert audio.start_time == 0.0
    assert abs(audio.end_time - 1.0) < DUR_TOL
    # duration and sample_count agree exactly: duration is the frame count over
    # the rate (#96; it was end_time - start_time in rounded milliseconds)
    assert audio.duration == audio.sample_count / audio.sample_rate
    assert abs(audio.sample_count - 44100) < 44100 * DUR_TOL


def test_stereo_channels_reported(make_tone_audio):
    audio = Audio(str(make_tone_audio(1.0, channels=2)))
    assert audio.channels == 2


def test_path_object_constructs_like_str(tone_audio):
    """A ``pathlib.Path`` is accepted as a file path, just like a ``str``.

    Pins the fix for the former footgun where a non-``str`` ``os.PathLike`` was
    silently routed into the AudioSegment branch (``src_path is None`` and a
    broken object). Now ``Audio(Path(p))`` behaves exactly like ``Audio(p)``.
    """
    from pathlib import Path

    path = Path(str(tone_audio))
    from_path = Audio(path)
    from_str = Audio(str(tone_audio))

    # src_path is coerced to the same str either way.
    assert from_path.src_path == str(tone_audio)
    assert from_path.src_path == from_str.src_path
    # Same observable audio properties.
    assert from_path.sample_rate == from_str.sample_rate
    assert abs(from_path.full_duration - from_str.full_duration) < DUR_TOL
    assert abs(from_path.full_duration - 1.0) < DUR_TOL


def test_audiosamples_path_object_constructs_like_str(tone_audio):
    """``AudioSamples`` likewise accepts a ``pathlib.Path`` like a ``str``."""
    from pathlib import Path

    from_path = AudioSamples(Path(str(tone_audio)))
    from_str = AudioSamples(str(tone_audio))
    assert from_path.audio_src == str(tone_audio)
    assert from_path.audio_src == from_str.audio_src
    assert len(from_path) == len(from_str)


# --------------------------------------------------------------------------- #
# Slicing: time_unit variants
# --------------------------------------------------------------------------- #


def test_slice_seconds_default_subduration(tone_audio):
    audio = Audio(str(tone_audio))  # default unit: seconds
    seg = audio[0.2:0.5]
    assert isinstance(seg, Audio)
    assert abs(seg.duration - 0.3) < DUR_TOL
    assert abs(seg.start_time - 0.2) < DUR_TOL
    assert abs(seg.end_time - 0.5) < DUR_TOL


def test_slice_milliseconds_subduration(tone_audio):
    audio = Audio(str(tone_audio), time_unit="milliseconds")
    seg = audio[200:500]  # 200ms..500ms -> 0.3s
    assert isinstance(seg, Audio)
    assert abs(seg.duration - 0.3) < DUR_TOL


def test_slice_samples_subduration(tone_audio):
    audio = Audio(str(tone_audio), time_unit="samples")
    # 0..22050 samples at 44100 Hz -> 0.5s
    seg = audio[0:22050]
    assert isinstance(seg, Audio)
    assert abs(seg.duration - 0.5) < DUR_TOL


def test_slice_inherits_time_unit(tone_audio):
    audio = Audio(str(tone_audio), time_unit="milliseconds")
    seg = audio[200:500]
    assert seg.time_unit == "milliseconds"


def test_negative_slice_start_is_relative_to_end(tone_audio):
    audio = Audio(str(tone_audio))  # 1s
    seg = audio[-0.3:]  # last 0.3s
    assert abs(seg.duration - 0.3) < DUR_TOL
    assert abs(seg.start_time - 0.7) < DUR_TOL


def test_open_ended_slices_clamp_to_audio_bounds(tone_audio):
    audio = Audio(str(tone_audio))
    seg = audio[0.5:]  # 0.5s .. end
    assert abs(seg.start_time - 0.5) < DUR_TOL
    assert abs(seg.end_time - 1.0) < DUR_TOL
    assert abs(seg.duration - 0.5) < DUR_TOL


def test_slice_step_rejected(tone_audio):
    audio = Audio(str(tone_audio))
    with pytest.raises(ValueError):
        _ = audio[0:1:2]


def test_inverted_range_rejected(tone_audio):
    audio = Audio(str(tone_audio))
    with pytest.raises(ValueError):
        _ = audio[0.6:0.3]


def test_nested_slice_chains_offsets(tone_audio):
    audio = Audio(str(tone_audio))
    seg = audio[0.2:0.8]  # 0.6s window starting at 0.2
    sub = seg[0.1:0.3]  # relative to the sub-window's start
    assert abs(sub.start_time - 0.3) < DUR_TOL
    assert abs(sub.duration - 0.2) < DUR_TOL


# --------------------------------------------------------------------------- #
# save()
# --------------------------------------------------------------------------- #


def test_save_with_output_path_writes_file(tone_audio, tmp_path):
    audio = Audio(str(tone_audio))
    out = tmp_path / "clip.mp3"
    returned = audio[0.2:0.7].save(output=str(out))
    from pathlib import Path

    assert isinstance(returned, Path)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0
    # Round-trips to roughly the requested sub-duration.
    saved = Audio(str(out))
    assert abs(saved.full_duration - 0.5) < 0.1


def test_save_creates_missing_parent_dirs(tone_audio, tmp_path):
    audio = Audio(str(tone_audio))
    out = tmp_path / "nested" / "deeper" / "clip.wav"
    returned = audio[0:0.5].save(output=str(out), format="wav")
    assert returned.exists()
    assert returned.suffix == ".wav"


# --------------------------------------------------------------------------- #
# fade_in / fade_out (chaining returns Audio)
# --------------------------------------------------------------------------- #


def test_fade_in_returns_audio(tone_audio):
    audio = Audio(str(tone_audio))
    faded = audio.fade_in(0.2)
    assert isinstance(faded, Audio)
    assert abs(faded.duration - 1.0) < DUR_TOL


def test_fade_out_returns_audio(tone_audio):
    audio = Audio(str(tone_audio))
    faded = audio.fade_out(0.2)
    assert isinstance(faded, Audio)
    assert abs(faded.duration - 1.0) < DUR_TOL


def test_fade_in_then_fade_out_chains(tone_audio):
    audio = Audio(str(tone_audio))
    faded = audio.fade_in(0.2).fade_out(0.2)
    assert isinstance(faded, Audio)
    assert abs(faded.duration - 1.0) < DUR_TOL
    # Result of fade is built from an AudioSegment -> src_path is None.
    assert faded.src_path is None


# --------------------------------------------------------------------------- #
# overlay / __add__
# --------------------------------------------------------------------------- #


def test_overlay_returns_audio_preserving_background_duration(tone_audio):
    audio = Audio(str(tone_audio))
    mixed = audio.overlay(audio)
    assert isinstance(mixed, Audio)
    # Overlay length is clamped to the background's length.
    assert abs(mixed.duration - 1.0) < DUR_TOL


def test_overlay_with_position_keeps_background_duration(tone_audio):
    audio = Audio(str(tone_audio))
    mixed = audio.overlay(audio, position=0.5)
    assert isinstance(mixed, Audio)
    assert abs(mixed.duration - 1.0) < DUR_TOL


def test_add_concatenates_durations(tone_audio):
    audio = Audio(str(tone_audio))
    combined = audio[0:0.5] + audio[0:0.5]
    assert isinstance(combined, Audio)
    assert abs(combined.duration - 1.0) < 0.1


# --------------------------------------------------------------------------- #
# .samples Mapping interface
# --------------------------------------------------------------------------- #


def test_samples_is_audiosamples_mapping(tone_audio):
    audio = Audio(str(tone_audio))
    from collections.abc import Mapping

    samples = audio.samples
    assert isinstance(samples, AudioSamples)
    assert isinstance(samples, Mapping)


def test_samples_len_matches_sample_count(tone_audio):
    audio = Audio(str(tone_audio))
    samples = audio.samples
    # Full 1s mono at 44100 Hz.
    assert abs(len(samples) - 44100) < 44100 * DUR_TOL


def test_samples_integer_index_returns_scalar_float(tone_audio):
    audio = Audio(str(tone_audio))
    samples = audio.samples
    value = samples[0]
    # Quirk: a NumPy scalar, not a builtin float, but float-coercible.
    assert isinstance(value, np.floating)
    assert isinstance(float(value), float)
    # Normalized to [-1, 1].
    assert -1.0 <= float(value) <= 1.0


def test_samples_slice_returns_ndarray(tone_audio):
    audio = Audio(str(tone_audio))
    samples = audio.samples
    chunk = samples[100:200]
    assert isinstance(chunk, np.ndarray)
    assert len(chunk) == 100


def test_samples_negative_index(tone_audio):
    audio = Audio(str(tone_audio))
    samples = audio.samples
    last = samples[-1]
    assert isinstance(last, np.floating)
    assert -1.0 <= float(last) <= 1.0


def test_samples_index_out_of_range_raises(tone_audio):
    audio = Audio(str(tone_audio))
    samples = audio.samples
    with pytest.raises(IndexError):
        _ = samples[len(samples)]


def test_samples_view_of_subsegment_is_shorter(tone_audio):
    audio = Audio(str(tone_audio))
    sub = audio[0.5:1.0]
    # The samples view of a sub-segment spans only that window.
    assert abs(len(sub.samples) - 22050) < 22050 * DUR_TOL


def test_audiosamples_direct_construction_from_path(tone_audio):
    samples = AudioSamples(str(tone_audio))
    assert abs(len(samples) - 44100) < 44100 * DUR_TOL
    assert isinstance(samples[0], np.floating)


# --------------------------------------------------------------------------- #
# Context manager protocol
# --------------------------------------------------------------------------- #


def test_audio_context_manager_yields_self(tone_audio):
    audio = Audio(str(tone_audio))
    with audio as ctx:
        assert ctx is audio
        assert abs(ctx.duration - 1.0) < DUR_TOL


def test_audio_context_manager_releases_on_exit(tone_audio):
    audio = Audio(str(tone_audio))
    with audio:
        pass
    # Exit calls close(), which drops the in-memory AudioSegment reference.
    assert audio._audio is None


def test_audio_close_is_idempotent(tone_audio):
    audio = Audio(str(tone_audio))
    audio.close()
    audio.close()  # second call must not raise
    assert audio._audio is None


# --------------------------------------------------------------------------- #
# normalize / to_mono / resample (return new Audio for chaining)
# --------------------------------------------------------------------------- #


def test_normalize_returns_new_audio_same_duration(tone_audio):
    audio = Audio(str(tone_audio))
    normalized = audio.normalize()
    assert isinstance(normalized, Audio)
    assert normalized is not audio
    assert abs(normalized.duration - audio.duration) < DUR_TOL
    # Same channel count / sample rate; only level changes.
    assert normalized.channels == audio.channels
    assert normalized.sample_rate == audio.sample_rate


def test_to_mono_downmixes_stereo(make_tone_audio):
    audio = Audio(str(make_tone_audio(1.0, channels=2)))
    assert audio.channels == 2
    mono = audio.to_mono()
    assert isinstance(mono, Audio)
    assert mono is not audio
    assert mono.channels == 1
    assert abs(mono.duration - audio.duration) < DUR_TOL


def test_to_mono_on_mono_is_noop_shaped(tone_audio):
    audio = Audio(str(tone_audio))  # mono
    mono = audio.to_mono()
    assert mono.channels == 1


def test_resample_changes_sample_rate(tone_audio):
    audio = Audio(str(tone_audio))
    assert audio.sample_rate == 44100
    resampled = audio.resample(16000)
    assert isinstance(resampled, Audio)
    assert resampled is not audio
    assert resampled.sample_rate == 16000
    assert abs(resampled.duration - audio.duration) < DUR_TOL


def test_convenience_methods_chain(make_tone_audio):
    audio = Audio(str(make_tone_audio(1.0, channels=2)))
    out = audio.to_mono().resample(22050).normalize()
    assert isinstance(out, Audio)
    assert out.channels == 1
    assert out.sample_rate == 22050
    assert abs(out.duration - 1.0) < DUR_TOL
