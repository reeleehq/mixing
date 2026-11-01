"""``mixing.audio.aligned_spans`` — which PARTS of a clip align, and at what offset each.

``align_clips_to_reference`` answers "where does this clip sit?" with one number. That is
the right answer only for a clip that is one continuous take. A clip that was stopped and
restarted has no such number, and the single-offset model does not say so — it returns the
offset of whichever part correlated best and describes the rest of the clip wrongly, with a
confidence that looks fine. Measured below: a clip holding two takes of the same song
scores 0.494 on the single-offset path, which passes any sane trust gate.

The fixtures are small on purpose. Each window is an FFT cross-correlation against the
whole reference, so window count is the cost axis and these run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import (
    AlignedSpan,
    AlignmentReference,
    prepare_reference,
    align_clips_to_reference,
    aligned_spans,
    find_audio_offset_detailed,
)
from mixing.errors import WindowTooWideForClip

SR = 16000
WIN, HOP = 10.0, 5.0
SPAN_WINDOW_S_DEFAULT = 20.0


def _reference(seconds: float = 90.0) -> np.ndarray:
    """A non-periodic broadband 'song' — one sharp autocorrelation peak, no ambiguity."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in [(180, 520), (440, 130), (700, 900), (110, 250)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    return x / np.max(np.abs(x))


@pytest.fixture(scope="module")
def ref() -> np.ndarray:
    return _reference()


@pytest.fixture(scope="module")
def song(tmp_path_factory, ref) -> str:
    p = tmp_path_factory.mktemp("spans") / "song.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p)


def _noisy(seg: np.ndarray, rng, snr_db: float = 15.0) -> np.ndarray:
    p_sig = np.mean(seg**2)
    return seg + rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), len(seg))


def _write(tmp_path, name: str, samples: np.ndarray) -> str:
    p = tmp_path / name
    wavfile.write(
        str(p), SR, ((samples / np.max(np.abs(samples))) * 32767).astype(np.int16)
    )
    return str(p)


def _take(ref, a: float, b: float, rng) -> np.ndarray:
    return _noisy(ref[int(a * SR) : int(b * SR)].copy(), rng)


def _spans(song, clip, **kw):
    return aligned_spans(song, clip, sample_rate=SR, window_s=WIN, hop_s=HOP, **kw)


# --------------------------------------------------------------------------
# The thing it exists for
# --------------------------------------------------------------------------


def test_a_stopped_and_restarted_clip_yields_one_span_per_take(song, ref, tmp_path):
    """The headline. Two takes, two spans, each with its OWN offset."""
    rng = np.random.default_rng(0)
    # Recorded song[5..30], stopped, then recorded song[60..85].
    clip = _write(
        tmp_path,
        "two.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 60, 85, rng)]),
    )
    spans = _spans(song, clip)

    assert len(spans) == 2, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    a, b = spans
    # abs=0.05, not abs=WIN: the observed error is 0.0000, and a WIN-wide tolerance
    # would let every reported offset be wrong by a whole window with the suite green.
    assert a.offset_s == pytest.approx(5.0, abs=0.05)
    assert b.offset_s == pytest.approx(35.0, abs=0.05)  # clip t=25 -> song t=60
    assert a.reference_span[0] == pytest.approx(5.0, abs=0.05)
    assert b.reference_span[1] == pytest.approx(85.0, abs=0.05)


def test_the_single_offset_model_describes_that_clip_wrongly_and_looks_confident(
    song, ref, tmp_path
):
    """Why this function had to exist, stated as a measurement rather than a claim.

    The single-offset answer is not merely imprecise — it is *right about one take and
    wrong about the other*, at a confidence that clears any threshold a caller would set.
    """
    rng = np.random.default_rng(0)
    clip = _write(
        tmp_path,
        "two.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 60, 85, rng)]),
    )
    one = find_audio_offset_detailed(song, clip, sample_rate=SR, feature="envelope")
    assert one.confidence > 0.3, "the wrong answer does not announce itself"
    spans = _spans(song, clip)
    # It matches ONE of the two spans and contradicts the other.
    matches = [s for s in spans if abs(s.offset_s - one.offset_s) < WIN]
    assert len(matches) == 1


def test_a_continuous_take_returns_exactly_one_span(song, ref, tmp_path):
    """The compatibility property every caller migrating off the single offset needs.

    Asserted rather than assumed: if a clip that IS one take came back as several, the
    migration would turn every healthy project into a fragmented one.
    """
    rng = np.random.default_rng(1)
    clip = _write(tmp_path, "one.wav", _take(ref, 20, 70, rng))
    spans = _spans(song, clip)

    assert len(spans) == 1
    assert spans[0].clip_start_s == pytest.approx(0.0, abs=0.01)
    # The most valuable assertion here — it cross-checks the windowed path against an
    # INDEPENDENT implementation. At abs=WIN it could not fire.
    single = find_audio_offset_detailed(song, clip, sample_rate=SR, feature="envelope")
    assert spans[0].offset_s == pytest.approx(single.offset_s, abs=0.05)


def test_spans_are_ordered_and_never_overlap(song, ref, tmp_path):
    """A clip instant claimed by two different offsets is not a fact about anything.

    Windows overlap by ``window - hop``, so the window straddling a boundary belongs to
    both takes; untrimmed, consecutive spans overlapped by a full hop.
    """
    rng = np.random.default_rng(2)
    clip = _write(
        tmp_path,
        "three.wav",
        np.concatenate(
            [_take(ref, 5, 25, rng), _take(ref, 55, 75, rng), _take(ref, 30, 50, rng)]
        ),
    )
    spans = _spans(song, clip)
    for a, b in zip(spans, spans[1:]):
        assert a.clip_end_s <= b.clip_start_s + 1e-9, (a, b)
        assert a.clip_start_s < a.clip_end_s


def test_a_clip_that_matches_nothing_returns_no_spans(song, tmp_path):
    """Empty is a real answer, not a failure."""
    rng = np.random.default_rng(3)
    clip = _write(tmp_path, "noise.wav", rng.normal(0, 1, int(30 * SR)))
    assert _spans(song, clip) == []


# --------------------------------------------------------------------------
# The parts that are easy to get wrong
# --------------------------------------------------------------------------


def test_the_offset_is_relative_to_the_clip_not_the_window(song, ref, tmp_path):
    """`offset = lag - window_start`, and dropping the subtraction is invisible at t=0.

    A clip whose match begins well INTO it is what makes the conversion observable: every
    window would otherwise report a different offset and no two would ever group, so a
    continuous take would come back shattered into one span per window.
    """
    rng = np.random.default_rng(4)
    clip = _write(tmp_path, "late.wav", _take(ref, 10, 60, rng))
    spans = _spans(song, clip)
    assert len(spans) == 1, (
        "the per-window offsets did not agree — check `lag - start_s`"
    )
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_silence_mid_take_does_not_read_as_a_stop_and_restart(song, ref, tmp_path):
    """One continuous take with a dead patch must come back as ONE span.

    Measured: 12 s of hard silence collapses the windows inside it to a confidence of
    exactly 0.0, the run breaks, and without the merge the result is two spans BOTH
    reporting offset 10.00 — a discontinuity that never happened. A stop and a restart
    cannot resume in sync, so two spans AGREEING on the offset are positive evidence of
    continuity.

    This replaces a hysteresis test that could not fail: on real correlations a degraded
    window does not dip into a band, it collapses, so no keep-threshold was ever what
    decided.
    """
    rng = np.random.default_rng(5)
    take = _take(ref, 10, 60, rng)
    take[int(20 * SR) : int(32 * SR)] = 0.0  # twelve dead seconds mid-take
    clip = _write(tmp_path, "dip.wav", take)

    spans = _spans(song, clip)
    assert len(spans) == 1, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(10.0, abs=0.05)
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_a_long_unverified_gap_is_not_merged_away(song, ref, tmp_path):
    """The bound on the merge, and why it is not optional.

    A clip that records the song, then a stretch of something else, then the song again
    AT THE SAME OFFSET would otherwise be merged into one span claiming the middle
    aligns. Correlation cannot tell "quiet" from "different material", so past the gap
    bound both are reported and the caller decides.
    """
    rng = np.random.default_rng(11)
    head = _take(ref, 10, 25, rng)
    middle = rng.normal(
        0, 0.5, int(30 * SR)
    )  # unrelated material, far longer than a window
    tail = _take(ref, 55, 70, rng)  # offset 10 again: clip t=45 -> song t=55
    clip = _write(tmp_path, "far.wav", np.concatenate([head, middle, tail]))

    spans = _spans(song, clip)
    assert len(spans) == 2, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(spans[1].offset_s, abs=1.0), (
        "the fixture is only meaningful if both spans DO agree on the offset — "
        "otherwise the offset rule would have kept them apart and the gap bound "
        "would not be what this test measures"
    )


def test_the_merge_is_bounded_by_merge_gap_s_and_the_caller_can_widen_it(
    song, ref, tmp_path
):
    """Same fixture, wider bound -> one span. The knob is what is under test."""
    rng = np.random.default_rng(11)
    clip = _write(
        tmp_path,
        "far.wav",
        np.concatenate(
            [
                _take(ref, 10, 25, rng),
                rng.normal(0, 0.5, int(30 * SR)),
                _take(ref, 55, 70, rng),
            ]
        ),
    )
    assert len(_spans(song, clip, merge_gap_s=60.0)) == 1


def test_a_clip_shorter_than_one_window_still_works(song, ref, tmp_path):
    rng = np.random.default_rng(6)
    clip = _write(tmp_path, "short.wav", _take(ref, 30, 36, rng))
    spans = _spans(song, clip)
    assert len(spans) == 1
    assert spans[0].offset_s == pytest.approx(30.0, abs=0.05)


def test_the_tail_of_a_clip_is_covered(song, ref, tmp_path):
    """A clip whose length is not a whole number of hops must not lose its end.

    Without the final catch-up window, up to ``window_s`` of every such clip is never
    looked at and a span ending there is silently truncated.
    """
    rng = np.random.default_rng(7)
    clip = _write(tmp_path, "tail.wav", _take(ref, 10, 47.5, rng))  # 37.5s on a 5s hop
    spans = _spans(song, clip)
    assert len(spans) == 1
    assert spans[0].clip_end_s == pytest.approx(37.5, abs=0.2)


def test_the_signals_are_decoded_once_not_once_per_window(
    song, ref, tmp_path, monkeypatch
):
    """Re-decoding per window would be N times the cost AND reintroduce issue #25.

    pydub's rate conversion has no anti-alias filter, so a per-window decode path is a
    per-window chance to halve the confidence. Counting the calls is the only way to keep
    a future refactor from quietly moving the decode inside the loop.
    """
    import mixing.audio.audio_ops as ops

    calls = []
    real = ops._load_mono_samples
    monkeypatch.setattr(
        ops, "_load_mono_samples", lambda src, sr: calls.append(1) or real(src, sr)
    )
    rng = np.random.default_rng(8)
    clip = _write(tmp_path, "count.wav", _take(ref, 10, 60, rng))
    ops.aligned_spans(song, clip, sample_rate=SR, window_s=WIN, hop_s=HOP)
    assert len(calls) == 2, "one decode for the reference, one for the clip — no more"


# --------------------------------------------------------------------------
# A prepared reference: the song decoded once for many clips (issue #111)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("feature", ["envelope", "waveform"])
def test_a_prepared_reference_gives_the_same_spans(song, ref, tmp_path, feature):
    rng = np.random.default_rng(11)
    clip = _write(
        tmp_path,
        "two_prep.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 60, 85, rng)]),
    )
    prepared = prepare_reference(song, sample_rate=SR, feature=feature)
    assert isinstance(prepared, AlignmentReference)
    assert prepared.sample_rate == SR
    assert prepared.duration_s == pytest.approx(len(ref) / SR)
    assert (prepared.envelope is None) == (feature == "waveform")
    kw = dict(window_s=WIN, hop_s=HOP, feature=feature)
    assert aligned_spans(prepared, clip, **kw) == _spans(song, clip, feature=feature)


def test_a_waveform_prepared_reference_still_serves_the_envelope_feature(
    song, ref, tmp_path
):
    rng = np.random.default_rng(12)
    clip = _write(tmp_path, "one_prep.wav", _take(ref, 10, 50, rng))
    prepared = prepare_reference(song, sample_rate=SR, feature="waveform")
    assert aligned_spans(prepared, clip, window_s=WIN, hop_s=HOP) == _spans(song, clip)


def test_many_clips_against_a_prepared_reference_decode_the_song_once(
    song, ref, tmp_path, monkeypatch
):
    import mixing.audio.audio_ops as ops

    decoded, analysed = [], []
    real_load, real_env = ops._load_mono_samples, ops.onset_envelope
    monkeypatch.setattr(
        ops,
        "_load_mono_samples",
        lambda src, sr: decoded.append(src) or real_load(src, sr),
    )
    monkeypatch.setattr(
        ops,
        "onset_envelope",
        lambda x, sr, **kw: analysed.append(len(x)) or real_env(x, sr, **kw),
    )
    prepared = ops.prepare_reference(song, sample_rate=SR)
    rng = np.random.default_rng(13)
    clips = [
        _write(tmp_path, f"c{i}.wav", _take(ref, 10 * i, 10 * i + 25, rng))
        for i in range(3)
    ]
    for clip in clips:
        ops.aligned_spans(prepared, clip, window_s=WIN, hop_s=HOP)
    assert decoded.count(song) == 1, "the song is decoded once, by prepare_reference"
    assert analysed.count(len(ref)) == 1, "and its envelope computed once"


def test_a_rate_that_disagrees_with_the_prepared_reference_is_refused(song, tmp_path):
    prepared = prepare_reference(song, sample_rate=SR)
    with pytest.raises(ValueError, match="prepared reference is at 16000"):
        aligned_spans(prepared, song, sample_rate=8000)
    with pytest.raises(ValueError, match="unknown feature"):
        prepare_reference(song, feature="mfcc")


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_reference_span_uses_the_same_offset_convention_as_ClipAlignment():
    """``reference_time = clip_time + offset_s``, as `ClipAlignment.offset_s` documents."""
    s = AlignedSpan(clip_start_s=4.0, clip_end_s=9.0, offset_s=100.0, confidence=0.5)
    assert s.reference_span == (104.0, 109.0)
    assert s.duration_s == 5.0


def test_an_unknown_feature_is_refused(song, tmp_path):
    rng = np.random.default_rng(9)
    clip = _write(tmp_path, "f.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="unknown feature"):
        aligned_spans(song, clip, feature="vibes")


@pytest.mark.parametrize("kw", [{"window_s": 0.0}, {"hop_s": -1.0}])
def test_a_non_positive_window_or_hop_is_refused(song, tmp_path, kw):
    rng = np.random.default_rng(10)
    clip = _write(tmp_path, "w.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="must be positive"):
        aligned_spans(song, clip, **kw)


# --------------------------------------------------------------------------
# Found by adversarial review: spans must lie ON the reference
# --------------------------------------------------------------------------


def test_a_span_never_reports_reference_time_the_reference_does_not_have(
    song, ref, tmp_path
):
    """The reviewer's reproducer, and the worst failure mode in this module.

    A window is admitted while it overlaps the reference by `min_overlap_ratio` of
    ITSELF, so a span could run past a reference edge by `(1-ratio)*window_s`. Measured
    before the fix on a 30 s reference: `reference_span == (-10.0, 40.0)` — a 50 s span
    of a 30 s reference, which is not imprecise but impossible.

    What made it dangerous is that it fails SILENTLY and in the wrong direction. A
    caller slicing `ref[int(-10.0*sr):int(40.0*sr)]` gets numpy's negative-index
    resolution: the start wraps to reference time 20 s and the end clamps to 30 s, so
    they receive the reference's TAIL for a span whose true content is its ENTIRE
    length. No exception either way.
    """
    rng = np.random.default_rng(20)
    # clip = 20 s of junk, then the whole reference, then 20 s of junk
    clip = _write(
        tmp_path,
        "overrun.wav",
        np.concatenate(
            [
                rng.normal(0, 0.5, int(20 * SR)),
                _noisy(ref.copy(), rng),
                rng.normal(0, 0.5, int(20 * SR)),
            ]
        ),
    )
    ref_dur = len(ref) / SR
    spans = _spans(song, clip)
    assert spans
    for sp in spans:
        a, b = sp.reference_span
        assert a >= -1e-6, f"reference start {a} is before the reference begins"
        assert b <= ref_dur + 1e-6, (
            f"reference end {b} is past the reference's {ref_dur}"
        )
        assert sp.duration_s <= ref_dur + 1e-6, (
            f"a {sp.duration_s}s span cannot align to a {ref_dur}s reference at one offset"
        )


def test_a_clip_that_starts_before_the_reference_is_trimmed_not_negated(
    song, ref, tmp_path
):
    """The preroll case — the common one, and the one that slices to EMPTY."""
    rng = np.random.default_rng(21)
    clip = _write(
        tmp_path,
        "preroll.wav",
        np.concatenate([rng.normal(0, 0.5, int(20 * SR)), _take(ref, 0, 30, rng)]),
    )
    spans = _spans(song, clip)
    assert spans
    assert spans[0].reference_span[0] >= -1e-6
    # ...and the CLIP edge moved with it, so the documented identity still holds.
    for sp in spans:
        assert sp.reference_span == pytest.approx(
            (sp.clip_start_s + sp.offset_s, sp.clip_end_s + sp.offset_s)
        )


def test_hop_wider_than_the_window_is_refused(song, ref, tmp_path):
    """It would leave clip time no correlation ever examined inside a single span."""
    rng = np.random.default_rng(22)
    clip = _write(tmp_path, "hop.wav", _take(ref, 10, 60, rng))
    with pytest.raises(ValueError, match="must not exceed"):
        aligned_spans(song, clip, sample_rate=SR, window_s=5.0, hop_s=55.0)


def test_the_earlier_spans_end_wins_the_disputed_overlap():
    """`_disjoin`'s documented rule, pinned directly.

    A pure list operation, so this needs no audio and no tolerance — it pins the POLICY
    rather than the correlation's accuracy, which is why the end-to-end tests (with their
    window-wide tolerances) could not.
    """
    from mixing.audio.audio_ops import _disjoin

    a = AlignedSpan(clip_start_s=0.0, clip_end_s=25.0, offset_s=5.0, confidence=0.9)
    b = AlignedSpan(clip_start_s=20.0, clip_end_s=50.0, offset_s=35.0, confidence=0.9)
    assert [(s.clip_start_s, s.clip_end_s) for s in _disjoin([a, b])] == [
        (0.0, 25.0),
        (25.0, 50.0),
    ]


def test_the_shipped_defaults_are_exercised(song, ref, tmp_path):
    """Every other test overrides window_s/hop_s; nothing ran what users get."""
    from mixing.audio.audio_ops import SPAN_HOP_S, SPAN_WINDOW_S

    rng = np.random.default_rng(23)
    clip = _write(tmp_path, "defaults.wav", _take(ref, 5, 85, rng))
    spans = aligned_spans(song, clip, sample_rate=SR)  # no overrides at all
    assert len(spans) == 1
    assert spans[0].offset_s == pytest.approx(5.0, abs=0.05)
    assert (SPAN_WINDOW_S, SPAN_HOP_S) == (20.0, 10.0)


def test_reference_duration_can_be_stated_by_the_caller(song, ref, tmp_path):
    """Parity with `align_clips_to_reference`, and it must actually bound the spans."""
    rng = np.random.default_rng(24)
    clip = _write(tmp_path, "rd.wav", _take(ref, 10, 80, rng))
    spans = _spans(song, clip, reference_duration=40.0)
    assert spans
    for sp in spans:
        assert sp.reference_span[1] <= 40.0 + 1e-6


# --------------------------------------------------------------------------
# Issue #30: a reference that repeats itself
#
# The fixtures above are deliberately non-repetitive ("one sharp autocorrelation peak,
# no alignment ambiguity"), which is why the suite was green while this was live. Music
# is the material this function exists for, and music repeats.
# --------------------------------------------------------------------------


def _motif(seconds: float, seed: int) -> np.ndarray:
    """A short broadband phrase with percussive onsets, so both features have a grip."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in rng.uniform(120, 900, (4, 2)):
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / max(t[-1], 1e-9))) * t)
    beat = np.zeros_like(t)
    for i in range(0, len(t), int(0.5 * SR)):
        n = min(400, len(t) - i)
        beat[i : i + n] += np.hanning(400)[:n]
    return (x / np.max(np.abs(x))) * 0.7 + beat * 0.5


#: Length of one motif in the repetitive fixtures, seconds. Six of them make a 90 s
#: "song" — the same length as the non-repetitive reference above, so the two are
#: comparable and the only variable is the repetition.
MOTIF_S = 15.0


@pytest.fixture(scope="module")
def repeating(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """Verse/chorus structure: A B C B A B. An offset into it is NOT unique."""
    a, b, c = (_motif(MOTIF_S, s) for s in (1, 2, 3))
    ref = np.concatenate([a, b, c, b, a, b])
    p = tmp_path_factory.mktemp("rep") / "repeating.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


@pytest.fixture(scope="module")
def tiling(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """One motif, nine times. Every offset that is right is right nine ways."""
    ref = np.concatenate([_motif(10.0, 11)] * 9)
    p = tmp_path_factory.mktemp("tile") / "tiling.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


def test_a_repeating_reference_no_longer_fragments_one_continuous_take(
    repeating, tmp_path
):
    """The headline of issue #30, as a before/after on the same clip.

    ``near_tie_ratio=0.0`` is the shipped-before behaviour — each window picks its lag by
    an independent argmax — and it is exercised here rather than described, because a
    regression test for a coin flip has to show the coin.

    Measured: the argmax path returns 4 spans for one continuous take, of which three
    carry a WRONG offset (80.0, 50.0 and -40.0 against a truth of 20.0) — and the worst
    of them scores 0.982, HIGHER than the span that is right. That is the failure a
    consumer cannot filter out. The count is asserted as a floor rather than pinned:
    which way a near-tie falls is exactly what is not stable here, and pinning it would
    make this test fail for being right.
    """
    song, ref = repeating
    rng = np.random.default_rng(30)
    clip = _write(
        tmp_path, "rep_one.wav", _take(ref, 20, 70, rng)
    )  # one take, offset 20

    argmax = _spans(song, clip, near_tie_ratio=0.0)
    assert len(argmax) >= 3, "the fixture must still reproduce the defect"
    wrong = [s for s in argmax if abs(s.offset_s - 20.0) > 1.0]
    assert wrong, "the defect is a WRONG offset, not merely a split"
    assert max(s.confidence for s in wrong) > 0.9, (
        "and it is wrong at a confidence no caller would filter out"
    )

    spans = _spans(song, clip)
    assert len(spans) == 1, [(s.clip_start_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(20.0, abs=0.05)
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_support_says_the_windows_needed_help_and_the_confidence_does_not(
    repeating, tiling, song, ref, tmp_path
):
    """The acceptance line: the offset is right AND the honest doubt is reported.

    ``confidence`` answers "how well does the clip match where we put it", and on a
    repeating reference that question has several excellent answers. Measured across
    these three references it reads 0.985 / 0.978 / 0.980 — it barely moves, and it does
    not even move the right way: the exactly tiling reference, where the offset is a
    free choice among nine, scores HIGHER than the verse/chorus one. A caller gating on
    it alone cannot tell them apart, which is the complaint in issue #30. ``support`` —
    the fraction of windows that reached the offset unaided — is what moves, and it
    moves with the repetition: 1.000 / 0.444 / 0.000.
    """
    rng = np.random.default_rng(31)
    cases = {}
    for name, (path, material) in {
        "none": (song, ref),
        "verse/chorus": repeating,
        "exact tiling": tiling,
    }.items():
        clip = _write(tmp_path, f"sup_{len(cases)}.wav", _take(material, 20, 70, rng))
        # The take is one span on the first two; on the tiling reference the answer is
        # arbitrary enough that a tail window can peel off, so take the longest.
        cases[name] = max(_spans(path, clip), key=lambda s: s.duration_s)

    assert cases["none"].support == 1.0, "no repetition, no disagreement"
    assert cases["verse/chorus"].support < 1.0, (
        "some windows correlated just as well against the wrong chorus"
    )
    assert cases["exact tiling"].support < cases["verse/chorus"].support, (
        "and where every offset is equally true, nothing agrees unaided"
    )
    # The point of the field: the confidence cannot make this distinction.
    assert min(s.confidence for s in cases.values()) > 0.9


def test_a_stop_and_restart_still_departs_on_a_repeating_reference(repeating, tmp_path):
    """The feature the fix must not eat.

    Consensus could trivially return one span for everything. It does not, because no
    window is ever moved to a lag its own correlation did not already rate a near-tie —
    and a restarted take has no such lag near the old offset. Both takes here span a
    motif boundary whose SEQUENCE occurs once (…C and B,A…), so each offset is unique
    and this asserts the answer rather than the tie-break.
    """
    song, ref = repeating
    rng = np.random.default_rng(32)
    clip = _write(
        tmp_path,
        "rep_two.wav",
        np.concatenate([_take(ref, 25, 45, rng), _take(ref, 55, 75, rng)]),
    )
    spans = _spans(song, clip)
    assert len(spans) == 2, [(s.clip_start_s, s.offset_s) for s in spans]
    a, b = spans
    assert a.offset_s == pytest.approx(25.0, abs=0.05)
    assert b.offset_s == pytest.approx(35.0, abs=0.05)  # clip t=20 -> song t=55


def test_near_tie_ratio_zero_is_the_old_behaviour_exactly(song, ref, tmp_path):
    """The knob is a seam, so it has to be pinned at both ends.

    On material with no repetition the consensus pass has nothing to decide, so the two
    settings must agree exactly — otherwise the default silently perturbs every existing
    caller's numbers, and the before/after test above would be measuring the wrong thing.
    """
    rng = np.random.default_rng(33)
    clip = _write(tmp_path, "seam.wav", _take(ref, 10, 60, rng))
    assert _spans(song, clip) == _spans(song, clip, near_tie_ratio=0.0)


def test_a_negative_near_tie_ratio_is_refused(song, tmp_path):
    rng = np.random.default_rng(34)
    clip = _write(tmp_path, "neg.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="near_tie_ratio"):
        aligned_spans(song, clip, sample_rate=SR, near_tie_ratio=-0.1)


# --------------------------------------------------------------------------
# The same defect in the whole-clip estimator (`align_clips_to_reference`)
#
# Same correlation, same coin flip — and it is the entry point the named consumer
# actually calls, so the fix has to reach it or it reaches no one.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_halves(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """A 90 s reference that is one 45 s half, twice."""
    ref = np.concatenate([_motif(45.0, 7)] * 2)
    p = tmp_path_factory.mktemp("halves") / "halves.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


def test_the_whole_clip_estimator_reaches_the_consensus_offset(two_halves, tmp_path):
    """Measured: the argmax puts this clip 45 s out, and its coefficient does not flinch.

    Offset 65 is not merely wrong, it is impossible — the clip is 50 s and the reference
    ends at 90 s, so the clip's last 25 s would sit past the end of the song. One number
    over the whole clip cannot notice that; the windows can, because the ones covering
    that tail have no candidate there at all.
    """
    song, ref = two_halves
    rng = np.random.default_rng(35)
    clip = _write(tmp_path, "halves_clip.wav", _take(ref, 20, 70, rng))  # offset 20

    (old,) = align_clips_to_reference(song, [clip], sample_rate=SR, consensus=False)
    assert abs(old.offset_s - 20.0) > 1.0, "the fixture must still reproduce the defect"
    assert old.confidence > 0.9, "at a confidence that clears any sane gate"
    assert old.support is None, (
        "and nothing measured agreed with it, which is what `support` must say — a 1.0 "
        "here would VOUCH for the wrong offset instead of merely failing to warn"
    )

    (new,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    assert new.offset_s == pytest.approx(20.0, abs=0.05)
    assert new.support < 1.0, "half of this reference genuinely does fit twice"


def test_support_localises_a_clip_that_is_only_partly_the_song(song, ref, tmp_path):
    """The second thing a coefficient cannot say: how MUCH of the clip is the reference.

    Half song, half unrelated noise. The confidence is measured where the clip matches,
    so it stays high — and should, the alignment really is that good. ``support`` is the
    field that reports that only half the clip voted for it.
    """
    rng = np.random.default_rng(36)
    clip = _write(
        tmp_path,
        "halfjunk.wav",
        np.concatenate([_take(ref, 10, 40, rng), rng.normal(0, 0.5, int(30 * SR))]),
    )
    (a,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    assert a.offset_s == pytest.approx(10.0, abs=0.05)
    assert a.confidence > 0.9
    assert 0.3 < a.support < 0.8, f"about half the clip should agree, got {a.support}"


def test_an_explicit_window_wider_than_the_clip_is_refused(song, ref, tmp_path):
    """A window the clip cannot hold is an error, not a silently ungoverned answer.

    This test used to assert the opposite: that consensus at ``window_s=SPAN_WINDOW_S``
    on a 15 s clip degraded to the single whole-clip correlation, same offset, same
    confidence, ``support=None``. It degraded correctly — and unreportably. The caller
    asked for a windowed vote, got one correlation, and the only trace was the ABSENCE
    of a support number, which is exactly what a clip too short to support at any window
    also shows (issue #43). So the explicit-window path now refuses.

    What is NOT refused, and is pinned separately below: the ``window_s=None`` path,
    which fits the window to the clip and may honestly land on a single window at the
    floor. That is the estimator doing its best for a short clip, not a caller's mistake.
    """
    from mixing.audio.audio_ops import SPAN_HOP_S, SPAN_WINDOW_S

    rng = np.random.default_rng(37)
    clip = _write(tmp_path, "short_clip.wav", _take(ref, 30, 45, rng))
    with pytest.raises(WindowTooWideForClip) as excinfo:
        align_clips_to_reference(
            song, [clip], sample_rate=SR, window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S
        )
    err = excinfo.value
    assert err.clip_index == 0
    assert err.window_s == SPAN_WINDOW_S and err.hop_s == SPAN_HOP_S
    assert err.clip_duration_s == pytest.approx(15.0, abs=0.05)
    # duration / (1 + MAX_SUPPORT_OVERLAP): the largest window that still leaves a
    # second INDEPENDENT look. NOT duration - hop_s — see the counterexamples in
    # test_adaptive_window.py, which the hop-based bound gets wrong in both directions.
    assert err.max_window_s == pytest.approx(10.0, abs=0.05)
    # The numbers must be IN the message too — the point of the error is that the
    # window, not the clip's length alone, is nameable as the reason.
    text = str(err)
    assert "window_s=20.000" in text and "clip 0" in text
    assert isinstance(err, ValueError), "existing ValueError handlers keep working"
    # And the window the error names must RESTORE what the refusal withheld: not merely
    # be accepted, but come back with the support the caller asked for in the first
    # place. An accepted-but-still-unmeasured ceiling would be the original bug wearing
    # an exception.
    (ok,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=err.max_window_s, hop_s=SPAN_HOP_S
    )
    assert ok.window_s == pytest.approx(err.max_window_s)
    assert ok.support is not None, "the remedy the error names must actually measure"


def test_the_bound_is_the_clips_length_and_the_overlap_rule_not_the_hop(
    song, ref, tmp_path
):
    """The two hop-based counterexamples, stated on real support rather than arithmetic.

    ``window_s > clip_duration - hop_s`` is the intuitive bound and it is wrong in both
    directions, because :func:`_window_offsets` appends a tail window at
    ``len(clip) - n_win`` whatever the hop:

    - a TINY hop does not buy a second look — ``window_s=11, hop_s=1`` on a 15 s clip
      satisfies the hop-based bound and still has nothing independent to vote with;
    - a hop LONGER than the clip does not cost one — ``window_s=6, hop_s=12`` violates
      the hop-based bound and measures a support perfectly well.

    Pinned on the estimator's own output, so the guard cannot drift back onto the
    intuitive rule without a test failing on the thing that actually matters.
    """
    rng = np.random.default_rng(41)
    clip = _write(tmp_path, "fifteen.wav", _take(ref, 30, 45, rng))

    with pytest.raises(WindowTooWideForClip):
        align_clips_to_reference(song, [clip], sample_rate=SR, window_s=11.0, hop_s=1.0)

    (wide_hop,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=6.0, hop_s=12.0
    )
    assert wide_hop.support is not None, "the tail window is a second look"


# --------------------------------------------------------------------------
# `support` must never vouch for something it did not measure
#
# Found by review of the first version of this fix, which reported ``support=1.0``
# wherever there was only one opinion to count. That is the same class of defect as
# issue #30 itself: a number that looks like evidence and is not.
# --------------------------------------------------------------------------


def test_the_one_window_reproduction_is_now_a_refusal(two_halves, tmp_path):
    """The reviewer's reproduction, converted from a silent ``None`` to a refusal.

    A 15 s clip taken from offset 30.0 of a reference that is one 45 s half twice comes
    back at offset **75.0** — wrong — at confidence 0.979 when it is measured as a single
    window, because a single window genuinely cannot tell those two halves apart. The
    original complaint was about what ``support`` said of that: reporting 1.0 turns
    "nobody checked" into "everybody agreed", so it reports ``None``.

    ``None`` was the right answer for the vote and the wrong answer for the CALL: asking
    for a 20 s window on a 15 s clip cannot produce the vote it names, and issue #43 says
    so out loud rather than handing back an uncorroborated 75.0. ``consensus=False`` still
    returns that same wrong offset with ``support=None`` — asked for explicitly, it is an
    answer and not a degradation, and it stays pinned here as the contrast.
    """
    from mixing.audio.audio_ops import SPAN_HOP_S, SPAN_WINDOW_S

    song, ref = two_halves
    rng = np.random.default_rng(38)
    clip = _write(tmp_path, "one_window.wav", _take(ref, 30, 45, rng))  # true offset 30

    with pytest.raises(WindowTooWideForClip) as excinfo:
        align_clips_to_reference(
            song, [clip], sample_rate=SR, window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S
        )
    assert excinfo.value.max_window_s == pytest.approx(10.0, abs=0.05)

    (single,) = align_clips_to_reference(song, [clip], sample_rate=SR, consensus=False)
    assert single.offset_s == 75.0, (
        "the fixture is only meaningful while the offset really is wrong"
    )
    assert single.confidence > 0.9
    assert single.support is None, "consensus=False put nothing to a vote at all"


def test_the_default_window_path_is_untouched_by_the_refusal(two_halves, ref, tmp_path):
    """The refusal is about EXPLICIT windows only — ``window_s=None`` is byte-identical.

    Issue #43 converts a working call into an error, so the boundary of that change is
    worth pinning rather than inferring: with no explicit ``window_s`` the window is
    fitted per clip, every clip from long to floor-short is accepted, and the numbers
    that come back are the ones the adaptive path produced before the refusal existed.

    The floor case is the one that would break first if the check ever leaked into the
    ``None`` path: a clip short enough that the fitted window is the floor may hold a
    single window, which is precisely the shape the explicit path now refuses — and here
    it must still be answered, because nobody asked for a window the clip cannot hold.
    """
    song, _ = two_halves
    rng = np.random.default_rng(39)
    clips = [
        _write(tmp_path, "long.wav", _take(ref, 5, 65, rng)),  # 60 s, default window
        _write(tmp_path, "mid.wav", _take(ref, 30, 45, rng)),  # 15 s, fitted window
        _write(tmp_path, "tiny.wav", _take(ref, 10, 12, rng)),  # 2 s, floor window
    ]
    got = align_clips_to_reference(song, clips, sample_rate=SR)
    assert [a.index for a in got] == [0, 1, 2], "no clip was refused"
    assert got[0].window_s == pytest.approx(SPAN_WINDOW_S_DEFAULT)
    assert got[1].window_s == pytest.approx(5.0, abs=0.05), "min(20, 15/3)"
    assert got[2].window_s < 5.0, "the floor window, fitted to a 2 s clip"
    assert all(a.hop_s == pytest.approx(a.window_s / 2) for a in got)
    # A single-window answer is still an ANSWER on this path, not an error.
    assert got[2].offset_s is not None and got[2].confidence is not None


def test_the_same_clip_gets_a_measured_support_once_its_window_fits_it(
    two_halves, tmp_path
):
    """The other half of the test above: the vote exists, so the number exists.

    Same fixture, same clip, same call — minus the pinned window. At the shipped default
    the window is now fitted to the clip, so the 15 s clip that had ONE opinion has
    several, and ``support`` is a fraction instead of ``None``. That is what this test is
    for, and all it is for.

    **This is a tie-break, not a wrong→right correction, and it is not evidence that the
    fix works.** The reference here is one 45 s half twice, so 30.0 and 75.0 are *both*
    places the clip genuinely fits: the material at those two times is identical, and no
    correlation can prefer one. 30.0 is where this clip was cut from, which the vote now
    settles on, but a fixture that cannot distinguish them cannot demonstrate an offset
    being corrected. The evidence that #41's rule fixes offsets is the measurement on real
    cross-device material (thorwhalen/muvid#59), where the wrong answer was wrong by 102 s
    against a reference that does not repeat verbatim.

    The support is deliberately not asserted to be 1.0: with two equally true answers some
    windows land on each, and a fraction strictly inside ``(0, 1]`` is the honest report of
    exactly that — a number a caller can gate on, which ``None`` was not.
    """
    song, ref = two_halves
    rng = np.random.default_rng(38)
    clip = _write(tmp_path, "fitted_window.wav", _take(ref, 30, 45, rng))

    (fitted,) = align_clips_to_reference(song, [clip], sample_rate=SR)

    assert fitted.offset_s == pytest.approx(30.0, abs=0.05)
    assert fitted.support is not None, (
        "a fitted window is a vote, and a vote is measured"
    )
    assert 0.0 < fitted.support <= 1.0


def test_a_span_too_short_to_disagree_reports_no_support(song, ref, tmp_path):
    """The same rule on the span side: one window, one opinion, no support fraction."""
    rng = np.random.default_rng(39)
    clip = _write(
        tmp_path, "tiny.wav", _take(ref, 30, 36, rng)
    )  # shorter than a window
    (span,) = _spans(song, clip)
    assert span.offset_s == pytest.approx(30.0, abs=0.05)
    assert span.support is None


def test_windows_that_are_the_same_look_twice_do_not_count_as_support(
    song, ref, tmp_path
):
    """Two windows sharing 95% of their samples are one opinion, not two.

    Reported from real material (issue #30): a 10 s clip at ``window_s=9.5, hop_s=0.5``
    produced two windows overlapping 95%, which agreed — of course they did, they saw the
    same audio — and the pair was reported as ``support=1.00`` for an offset 102 s from
    the truth. That is the shared-bias failure the consensus pass exists to avoid, one
    level up: a count of agreeing windows only means something if the windows could have
    disagreed. So the TALLY runs over windows separated by at least
    :data:`~mixing.audio.audio_ops.MAX_SUPPORT_OVERLAP`, and below a quorum of those the
    answer is ``None`` — unmeasured — rather than a manufactured 1.0.

    Since issue #43 this exact call does not return an unmeasured answer — it REFUSES.
    ``window_s=9.5`` on a 10 s clip is above the largest window that leaves a second
    independent look (10 / 1.5 = 6.67), so the vote it names cannot be held, and the
    ``support=None`` this test was written to pin is precisely the unexplained silence
    #43 is about. The tally rule is unchanged and still what makes the answer ``None``;
    what changed is that the caller is now told, and told at what window to ask again.

    The offset itself is unaffected: every window still votes, which the retry shows.
    """
    rng = np.random.default_rng(41)
    clip = _write(tmp_path, "overlapped.wav", _take(ref, 20, 30, rng))
    with pytest.raises(WindowTooWideForClip) as excinfo:
        align_clips_to_reference(song, [clip], sample_rate=SR, window_s=9.5, hop_s=0.5)
    assert excinfo.value.max_window_s == pytest.approx(10.0 / 1.5, abs=0.05)

    (retried,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=excinfo.value.max_window_s, hop_s=0.5
    )
    assert retried.offset_s == pytest.approx(20.0, abs=0.05), (
        "the offset still measures"
    )
    assert retried.support is not None, "and now it is corroborated as well"

    (spaced,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=5.0, hop_s=2.5
    )
    assert spaced.offset_s == pytest.approx(20.0, abs=0.05)
    assert spaced.support == 1.0, "at half-window hops every window still counts"


def _spaced_windows(starts, length: float):
    """One ``_WindowMeasurement`` per start — only the extents matter to the tally."""
    from mixing.audio.audio_ops import _WindowMeasurement

    return [
        _WindowMeasurement(
            clip_start_s=s0,
            clip_end_s=s0 + length,
            candidates=((0.0, 1.0),),
            vote_offset_s=0.0,
        )
        for s0 in starts
    ]


@pytest.mark.parametrize(
    "starts,kept,why",
    [
        ([0.0, 10.0, 20.0, 30.0], 4, "a hop of exactly half a window keeps them all"),
        ([0.0, 10.1, 20.2, 30.3], 4, "and anything wider certainly does"),
        (
            [0.0, 5.0, 10.0, 15.0, 20.0],
            3,
            "a quarter-window hop counts every other one",
        ),
        ([0.0, 0.5], 1, "95% overlap is one look, not two — the real-material case"),
        (
            [0.0, 10.0, 20.0, 30.0, 35.0],
            4,
            "the appended TAIL window is not a new look",
        ),
    ],
)
def test_where_the_independence_bound_falls(starts, kept, why):
    """The bound is asserted ON the boundary, not on a clip that happens to sit near it.

    ``SPAN_HOP_S`` is exactly half of ``SPAN_WINDOW_S``, so whether the comparison is
    ``>=`` or ``>`` decides whether every default caller tallies all of its windows or
    every other one — a one-character difference that would halve every documented
    support number. A test on representative audio cannot see that; these cases can,
    because they sit exactly on it.

    The last case is the one that costs something. ``_window_offsets`` appends a tail
    window so a clip whose length is not a whole number of hops is still covered to its
    end, and that window usually lands less than half a window after its neighbour. It
    still votes and still sets the offset — it is only excluded from the TALLY, which
    lowers the denominator by one. Measured on real material, that moved one clip's
    support from 0.45 to 0.50.
    """
    from mixing.audio.audio_ops import _independent_windows

    assert len(_independent_windows(_spaced_windows(starts, 20.0))) == kept, why


def test_the_default_settings_tally_the_whole_regular_grid(song, ref, tmp_path):
    """The same boundary, reached through the real windowing code rather than by hand."""
    from mixing.audio.audio_ops import (
        SPAN_HOP_S,
        SPAN_WINDOW_S,
        _independent_windows,
        _load_mono_samples,
        _window_offsets,
    )

    rng = np.random.default_rng(42)
    clip = _write(tmp_path, "default_hops.wav", _take(ref, 10, 70, rng))
    windows = _window_offsets(
        _load_mono_samples(song, SR),
        _load_mono_samples(clip, SR),
        SR,
        window_s=SPAN_WINDOW_S,
        hop_s=SPAN_HOP_S,
        feature="envelope",
        min_overlap_ratio=0.5,
    )
    assert len(windows) > 2, "the fixture must produce a real sequence of windows"
    assert _independent_windows(windows) == windows


def test_an_unmeasured_span_does_not_inherit_support_when_merged(song, ref, tmp_path):
    """A merge must not launder an unmeasured half through a measured one.

    ``_merge_same_offset`` rejoins two spans that agree across a short unverified gap.
    If one of them never measured a support, averaging the pair would hand the merged
    span a fraction that describes only part of it — the vouching problem again, one
    level up. Unmeasured plus measured is unmeasured.
    """
    from mixing.audio.audio_ops import _merge_same_offset

    measured = AlignedSpan(0.0, 30.0, offset_s=5.0, confidence=0.9, support=1.0)
    unmeasured = AlignedSpan(31.0, 37.0, offset_s=5.0, confidence=0.9, support=None)
    (merged,) = _merge_same_offset(
        [measured, unmeasured], offset_tolerance_s=0.25, merge_gap_s=10.0
    )
    assert merged.clip_end_s == 37.0, "the fixture must actually merge"
    assert merged.support is None

    both = AlignedSpan(31.0, 37.0, offset_s=5.0, confidence=0.9, support=0.5)
    (merged_both,) = _merge_same_offset(
        [measured, both], offset_tolerance_s=0.25, merge_gap_s=10.0
    )
    assert merged_both.support == pytest.approx((1.0 * 30.0 + 0.5 * 6.0) / 36.0)


def test_two_offsets_that_genuinely_tie_report_the_tie(song, ref, tmp_path):
    """A 50/50 is reported as a 50/50, not resolved into a confident single answer.

    Two takes of equal length is the cleanest tie there is: half the windows say one
    offset, half say the other, and asking `align_clips_to_reference` for ONE number has
    no better answer than picking one. Measured, it returns a true offset — 5.0, one of
    the two — at confidence 0.985, which is exactly as high as it would be if the clip
    were unambiguous. ``support`` is the only thing that reports the split, at 0.444.

    Which of the two wins is arbitrary and deliberately not pinned; that it is one of
    them, and that the support says half the clip disagrees, is the contract.
    """
    rng = np.random.default_rng(40)
    clip = _write(
        tmp_path,
        "tie.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 55, 80, rng)]),
    )
    (a,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    assert a.offset_s == pytest.approx(5.0, abs=0.05) or a.offset_s == pytest.approx(
        30.0, abs=0.05
    ), f"neither true offset was returned: {a.offset_s}"
    assert a.confidence > 0.9, "the tie does not show up in the coefficient"
    assert 0.35 < a.support < 0.65, f"a 50/50 must read as one, got {a.support}"


# --------------------------------------------------------------------------
# `consensus=False` is a characterization guardrail, not a convenience flag
# --------------------------------------------------------------------------

#: ``(offset_s, confidence)`` produced by ``align_clips_to_reference`` at commit ad8f056
#: — the last commit BEFORE the consensus pass — for each fixture below. Captured by
#: running that checkout, not by running the current code, so this pins the new
#: ``consensus=False`` path against the OLD function rather than against itself.
#: Keyed by ``(fixture, take_start, take_end, seed, feature)``.
PRE_CONSENSUS_RESULTS = {
    ("plain", 30, 45, 37, "envelope"): (30.0, 0.9845849122668827),
    ("plain", 30, 45, 37, "waveform"): (30.0, 0.9845842060773334),
    ("plain", 10, 60, 40, "envelope"): (10.0, 0.9845698125115664),
    ("plain", 10, 60, 40, "waveform"): (10.0, 0.9845689746370594),
    ("halves", 20, 70, 35, "envelope"): (-25.0, 0.9781992781864717),
    ("halves", 20, 70, 35, "waveform"): (-25.0, 0.9781992794529429),
    ("halves", 30, 45, 38, "envelope"): (75.0, 0.9792068725230924),
    ("halves", 30, 45, 38, "waveform"): (75.0, 0.9792019527804959),
}


@pytest.mark.parametrize("key", sorted(PRE_CONSENSUS_RESULTS))
def test_consensus_false_reproduces_the_pre_consensus_function(
    key, song, ref, two_halves, tmp_path
):
    """``consensus=False`` must be the OLD behaviour, measured against the old code.

    A new-vs-new comparison could not catch this: if the consensus path and the
    single-correlation path drifted together, both would move and the test would stay
    green. So the expected values were captured by running the pre-change checkout
    (ad8f056) and are pinned here as literals — including the two WRONG offsets that
    change motivated (-25.0 and 75.0), because a guardrail records what the code did,
    not what it should have done.

    The confidence is compared to 1e-9 rather than exactly: an FFT correlation can
    differ in its last bits across platforms and library versions, and a bit-exact pin
    would fail for reasons that have nothing to do with this function.
    """
    fixture, a, b, seed, feature = key
    path, material = {"plain": (song, ref), "halves": two_halves}[fixture]
    rng = np.random.default_rng(seed)
    clip = _write(
        tmp_path, f"pre_{fixture}_{a}_{b}_{seed}.wav", _take(material, a, b, rng)
    )

    (got,) = align_clips_to_reference(
        path, [clip], sample_rate=SR, feature=feature, consensus=False
    )
    want_offset, want_confidence = PRE_CONSENSUS_RESULTS[key]
    assert got.offset_s == want_offset
    assert got.confidence == pytest.approx(want_confidence, abs=1e-9)
    assert got.support is None, "the old function had no support to report"
