"""``mixing.splice`` (#167): replace a span of a take, joins faded, streamed for files."""

import subprocess
import sys

import numpy as np
import pytest

from mixing.splice import (
    JOIN_FADE_S,
    replace_span,
    replace_span_file,
    spliced_chunks,
)
from mixing.util import has_ffmpeg

SR = 8000
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg not on PATH")


def _noise(n, ch=2, seed=0, amp=0.1):  # well inside full scale: a file must not clip
    return np.random.default_rng(seed).standard_normal((n, ch)) * amp


def test_outside_the_fades_everything_is_copied_exactly():
    base, insert = _noise(SR), _noise(SR // 4, seed=1)
    out = replace_span(base, (0.4, 0.6), insert, sr=SR)
    a, b, f = int(0.4 * SR), int(0.6 * SR), int(round(JOIN_FADE_S * SR))
    assert len(out) == len(base) - (b - a) + len(insert)
    np.testing.assert_array_equal(out[: a - f], base[: a - f])
    np.testing.assert_array_equal(out[a + len(insert) + f :], base[b + f :])
    mid = out[a + f : a + len(insert) - f]
    np.testing.assert_array_equal(mid, insert[f:-f])  # the insert's body, untouched


def test_every_join_is_faded_to_silence():
    base, insert = np.ones((SR, 1)), np.ones((SR // 4, 1))
    out = replace_span(base, (0.4, 0.6), insert, sr=SR)[:, 0]
    a, n = int(0.4 * SR), len(insert)
    # the base fades out to 0 at the cut, the insert in from 0 and out to ~0,
    # and the base back in from 0: no step anywhere
    assert out[a - 1] < 0.2 and out[a] == 0 and out[a + n - 1] < 0.2
    assert out[a + n] == 0
    assert np.abs(np.diff(out)).max() < 2.0 / (JOIN_FADE_S * SR)


def test_the_result_does_not_depend_on_where_the_chunks_split():
    base, insert = _noise(5000), _noise(700, seed=1)
    whole = np.concatenate(list(spliced_chunks([base], (2000, 2600), insert, fade=64)))
    rng = np.random.default_rng(2)
    for _ in range(20):
        cuts = np.sort(rng.integers(0, len(base), 6))
        parts = np.split(base, cuts)  # empty pieces and splits inside the fades
        split = np.concatenate(
            list(spliced_chunks(parts, (2000, 2600), insert, fade=64))
        )
        np.testing.assert_array_equal(split, whole)


@pytest.mark.parametrize(
    "span", [(0.0, 0.0), (0.0, 0.2), (0.9, 1.0), (1.0, 1.0), (2.0, 3.0)]
)
def test_spans_at_the_edges(span):
    base, insert = _noise(SR), _noise(100, seed=1)
    out = replace_span(base, span, insert, sr=SR, fade_s=0)
    a = min(len(base), int(span[0] * SR))
    b = min(len(base), int(span[1] * SR))
    np.testing.assert_array_equal(out, np.concatenate([base[:a], insert, base[b:]]))


def test_mono_insert_into_stereo_and_mono_base_stays_mono():
    out = replace_span(_noise(SR), (0.1, 0.2), np.ones(50), sr=SR, fade_s=0)
    np.testing.assert_array_equal(out[800:850], np.ones((50, 2)))
    mono = replace_span(np.zeros(SR), (0.1, 0.2), np.ones(50), sr=SR)
    assert mono.ndim == 1 and len(mono) == SR - 800 + 50


@pytest.mark.parametrize(
    "span, insert, match",
    [
        ((0.5, 0.2), np.ones(5), "start <= end"),
        ((-0.1, 0.2), np.ones(5), "start <= end"),
        ((0.1, float("nan")), np.ones(5), "finite"),
        ((0.1, 0.2), np.ones((5, 3)), "mono or have the base's 2 channels"),
    ],
)
def test_bad_spans_and_inserts_are_refused(span, insert, match):
    with pytest.raises(ValueError, match=match):
        replace_span(_noise(SR), span, insert, sr=SR)


def _wav(path, samples, sr=SR):
    from mixing.decode import encode_float_chunks

    return encode_float_chunks([samples], sr, path)


def _read(path, **kw):
    from mixing.decode import decode_float_chunks

    rate, chunks = decode_float_chunks(path, **kw)
    return np.concatenate(list(chunks)), rate


@needs_ffmpeg
def test_a_lossless_file_keeps_every_sample_outside_the_joins(tmp_path):
    src = _wav(tmp_path / "take.wav", _noise(3 * SR))
    before, _ = _read(src)  # the 16-bit values the file holds
    insert = _noise(SR // 2, seed=1)
    written = replace_span_file(src, (1.0, 1.5), insert, output=tmp_path / "out.wav")
    after, rate = _read(written)
    assert rate == SR
    expected = replace_span(before, (1.0, 1.5), insert, sr=SR)
    np.testing.assert_allclose(after, expected, atol=1 / 32768)  # one 16-bit step
    a, f = SR, int(round(JOIN_FADE_S * SR))
    np.testing.assert_array_equal(after[: a - f], before[: a - f])  # bit-exact
    np.testing.assert_array_equal(
        after[a + len(insert) + f :], before[int(1.5 * SR) + f :]
    )


@needs_ffmpeg
def test_a_file_insert_is_resampled_and_the_default_output_sits_beside(
    tmp_path, monkeypatch
):
    import mixing.decode as decode

    monkeypatch.setattr(decode, "DECODE_CHUNK_S", 0.1)  # many chunks, many joins
    src = _wav(tmp_path / "take.wav", _noise(2 * SR))
    line = _wav(tmp_path / "line.wav", _noise(16000 // 2, ch=1, seed=3), sr=16000)
    written = replace_span_file(src, (0.5, 1.0), line)
    assert written == tmp_path / "take_spliced.wav"
    after, rate = _read(written)
    assert rate == SR and after.shape == (2 * SR, 2)  # half a second for half a second


@needs_ffmpeg
def test_in_place_is_allowed_for_audio_and_refused_over_a_video(tmp_path):
    from mixing.util import ffmpeg_exe

    src = _wav(tmp_path / "take.wav", _noise(SR))
    assert replace_span_file(src, (0.2, 0.4), np.zeros(10), output=src) == src
    assert _read(src)[0].shape == (SR - int(0.2 * SR) + 10, 2)
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=red:s=32x32:d=1",
         "-f", "lavfi", "-i", "sine=d=1", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    with pytest.raises(ValueError, match="drop its video"):
        replace_span_file(video, (0.2, 0.4), np.zeros(10), output=video)


def test_import_stays_light():
    code = (
        "import sys, mixing.splice; "
        "heavy = {'moviepy', 'cv2', 'pydub', 'PIL', 'scipy'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_span_past_the_end_still_fades_the_base_out():
    base = np.ones((SR, 1))
    out = replace_span(base, (2.0, 3.0), np.ones((100, 1)), sr=SR)[:, 0]
    f = int(round(JOIN_FADE_S * SR))
    assert len(out) == SR + 100
    assert out[SR - 1] < 0.2 and out[SR] == 0  # no step from 1.0 into the insert
    np.testing.assert_array_equal(out[: SR - f], 1.0)


def test_the_engine_checks_its_cut_and_fade():
    with pytest.raises(ValueError, match="0 <= start <= end"):
        list(spliced_chunks([np.zeros((10, 1))], (6, 3), np.zeros(2), fade=1))
    with pytest.raises(ValueError, match="fade must be >= 0"):
        list(spliced_chunks([np.zeros((10, 1))], (3, 6), np.zeros(2), fade=-1))
    with pytest.raises(ValueError, match="fade_s must be >= 0"):
        replace_span(np.zeros(10), (0.1, 0.2), np.zeros(2), sr=10, fade_s=-1)


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["pcm_s24le", "pcm_f32le"])
def test_a_deep_wav_master_keeps_its_depth_and_its_samples(tmp_path, codec):
    from mixing.decode import encode_float_chunks
    from mixing.probe import probe

    src = encode_float_chunks(
        [_noise(2 * SR)], SR, tmp_path / "m.wav", args=("-c:a", codec)
    )
    before, _ = _read(src)
    replace_span_file(
        src, (0.5, 1.0), [0.0] * 40, output=src
    )  # a list insert, in place
    assert [s.codec for s in probe(src).streams] == [codec]
    after, _ = _read(src)
    a, f = SR // 2, int(round(JOIN_FADE_S * SR))
    np.testing.assert_array_equal(after[: a - f], before[: a - f])  # bit-exact
    np.testing.assert_array_equal(after[a + 40 + f :], before[SR + f :])
