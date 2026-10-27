"""The pure filter builders (an#611): pinned strings, and refused inputs."""

import pytest

from mixing.filtergraph import atempo_chain, resample_pitch_filter


def test_the_strings_are_pinned_because_an_keys_on_them():
    """`an` keys every pitch- or tempo-shifted voice line on these strings
    (an.audio.effects, an#611). Changing one re-processes those lines there:
    change it on purpose, and update this test with that in mind."""
    assert resample_pitch_filter(4, 44100) == (
        "asetrate=55563,aresample=44100,atempo=0.7936936451"
    )
    assert resample_pitch_filter(-12, 44100, tempo=1.1, flush_s=0.25) == (
        "asetrate=22050,aresample=44100,apad=pad_dur=0.25,"
        "atempo=2.0000000000,atempo=1.1000000000"
    )
    assert resample_pitch_filter(0, 44100, tempo=1.25, flush_s=0.25) == (
        "apad=pad_dur=0.25,atempo=1.2500000000"
    )


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((0, 44100), {"tempo": 0.0}),
        ((0, 44100), {"tempo": -1.0}),
        ((0, 44100), {"tempo": float("inf")}),
        ((0, 44100), {"tempo": float("nan")}),
        ((3, 0), {}),
        ((-24, 1), {}),
        ((3, 44100), {"flush_s": -0.1}),
    ],
)
def test_inputs_that_build_no_chain_are_refused(args, kwargs):
    with pytest.raises(ValueError):
        resample_pitch_filter(*args, **kwargs)


def test_a_tiny_pad_is_written_in_fixed_point():
    assert "pad_dur=0.00001," in resample_pitch_filter(3, 44100, flush_s=1e-5)


def test_atempo_chain_refuses_what_would_never_end():
    with pytest.raises(ValueError):
        atempo_chain(0.0)
