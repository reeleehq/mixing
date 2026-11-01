"""``mixing.errors``: the error classes survive pickle and copy, numbers intact.

They cross process boundaries (multiprocessing, concurrent.futures) and get
copied by test tools; an exception that cannot be rebuilt from its pickle turns
the real error into a ``TypeError`` about missing arguments.
"""

import copy
import pickle

import pytest

from mixing.errors import (
    GainCappedWarning,
    MediaNotFoundError,
    ProbeError,
    WindowTooWideForClip,
)


def _too_wide():
    return WindowTooWideForClip(
        clip_index=3, clip_duration_s=12.5, window_s=10.0, hop_s=5.0, max_window_s=6.25
    )


@pytest.mark.parametrize("twin", [pickle.loads, copy.copy, copy.deepcopy])
def test_window_too_wide_rebuilds_with_its_numbers(twin):
    error = _too_wide()
    again = twin(pickle.dumps(error)) if twin is pickle.loads else twin(error)
    assert type(again) is WindowTooWideForClip and str(again) == str(error)
    fields = ("clip_index", "clip_duration_s", "window_s", "hop_s", "max_window_s")
    assert [getattr(again, f) for f in fields] == [getattr(error, f) for f in fields]
    assert isinstance(again, ValueError)


@pytest.mark.parametrize(
    "error",
    [
        ProbeError("cannot probe x"),
        MediaNotFoundError("no such media file: x"),
        GainCappedWarning("capped", subject="v", measured_lufs=-50.0, max_gain_db=6.0),
    ],
)
def test_the_other_classes_round_trip_too(error):
    again = pickle.loads(pickle.dumps(error))
    assert type(again) is type(error) and str(again) == str(error)
    assert vars(again) == vars(error)
