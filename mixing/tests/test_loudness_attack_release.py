"""``mixing.loudness.attack_release``: its closed form per run of equal targets
is the per-step loop, so one follower serves frames and per-sample gates alike
(reeleehq/foley#306)."""

import numpy as np
import pytest

from mixing.loudness import (

    _attack_release_loop,
    _one_pole_coefficient,
    attack_release,
)


def _looped(t, *, step_s, attack_s, release_s, initial):
    a, r = _one_pole_coefficient(step_s, attack_s), _one_pole_coefficient(step_s, release_s)
    return _attack_release_loop(np.asarray(t, dtype=np.float64), a, r, initial)


@pytest.mark.parametrize("seed", range(8))
def test_the_closed_form_on_gates_is_the_loop(seed):
    rng = np.random.default_rng(seed)
    runs = int(rng.integers(1, 20))
    t = np.repeat(rng.choice([1.0, 0.3, 0.0, -12.0], size=runs), rng.integers(1, 4000, size=runs))
    kw = dict(
        step_s=1 / 48000,
        attack_s=float(rng.uniform(0.0, 0.2)),
        release_s=float(rng.uniform(0.0, 0.5)),
        initial=float(rng.choice([1.0, 0.0])),
    )
    assert np.max(np.abs(attack_release(t, **kw) - _looped(t, **kw))) < 1e-9


def test_a_target_that_changes_every_step_takes_the_loop_and_agrees():
    t = np.random.default_rng(0).normal(size=500)
    kw = dict(step_s=0.01, attack_s=0.05, release_s=0.2, initial=0.0)
    np.testing.assert_allclose(attack_release(t, **kw), _looped(t, **kw), atol=1e-12)


def test_edges_empty_and_instant():
    assert attack_release(np.array([]), step_s=0.01, attack_s=0.1, release_s=0.1).shape == (0,)
    gate = np.repeat([1.0, 0.0], 50)
    np.testing.assert_array_equal(
        attack_release(gate, step_s=0.01, attack_s=0.0, release_s=0.0, initial=0.5), gate
    )
