import numpy as np

from gesture_bc.teleop.one_euro import OneEuroFilter, smoothing_factor

RATE_HZ = 30.0  # a typical webcam


def run(filter_: OneEuroFilter, signal: np.ndarray) -> np.ndarray:
    return np.array([filter_(x, i / RATE_HZ) for i, x in enumerate(signal)])


def test_smoothing_factor_grows_with_the_cutoff():
    dt = 1 / RATE_HZ
    assert 0 < smoothing_factor(dt, 0.5) < smoothing_factor(dt, 5.0) < 1


def test_first_sample_passes_through_and_a_constant_stays_constant():
    out = run(OneEuroFilter(), np.full((20, 3), [0.5, 0.3, 0.1]))
    np.testing.assert_allclose(out, np.full((20, 3), [0.5, 0.3, 0.1]))


def test_jitter_of_a_still_hand_is_reduced():
    rng = np.random.default_rng(0)
    noisy = 0.5 + rng.normal(0.0, 0.01, size=(300, 1))
    out = run(OneEuroFilter(min_cutoff_hz=1.0, beta=0.0), noisy)
    assert out[30:].std() < 0.4 * noisy[30:].std()


def test_beta_reduces_the_lag_of_fast_movements():
    ramp = np.linspace(0.0, 1.0, 30)[:, None]  # moves one unit in one second
    slow = run(OneEuroFilter(min_cutoff_hz=1.0, beta=0.0), ramp)
    adaptive = run(OneEuroFilter(min_cutoff_hz=1.0, beta=5.0), ramp)
    lag_slow, lag_adaptive = ramp[-1, 0] - slow[-1, 0], ramp[-1, 0] - adaptive[-1, 0]
    assert 0 < lag_adaptive < 0.5 * lag_slow


def test_reset_forgets_the_history():
    filter_ = OneEuroFilter()
    run(filter_, np.zeros((10, 1)))
    filter_.reset()
    np.testing.assert_allclose(filter_(np.array([1.0]), 100.0), [1.0])


def test_repeated_timestamp_does_not_break_the_filter():
    filter_ = OneEuroFilter(beta=5.0)
    filter_(np.array([0.0]), 1.0)
    out = filter_(np.array([1.0]), 1.0)  # dt = 0
    assert np.all(np.isfinite(out))
