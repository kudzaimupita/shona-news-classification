import pytest

from shona_news.metrics import bootstrap, holm, macro_f1, mcnemar_exact, paired_bootstrap

GOLD = ["a", "b", "c", "d"] * 25


def test_mcnemar_is_one_for_identical_systems() -> None:
    assert mcnemar_exact(GOLD, GOLD, GOLD) == 1.0


def test_mcnemar_detects_a_clear_difference() -> None:
    worse = ["a"] * len(GOLD)
    assert mcnemar_exact(GOLD, GOLD, worse) < 1e-6


def test_mcnemar_exact_small_case() -> None:
    # 3 items only A gets right, 0 only B: p = 2 * 0.5**3
    gold, a, b = ["x", "x", "x"], ["x", "x", "x"], ["y", "y", "y"]
    assert mcnemar_exact(gold, a, b) == pytest.approx(0.25)


def test_holm_matches_hand_computation() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_and_paired_bootstrap_bracket_estimates() -> None:
    noisy = GOLD[:]
    noisy[::6] = ["a"] * len(noisy[::6])
    interval = bootstrap(GOLD, noisy, macro_f1, n_resamples=300)
    assert interval.low <= interval.estimate <= interval.high
    difference = paired_bootstrap(GOLD, GOLD, noisy, macro_f1, n_resamples=300)
    assert difference.low > 0
