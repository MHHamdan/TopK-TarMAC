import pytest

from scripts.analyse_timing_robustness import paired_summary


def test_fixed_baseline_and_oracle_are_distinct_estimands():
    raw = {"dense_a": [[1., 1., 10., 10.]],
           "dense_b": [[10., 10., 1., 1.]],
           "sparse": [[2., 2., 2., 2.]]}
    result = paired_summary(raw)
    assert result["fixed_baseline"] == "dense_a"
    ratio = result["ratios"]["sparse"]
    assert ratio["fixed_baseline_paired_median"] == pytest.approx(1.1)
    assert ratio["pointwise_best_dense_paired_median"] == 2.
    assert ratio["ratio_of_medians_fixed_baseline"] == pytest.approx(2 / 5.5)


def test_unpaired_lengths_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        paired_summary({"dense_a": [[1., 2.]], "sparse": [[1.]]})


def test_same_total_length_but_different_run_layout_rejected():
    with pytest.raises(ValueError, match="run layout"):
        paired_summary({"dense_a": [[1.], [2.]], "sparse": [[1., 2.]]})
