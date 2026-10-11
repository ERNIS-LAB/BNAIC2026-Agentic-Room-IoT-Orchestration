from collections import Counter

from benchmark import _check_answer, _compute_metrics
from benchmark_cases import TEST_CASES


def test_benchmark_suite_retains_unique_cases_and_expected_composition():
    assert len({case["id"] for case in TEST_CASES}) == 300
    assert Counter(case["category"] for case in TEST_CASES) == {
        "single_sensor": 126,
        "multi_sensor": 84,
        "out_of_scope": 45,
        "adversarial": 45,
    }


def test_answer_matching_accepts_all_required_facts():
    assert _check_answer("31.5 C, 450 lux", ["31.5", "450"])
    assert not _check_answer("31.5 C", ["31.5", "450"])


def test_empty_metrics_are_defined():
    metrics = _compute_metrics([])
    assert metrics["n_tasks"] == 0
    assert metrics["success_rate"] == 0
