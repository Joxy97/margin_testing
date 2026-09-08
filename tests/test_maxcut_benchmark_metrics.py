import math
import unittest

from maxcut_solver_benchmark.metrics import (
    aggregate_rows,
    attempts_to_solution,
    cut_value,
    quality_metrics,
    time_to_solution,
)


class MaxCutBenchmarkMetricsTest(unittest.TestCase):
    def test_quality_uses_positive_maxcut_reference(self) -> None:
        quality = quality_metrics(9.0, 10.0)
        self.assertAlmostEqual(quality.approximationRatio, 0.9)
        self.assertAlmostEqual(quality.relativeGap, 0.1)
        self.assertFalse(quality.success)
        self.assertTrue(quality_metrics(10.0 - 1e-10, 10.0).success)

    def test_tts_is_measured_in_complete_solver_invocations(self) -> None:
        self.assertEqual(attempts_to_solution(1.0), 1)
        self.assertEqual(attempts_to_solution(0.5), 7)
        self.assertAlmostEqual(time_to_solution(0.5, 0.25), 1.75)
        self.assertTrue(math.isinf(time_to_solution(0.0, 0.25)))

    def test_aggregate_recomputes_success_and_uses_amortized_time(self) -> None:
        rows = [
            {"cut": 10, "reference_cut": 10, "amortized_seconds": 0.2, "success": False},
            {"cut": 9, "reference_cut": 10, "amortized_seconds": 0.4, "success": True},
        ]
        result = aggregate_rows(rows)
        self.assertEqual(result.trials, 2)
        self.assertEqual(result.successes, 1)
        self.assertAlmostEqual(result.meanAmortizedSeconds, 0.3)
        self.assertAlmostEqual(result.meanApproximationRatio, 0.95)
        self.assertAlmostEqual(result.timeToSolution99Seconds, 2.1)

    def test_aggregate_rejects_mixed_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "same reference_cut"):
            aggregate_rows(
                [
                    {"cut": 4, "reference_cut": 5, "amortized_seconds": 1},
                    {"cut": 4, "reference_cut": 6, "amortized_seconds": 1},
                ]
            )

    def test_cut_value_is_independent_of_qubo_sign_convention(self) -> None:
        edges = [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 7.0)]
        self.assertEqual(cut_value([0, 1, 1], edges), 9.0)
        with self.assertRaisesRegex(ValueError, "binary"):
            cut_value([0, 2, 1], edges)


if __name__ == "__main__":
    unittest.main()
