# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd
from hta.analyzers.breakdown_analysis import BreakdownAnalysis


class KernelDurationAggregationTest(unittest.TestCase):
    def test_kernel_type_totals_preserve_submicrosecond_intervals(self):
        events = pd.DataFrame(
            {
                "ts": [0.0, 0.75, 1.5, 1.75, 2.0, 2.5],
                "dur": [0.5, 0.5, 0.75, 0.5, 0.5, 0.25],
                "kernel_type": [
                    "COMPUTATION",
                    "COMPUTATION",
                    "COMPUTATION",
                    "COMPUTATION",
                    "COMMUNICATION",
                    "MEMORY",
                ],
            }
        )

        totals = BreakdownAnalysis._get_gpu_kernel_type_time(
            events, ["COMPUTATION", "COMMUNICATION", "MEMORY"]
        ).set_index("kernel_type")["sum"]
        self.assertEqual(
            totals.to_dict(),
            {
                "COMPUTATION": 1.5,
                "COMPUTATION overlapping COMMUNICATION": 0.25,
                "COMMUNICATION": 0.25,
                "MEMORY": 0.25,
            },
        )

    def test_grouped_statistics_describe_individual_invocations(self):
        events = pd.DataFrame(
            {
                "name": ["gemm", "gemm", "copy", "copy", "reduce", "norm"],
                "dur": [30, 50, 4, 8, 6, 3],
            }
        )

        actual = BreakdownAnalysis._aggr_gpu_kernel_time(
            events, num_kernels=1, duration_ratio=1.0, allowlist_names=["norm"]
        ).set_index("name")
        expected_durations = {"gemm": [30, 50], "norm": [3], "others": [4, 8, 6]}
        for name, durations in expected_durations.items():
            with self.subTest(name=name):
                expected = pd.Series(durations).agg(["sum", "max", "min", "mean", "std"])
                pd.testing.assert_series_equal(
                    actual.loc[name],
                    expected.fillna(0),
                    check_names=False,
                )
