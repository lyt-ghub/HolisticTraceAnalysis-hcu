# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd
from hta.analyzers.breakdown_analysis import BreakdownAnalysis


class KernelDurationAggregationTest(unittest.TestCase):
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
