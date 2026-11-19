# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from hta.megatron_profile import _profile_rows


class MegatronProfileTest(unittest.TestCase):
    def test_retains_unlinked_kernels_and_distinct_input_shapes(self):
        def activity(name, cat, start, duration, args=None):
            return {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": 1,
                "tid": 1,
                "ts": start,
                "dur": duration,
                "args": args or {},
            }

        trace = {"traceEvents": [
            activity("parent", "user_annotation", 10, 20),
            activity("matmul", "cpu_op", 11, 4, {"Input Dims": [[2]], "External id": 1}),
            activity("matmul", "cpu_op", 16, 5, {"Input Dims": [[3]], "External id": 2}),
            activity("hipLaunchKernel", "cuda_runtime", 17, 1,
                     {"External id": 2, "correlation": 9}),
            activity("gemm", "kernel", 22, 8, {"External id": 1}),
            activity("gemm", "kernel", 32, 7, {"correlation": 9}),
            activity("unlinked kernel", "kernel", 41, 3),
            activity("parent", "gpu_user_annotation", 42, 3),
        ]}
        rows = {(row[0], row[11]): row for row in _profile_rows(trace)}
        self.assertEqual(rows["parent", ""][2:10], (11, 1.0, 20, 20, 0, 0, 15, 15))
        self.assertEqual(rows["matmul", "[[2]]"][6:11], (8, 8/18, 8, 8, 1))
        self.assertEqual(rows["matmul", "[[3]]"][6:11], (7, 7/18, 7, 7, 1))
        self.assertEqual(rows["gemm", ""][6:11], (15, 15/18, 15, 7.5, 2))
        self.assertEqual(rows["unlinked kernel", ""][6:11], (3, 3/18, 3, 3, 1))
