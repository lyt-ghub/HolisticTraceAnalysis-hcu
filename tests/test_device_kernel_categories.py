# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from hta.trace_analysis import TraceAnalysis


class DeviceKernelCategoryTest(unittest.TestCase):
    def _analyze(self, event_specs):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        events = [
            {
                "ph": "X",
                "cat": category,
                "name": name,
                "ts": start,
                "dur": duration,
                "pid": 123 if category == "cpu_op" else 0,
                "tid": 1,
                "args": {"stream": 0, "correlation": index},
            }
            for index, (category, name, start, duration) in enumerate(event_specs, 1)
        ]
        path = Path(directory.name) / "rank-0.json"
        path.write_text(
            json.dumps({"distributedInfo": {"rank": 0}, "traceEvents": events}, indent=2),
            encoding="utf-8",
        )
        return TraceAnalysis(trace_dir=str(path.parent))

    def test_excludes_cpu_events_with_a_device_stream_id(self):
        analysis = self._analyze(
            [
                ("Kernel", "triton_mm", 1, 10),
                ("cuda_sync", "Stream Wait Event", 13, 2),
                ("cpu_op", "triton_mm", 16, 10),
                ("kernel", "ncclAllReduce", 21, 10),
                ("Memcpy", "Memcpy HtoD", 31, 5),
            ]
        )
        _, kernels = analysis.get_gpu_kernel_breakdown(
            visualize=False, include_memory_kernels=True
        )
        self.assertEqual(
            kernels.set_index("name")["sum (us)"].to_dict(),
            {"triton_mm": 10, "ncclAllReduce": 10, "Memcpy HtoD": 5},
        )
        time = analysis.get_temporal_breakdown(visualize=False).iloc[0]
        self.assertEqual(time["compute_time(us)"], 10)
        self.assertEqual(time["non_compute_time(us)"], 17)
        overlap = analysis.get_comm_comp_overlap(visualize=False).iloc[0]
        self.assertEqual(overlap["comp_comm_overlap_pctg"], 0)

    def test_stream_tagged_cpu_event_without_device_activity(self):
        analysis = self._analyze([("cpu_op", "triton_mm", 1, 10)])
        _, kernels = analysis.get_gpu_kernel_breakdown(visualize=False)
        self.assertTrue(kernels.empty)
        time = analysis.get_temporal_breakdown(visualize=False).iloc[0]
        self.assertEqual(time["kernel_time(us)"], 0)
        self.assertTrue(pd.isna(time["compute_time_pctg"]))
        overlap = analysis.get_comm_comp_overlap(visualize=False).iloc[0]
        self.assertTrue(pd.isna(overlap["comp_comm_overlap_pctg"]))
