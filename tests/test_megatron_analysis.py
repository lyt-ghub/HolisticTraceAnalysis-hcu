# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hta.megatron_analysis import export_megatron_analysis


@unittest.skipUnless(importlib.util.find_spec("xlsxwriter"), "requires the Excel extra")
class MegatronAnalysisTest(unittest.TestCase):
    def test_report_names_and_index(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "rank0.json"
            events = [
                {
                    "ph": "X", "cat": "kernel", "name": name, "ts": start,
                    "dur": 10, "pid": 1, "tid": stream, "args": {"stream": stream},
                }
                for name, start, stream in [("matmul", 10, 0), ("ncclAllReduce", 15, 4)]
            ]
            trace.write_text(
                json.dumps({"distributedInfo": {"rank": 0}, "traceEvents": events}),
                encoding="utf-8",
            )
            report_dir = root / "reports"
            export_megatron_analysis(trace, report_dir)
            expected = {
                "hcu_time_breakdown_by_rank.xlsx",
                "hcu_activity_time_by_type.xlsx",
                "hcu_kernel_hotspots_by_rank.xlsx",
                "hcu_stream_idle_time_by_cause.xlsx",
                "hcu_stream_idle_gap_statistics.csv",
                "communication_compute_overlap_by_rank.csv",
                "trace_capture_and_analysis_summary.json",
                "megatron_analysis_guide.md",
            }
            self.assertEqual({file.name for file in report_dir.iterdir()}, expected)

            export_megatron_analysis(trace, report_dir, compare=trace)
            expected.update({
                "gradient_bucket_overlap_comparison.xlsx",
                "gradient_bucket_overlap_comparison.csv",
            })
            self.assertEqual({file.name for file in report_dir.iterdir()}, expected)
            summary_file = "trace_capture_and_analysis_summary.json"
            guide_file = "megatron_analysis_guide.md"
            summary = json.loads((report_dir / summary_file).read_text(encoding="utf-8"))
            self.assertEqual(
                {entry["file"] for entry in summary["artifacts"].values()},
                expected - {summary_file, guide_file},
            )
            guide = (report_dir / guide_file).read_text(encoding="utf-8")
            for filename in expected - {guide_file}:
                self.assertIn(f"]({filename})", guide)
