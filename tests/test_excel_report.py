# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree
from zipfile import ZipFile

from hta.configs.env_options import HTAEnvOptions
from hta.analyzers.training_analysis import read_training_log
from hta.excel_report import export_excel_report


_NS = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def _charts_by_title(archive):
    charts = {}
    for filename in archive.namelist():
        if filename.startswith("xl/charts/chart") and filename.endswith(".xml"):
            chart = ElementTree.fromstring(archive.read(filename))
            title = "".join(node.text for node in chart.findall(".//c:chart/c:title//a:t", _NS))
            charts[title] = chart
    return charts


@unittest.skipUnless(importlib.util.find_spec("xlsxwriter"), "requires the Excel extra")
class ExcelReportTest(unittest.TestCase):
    def test_training_throughput_preserves_units_and_actual_batch_size(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            megatron, bert, resnet = (root / name for name in ("megatron.log", "dllogger.json", "resnet.log"))
            megatron.write_text(
                " iteration 1/1 | consumed samples: 64 | elapsed time per iteration (ms): 200 | "
                "throughput per GPU (TFLOP/s/GPU): 2.5 | global batch size: 64 |\n"
            )
            bert.write_text(
                'DLLL {"step": [0,1], "data": {"training_step_sequences": 4, "training_step_seconds": 0.5}}\n'
                'DLLL {"step": [0,2], "data": {"training_step_sequences": 4, "training_step_seconds": 0.25}}\n'
                'DLLL {"step": [], "data": {"training_sequences_per_second": 8, "raw_train_time": 4}}\n'
            )
            resnet.write_text("Epoch: [0][0/2] Time 0.500 (0.500) Loss 2.0 (2.0)\n"
                              "Epoch: [0][1/2] Time 0.250 (0.375) Loss 1.5 (1.8)\n")
            (root / "capture.json").write_text('{"batch_size": 4, "train_images": 5}')
            for path, unit, rates in ((megatron, "TFLOP/s/HCU", [2.5]), (bert, "序列/秒", [8, 16]),
                                      (resnet, "图像/秒", [8, 4])):
                metrics = read_training_log(path)
                self.assertEqual(metrics.unit, unit)
                self.assertEqual([sample.rate for sample in metrics.throughput], rates)

    def test_training_phase_chart_follows_cpu_submissions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            output = root / "report.xlsx"
            events = [
                {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur,
                 "pid": pid, "tid": tid, "args": {"stream": stream, "correlation": correlation}}
                for cat, name, ts, dur, pid, tid, stream, correlation in (
                    ("user_annotation", "ProfilerStep#5", 100, 50, 1, 1, -1, -1),
                    ("user_annotation", "BERT.forward_and_loss", 100, 10, 1, 1, -1, -1),
                    ("user_annotation", "BERT.backward", 110, 10, 1, 1, -1, -1),
                    ("user_annotation", "BERT.optimizer", 120, 6, 1, 1, -1, -1),
                    ("cuda_runtime", "hipLaunchKernel", 104, 1, 1, 1, -1, 11),
                    ("cuda_runtime", "hipLaunchKernel", 112, 1, 1, 2, -1, 12),
                    ("kernel", "forward_kernel", 111, 4, 2, 0, 0, 11),
                    ("kernel", "backward_kernel", 121, 7, 2, 0, 0, 12),
                    ("gpu_user_annotation", "BERT.backward", 121, 1, 2, 0, 0, -1),
                )
            ]
            trace.write_text(json.dumps({"distributedInfo": {"rank": 0}, "traceEvents": events}))
            log = root / "dllogger.json"
            records = [
                {"step": [0, 1], "data": {"step_loss": 12.0}},
                {"step": [0, 3], "data": {"step_loss": 9.0}},
                {"step": [0, 4], "data": {"final_loss": 8.0}},
                {"step": [], "data": {"final_loss": 8.0, "training_sequences_per_second": 8}},
            ]
            log.write_text("\n".join(f"DLLL {json.dumps(record)}" for record in records))
            self.assertFalse(read_training_log(log).throughput)
            timing = root / "throughput.json"
            timing.write_text(
                'DLLL {"step": [0,1], "data": {"training_step_sequences": 4, "training_step_seconds": 0.5}}\n'
                'DLLL {"step": [0,2], "data": {"training_step_sequences": 4, "training_step_seconds": 0.125}}\n'
            )
            export_excel_report(trace, output, training_log=log, throughput_log=timing)
            with ZipFile(output) as archive:
                charts = _charts_by_title(archive)
                chart = charts["BERT / Rank 0：训练阶段耗时（ms）"]
                cpu, hcu = chart.findall(".//c:val/c:numRef", _NS)
                self.assertEqual(cpu.findtext("c:f", namespaces=_NS).replace("'", ""), "训练阶段明细!$G$2:$G$4")
                self.assertEqual(hcu.findtext("c:f", namespaces=_NS).replace("'", ""), "训练阶段明细!$H$2:$H$4")
                for series, expected in ((cpu, [0.010, 0.010, 0.006]), (hcu, [0.004, 0.007])):
                    values = [float(point.text) for point in series.findall("c:numCache/c:pt/c:v", _NS)]
                    self.assertEqual(values, expected)
                loss_chart = charts["训练损失变化"]
                throughput = charts["BERT 训练吞吐（序列/秒）"]
                self.assertIsNotNone(throughput.find(".//c:scatterChart", _NS))
                points = throughput.findall(".//c:yVal/c:numRef/c:numCache/c:pt/c:v", _NS)
                self.assertEqual([float(point.text) for point in points], [8, 32])
                for axis, expected in (("xVal", [[1, 3, 4], [1, 3, 4]]), ("yVal", [[12, 9], [8]])):
                    series = loss_chart.findall(f".//c:{axis}/c:numRef/c:numCache", _NS)
                    values = [[float(point.text) for point in item.findall("c:pt/c:v", _NS)] for item in series]
                    self.assertEqual(values, expected)

    def test_inference_charts_use_cpu_phase_markers(self):
        def write_trace(path, annotations):
            events = [
                {"ph": "X", "cat": category, "name": name, "ts": 100 + index * 25,
                 "dur": duration, "pid": 1, "tid": 1, "args": {"stream": -1}}
                for index, (category, name, duration) in enumerate(annotations)
            ]
            events.append({
                "ph": "X", "cat": "kernel", "name": "matmul", "ts": 102, "dur": 5,
                "pid": 2, "tid": 0, "args": {"stream": 0},
            })
            path.write_text(json.dumps({"distributedInfo": {"rank": 0}, "traceEvents": events}))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vllm, sglang, output = (root / name for name in ("vllm.json", "sglang.json", "report.xlsx"))
            decode = "execute_context_0(0)_generation_1(1)"
            write_trace(vllm, [
                ("user_annotation", "execute_context_1(14)_generation_0(0)", 20),
                ("user_annotation", decode, 8),
                ("gpu_user_annotation", decode, 5),
                ("user_annotation", "execute_context_0(0)_generation_0(0)", 5),
            ])
            write_trace(sglang, [
                ("user_annotation", "step[EXTEND bs=1 toks=14]", 19),
                ("user_annotation", "step[DECODE bs=1]", 9),
            ])
            export_excel_report(vllm, output, label="vLLM", compare=sglang, compare_label="SGLang")
            with ZipFile(output) as archive:
                charts = _charts_by_title(archive)
                labels = archive.read("xl/sharedStrings.xml").decode("utf-8")
                self.assertIn("按开始时间排列的第 N 步", labels)
                self.assertIn("步骤 1", labels)
                self.assertNotIn("标记 1", labels)
                self.assertIn("一步可以处理一个或多个 token", labels)
                self.assertIn("阶段内步骤序号", labels)
                for name, phase, expected in (
                    ("vLLM", "Prefill", 0.02), ("vLLM", "Decode", 0.008),
                    ("SGLang", "Prefill", 0.019), ("SGLang", "Decode", 0.009),
                ):
                    series = charts[f"{name} / Rank 0：{phase} 步骤用时（ms）"]
                    values = series.findall(".//c:val/c:numRef/c:numCache/c:pt/c:v", _NS)
                    self.assertEqual([float(value.text) for value in values], [expected])
                self.assertNotIn("各训练步的用时分布（ms）", charts)

    def test_step_chart_uses_host_boundaries_and_clips_activity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            output = root / "report.xlsx"
            events = [
                {
                    "ph": "X", "cat": category, "name": name, "ts": start, "dur": duration,
                    "pid": 1 if stream < 0 else 2, "tid": stream, "args": {"stream": stream},
                }
                for category, name, start, duration, stream in (
                    ("user_annotation", "ProfilerStep#2", 100, 10, -1),
                    ("user_annotation", "ProfilerStep#3", 110, 10, -1),
                    ("gpu_user_annotation", "ProfilerStep#2", 104, 6, -1),
                    ("kernel", "matmul", 104, 10, 0),
                    ("kernel", "ncclAllReduce", 108, 8, 4),
                )
            ]
            trace.write_text(
                json.dumps({"distributedInfo": {"rank": 0}, "traceEvents": events}),
                encoding="utf-8",
            )
            export_excel_report(trace, output)
            with ZipFile(output) as archive:
                labels = archive.read("xl/sharedStrings.xml").decode("utf-8")
                self.assertIn("采样训练步  ProfilerStep#2、#3", labels)
                self.assertIn("跨步活动按落在各步里的实际时长计入", labels)
                chart = _charts_by_title(archive)["各训练步的用时分布（ms）"]
                numeric = chart.findall(".//c:val/c:numRef", _NS)
                for values, column, expected in zip(
                    numeric, "EFG", ([0.006, 0.004], [0, 0.002], [0.004, 0.004]), strict=True,
                ):
                    self.assertEqual(
                        values.findtext("c:f", namespaces=_NS),
                        f"'HCU 按训练步'!${column}$2:${column}$3",
                    )
                    self.assertEqual(
                        [float(point.findtext("c:v", namespaces=_NS))
                         for point in values.findall("c:numCache/c:pt", _NS)],
                        expected,
                    )

    def test_charts_include_only_measured_overlap_and_use_workbook_cells(self):
        def trace(path, events):
            path.write_text(
                json.dumps(
                    {
                        "distributedInfo": {"rank": 0},
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": name,
                                "ts": start,
                                "dur": duration,
                                "pid": 1,
                                "tid": stream,
                                "args": {"stream": stream},
                            }
                            for name, start, duration, stream in events
                        ],
                    }
                ),
                encoding="utf-8",
            )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            main, solo, output = (root / name for name in ("main.json", "solo.json", "report.xlsx"))
            trace(main, [
                ("matmul", 15.25, 10.5, 0), ("ncclAllReduce", 10.25, 10, 4),
                ("matmul", 30.25, 1, 0),
            ])
            trace(solo, [("matmul", 10.25, 7.5, 0)])
            rounding = HTAEnvOptions.instance().disable_ns_rounding()
            export_excel_report(
                main, output, label="默认分桶", bucket_size=40_000_000,
                compare=solo, compare_label="小分桶", compare_bucket_size=65_536,
            )
            self.assertEqual(HTAEnvOptions.instance().disable_ns_rounding(), rounding)
            with ZipFile(output) as archive:
                charts = _charts_by_title(archive)
                self.assertEqual(len(charts), 15)
                dashboard = ElementTree.fromstring(archive.read("xl/drawings/drawing1.xml"))
                self.assertEqual(len(dashboard.findall(".//c:chart", _NS)), len(charts))
                self.assertNotIn("Kernel 诊断", archive.read("xl/workbook.xml").decode("utf-8"))
                rank_label = "默认分桶 · 40,000,000 个梯度元素 / Rank 0"
                self.assertIn("小分桶 · 65,536 个梯度元素 / Rank 0：Kernel 累计耗时（ms）", charts)
                chart = charts["通信与计算重叠率（%）"]
                values = chart.find(".//c:val/c:numRef", _NS)
                self.assertEqual(values.findtext("c:f", namespaces=_NS), "通信重叠!$G$2:$G$3")
                self.assertEqual(
                    [point.findtext("c:v", namespaces=_NS)
                     for point in values.findall("c:numCache/c:pt", _NS)],
                    ["0.5"],
                )
                for chart_title, formula in [
                    ("通信时长（ms）", "通信重叠!$E$2:$E$3"),
                    ("HCU 分析窗口内的时间占比（%）", "'HCU 时间分布'!$H$2:$H$3"),
                    ("HCU 活动分类占比（%）", "'HCU 活动分类'!$L$2:$L$3"),
                    ("HCU 各流的空档原因占比（%）", "'HCU 流空档'!$I$2:$I$4"),
                    (f"{rank_label}：Kernel 调用次数（次）", "'Kernel 热点'!$F$2:$F$3"),
                    (f"{rank_label}：Kernel 单次耗时（µs）", "'Kernel 热点'!$H$2:$H$3"),
                ]:
                    chart = charts[chart_title]
                    self.assertIn(formula, [f.text for f in chart.findall(".//c:val/c:numRef/c:f", _NS)])
                allreduce = charts["AllReduce 次数（次）"].find(".//c:val/c:numRef", _NS)
                self.assertEqual(allreduce.findtext("c:f", namespaces=_NS), "通信重叠!$I$2:$I$3")
                self.assertEqual(
                    [point.findtext("c:v", namespaces=_NS)
                     for point in allreduce.findall("c:numCache/c:pt", _NS)],
                    ["1"],
                )
                latency = charts[f"{rank_label}：Kernel 单次耗时（µs）"]
                self.assertEqual(latency.find(".//c:grouping", _NS).get("val"), "clustered")
            export_excel_report(solo, output)
            with ZipFile(output) as archive:
                charts = _charts_by_title(archive)
                self.assertEqual(len(charts), 7)
                for title in (
                    "Kernel 累计耗时（ms）", "Kernel 调用次数（次）", "Kernel 单次耗时（µs）",
                ):
                    self.assertIn(f"本次采样里 Rank 0：{title}", charts)
                labels = archive.read("xl/sharedStrings.xml").decode("utf-8")
                self.assertIn("本次采样", labels)
                self.assertIn("本次采样里 Rank 0", labels)
                self.assertNotIn("样本 1", labels)
                self.assertIn("从第一项 HCU 活动开始算到最后一项活动结束", labels)
                self.assertIn("如单独发生的通信或内存操作", labels)
                self.assertNotIn("AllReduce 次数（次）", charts)
                self.assertNotIn("HCU 各流的空档原因占比（%）", charts)
