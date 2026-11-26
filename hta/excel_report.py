# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export HTA time breakdowns and editable Excel charts from profiler traces."""

import argparse
from itertools import groupby
import math
from pathlib import Path

from hta.analyzers.inference_analysis import get_inference_phase_spans
from hta.analyzers.training_analysis import get_training_phase_spans, read_training_log
from hta.common.atomic_file import atomic_binary_output
from hta.common.trace_file import create_rank_to_trace_dict
from hta.configs.env_options import HTAEnvOptions
from hta.trace_analysis import TraceAnalysis
from hta.utils.utils import is_allreduce_kernel


_TOP_KERNELS = 8
_RANKS_PER_CHART = 12
_PHASE_STEPS_PER_CHART = 16
_MAX_THROUGHPUT_LABELS = 12
_DEFAULT_SAMPLE_LABEL = "样本 1"
_COMPUTE = "#277A73"
_COMM = "#5668BD"
_NEUTRAL = "#A2AFBF"
_AMBER = "#BF8238"
_TIME_TYPES = [("计算", _COMPUTE), ("其他活动", _COMM), ("未记录到活动", _NEUTRAL)]
_ACTIVITY_TYPES = [
    ("COMPUTATION", "仅计算", _COMPUTE),
    ("COMMUNICATION", "仅通信", _COMM),
    ("MEMORY", "仅内存操作", _AMBER),
    ("COMPUTATION overlapping COMMUNICATION", "计算与通信", "#544572"),
    ("COMPUTATION overlapping MEMORY", "计算与内存操作", "#80A65F"),
    ("COMMUNICATION overlapping MEMORY", "通信与内存操作", "#BF6985"),
    ("COMPUTATION overlapping COMMUNICATION overlapping MEMORY", "三者同时发生", "#635A4D"),
]
_IDLE_TYPES = [
    ("host_wait", "等 CPU 提交", _COMM),
    ("kernel_wait", "任务间短空档", _AMBER),
    ("other", "其他空档", _NEUTRAL),
]
_TIME_COLUMNS = [
    "idle_time(us)",
    "compute_time(us)",
    "non_compute_time(us)",
    "kernel_time(us)",
]


def _load_trace(path: Path) -> TraceAnalysis:
    if path.is_dir():
        analysis = TraceAnalysis(trace_dir=str(path), include_last_profiler_step=True)
    elif path.is_file():
        if not path.name.endswith((".json", ".json.gz")):
            raise ValueError(f"Expected a .json or .json.gz trace: {path}")
        _, files = create_rank_to_trace_dict([str(path)])
        analysis = TraceAnalysis(
            trace_dir=str(path.parent),
            trace_files=files,
            include_last_profiler_step=True,
        )
    else:
        raise ValueError(f"Trace does not exist: {path}")
    if not analysis.t.get_ranks():
        raise ValueError(f"No analyzable trace events in {path}")
    return analysis


def _write_table(sheet, headers, rows, widths, header, formats):
    if len(rows) >= 1_048_576:
        raise ValueError(f"Too many rows for Excel sheet {sheet.name}")
    sheet.hide_gridlines(2)
    sheet.freeze_panes(1, 1)
    sheet.set_row(0, 24)
    sheet.write_row(0, 0, headers, header)
    for col, width in enumerate(widths):
        sheet.set_column(col, col, width, formats.get(col))
    for idx, row in enumerate(rows, 1):
        sheet.write_row(idx, 0, row)
    if rows:
        sheet.autofilter(0, 0, len(rows), len(headers) - 1)


def _add_chart(
    book, target, anchor, sheet, start, end, series, title, unit,
    *, category_col=2, signed=False, stacked=True,
):
    chart = book.add_chart(
        {"type": "bar", **({"subtype": "stacked"} if stacked and len(series) > 1 else {})}
    )
    for col, name, color in series:
        options = {
            "name": name,
            "categories": [sheet, start, category_col, end, category_col],
            "values": [sheet, start, col, end, col],
            "fill": {"color": color},
            "border": {"none": True},
        }
        if unit == "%" and len(series) == 1:
            options["data_labels"] = {"value": True, "num_format": "0.00%"}
        elif unit == "次":
            options["data_labels"] = {"value": True, "num_format": "0"}
        chart.add_series(options)
    chart.set_title({"name": f"{title}（{unit}）", "name_font": {"size": 13}})
    axis = {
        "num_format": "General",
        "num_font": {"size": 10},
        "label_position": "low",
    }
    if not signed:
        axis["min"] = 0
    if unit == "%":
        axis.update(max=1, num_format="0%")
    chart.set_x_axis(axis)
    chart.set_y_axis({"reverse": True, "crossing": "max", "num_font": {"size": 10}})
    chart.set_legend({"position": "top"} if len(series) > 1 else {"none": True})
    height = 144 + (32 if stacked else 42) * (end - start + 1)
    chart.set_size({"width": 830, "height": height})
    target.insert_chart(anchor, 0, chart)
    return anchor + math.ceil(height / 20) + 2


def _add_training_loss_chart(book, overview, anchor, rows, header, numeric, caption):
    sheet = book.add_worksheet("训练损失明细")
    _write_table(
        sheet, ["训练步", "Epoch", "当前步训练损失", "训练结束损失"],
        rows, [14, 13, 22, 22], header, {2: numeric, 3: numeric},
    )
    chart = book.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    for column, name, color in ((2, "当前步训练损失", _COMPUTE), (3, "训练结束损失", _AMBER)):
        if any(row[column] is not None for row in rows):
            chart.add_series({
                "name": name,
                "categories": [sheet.name, 1, 0, len(rows), 0],
                "values": [sheet.name, 1, column, len(rows), column],
                "line": {"color": color} if column == 2 else {"none": True},
                "marker": {"type": "circle" if column == 2 else "diamond", "size": 6,
                           "fill": {"color": color}, "border": {"none": True}},
            })
    chart.set_title({"name": "训练损失变化", "name_font": {"size": 13}})
    step_interval = max(1, math.ceil((rows[-1].step - rows[0].step) / 10))
    chart.set_x_axis({"name": "训练步", "num_format": "0", "major_unit": step_interval})
    chart.set_y_axis({"name": "损失值（Loss）"})
    chart.set_legend({"position": "top"})
    height = 276
    chart.set_size({"width": 830, "height": height})
    overview.insert_chart(anchor, 0, chart)
    caption_row = anchor + math.ceil(height / 20)
    overview.write(caption_row, 0, "数据来自训练日志，包含日志中已记录的预热步。", caption)
    if any(row[3] is not None for row in rows):
        overview.write(caption_row + 1, 0, "训练结束损失按日志的 final_loss 单独标记。", caption)
    return caption_row + 3


def _add_training_throughput(book, overview, anchor, metrics, header, numeric, caption):
    rows = [
        [metrics.model, sample.step, f"训练步 {sample.step}",
         metrics.unit, sample.rate, sample.elapsed_ms, sample.items_per_step]
        for sample in metrics.throughput
    ]
    headers = ["模型", "训练步", "统计范围", "吞吐单位", "训练吞吐", "计时窗口 (ms)", "每步处理样本数"]
    widths = [18, 12, 18, 21, 17, 21, 20]
    number_formats = {4: numeric, 5: numeric, 6: numeric}
    if metrics.model == "Megatron" and metrics.unit == "TFLOP/s/HCU":
        headers.append("全局样本/秒")
        widths.append(18)
        number_formats[7] = numeric
        for row, sample in zip(rows, metrics.throughput):
            row.append(1000 * sample.items_per_step / sample.elapsed_ms)
    sheet = book.add_worksheet("训练吞吐明细")
    _write_table(sheet, headers, rows, widths, header, number_formats)
    chart = book.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    point_labels = {"value": True, "num_format": "0.0" if metrics.unit == "TFLOP/s/HCU" else "0.00", "position": "above"}
    chart.add_series({
        "name": "训练吞吐",
        "categories": [sheet.name, 1, 1, len(rows), 1],
        "values": [sheet.name, 1, 4, len(rows), 4],
        "line": {"color": _COMPUTE, "width": 2},
        "marker": {"type": "circle", "size": 6, "fill": {"color": _COMPUTE}, "border": {"none": True}},
        **({"data_labels": point_labels} if len(rows) <= _MAX_THROUGHPUT_LABELS else {}),
    })
    chart.set_title({"name": f"{metrics.model} 训练吞吐（{metrics.unit}）", "name_font": {"size": 13}})
    step_interval = max(1, math.ceil((rows[-1][1] - rows[0][1]) / 10))
    chart.set_x_axis({"name": "训练步", "num_format": "0", "major_unit": step_interval})
    chart.set_y_axis({"name": metrics.unit, "min": 0})
    chart.set_legend({"none": True})
    height = 276
    chart.set_size({"width": 830, "height": height})
    overview.insert_chart(anchor, 0, chart)
    caption_row = anchor + math.ceil(height / 20)
    overview.write(caption_row, 0, metrics.rate_source, caption)
    return caption_row + 2


def _shares(values, total):
    # A zero window or a signed idle breakdown has no meaningful composition.
    if total <= 0 or any(value < 0 for value in values):
        return [None] * len(values)
    return [value / total for value in values]


def _bucket_label(label: str, bucket_size: int | None) -> str:
    if bucket_size is None:
        return label
    if bucket_size <= 0:
        raise ValueError("Bucket size must be positive")
    return f"{label} · {bucket_size:,} 个梯度元素"


def _collect_steps(analysis: TraceAnalysis, label: str, *, single_capture_title: bool):
    steps = analysis.get_profiler_step_breakdown()
    rows = []
    for step in steps.itertuples(index=False):
        name = f"ProfilerStep#{step.profiler_step} / Rank {step.rank}"
        chart_label = name if single_capture_title else f"{label} / {name}"
        rows.append([
            label, step.rank, chart_label, step.profiler_step,
            step.compute_time_us / 1000, step.other_activity_us / 1000,
            step.no_activity_us / 1000, step.step_time_us / 1000, step.start_us / 1000,
        ])
    return rows


def _collect(analysis: TraceAnalysis, label: str, *, single_capture_title: bool):
    times = analysis.get_temporal_breakdown(visualize=False).set_index("rank")
    overlap = analysis.get_comm_comp_overlap(
        visualize=False, include_times=True
    ).set_index("rank")
    symbols = analysis.t.symbol_table
    names = symbols.get_sym_table()
    # Request every name so the ranked chart never treats "others" as a kernel.
    activity, kernels = analysis.get_gpu_kernel_breakdown(
        visualize=False, num_kernels=len(names)
    )
    activity_us = activity.set_index("kernel_type")["sum"]
    activity_values = [activity_us.get(key, 0) / 1000 for key, _, _ in _ACTIVITY_TYPES]
    activity_row = [label, *activity_values, *_shares(activity_values, sum(activity_values))]
    time_rows, overlap_rows, kernel_rows, hotspot_rows, hotspots = [], [], [], [], []
    idle_rows = []
    for rank in sorted(analysis.t.get_ranks()):
        rank_label = f"{label}里 Rank {rank}" if single_capture_title else f"{label} / Rank {rank}"
        idle, compute, other, span = times.loc[rank, _TIME_COLUMNS]
        time_rows.append([
            label, rank, rank_label, compute / 1000, other / 1000, idle / 1000, span / 1000,
            *_shares((compute, other, idle), span),
        ])
        frame = analysis.t.get_trace(rank)
        counts = frame.loc[symbols.get_gpu_kernel_mask(frame), "name"].value_counts()
        comm, together = overlap.loc[
            rank, ["communication_time_us", "overlap_time_us"]
        ]
        allreduce = sum(int(count) for key, count in counts.items() if is_allreduce_kernel(names[key]))
        overlap_rows.append(
            [
                label, rank, rank_label, comm / 1000, together / 1000,
                (comm - together) / 1000, together / comm if comm else None,
                "已采到通信" if comm else "未采到可计时的通信",
                allreduce if allreduce or comm else None,
            ]
        )
        if not counts.empty:
            gaps, _ = analysis.get_idle_time_breakdown(ranks=[rank], visualize=False)
            streams = gaps.pivot(index="stream", columns="idle_category", values="idle_time")
            for stream, gaps_by_type in streams.fillna(0).sort_index().iterrows():
                values = [gaps_by_type.get(key, 0) / 1000 for key, _, _ in _IDLE_TYPES]
                idle_rows.append([
                    label, rank, f"{rank_label} / 流 {stream}", stream,
                    *values, *_shares(values, sum(values)),
                ])
        rank_kernels = kernels[kernels["rank"].eq(rank)].sort_values(
            ["sum (us)", "name"], ascending=[False, True]
        )
        start = len(hotspot_rows) + 1
        for row in rank_kernels.to_dict("records"):
            name = row["name"]
            category = row["kernel_type"]
            count = int(counts[symbols.sym_index[name]])
            total, mean, maximum = (row[c] for c in ("sum (us)", "mean (us)", "max (us)"))
            kernel_rows.append([
                label, rank, name, category, count, total, mean, row["min (us)"], maximum,
                row["stddev"],
            ])
            number = len(hotspot_rows) + 1 - start
            if number < _TOP_KERNELS:
                short = name if len(name) <= 38 else f"{name[:35]}…"
                hotspot_rows.append(
                    [label, rank, f"{number + 1}. {short}", total / 1000,
                     name, count, mean, maximum]
                )
        if len(hotspot_rows) >= start:
            hotspots.append((rank_label, start, len(hotspot_rows)))
    return time_rows, overlap_rows, kernel_rows, hotspot_rows, hotspots, activity_row, idle_rows


def export_excel_report(
    trace: Path,
    output: Path,
    *,
    label: str = _DEFAULT_SAMPLE_LABEL,
    compare: Path | None = None,
    compare_label: str = "样本 2",
    bucket_size: int | None = None,
    compare_bucket_size: int | None = None,
    training_log: Path | None = None,
    throughput_log: Path | None = None,
) -> Path:
    """Analyze one or two captures and create a workbook with native charts."""
    import xlsxwriter

    trace, output = Path(trace).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".xlsx":
        raise ValueError("Choose an .xlsx output path")
    if compare_bucket_size is not None and compare is None:
        raise ValueError("Specify a comparison capture with --compare-bucket-size")
    label = _bucket_label(label, bucket_size)
    single_capture_title = compare is None and label == _DEFAULT_SAMPLE_LABEL
    if single_capture_title:
        label = "本次采样"
    inputs = [(label, trace)]
    if compare is not None:
        compare_label = _bucket_label(compare_label, compare_bucket_size)
        if label == compare_label:
            raise ValueError("Use distinct labels for the two captures")
        inputs.append((compare_label, Path(compare).resolve()))
    time_rows, overlap_rows, kernel_rows, hotspot_rows, hotspots = [], [], [], [], []
    activity_rows, idle_rows, step_rows = [], [], []
    phase_rows = []
    training_phase_rows = []
    metrics = read_training_log(Path(training_log)) if training_log is not None else None
    loss_rows = metrics.losses if metrics is not None else []
    if throughput_log is not None:
        benchmark = read_training_log(Path(throughput_log))
        if not benchmark.throughput:
            raise ValueError(f"No per-step throughput in {throughput_log}")
        if metrics is not None and benchmark.model != metrics.model:
            raise ValueError("The throughput log and training log must come from the same framework")
        metrics = benchmark
    steps_by_capture = {}
    options = HTAEnvOptions.instance()
    old_rounding = options.disable_ns_rounding()
    options.set_disable_ns_rounding(True)
    try:
        for name, path in inputs:
            analysis = _load_trace(path)
            times, overlaps, kernels, top, charts, activity, idle = _collect(
                analysis, name, single_capture_title=single_capture_title,
            )
            steps = _collect_steps(analysis, name, single_capture_title=single_capture_title)
            phases = get_inference_phase_spans(analysis.t)
            for phase in phases.itertuples(index=False):
                phase_rows.append([
                    name, phase.rank, f"步骤 {phase.phase_index}", phase.framework, phase.phase,
                    phase.phase_index, phase.duration_us / 1000, phase.start_us / 1000,
                    phase.annotation,
                ])
            training_phases = get_training_phase_spans(analysis.t)
            for phase in training_phases.itertuples(index=False):
                kernel_count = phase.kernel_count
                if kernel_count is not None and math.isnan(kernel_count):
                    kernel_count = None
                training_phase_rows.append([
                    name, phase.rank, " · ".join(filter(None, [phase.step, phase.phase])),
                    phase.model, phase.step, phase.phase, phase.cpu_us / 1000,
                    phase.kernel_us / 1000 if kernel_count is not None else None,
                    kernel_count, phase.start_us / 1000, phase.annotation,
                ])
            offset = len(hotspot_rows)
            hotspots.extend((key, a + offset, b + offset) for key, a, b in charts)
            time_rows.extend(times)
            overlap_rows.extend(overlaps)
            kernel_rows.extend(kernels)
            hotspot_rows.extend(top)
            activity_rows.append(activity)
            idle_rows.extend(idle)
            step_rows.extend(steps)
            steps_by_capture[name] = sorted({row[3] for row in steps})
    finally:
        options.set_disable_ns_rounding(old_rounding)

    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_binary_output(output) as temporary:
        with xlsxwriter.Workbook(
            temporary, {"strings_to_formulas": False, "strings_to_urls": False}
        ) as book:
            header = book.add_format(
                {"bold": True, "bg_color": "#203C55", "font_color": "#FFFFFF"}
            )
            title = book.add_format({"bold": True, "font_size": 16})
            caption = book.add_format({"font_color": "#43536A"})
            numeric = book.add_format({"num_format": "#,##0.000"})
            pct = book.add_format({"num_format": "0.00%"})
            overview = book.add_worksheet("图表")
            overview.hide_gridlines(2)
            overview.write("A1", "HTA 活动分析", title)
            overview.set_row(0, 24)
            overview_row = 1
            for name, steps in steps_by_capture.items() if step_rows else ():
                prefix = f"{name}  " if len(inputs) > 1 else ""
                if steps:
                    markers = "、".join(f"#{step}" for step in steps[:10])
                    tail = f"等 {len(steps)} 步" if len(steps) > 10 else ""
                    text = f"{prefix}采样训练步  ProfilerStep{markers}{tail}。各 Rank 的用时见「HCU 按训练步」。"
                else:
                    text = f"{prefix}采样训练步  trace 中未记录 CPU 上的 ProfilerStep。"
                overview.write(overview_row, 0, text, caption)
                overview_row += 1
            notes = [
                "HCU 时间跨度：从首项活动开始到末项活动结束。",
                "流空档原因由 HTA 按时序推断。同一卡上不同流的空档可以重叠。",
                "活动分类占比以活动总时长为分母。",
                "流空档原因占比以该流的总空档时长为分母。零总时长或含负值时留空。",
            ]
            if any(row[6] is not None for row in overlap_rows):
                notes[1:1] = [
                    "仅通信表示没有与计算重叠的通信时间。它仍是活动时间。",
                    "重叠率以通信总时长为分母。未采到可计时的通信时留空。",
                ]
            if phase_rows:
                notes[0:0] = [
                    "Prefill 和 Decode 图取自框架记录的 CPU 标记跨度。请求端延迟需另外测量。",
                ]
            if training_phase_rows:
                notes[0:0] = ["前向、反向和参数更新的用时见「训练阶段明细」。"]
            if loss_rows:
                notes.append("训练日志记录的损失值见「训练损失明细」。")
            if metrics and metrics.throughput:
                notes.append("训练吞吐及计时窗口见「训练吞吐明细」。")
                if throughput_log is not None:
                    notes.append("吞吐图使用单独传入的复测日志。损失图和轨迹分析沿用原训练数据。")
            for note in notes:
                overview.write(overview_row, 0, note)
                overview_row += 1
            kernel_link_row = overview_row
            if hotspots:
                overview_row += 1
            if bucket_size is not None or compare_bucket_size is not None:
                overview.write(overview_row, 0, "标签中的梯度元素数表示 Megatron 的分桶阈值。")
                overview_row += 1
            time = book.add_worksheet("HCU 时间分布")
            _write_table(
                time,
                ["样本", "Rank", "样本与 Rank", "计算 (ms)", "其他活动 (ms)",
                 "未记录到活动 (ms)", "分析窗口 (ms)",
                 *[f"{name}占比" for name, _ in _TIME_TYPES]],
                time_rows, [30, 9, 42, 17, 20, 22, 18, 16, 18, 22], header,
                {**dict.fromkeys(range(3, 7), numeric), **dict.fromkeys(range(7, 10), pct)},
            )
            overlap = book.add_worksheet("通信重叠")
            _write_table(
                overlap,
                ["样本", "Rank", "样本与 Rank", "通信 (ms)", "重叠 (ms)",
                 "仅通信 (ms)", "重叠率", "通信记录", "AllReduce 次数"],
                overlap_rows, [30, 9, 42, 17, 17, 17, 14, 28, 20], header,
                {3: numeric, 4: numeric, 5: numeric, 6: pct},
            )
            activity = book.add_worksheet("HCU 活动分类")
            _write_table(
                activity, ["样本", *[f"{name} (ms)" for _, name, _ in _ACTIVITY_TYPES],
                           *[f"{name}占比" for _, name, _ in _ACTIVITY_TYPES]],
                activity_rows, [30, 16, 16, 20, 20, 24, 24, 21, 16, 16, 20, 20, 24, 24, 21], header,
                {**dict.fromkeys(range(1, 8), numeric), **dict.fromkeys(range(8, 15), pct)},
            )
            idle = book.add_worksheet("HCU 流空档")
            _write_table(
                idle,
                ["样本", "Rank", "样本、Rank 与流", "流编号",
                 *[f"{name} (ms)" for _, name, _ in _IDLE_TYPES],
                 *[f"{name}占比" for _, name, _ in _IDLE_TYPES]],
                idle_rows, [30, 9, 49, 12, 20, 21, 18, 20, 21, 18], header,
                {**dict.fromkeys(range(4, 7), numeric), **dict.fromkeys(range(7, 10), pct)},
            )
            detail = book.add_worksheet("Kernel 明细")
            _write_table(
                detail,
                ["样本", "Rank", "内核名称", "活动类别", "调用次数", "累计 (µs)",
                 "平均 (µs)", "最短单次 (µs)", "最长单次 (µs)", "标准差 (µs)"],
                kernel_rows, [30, 9, 70, 18, 14, 18, 18, 18, 18, 18], header,
                {c: numeric for c in range(5, 10)},
            )
            top = book.add_worksheet("Kernel 热点")
            _write_table(
                top,
                ["样本", "Rank", "图中名称", "累计 (ms)", "内核全名", "调用次数",
                 "平均单次 (µs)", "最长单次 (µs)"],
                hotspot_rows, [30, 9, 46, 18, 70, 14, 18, 18], header,
                {3: numeric, 6: numeric, 7: numeric},
            )
            anchor = overview_row + 1
            if training_phase_rows:
                training_sheet = book.add_worksheet("训练阶段明细")
                _write_table(
                    training_sheet,
                    ["样本", "Rank", "图中名称", "模型", "训练步", "阶段", "CPU 标记跨度 (ms)",
                     "HCU kernel 累计耗时 (ms)", "已匹配 kernel 次数", "标记开始时间 (相对轨迹起点, ms)",
                     "原始 trace 标记"],
                    training_phase_rows, [22, 9, 34, 14, 20, 16, 23, 25, 22, 38, 33],
                    header, {6: numeric, 7: numeric, 9: numeric},
                )
                first = 1
                for (sample, rank, model), group in groupby(
                    training_phase_rows, key=lambda row: (row[0], row[1], row[3])
                ):
                    end = first + sum(1 for _ in group) - 1
                    name = model if single_capture_title else f"{sample} / {model}"
                    for start in range(first, end + 1, _RANKS_PER_CHART):
                        last = min(start + _RANKS_PER_CHART - 1, end)
                        series = [(6, "CPU 标记跨度", _COMM)]
                        if any(row[7] is not None for row in training_phase_rows[start - 1:last]):
                            series.append((7, "HCU kernel 累计耗时", _COMPUTE))
                        anchor = _add_chart(
                            book, overview, anchor, training_sheet.name, start, last, series,
                            f"{name} / Rank {rank}：训练阶段耗时", "ms", stacked=False,
                        )
                        overview.write(anchor - 2, 0, "HCU kernel 按 CPU 提交所在阶段归类。仅累计成功匹配的 kernel。", caption)
                        overview.write(anchor - 1, 0, "两种时间会重叠。HCU kernel 可以在 CPU 标记结束后继续运行。", caption)
                        anchor += 1
                    first = end + 1
            if metrics and metrics.throughput:
                anchor = _add_training_throughput(book, overview, anchor, metrics, header, numeric, caption)
            if loss_rows:
                anchor = _add_training_loss_chart(book, overview, anchor, loss_rows, header, numeric, caption)
            if phase_rows:
                sample_order = {name: index for index, (name, _) in enumerate(inputs)}
                phase_order = {"Prefill": 0, "Decode": 1, "Prefill + Decode": 2}
                phase_rows.sort(key=lambda row: (sample_order[row[0]], row[1], row[3], phase_order[row[4]], row[5]))
                phase_sheet = book.add_worksheet("推理阶段明细")
                _write_table(
                    phase_sheet,
                    ["样本", "Rank", "图中步骤", "推理框架", "阶段", "阶段内步骤序号",
                     "CPU 标记用时 (ms)", "标记开始时间 (相对轨迹起点, ms)", "原始 trace 标记"],
                    phase_rows, [22, 9, 17, 18, 21, 19, 23, 38, 58], header, {6: numeric, 7: numeric},
                )
                start = 1
                for (sample, rank, framework, phase), group in groupby(
                    phase_rows, key=lambda row: (row[0], row[1], row[3], row[4])
                ):
                    end = start + sum(1 for _ in group) - 1
                    name = sample if sample != "本次采样" else framework
                    for first in range(start, end + 1, _PHASE_STEPS_PER_CHART):
                        last = min(end, first + _PHASE_STEPS_PER_CHART - 1)
                        anchor = _add_chart(
                            book, overview, anchor, phase_sheet.name, first, last,
                            [(6, "步骤用时", _COMPUTE)],
                            f"{name} / Rank {rank}：{phase} 步骤用时", "ms",
                        )
                        overview.write(
                            anchor - 2, 0,
                            "「步骤 N」是该 Rank 在当前阶段按开始时间排列的第 N 步。一步可以处理一个或多个 token。",
                            caption,
                        )
                    start = end + 1
            if step_rows:
                step_sheet = book.add_worksheet("HCU 按训练步")
                _write_table(
                    step_sheet,
                    ["样本", "Rank", "图中名称", "ProfilerStep 编号", "计算 (ms)", "其他活动 (ms)",
                     "未记录到活动 (ms)", "训练步用时 (ms)", "步开始时间 (相对轨迹起点, ms)"],
                    step_rows, [30, 9, 46, 20, 17, 20, 22, 20, 33], header,
                    dict.fromkeys(range(4, 9), numeric),
                )
                for start in range(1, len(step_rows) + 1, _RANKS_PER_CHART):
                    end = min(start + _RANKS_PER_CHART - 1, len(step_rows))
                    anchor = _add_chart(
                        book, overview, anchor, step_sheet.name, start, end,
                        [(col, *item) for col, item in enumerate(_TIME_TYPES, 4)],
                        "各训练步的用时分布", "ms",
                    )
                    overview.write(
                        anchor - 2, 0,
                        "每条横条对应一个 Rank 的一个训练步。整条长度是 CPU 上的 ProfilerStep 用时。",
                        caption,
                    )
                    overview.write(
                        anchor - 1, 0,
                        "颜色显示这一步里 HCU 的活动。跨步活动按落在各步里的实际时长计入。",
                        caption,
                    )
                    anchor += 1
            for start in range(1, len(time_rows) + 1, _RANKS_PER_CHART):
                end = min(start + _RANKS_PER_CHART - 1, len(time_rows))
                if any(r[6] is not None for r in overlap_rows[start - 1:end]):
                    anchor = _add_chart(
                        book, overview, anchor, overlap.name, start, end,
                        [(6, "重叠率", _COMM)], "通信与计算重叠率", "%",
                    )
                    anchor = _add_chart(
                        book, overview, anchor, overlap.name, start, end,
                        [(4, "与计算重叠", _COMPUTE), (5, "仅通信", _COMM)],
                        "通信时长", "ms",
                    )
                if any(row[8] is not None for row in overlap_rows[start - 1:end]):
                    anchor = _add_chart(
                        book, overview, anchor, overlap.name, start, end,
                        [(8, "AllReduce 次数", _COMM)], "AllReduce 次数", "次",
                    )
                anchor = _add_chart(
                    book, overview, anchor, time.name, start, end,
                    [(col, *item) for col, item in enumerate(_TIME_TYPES, 3)],
                    "HCU 时间分布", "ms",
                )
                if any(row[7] is not None for row in time_rows[start - 1:end]):
                    anchor = _add_chart(
                        book, overview, anchor, time.name, start, end,
                        [(col, *item) for col, item in enumerate(_TIME_TYPES, 7)],
                        "HCU 分析窗口内的时间占比", "%",
                    )
                    overview.write(
                        anchor - 2, 0,
                        "对每个 Rank，HTA 从第一项 HCU 活动开始算到最后一项活动结束，这段时间为 100%。",
                        caption,
                    )
                    overview.write(
                        anchor - 1, 0,
                        "三种颜色分别表示计算、其他活动（如单独发生的通信或内存操作），以及未记录到 HCU 活动的时间。",
                        caption,
                    )
                    anchor += 1
            if any(row[8] is not None for row in activity_rows):
                activity_series = [
                    (col, name, color)
                    for col, (_, name, color) in enumerate(_ACTIVITY_TYPES, start=1)
                    if col <= 3 or any(row[col] for row in activity_rows)
                ]
                anchor = _add_chart(
                    book, overview, anchor, activity.name, 1, len(activity_rows),
                    activity_series, "HCU 活动分类（各 Rank 时长相加）", "ms", category_col=0,
                )
                anchor = _add_chart(
                    book, overview, anchor, activity.name, 1, len(activity_rows),
                    [(col + len(_ACTIVITY_TYPES), name, color)
                     for col, name, color in activity_series],
                    "HCU 活动分类占比", "%", category_col=0,
                )
            idle_series = [(col, name, color) for col, (_, name, color) in enumerate(_IDLE_TYPES, 4)]
            for start in range(1, len(idle_rows) + 1, _RANKS_PER_CHART):
                end = min(start + _RANKS_PER_CHART - 1, len(idle_rows))
                visible_rows = idle_rows[start - 1:end]
                values = [value for row in visible_rows for value in row[4:7]]
                if any(values):
                    anchor = _add_chart(
                        book, overview, anchor, idle.name, start, end, idle_series,
                        "HCU 各流的空档时间", "ms", signed=min(values) < 0,
                    )
                if any(row[7] is not None for row in visible_rows):
                    anchor = _add_chart(
                        book, overview, anchor, idle.name, start, end,
                        [(col + len(_IDLE_TYPES), name, color) for col, name, color in idle_series],
                        "HCU 各流的空档原因占比", "%",
                    )
            if hotspots:
                overview.write_url(
                    kernel_link_row, 0, f"internal:'图表'!A{anchor + 1}", string="跳到 Kernel 热点图表"
                )
                overview.write(anchor, 0, "每组图使用同一批 Kernel，均按累计耗时排序。", caption)
                anchor += 1
            for rank_label, start, end in hotspots:
                anchor = _add_chart(
                    book, overview, anchor, top.name, start, end,
                    [(3, "累计时长", _COMPUTE)], f"{rank_label}：Kernel 累计耗时", "ms",
                )
                anchor = _add_chart(
                    book, overview, anchor, top.name, start, end,
                    [(5, "调用次数", _COMM)], f"{rank_label}：Kernel 调用次数", "次",
                )
                anchor = _add_chart(
                    book, overview, anchor, top.name, start, end,
                    [(6, "平均单次", _COMPUTE), (7, "最长单次", _AMBER)],
                    f"{rank_label}：Kernel 单次耗时", "µs", stacked=False,
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Chrome trace file or trace directory")
    parser.add_argument("output", type=Path, help="Output .xlsx workbook")
    parser.add_argument("--training-log", type=Path, help="Megatron, BERT, or ResNet50 training log for the first capture")
    parser.add_argument("--throughput-log", type=Path, help="Optional per-step throughput log from a separate run of the same framework")
    parser.add_argument("--label", default=_DEFAULT_SAMPLE_LABEL, help="Label for the first capture")
    parser.add_argument("--bucket-size", type=int, help="Bucket threshold to show for the first capture")
    parser.add_argument("--compare", type=Path, help="Optional second file or directory")
    parser.add_argument(
        "--compare-label", default="样本 2", help="Label for the second capture"
    )
    parser.add_argument(
        "--compare-bucket-size", type=int, help="Bucket threshold to show for the second capture"
    )
    args = parser.parse_args()
    try:
        result = export_excel_report(
            args.trace, args.output, label=args.label, compare=args.compare,
            compare_label=args.compare_label, bucket_size=args.bucket_size,
            compare_bucket_size=args.compare_bucket_size,
            training_log=args.training_log,
            throughput_log=args.throughput_log,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "xlsxwriter":
            raise
        parser.error("Install the Excel extra with pip install 'traceinsight[excel]'")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported HTA data and charts to {result}")


if __name__ == "__main__":
    main()
