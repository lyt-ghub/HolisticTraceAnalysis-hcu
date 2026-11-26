# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export individual Megatron trace analysis reports with descriptive names."""

import argparse
import json
from pathlib import Path

import pandas as pd

from hta.common.atomic_file import atomic_binary_output
from hta.configs.env_options import HTAEnvOptions
from hta.excel_report import _load_trace
from hta.utils.utils import is_allreduce_kernel


REPORT_FILES = {
    "temporal_0": "hcu_time_breakdown_by_rank.xlsx",
    "kernels_0": "hcu_activity_time_by_type.xlsx",
    "kernels_1": "hcu_kernel_hotspots_by_rank.xlsx",
    "idle_0": "hcu_stream_idle_time_by_cause.xlsx",
    "idle_1": "hcu_stream_idle_gap_statistics.csv",
    "overlap_0": "communication_compute_overlap_by_rank.csv",
    "bucket_overlap": "gradient_bucket_overlap_comparison.xlsx",
    "bucket_overlap_csv": "gradient_bucket_overlap_comparison.csv",
    "summary": "trace_capture_and_analysis_summary.json",
    "guide": "megatron_analysis_guide.md",
}

_DESCRIPTIONS = {
    "temporal_0": "各 Rank 的计算、其他活动和空闲时长。",
    "kernels_0": "各 Rank 合计的计算、通信、内存操作和重叠时长。",
    "kernels_1": "按 Rank 和活动类别汇总累计耗时较高的 HCU kernel。",
    "idle_0": "各 HCU 流按原因汇总的空闲时长和占比。",
    "idle_1": "各 HCU 流的空档次数、平均时长、分位数和极值。",
    "overlap_0": "各 Rank 的通信与计算重叠率。没有可计时的通信时留空。",
    "bucket_overlap": "两组梯度分桶配置的通信时长和重叠率。",
    "bucket_overlap_csv": "分桶重叠对照表的 CSV 版本。",
    "summary": "采样 Rank、解析事件数量和报表清单。",
}

_HEADERS = {
    "rank": "训练进程编号", "stream": "设备流编号", "idle_category": "空闲原因",
    "idle_time": "空闲时长 (µs)", "idle_time_ratio": "流内空闲占比",
    "count": "空档次数", "mean": "平均空档时长 (µs)", "std": "空档时长标准差 (µs)",
    "min": "最短空档 (µs)", "max": "最长空档 (µs)",
    "25%": "第 25 百分位", "50%": "中位数", "75%": "第 75 百分位",
    "idle_time(us)": "空闲时长", "compute_time(us)": "计算时长",
    "non_compute_time(us)": "非计算时长", "kernel_time(us)": "分析窗口时长",
    "idle_time_pctg": "空闲占比 (%)", "compute_time_pctg": "计算占比 (%)",
    "non_compute_time_pctg": "非计算占比 (%)",
    "kernel_type": "活动类别", "name": "Kernel 名称", "sum": "活动时长 (µs)",
    "sum (us)": "累计耗时", "max (us)": "最长单次耗时", "min (us)": "最短单次耗时",
    "mean (us)": "平均单次耗时", "stddev": "耗时标准差 (µs)", "percentage": "时间占比 (%)",
    "comp_comm_overlap_pctg": "通信与计算重叠率 (%)",
}


def _write_report(frame: pd.DataFrame, output: Path, sheet: str) -> None:
    headers = [f"{col} {_HEADERS[col]}" if col in _HEADERS else col for col in frame]
    with atomic_binary_output(output) as stream:
        if output.suffix == ".csv":
            stream.write(frame.to_csv(index=False, header=headers).encode("utf-8-sig"))
            return

        import xlsxwriter

        options = {"strings_to_formulas": False, "strings_to_urls": False}
        with xlsxwriter.Workbook(stream, options) as workbook:
            table = workbook.add_worksheet(sheet)
            header = workbook.add_format(
                {"bold": True, "bg_color": "#203C55", "font_color": "#FFFFFF"}
            )
            table.write_row(0, 0, headers, header)
            table.freeze_panes(1, 0)
            table.autofilter(0, 0, len(frame), len(headers) - 1)
            for column, name in enumerate(headers):
                width = 65 if frame.columns[column] == "name" else max(16, len(name) + 5)
                table.set_column(column, column, width)
            values = frame.replace([float("inf"), float("-inf")], float("nan"))
            values = values.astype(object).where(values.notna(), None)
            for index, row in enumerate(values.itertuples(index=False, name=None), start=1):
                table.write_row(index, 0, row)


def _comparison_rows(analysis, label):
    overlap = analysis.get_comm_comp_overlap(visualize=False, include_times=True)
    symbols = analysis.t.symbol_table
    rows = []
    for row in overlap.itertuples(index=False):
        frame = analysis.t.get_trace(row.rank)
        gpu = frame.loc[symbols.get_gpu_kernel_mask(frame)]
        names = gpu["name"].map(symbols.get_sym_table().__getitem__)
        rows.append({
            "配置": label,
            "训练进程 Rank": row.rank,
            "AllReduce 次数": int(names.map(is_allreduce_kernel).sum()),
            "通信时间 (µs)": row.communication_time_us,
            "通信与计算重叠时间 (µs)": row.overlap_time_us,
            "HTA 重叠率 (%)": row.comp_comm_overlap_pctg,
        })
    return rows


def export_megatron_analysis(
    trace: Path,
    output: Path,
    *,
    compare: Path | None = None,
    label: str = "样本 1",
    compare_label: str = "样本 2",
) -> dict:
    """Generate separate tables and an index from one or two profiler captures."""
    options = HTAEnvOptions.instance()
    old_rounding = options.disable_ns_rounding()
    options.set_disable_ns_rounding(True)
    try:
        analysis = _load_trace(trace)
        symbols = analysis.t.symbol_table
        names = symbols.get_sym_table()
        ranks = analysis.t.get_ranks()
        event_counts = {}
        activity_counts = {}
        profile_steps = {}
        for rank in ranks:
            frame = analysis.t.get_trace(rank)
            event_counts[rank] = len(frame)
            activity_counts[rank] = int(symbols.get_gpu_kernel_mask(frame).sum())
            profile_steps[rank] = sorted(
                {names[name] for name in frame["name"].unique()
                 if names[name].startswith("ProfilerStep#")}
            )
        active_ranks = [rank for rank in ranks if activity_counts[rank]]
        if not active_ranks:
            raise ValueError("The trace contains no HCU activity to analyze")
        categories, kernels = analysis.get_gpu_kernel_breakdown(visualize=False)
        idle, intervals = analysis.get_idle_time_breakdown(
            ranks=active_ranks, visualize=False, show_idle_interval_stats=True,
        )
        tables = {
            "temporal_0": analysis.get_temporal_breakdown(visualize=False),
            "kernels_0": categories,
            "kernels_1": kernels,
            "idle_0": idle,
            "idle_1": intervals.rename_axis("idle_category").reset_index(),
            "overlap_0": analysis.get_comm_comp_overlap(visualize=False),
        }
        if compare is not None:
            if label == compare_label:
                raise ValueError("Use distinct labels for the two captures")
            comparison = pd.DataFrame(
                _comparison_rows(analysis, label)
                + _comparison_rows(_load_trace(compare), compare_label)
            )
            tables.update(bucket_overlap=comparison, bucket_overlap_csv=comparison)
    finally:
        options.set_disable_ns_rounding(old_rounding)

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "trace_dir": str(trace if trace.is_dir() else trace.parent),
        "trace_files": [Path(analysis.t.trace_files[rank]).name for rank in ranks],
        "ranks": ranks,
        "trace_events_after_parsing_by_rank": event_counts,
        "hcu_activity_events_by_rank": activity_counts,
        "profile_steps_by_rank": profile_steps,
        "artifacts": {
            key: {"file": REPORT_FILES[key], "rows": len(table)}
            for key, table in tables.items()
        },
    }
    for key, table in tables.items():
        _write_report(table, output / REPORT_FILES[key], key)
    with atomic_binary_output(output / REPORT_FILES["summary"]) as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))

    guide = ["# Megatron 分析报表", "", "| 文件 | 内容 |", "|---|---|"]
    for key in [*tables, "summary"]:
        filename = REPORT_FILES[key]
        guide.append(f"| [{filename}]({filename}) | {_DESCRIPTIONS[key]} |")
    with atomic_binary_output(output / REPORT_FILES["guide"]) as stream:
        stream.write(("\n".join(guide) + "\n").encode("utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="PyTorch Profiler trace file or directory")
    parser.add_argument("output", type=Path, help="Directory for the reports")
    parser.add_argument(
        "--compare", type=Path, help="Second capture for the bucket overlap comparison"
    )
    parser.add_argument("--label", default="样本 1", help="Label for the first capture")
    parser.add_argument("--compare-label", default="样本 2", help="Label for the second capture")
    args = parser.parse_args()
    try:
        export_megatron_analysis(
            args.trace, args.output, compare=args.compare,
            label=args.label, compare_label=args.compare_label,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "xlsxwriter":
            raise
        parser.error("Install the Excel extra with pip install 'traceinsight[excel]'")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported Megatron analysis reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()
