# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export Megatron PyTorch Profiler traces in profiler table format."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hta.common.trace_file import read_trace


_HEADERS = (
    "Name",
    "Self CPU %",
    "Self CPU",
    "CPU total %",
    "CPU total",
    "CPU time avg",
    "Self CUDA",
    "Self CUDA %",
    "CUDA total",
    "CUDA time avg",
    "# of Calls",
    "Input Shapes",
)
_CPU_ACTIVITIES = {
    "cpu_op", "user_annotation", "cuda_runtime", "cuda_driver", "hip_runtime", "hip_driver"
}
_DEVICE_ACTIVITIES = {"kernel", "gpu_memcpy", "gpu_memset"}
_ACTIVITIES = _CPU_ACTIVITIES | _DEVICE_ACTIVITIES


@dataclass
class _Activity:
    start: float
    duration: float
    category: str
    name: str
    shapes: str
    direct_device: float = 0.0
    inclusive_device: float = 0.0
    covered_children: float = 0.0
    covered_until: float = float("-inf")
    parent: int | None = None

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class _Totals:
    calls: int = 0
    self_cpu: float = 0.0
    cpu: float = 0.0
    self_device: float = 0.0
    device: float = 0.0


def _profile_rows(trace: dict) -> list[tuple]:
    """Aggregate recorded activities by name, event kind, and recorded shapes."""
    cpu_threads = defaultdict(list)
    cpu_by_external_id = defaultdict(list)
    launch_external_ids = defaultdict(set)
    device_events = []
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Missing PyTorch Profiler traceEvents array")

    origin = min((event["ts"] for event in events if event.get("ph") == "X"), default=0)
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in _ACTIVITIES:
            continue
        args = event.get("args", {})
        shapes = args.get("Input Dims", args.get("Input Shapes"))
        shapes = "" if shapes is None else json.dumps(shapes, ensure_ascii=False)
        activity = _Activity(
            event["ts"] - origin, event["dur"], category, event["name"], shapes
        )
        external_id = args.get("External id")
        correlation = args.get("correlation")
        if category in _DEVICE_ACTIVITIES:
            device_events.append((activity, external_id, correlation))
        else:
            cpu_threads[event["pid"], event["tid"]].append(activity)
            if category in {"cpu_op", "user_annotation"} and external_id:
                cpu_by_external_id[external_id].append(activity)
            elif external_id and correlation is not None:
                launch_external_ids[correlation].add(external_id)

    totals: dict[tuple[str, str, str], _Totals] = defaultdict(_Totals)
    device_total = 0.0
    for activity, external_id, correlation in device_events:
        owners = cpu_by_external_id.get(external_id, [])
        if not owners:
            launch_ids = launch_external_ids.get(correlation, set())
            if len(launch_ids) == 1:
                owners = cpu_by_external_id.get(next(iter(launch_ids)), [])
        if len(owners) == 1:
            owners[0].direct_device += activity.duration
        summary = totals[activity.name, activity.category, activity.shapes]
        summary.calls += 1
        summary.self_device += activity.duration
        summary.device += activity.duration
        device_total += activity.duration

    for activities in cpu_threads.values():
        activities.sort(key=lambda event: (event.start, -event.end))
        ancestors: list[int] = []
        for index, activity in enumerate(activities):
            while ancestors and activity.end > activities[ancestors[-1]].end:
                ancestors.pop()
            if ancestors:
                activity.parent = ancestors[-1]
                parent = activities[activity.parent]
                parent.covered_children += max(
                    0, activity.end - max(activity.start, parent.covered_until)
                )
                parent.covered_until = max(parent.covered_until, activity.end)
            ancestors.append(index)

        for activity in reversed(activities):
            activity.inclusive_device += activity.direct_device
            if activity.parent is not None:
                activities[activity.parent].inclusive_device += activity.inclusive_device
            summary = totals[activity.name, activity.category, activity.shapes]
            summary.calls += 1
            summary.self_cpu += max(0, activity.duration - activity.covered_children)
            summary.cpu += activity.duration
            summary.self_device += activity.direct_device
            summary.device += activity.inclusive_device

    cpu_total = sum(summary.self_cpu for summary in totals.values())
    rows = []
    for (name, _, shapes), summary in sorted(
        totals.items(), key=lambda item: (-item[1].self_device, item[0])
    ):
        rows.append(
            (
                name,
                summary.self_cpu / cpu_total if cpu_total else 0,
                summary.self_cpu,
                summary.cpu / cpu_total if cpu_total else 0,
                summary.cpu,
                summary.cpu / summary.calls,
                summary.self_device,
                summary.self_device / device_total if device_total else 0,
                summary.device,
                summary.device / summary.calls,
                summary.calls,
                shapes,
            )
        )
    return rows


def export_megatron_profile(trace_path: str | Path, output_path: str | Path) -> int:
    """Write one worksheet per Chrome trace and return the number of data rows."""
    import xlsxwriter

    trace_path, output_path = Path(trace_path), Path(output_path)
    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("Choose an .xlsx output path")
    sources = (
        sorted(
            path
            for path in trace_path.iterdir()
            if path.is_file() and path.name.endswith((".json", ".json.gz"))
        )
        if trace_path.is_dir()
        else [trace_path]
    )
    if not sources:
        raise ValueError(f"No .json or .json.gz trace files in {trace_path}")

    count = 0
    sheet_names: set[str] = set()
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temporary_output = Path(temp_dir) / "profile.xlsx"
        with xlsxwriter.Workbook(temporary_output) as workbook:
            header = workbook.add_format(
                {"bold": True, "font_color": "#FFFFFF", "bg_color": "#203C55"}
            )
            percent = workbook.add_format({"num_format": "0.00%"})
            timing = workbook.add_format({"num_format": '#,##0.000###" µs"'})
            text = workbook.add_format({"text_wrap": True, "valign": "top"})
            for index, source in enumerate(sources, start=1):
                trace = read_trace(str(source))
                rows = _profile_rows(trace)
                if not rows or len(rows) > 1048575:
                    raise ValueError(
                        f"Unsupported profiler row count in {source}: {len(rows)}"
                    )
                rank = (trace.get("distributedInfo") or {}).get("rank")
                name = f"Rank {rank}" if isinstance(rank, int) else f"Trace {index}"
                if name in sheet_names:
                    name = f"{name} ({index})"
                sheet_names.add(name)
                sheet = workbook.add_worksheet(name)
                sheet.write_row(0, 0, _HEADERS, header)
                sheet.hide_gridlines(2)
                sheet.freeze_panes(1, 1)
                sheet.autofilter(0, 0, len(rows), len(_HEADERS) - 1)
                sheet.set_column(0, 0, 62, text)
                sheet.set_column(10, 10, 15)
                sheet.set_column(11, 11, 44, text)
                for column in range(1, 10):
                    column_format = percent if column in {1, 3, 7} else timing
                    sheet.set_column(column, column, 19, column_format)
                for row_index, row in enumerate(rows, start=1):
                    if any(len(row[column]) > 32767 for column in (0, 11)):
                        raise ValueError(
                            f"Cell exceeds the Excel text limit in {source}, "
                            f"row {row_index + 1}"
                        )
                    sheet.write_string(row_index, 0, row[0])
                    sheet.write_row(row_index, 1, row[1:11])
                    if row[11]:
                        sheet.write_string(row_index, 11, row[11])
                count += len(rows)
        temporary_output.replace(output_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Chrome trace file or trace directory")
    parser.add_argument("output", type=Path, help="Path to a standalone .xlsx file")
    args = parser.parse_args()
    try:
        count = export_megatron_profile(args.trace, args.output)
    except ModuleNotFoundError as exc:
        if exc.name != "xlsxwriter":
            raise
        parser.error("Install the Excel extra with pip install 'traceinsight[excel]'")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported {count} profiler rows to {args.output}")


if __name__ == "__main__":
    main()
