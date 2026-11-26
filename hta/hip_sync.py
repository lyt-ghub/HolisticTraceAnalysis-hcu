# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Add HCU stream waits to a Kineto trace using a same-process DTK hipprof trace."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

from hta.common.trace_file import read_trace, write_trace


HIP_RECORD_NAMES = ("hipEventRecord", "hipEventRecordWithFlags")
HIP_COMPLETION_LAUNCH_NAMES = ("hipExtLaunchKernel", "hipExtModuleLaunchKernel")
HIP_WAIT_NAMES = ("hipStreamWaitEvent",)
HIP_SYNC_CATEGORY = "hip_sync"
_EVENT_APIS = (
    *HIP_RECORD_NAMES,
    *HIP_WAIT_NAMES,
    "hipEventCreate",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
)
_EVENT_HANDLE = re.compile(r"[,(]\s*event\s*=\s*(0[xX][\da-fA-F]+|[1-9]\d*)\b")
_LAUNCH_EVENT_HANDLE = re.compile(
    r"[,(]\s*(startEvent|stopEvent)\s*=\s*(0[xX][\da-fA-F]+|[1-9]\d*)\b"
)
_STREAM_HANDLE = re.compile(
    r"[,(]\s*(?:stream|hStream)\s*=\s*(0[xX][\da-fA-F]+|0|[1-9]\d*)\b"
)
_DEVICE_CATEGORIES = {
    "kernel", "gpu_memcpy", "gpu_memset", "Kernel", "Memcpy", "Memset",
}
_HIPPROF_CATEGORIES = {"HIP", "BARRIER", "HIPOPS", "HIPCOPY"}
_RUNTIME_CATEGORIES = {"cuda_runtime", "hip_runtime"}
_MAX_CLOCK_SKEW_NS = 1_000_000


@dataclass(frozen=True)
class _HipCall:
    name: str
    tid: int
    correlation: int
    begin_ns: int
    end_ns: int
    stream_handle: int | None
    completes_kernel: bool = False


@dataclass(frozen=True)
class _Barrier:
    queue: tuple[int, int]
    begin_ns: int | None
    end_ns: int | None


@dataclass(frozen=True)
class _RelayEndpoint:
    barrier_ns: int
    host_ns: int
    workload_queue: tuple[int, int]
    workload_stream: tuple[int, int]

    @property
    def workload(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.workload_queue, self.workload_stream


_RelayKey = tuple[tuple[int, int], int]


def _count_round_trip_waits(
    incoming: dict[_RelayKey, list[_RelayEndpoint]],
    outgoing: dict[_RelayKey, list[_RelayEndpoint]],
) -> int:
    """Count unique pairs where a signal returns to the originating workload queue."""
    total = 0
    for relay, entries in incoming.items():
        entries.sort(key=lambda entry: entry.barrier_ns)
        ends = [entry.barrier_ns for entry in entries]
        paired = defaultdict(list)
        for signal in outgoing.get(relay, ()):
            index = bisect_right(ends, signal.barrier_ns) - 1
            if index < 0 or (index > 0 and ends[index - 1] == ends[index]):
                continue
            wait = entries[index]
            if wait.host_ns <= signal.host_ns:
                paired[index].append(signal)
        total += 2 * sum(
            len(signals) == 1 and entries[index].workload == signals[0].workload
            for index, signals in paired.items()
        )
    return total


def _kineto_time_ns(event: dict[str, Any], base_time_ns: int) -> int:
    return base_time_ns + round(event["ts"] * 1000)


def augment_hip_stream_waits(
    trace: dict[str, Any], hipprof_events: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Link HIP stream waits to recorded events and keep the original Kineto events."""
    events = trace["traceEvents"]
    runtime = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in _RUNTIME_CATEGORIES
    ]
    waits = [event for event in runtime if event["name"] in HIP_WAIT_NAMES]
    counts = {
        "waits": len(waits),
        "added": 0,
        "already_present": 0,
        "same_stream": 0,
        "round_trip_waits": 0,
        "kernel_completion_waits": 0,
        "unresolved": 0,
    }
    if not waits:
        return trace, counts
    pids = {event["pid"] for event in runtime}
    if len(pids) != 1:
        raise ValueError("Expected one HIP process per Kineto trace")
    pid = next(iter(pids))
    base_time_ns = trace["baseTimeNanoseconds"]

    # Match hipprof queues to Kineto streams using the observed dispatches.
    device_activity = defaultdict(set)
    kernels_by_correlation = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in _DEVICE_CATEGORIES:
            continue
        args = event.get("args", {})
        stream = args.get("stream")
        if "correlation" in args and isinstance(stream, int) and stream >= 0:
            device_activity[args["correlation"]].add((event["pid"], stream))
            if event["cat"] in {"kernel", "Kernel"}:
                kernels_by_correlation[args["correlation"]].append((event["pid"], stream))

    stream_map = defaultdict(set)
    stream_handles = defaultdict(set)
    barriers = defaultdict(set)
    workload_queues = set()
    calls_by_handle = defaultdict(set)
    handles_by_wait = defaultdict(set)
    runtime_keys = {
        (event["tid"], event.get("args", {}).get("correlation"), event["name"])
        for event in runtime
    }
    for event in hipprof_events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in _HIPPROF_CATEGORIES:
            continue
        args = event["args"]
        if category == "HIP":
            if args["pid"] != pid or args.get("state") not in (0, "0"):
                continue
            name = event["name"]
            tid, corr = args["tid"], args["index"]
            arguments = args.get("args") or ""
            stream = _STREAM_HANDLE.search(arguments)
            stream_handle = int(stream[1], 0) if stream is not None else None
            if stream_handle is not None and (tid, corr, name) in runtime_keys:
                stream_handles[stream_handle].update(device_activity.get(corr, ()))
            handle = _EVENT_HANDLE.search(arguments) if name in _EVENT_APIS else None
            launch_handles = defaultdict(set)
            if "startEvent" in arguments or "stopEvent" in arguments:
                for match in _LAUNCH_EVENT_HANDLE.finditer(arguments):
                    if launch_handle := int(match[2], 0):
                        launch_handles[launch_handle].add(match[1])
            handle_id = int(handle[1], 0) if handle is not None else 0
            if not handle_id and not launch_handles:
                continue
            call = _HipCall(
                name, tid, corr, int(args["BeginNs"]), int(args["EndNs"]), stream_handle
            )
            if handle_id and name in HIP_WAIT_NAMES:
                handles_by_wait[call.tid, call.correlation].add((handle_id, call))
            elif handle_id:
                calls_by_handle[handle_id].add(call)
            for launch_handle, parameters in launch_handles.items():
                completes_kernel = (
                    parameters == {"stopEvent"} and name in HIP_COMPLETION_LAUNCH_NAMES
                )
                calls_by_handle[launch_handle].add(
                    replace(call, completes_kernel=completes_kernel)
                )
        elif args.get("procId", args.get("pid")) == pid:
            corr = args["index"]
            device, queue = int(args["devId"]), int(args["queueId"])
            if category == "BARRIER":
                begin_ns, end_ns = args.get("BeginNs"), args.get("EndNs")
                barriers[corr].add(
                    _Barrier(
                        (device, queue),
                        int(begin_ns) if begin_ns not in (None, "") else None,
                        int(end_ns) if end_ns not in (None, "") else None,
                    )
                )
            else:
                workload_queues.add((device, queue))
                stream_map[device, queue].update(device_activity.get(corr, ()))

    timelines = {}
    for handle, calls in calls_by_handle.items():
        ordered = sorted(
            calls, key=lambda call: (call.begin_ns, call.end_ns, call.correlation)
        )
        timelines[handle] = ([call.begin_ns for call in ordered], ordered)
    producers = {
        (event["tid"], event.get("args", {}).get("correlation")): event
        for event in runtime if event["name"] in (*HIP_RECORD_NAMES, *HIP_COMPLETION_LAUNCH_NAMES)
    }
    existing = {
        event.get("args", {}).get("correlation")
        for event in events
        if event.get("cat") in {"cuda_sync", HIP_SYNC_CATEGORY}
        and event.get("name") == "Stream Wait Event"
    }

    def resolve_stream(call: _HipCall) -> tuple[set[tuple[int, int]], bool]:
        by_handle = stream_handles.get(call.stream_handle, set())
        by_queue = set().union(
            *(
                stream_map.get(barrier.queue, ())
                for barrier in barriers.get(call.correlation, ())
            )
        )
        if by_handle and by_queue:
            return by_handle & by_queue, True
        return by_handle or by_queue, bool(by_handle or by_queue)

    incoming: dict[_RelayKey, list[_RelayEndpoint]] = defaultdict(list)
    outgoing: dict[_RelayKey, list[_RelayEndpoint]] = defaultdict(list)

    def observe_relay(
        record: _HipCall,
        wait: _HipCall,
        src: set[tuple[int, int]],
        dst: set[tuple[int, int]],
        src_mapped: bool,
        dst_mapped: bool,
    ) -> None:
        if src_mapped == dst_mapped:
            return
        visible, hidden_call = (src, wait) if src_mapped else (dst, record)
        if len(visible) != 1 or hidden_call.stream_handle is None:
            return
        record_barriers = barriers.get(record.correlation, ())
        wait_barriers = barriers.get(wait.correlation, ())
        if len(record_barriers) != 1 or len(wait_barriers) != 1:
            return
        before, after = next(iter(record_barriers)), next(iter(wait_barriers))
        if before.queue == after.queue or before.queue[0] != after.queue[0]:
            return
        if not (
            before.begin_ns is not None and before.end_ns is not None
            and after.begin_ns is not None and after.end_ns is not None
            and before.begin_ns <= before.end_ns <= after.end_ns
            and after.begin_ns <= after.end_ns
        ):
            return
        relay, anchor = (after, before) if src_mapped else (before, after)
        if relay.queue in workload_queues:
            return
        key = (relay.queue, hidden_call.stream_handle)
        anchor_stream = next(iter(visible))
        if src_mapped:
            incoming[key].append(
                _RelayEndpoint(after.end_ns, wait.end_ns, anchor.queue, anchor_stream)
            )
        else:
            outgoing[key].append(
                _RelayEndpoint(before.begin_ns, record.begin_ns, anchor.queue, anchor_stream)
            )

    additions = []
    for event in waits:
        corr = event.get("args", {}).get("correlation")
        if corr in existing:
            counts["already_present"] += 1
            continue
        matches = handles_by_wait.get((event["tid"], corr), set())
        if len(matches) != 1:
            continue
        handle, wait = next(iter(matches))
        wait_time_ns = _kineto_time_ns(event, base_time_ns)
        if abs(wait.begin_ns - wait_time_ns) > _MAX_CLOCK_SKEW_NS:
            continue
        starts, calls = timelines.get(handle, ([], []))
        index = bisect_left(starts, wait.begin_ns) - 1
        if index < 0 or (index + 1 < len(starts) and starts[index + 1] < wait.end_ns):
            continue
        record = calls[index]
        # Overlapping API submissions leave the generation of the event ambiguous.
        if record.end_ns > wait.begin_ns:
            continue
        if index > 0 and calls[index - 1].end_ns > record.begin_ns:
            continue
        if not record.completes_kernel and record.name not in HIP_RECORD_NAMES:
            continue
        record_event = producers.get((record.tid, record.correlation))
        if record_event is None or record_event["name"] != record.name:
            continue
        record_time_ns = _kineto_time_ns(record_event, base_time_ns)
        if abs(record.begin_ns - record_time_ns) > _MAX_CLOCK_SKEW_NS:
            continue

        src, src_mapped = resolve_stream(record)
        dst, dst_mapped = resolve_stream(wait)
        if record.completes_kernel:
            # The launch itself records the completion event. Require exactly
            # one corresponding HCU kernel on the resolved stream.
            kernels = kernels_by_correlation.get(record.correlation, ())
            if len(kernels) != 1 or kernels[0] not in src:
                continue
            src = {kernels[0]}
        if len(src) != 1 or len(dst) != 1:
            if not record.completes_kernel:
                observe_relay(record, wait, src, dst, src_mapped, dst_mapped)
            continue
        src_device, src_stream = next(iter(src))
        dst_device, dst_stream = next(iter(dst))
        if src_device != dst_device:
            continue
        if src_stream == dst_stream:
            counts["same_stream"] += 1
            continue

        additions.append(
            {
                "ph": "X",
                "cat": HIP_SYNC_CATEGORY,
                "name": "Stream Wait Event",
                "pid": dst_device,
                "tid": dst_stream,
                # Zero duration keeps the CPU submission marker out of HCU timings.
                "ts": event["ts"],
                "dur": 0,
                "args": {
                    "correlation": wait.correlation,
                    "stream": dst_stream,
                    "wait_on_stream": src_stream,
                    "wait_on_cuda_event_record_corr_id": record.correlation,
                },
            }
        )
        existing.add(wait.correlation)
        counts["kernel_completion_waits"] += int(record.completes_kernel)

    counts["added"] = len(additions)
    counts["round_trip_waits"] = _count_round_trip_waits(incoming, outgoing)
    counts["unresolved"] = counts["waits"] - sum((
        counts["added"], counts["already_present"], counts["same_stream"],
        counts["round_trip_waits"],
    ))
    return {**trace, "traceEvents": events + additions}, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kineto", type=Path, help="PyTorch Profiler .json or .json.gz")
    parser.add_argument(
        "hipprof", type=Path, nargs="+", help="Same-run DTK hipprof JSON files"
    )
    parser.add_argument("output", type=Path, help="A separate output .json or .json.gz")
    args = parser.parse_args()
    inputs = (args.kineto, *args.hipprof)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("Choose an output path separate from both input files")
    try:
        def hipprof_events():
            for path in args.hipprof:
                yield from read_trace(str(path))["traceEvents"]

        trace, counts = augment_hip_stream_waits(
            read_trace(str(args.kineto)), hipprof_events()
        )
        with TemporaryDirectory(dir=args.output.parent) as directory:
            temp = Path(directory) / args.output.name
            write_trace(trace, str(temp))
            temp.replace(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
