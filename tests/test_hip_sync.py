# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hta.analyzers.critical_path_analysis import CPEdgeType
from hta.hip_sync import augment_hip_stream_waits
from hta.trace_analysis import TraceAnalysis


BASE = 1_000_000_000
PID = 100


def _recorded_workload():
    trace = {"baseTimeNanoseconds": BASE, "distributedInfo": {"rank": 0}, "traceEvents": []}
    hipprof = []

    def event(category, name, start, duration, pid=PID, **args):
        trace["traceEvents"].append({
            "ph": "X", "cat": category, "name": name, "pid": pid, "tid": pid,
            "ts": start, "dur": duration, "args": args,
        })

    def sidecar(category, name, index, start, end, **args):
        hipprof.append({
            "ph": "X", "cat": category, "name": name,
            "args": {
                "pid": PID, "tid": PID, "index": index, "state": 0,
                "BeginNs": str(BASE + start * 1000), "EndNs": str(BASE + end * 1000),
                **args,
            },
        })

    def dispatch(name, corr, launch, start, stream, queue):
        event("cuda_runtime", "hipLaunchKernel", launch, 1, correlation=corr)
        event("kernel", name, start, 3, pid=0, correlation=corr, stream=stream)
        sidecar("HIP", "hipLaunchKernel", corr, launch, launch + 1,
                args=f"hipLaunchKernel(stream=0x{queue:x})")
        sidecar("HIPOPS", name, corr, start, start + 3, devId=0, queueId=str(queue))

    def event_api(name, corr, start, handle, queue, barrier_start, barrier_end):
        event("cuda_runtime", name, start, 1, correlation=corr)
        sidecar("HIP", name, corr, start, start + 1,
                args=f"{name}(event={handle}, stream=0x{queue:x})")
        sidecar("BARRIER", "BARRIER", corr, barrier_start, barrier_end,
                devId=0, queueId=str(queue))

    event("user_annotation", "ProfilerStep#0", 1, 80)
    dispatch("producer_a", 1, 5, 6, 1, 101)
    dispatch("producer_b", 2, 12, 13, 3, 103)
    event_api("hipEventRecord", 11, 19, "0xa", 101, 30, 31)
    event_api("hipEventRecordWithFlags", 12, 22, "0xb", 103, 31, 32)
    event_api("hipStreamWaitEvent", 21, 24, "0xa", 102, 32, 34)
    event_api("hipStreamWaitEvent", 22, 27, "0xb", 102, 34, 51)
    dispatch("consumer", 3, 40, 50, 2, 102)
    return trace, hipprof


def _round_trip_workload():
    trace, hipprof = _recorded_workload()
    # 101 signals 109, then 109 signals 101 after the first barrier completes.
    route = {21: (109, 24, 32, 34), 12: (109, 27, 36, 37), 22: (101, 30, 38, 39)}
    for event in trace["traceEvents"]:
        if event["args"].get("correlation") in route:
            event["ts"] = route[event["args"]["correlation"]][1]
    for event in hipprof:
        args = event["args"]
        if args["index"] not in route:
            continue
        queue, start, barrier_start, barrier_end = route[args["index"]]
        if event["cat"] == "HIP":
            args["args"] = args["args"].split("stream=")[0] + f"stream=0x{queue:x})"
            args["BeginNs"], args["EndNs"] = (BASE + start * 1000, BASE + (start + 1) * 1000)
        elif event["cat"] == "BARRIER":
            args["queueId"] = str(queue)
            args["BeginNs"], args["EndNs"] = (
                BASE + barrier_start * 1000, BASE + barrier_end * 1000,
            )
    return trace, hipprof


def _completion_workload(parameter):
    trace, hipprof = _recorded_workload()
    trace["traceEvents"] = [
        e for e in trace["traceEvents"] if e["args"].get("correlation") not in {11, 12, 22}
    ]
    hipprof = [e for e in hipprof if e["args"]["index"] not in {11, 12, 22}]
    for event in trace["traceEvents"]:
        if event["name"] == "hipLaunchKernel" and event["args"]["correlation"] in (1, 3):
            event["name"] = "hipExtLaunchKernel"
    for event in hipprof:
        if event["name"] == "hipLaunchKernel" and event["args"]["index"] in (1, 3):
            event["name"] = "hipExtLaunchKernel"
            args = event["args"]
            args["args"] = args["args"].replace("hipLaunchKernel", "hipExtLaunchKernel")
            if args["index"] == 1:
                args["args"] = args["args"][:-1] + f", {parameter}=0xa)"
    return trace, hipprof


def _stream_wait_edges(trace):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rank.json"
        path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        analysis = TraceAnalysis(trace_dir=directory)
        graph, success = analysis.critical_path_analysis(0, "ProfilerStep#0", 0)
    if not success:
        raise AssertionError("Could not analyze the training step")
    frame = analysis.t.get_trace(0)
    edges = []
    for _, _, data in graph.edges(data=True):
        edge = data["object"]
        if edge.type == CPEdgeType.SYNC_DEPENDENCY:
            src, dst = (graph.node_list[i].ev_idx for i in (edge.begin, edge.end))
            edges.append((frame.loc[src], frame.loc[dst]))
    return edges


class HipSyncTest(unittest.TestCase):
    def test_adds_both_cross_stream_edges_before_consuming_the_kernel(self):
        trace, hipprof = _recorded_workload()
        augmented, counts = augment_hip_stream_waits(trace, hipprof)
        self.assertEqual(counts, {
            "waits": 2, "added": 2, "already_present": 0, "same_stream": 0,
            "round_trip_waits": 0, "kernel_completion_waits": 0, "unresolved": 0,
        })
        self.assertEqual(augment_hip_stream_waits(augmented, hipprof)[1]["already_present"], 2)
        pairs = {(src["stream"], dst["stream"]) for src, dst in _stream_wait_edges(augmented)}
        self.assertEqual(pairs, {(1, 2), (3, 2)})

    def test_skips_incomplete_or_mismatched_sidecar(self):
        trace, hipprof = _recorded_workload()
        without_queue_records = [e for e in hipprof if e["cat"] == "HIP"]
        self.assertEqual(augment_hip_stream_waits(trace, without_queue_records)[1]["added"], 2)
        without_barrier_times = deepcopy(hipprof)
        for event in without_barrier_times:
            if event["cat"] == "BARRIER":
                del event["args"]["BeginNs"], event["args"]["EndNs"]
        self.assertEqual(augment_hip_stream_waits(trace, without_barrier_times)[1]["added"], 2)
        incomplete = [e for e in without_queue_records if e["args"]["index"] != 1]
        self.assertEqual(augment_hip_stream_waits(trace, incomplete)[1]["unresolved"], 1)
        stale = deepcopy(hipprof)
        for event in stale:
            event["args"]["BeginNs"] = str(int(event["args"]["BeginNs"]) + 10_000_000)
            event["args"]["EndNs"] = str(int(event["args"]["EndNs"]) + 10_000_000)
        out, counts = augment_hip_stream_waits(trace, stale)
        self.assertEqual(counts["unresolved"], 2)
        self.assertEqual(out["traceEvents"], trace["traceEvents"])

    def test_accounts_for_round_trip_without_creating_a_kernel_edge(self):
        trace, hipprof = _round_trip_workload()
        original = deepcopy(trace)
        result, counts = augment_hip_stream_waits(trace, hipprof)
        self.assertEqual(counts["round_trip_waits"], 2)
        self.assertEqual(counts["unresolved"], 0)
        self.assertEqual(result, original)
        self.assertEqual(augment_hip_stream_waits(result, hipprof)[1], counts)

    def test_requires_ordered_round_trip_on_the_same_workload_queue(self):
        for scenario in ("missing", "reversed", "different queue", "relay has work"):
            with self.subTest(scenario=scenario):
                trace, hipprof = _round_trip_workload()
                if scenario == "missing":
                    hipprof = [e for e in hipprof if e["args"]["index"] != 12]
                elif scenario == "reversed":
                    barrier = next(e for e in hipprof if e["cat"] == "BARRIER" and e["args"]["index"] == 12)
                    barrier["args"]["BeginNs"] = BASE + 33_000
                elif scenario == "different queue":
                    for e in hipprof:
                        if e["args"]["index"] == 22:
                            if e["cat"] == "HIP":
                                e["args"]["args"] = e["args"]["args"].replace("stream=0x65", "stream=0x67")
                            else:
                                e["args"]["queueId"] = "103"
                else:
                    dispatch = deepcopy(next(e for e in hipprof if e["cat"] == "HIPOPS"))
                    dispatch["args"].update(index=99, queueId="109")
                    hipprof.append(dispatch)
                result, counts = augment_hip_stream_waits(trace, hipprof)
                self.assertEqual(counts["round_trip_waits"], 0)
                self.assertEqual(counts["unresolved"], 2)
                self.assertEqual(result, trace)

    def test_stop_event_points_to_the_kernel_that_signals_completion(self):
        trace, hipprof = _completion_workload("stopEvent")
        later_launch = deepcopy(next(e for e in trace["traceEvents"] if e["cat"] == "cuda_runtime"))
        later_launch.update(name="hipLaunchKernel", ts=18, args={"correlation": 4})
        later_kernel = deepcopy(next(e for e in trace["traceEvents"] if e["cat"] == "kernel"))
        later_kernel.update(name="unrelated_kernel", ts=20, args={"correlation": 4, "stream": 1})
        trace["traceEvents"].extend([later_launch, later_kernel])
        augmented, counts = augment_hip_stream_waits(trace, hipprof)
        self.assertEqual(counts["added"], 1)
        self.assertEqual(counts["kernel_completion_waits"], 1)
        self.assertEqual(counts["unresolved"], 0)
        pairs = {(src["correlation"], dst["correlation"]) for src, dst in _stream_wait_edges(augmented)}
        self.assertEqual(pairs, {(1, 3)})

    def test_skips_start_event_or_ambiguous_kernel(self):
        trace, hipprof = _completion_workload("startEvent")
        self.assertEqual(augment_hip_stream_waits(trace, hipprof)[1]["unresolved"], 1)

        trace, hipprof = _completion_workload("stopEvent")
        kernel = next(e for e in trace["traceEvents"] if e["cat"] == "kernel")
        trace["traceEvents"].append({**kernel, "ts": kernel["ts"] + 1})
        self.assertEqual(augment_hip_stream_waits(trace, hipprof)[1]["unresolved"], 1)
