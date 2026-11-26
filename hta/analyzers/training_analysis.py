# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from hta.common.types import ProfilerStepGroupingPattern

if TYPE_CHECKING:
    from hta.common.trace import Trace


_TRAINING_MARKERS = {
    "BERT.forward_and_loss": ("BERT", "前向传播（含损失计算）"),
    "BERT.backward": ("BERT", "反向传播"),
    "BERT.optimizer": ("BERT", "参数更新"),
    "ResNet50.forward": ("ResNet50", "前向传播（含损失计算）"),
    "ResNet50.backward": ("ResNet50", "反向传播"),
    "ResNet50.optimizer": ("ResNet50", "参数更新"),
}
_RESNET_STEP = re.compile(
    r"^Epoch:\s*\[(\d+)\]\s*\[(\d+)/(\d+)\]\s+Time\s+([^\s(]+).*?\bLoss\s+([^\s(]+)"
)
_MEGATRON_STEP = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?iteration\s+(\d+)\s*/\s*\d+\s*\|")
_MEGATRON_TFLOPS = "throughput per GPU (TFLOP/s/GPU)"


class TrainingLoss(NamedTuple):
    step: int
    epoch: int | None
    step_loss: float | None
    final_loss: float | None


class TrainingThroughput(NamedTuple):
    step: int
    elapsed_ms: float
    items_per_step: float
    rate: float


class TrainingLog(NamedTuple):
    model: str
    losses: list[TrainingLoss]
    throughput: list[TrainingThroughput]
    unit: str
    rate_source: str = ""


def _logged_number(value: str | int | float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resnet_rates(log: Path, timing: list[tuple[int, int, int, float]]) -> list[TrainingThroughput]:
    capture = log.with_name("capture.json")
    if not capture.is_file():
        return []
    metadata = json.loads(capture.read_text(encoding="utf-8-sig"))
    batch_size, images = metadata.get("batch_size"), metadata.get("train_images")
    if not (isinstance(batch_size, int) and batch_size > 0 and isinstance(images, int) and images > 0):
        return []
    rows = []
    for step, batch, batches, seconds in timing:
        if batches != math.ceil(images / batch_size) or not 0 <= batch < batches:
            raise ValueError(f"ResNet50 batch counts do not match {capture}")
        actual_batch = min(batch_size, images - batch * batch_size)
        rows.append(TrainingThroughput(step, seconds * 1000, actual_batch, actual_batch / seconds))
    return rows


def _megatron_rates(timing, losses) -> TrainingLog:
    has_logged_tflops = any(logged is not None for _, _, _, _, logged in timing)
    rows = []
    prior_step, prior_consumed = 0, None
    for step, elapsed, batch_size, consumed, logged in timing:
        global_samples = batch_size
        if prior_consumed is not None and consumed is not None and step > prior_step:
            global_samples = (consumed - prior_consumed) / (step - prior_step)
        prior_step, prior_consumed = step, consumed
        rate = logged if has_logged_tflops else 1000 * global_samples / elapsed
        if rate is not None and rate >= 0:
            rows.append(TrainingThroughput(step, elapsed, global_samples, rate))
    if has_logged_tflops:
        return TrainingLog("Megatron", losses, rows, "TFLOP/s/HCU",
                           "直接读取 Megatron --log-throughput 日志的每卡值，沿用日志显示精度。")
    return TrainingLog("Megatron", losses, rows, "样本/秒", "全局每步处理样本数 ÷ 日志的每步耗时。")


def read_training_log(log: Path) -> TrainingLog:
    """Read losses and measured or derivable throughput for one training run."""
    losses = []
    resnet_timing = []
    megatron_timing = []
    bert_rates = []
    model = ""
    with log.open(encoding="utf-8-sig") as handle:
        for line in handle:
            final = None
            if match := _RESNET_STEP.match(line):
                model = "ResNet50"
                epoch, batch, batches = map(int, match.groups()[:3])
                step = epoch * batches + batch + 1
                seconds = _logged_number(match.group(4))
                if seconds and seconds > 0:
                    resnet_timing.append((step, batch, batches, seconds))
                loss = _logged_number(match.group(5))
            elif match := _MEGATRON_STEP.match(line):
                model = "Megatron"
                epoch, step = None, int(match.group(1))
                fields = {
                    key.strip(): value.strip()
                    for key, separator, value in (part.partition(":") for part in line[match.end():].split("|"))
                    if separator
                }
                elapsed = _logged_number(fields.get("elapsed time per iteration (ms)"))
                batch_size = _logged_number(fields.get("global batch size"))
                consumed = _logged_number(fields.get("consumed samples"))
                tflops = _logged_number(fields.get(_MEGATRON_TFLOPS))
                if elapsed and elapsed > 0 and batch_size and batch_size > 0:
                    megatron_timing.append((step, elapsed, batch_size, consumed, tflops))
                loss = _logged_number(fields.get("lm loss"))
            elif line.startswith("DLLL "):
                record = json.loads(line.removeprefix("DLLL "))
                indices, data = record.get("step"), record.get("data")
                if not isinstance(indices, list) or not isinstance(data, dict):
                    continue
                model = "BERT"
                if len(indices) != 2:
                    continue
                epoch, step = indices
                if not isinstance(epoch, int) or not isinstance(step, int):
                    continue
                sequences = _logged_number(data.get("training_step_sequences"))
                seconds = _logged_number(data.get("training_step_seconds"))
                if sequences and seconds and sequences > 0 and seconds > 0:
                    if bert_rates and step <= bert_rates[-1].step:
                        raise ValueError(f"Expected increasing BERT throughput steps in one run: {log}")
                    bert_rates.append(TrainingThroughput(step, seconds * 1000, sequences, sequences / seconds))
                loss = _logged_number(data.get("step_loss"))
                final = _logged_number(data.get("final_loss"))
            else:
                continue
            if loss is None and final is None:
                continue
            if losses and step <= losses[-1].step:
                raise ValueError(f"Expected increasing training steps in one run: {log}")
            losses.append(TrainingLoss(step, epoch, loss, final))

    if model == "Megatron":
        result = _megatron_rates(megatron_timing, losses)
    elif model == "ResNet50":
        result = TrainingLog(model, losses, _resnet_rates(log, resnet_timing), "图像/秒",
                             "每批实际图像数 ÷ 日志的当前批 Time。图像数来自同次采集的 capture.json。")
    else:
        result = TrainingLog(model, losses, bert_rates, "序列/秒",
                             "每步的实际序列数 ÷ 从取得批次到 HCU 完成参数更新的实测耗时。")
    if not result.losses and not result.throughput:
        raise ValueError(f"No recognized training metrics in {log}. Use a Megatron, BERT DLLLogger, or ResNet50 log.")
    return result


def get_training_phase_spans(trace: "Trace") -> pd.DataFrame:
    """Summarize explicit CPU stages and the HCU kernels submitted in them."""
    symbols = trace.symbol_table
    names = symbols.get_sym_table()
    ids = symbols.get_sym_id_map()
    phase_ids = {ids[name] for name in _TRAINING_MARKERS if name in ids}
    step_ids = set(symbols.get_profiler_step_ids())
    runtime_ids = {ids[name] for name in ("cuda_runtime", "cuda_driver", "hip_runtime") if name in ids}
    rows = []

    for rank in trace.get_ranks():
        frame = trace.get_trace(rank)
        annotations = frame.loc[
            frame["cat"].eq(ids.get("user_annotation", -1)) & frame["dur"].gt(0),
            ["pid", "tid", "name", "ts", "dur"],
        ].sort_values("ts", kind="stable")
        marks = annotations.loc[annotations["name"].isin(phase_ids)].drop_duplicates()
        if marks.empty:
            continue

        steps = annotations.loc[annotations["name"].isin(step_ids)].rename(
            columns={"ts": "step_ts", "dur": "step_dur", "name": "step_id"},
        )
        marks = pd.merge_asof(
            marks, steps, left_on="ts", right_on="step_ts", by=["pid", "tid"],
        )

        by_process = {}
        if "index_correlation" in frame:
            kernels = frame.loc[
                frame["cat"].eq(ids.get("kernel", -1))
                & frame["index_correlation"].gt(0) & frame["dur"].gt(0),
                ["index_correlation", "dur"],
            ]
            runtime = frame.loc[frame["cat"].isin(runtime_ids), ["index", "pid", "ts", "correlation"]]
            runtime = runtime.drop_duplicates(subset="correlation", keep=False).set_index("index")
            launches = kernels.join(runtime, on="index_correlation", how="inner").sort_values("ts")
            by_process = {pid: group for pid, group in launches.groupby("pid", sort=False)}

        for pid, spans in marks.groupby("pid", sort=False):
            launches = by_process.get(pid)
            # Concurrent stage marks in one process cannot unambiguously own its submissions.
            if spans["ts"].lt((spans["ts"] + spans["dur"]).cummax().shift()).any():
                launches = None
            for span in spans.itertuples(index=False):
                annotation = names[span.name]
                model, phase = _TRAINING_MARKERS[annotation]
                step = ""
                if span.ts + span.dur <= span.step_ts + span.step_dur:
                    marker = ProfilerStepGroupingPattern.pattern.match(names[int(span.step_id)])
                    if marker:
                        step = f"ProfilerStep#{marker.group(1)}"
                kernel_count, kernel_us = None, None
                if launches is not None:
                    first, last = launches["ts"].searchsorted([span.ts, span.ts + span.dur])
                    if last > first:
                        kernel_count = int(last - first)
                        kernel_us = launches["dur"].iloc[first:last].sum()
                rows.append((rank, model, phase, step, span.ts, span.dur, kernel_count, kernel_us, annotation))

    return pd.DataFrame(rows, columns=[
        "rank", "model", "phase", "step", "start_us", "cpu_us", "kernel_count", "kernel_us", "annotation",
    ]).sort_values(["rank", "start_us"], ignore_index=True)
