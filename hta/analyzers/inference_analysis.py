# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from hta.common.trace import Trace


_VLLM_STEP = re.compile(r"execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)")
_SGLANG_STEP = re.compile(r"step\[(EXTEND|DECODE|MIXED)(?: [^\]]*)?\]")
_SGLANG_PHASES = {"EXTEND": "Prefill", "DECODE": "Decode", "MIXED": "Prefill + Decode"}


def _inference_phase(name: str) -> tuple[str, str] | None:
    if match := _VLLM_STEP.fullmatch(name):
        context, generation = map(int, match.groups())
        if context and generation:
            return "vLLM", "Prefill + Decode"
        if context:
            return "vLLM", "Prefill"
        if generation:
            return "vLLM", "Decode"
    elif match := _SGLANG_STEP.fullmatch(name):
        return "SGLang", _SGLANG_PHASES[match.group(1)]
    return None


def get_inference_phase_spans(trace: "Trace") -> pd.DataFrame:
    """Return recognized vLLM/SGLang CPU phase annotations, in microseconds."""
    symbols = trace.symbol_table
    names = symbols.get_sym_table()
    user_annotation = symbols.get_sym_id_map().get("user_annotation", -1)
    phase_by_symbol = {
        index: phase
        for index, name in enumerate(names)
        if (phase := _inference_phase(name)) is not None
    }
    rows = []
    for rank in trace.get_ranks():
        frame = trace.get_trace(rank)
        marks = frame.loc[
            frame["cat"].eq(user_annotation)
            & frame["name"].isin(phase_by_symbol)
            & frame["dur"].gt(0),
            ["name", "ts", "dur"],
        ].sort_values("ts", kind="stable")
        counters: Counter[tuple[str, str]] = Counter()
        for index, start, duration in marks.itertuples(index=False, name=None):
            framework, phase = phase_by_symbol[index]
            counters[framework, phase] += 1
            rows.append((rank, framework, phase, counters[framework, phase], start, duration, names[index]))
    return pd.DataFrame(rows, columns=[
        "rank", "framework", "phase", "phase_index", "start_us", "duration_us", "annotation",
    ])
