# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, TYPE_CHECKING

import pandas as pd
import plotly.express as px
from hta.utils.utils import get_kernel_type, KernelType, merge_kernel_intervals

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.common.trace import Trace


class CommunicationAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_comm_comp_overlap(
        cls, t: "Trace", visualize: bool = True, include_times: bool = False
    ) -> pd.DataFrame:
        """
        Communication analysis implementation. See `get_comm_comp_overlap` in `trace_analysis.py` for details.
        """
        sym_table = t.symbol_table.get_sym_table()

        def get_comm_comp_overlap_times(
            trace_df: pd.DataFrame,
        ) -> tuple[float, float]:
            """
            Return the communication and overlapping durations, in microseconds.
            """
            gpu_kernels = trace_df.loc[t.symbol_table.get_gpu_kernel_mask(trace_df)].copy()
            names = gpu_kernels["name"].map(sym_table.__getitem__)
            gpu_kernels["kernel_type"] = names.map(get_kernel_type)

            # Isolate communication and computation kernels and merge each one of them.
            comp_kernels = merge_kernel_intervals(
                gpu_kernels[
                    gpu_kernels["kernel_type"].eq(KernelType.COMPUTATION.name)
                ].copy()
            )
            comm_kernels = merge_kernel_intervals(
                gpu_kernels[
                    gpu_kernels["kernel_type"].eq(KernelType.COMMUNICATION.name)
                ].copy()
            )
            communication_time = (comm_kernels["end"] - comm_kernels["ts"]).sum()
            if communication_time == 0:
                return 0.0, 0.0

            # When a communication kernel starts and ends, the cumulative status is changed by 1 and -1;
            # when a computation kernel starts and ends, the cumulative status is changed by 2 and -2.
            status_df = (
                pd.concat(
                    [
                        comm_kernels.melt(var_name="status", value_name="time").replace(
                            {"ts": 1, "end": -1}
                        ),
                        comp_kernels.melt(var_name="status", value_name="time").replace(
                            {"ts": 2, "end": -2}
                        ),
                    ]
                )
                .sort_values(by="time")
                .reset_index(drop=True)
            )
            status_df["running"] = status_df["status"].cumsum()
            # Time intervals when status is 3 indicate overlapping communication and computation kernels.
            overlap = status_df[status_df["running"].eq(3)]
            shifted_overlap = overlap.merge(
                status_df.shift(-1).dropna(), left_index=True, right_index=True
            )
            overlap_time = (shifted_overlap["time_y"] - shifted_overlap["time_x"]).sum()
            return communication_time, overlap_time

        result: Dict[str, List[float]] = defaultdict(list)
        for rank, trace_df in t.traces.items():
            communication_time, overlap_time = get_comm_comp_overlap_times(trace_df)
            result["rank"].append(rank)
            result["communication_time_us"].append(communication_time)
            result["overlap_time_us"].append(overlap_time)
            result["comp_comm_overlap_ratio"].append(
                overlap_time / communication_time if communication_time else float("nan")
            )
        result_df = pd.DataFrame(result)
        result_df["comp_comm_overlap_pctg"] = round(
            100 * result_df["comp_comm_overlap_ratio"], 2
        )

        if visualize:  # pragma: no cover
            fig = px.bar(
                result_df,
                x="rank",
                y="comp_comm_overlap_ratio",
                title="Computation-Communication Overlap",
                labels={
                    "rank": "Rank",
                    "comp_comm_overlap_ratio": (
                        "Computation-Communication Overlap Percentage"
                    ),
                },
            )

            fig.update_layout(yaxis_tickformat=".2%")
            fig.show()

        columns = ["rank", "comp_comm_overlap_pctg"]
        if include_times:
            columns += ["communication_time_us", "overlap_time_us"]
        return result_df[columns]
