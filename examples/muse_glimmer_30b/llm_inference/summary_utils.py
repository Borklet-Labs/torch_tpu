# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for formatting and printing benchmark latency summary tables."""

from collections.abc import Mapping
from typing import Any


def print_summary_table(
    all_metrics: Mapping[str, Mapping[str, Any]], num_decode_steps: int
) -> None:
  """Prints a markdown summary table of prefill and decode latency metrics."""
  headers = ["Metric"] + list(all_metrics.keys())
  table_data: list[list[Any]] = []

  prefill_row: list[Any] = ["Prefill time (ms)"]
  for _, metrics in all_metrics.items():
    prefill_row.append(metrics["prefill_time_ms"])
  table_data.append(prefill_row)

  for i in range(num_decode_steps):
    step_row: list[Any] = [f"Decode step {i + 1} (ms)"]
    for _, metrics in all_metrics.items():
      step_times = metrics["decode_step_times_ms"]
      if i < len(step_times):
        step_row.append(step_times[i])
      else:
        step_row.append(None)
    table_data.append(step_row)

  avg_row: list[Any] = ["Avg decode time / token (ms)"]
  for _, metrics in all_metrics.items():
    avg_row.append(metrics["avg_decode_time_per_token_ms"])
  table_data.append(avg_row)

  avg_row = ["Avg decode time / token (no warmup) (ms)"]
  for _, metrics in all_metrics.items():
    avg_row.append(metrics["avg_decode_time_per_token_no_warmup_ms"])
  table_data.append(avg_row)

  print("\n=== Latency Summary Table ===")

  def format_cell(val):
    if val is None:
      return ""
    if isinstance(val, float):
      return f"{val:.2f}"
    return str(val)

  formatted_headers = [format_cell(h) for h in headers]
  formatted_rows = [[format_cell(cell) for cell in row] for row in table_data]

  col_widths = [len(h) for h in formatted_headers]
  for row in formatted_rows:
    for i, cell in enumerate(row):
      col_widths[i] = max(col_widths[i], len(cell))

  header_line = (
      "| "
      + " | ".join(h.ljust(w) for h, w in zip(formatted_headers, col_widths))
      + " |"
  )
  print(header_line)

  separator_parts = []
  for i, w in enumerate(col_widths):
    if i == 0:
      separator_parts.append(":" + "-" * (w - 1) if w > 0 else "")
    else:
      separator_parts.append("-" * (w - 1) + ":" if w > 0 else "")
  separator_line = "| " + " | ".join(separator_parts) + " |"
  print(separator_line)

  for row in formatted_rows:
    row_cells = []
    for i, cell in enumerate(row):
      if i == 0:
        row_cells.append(cell.ljust(col_widths[i]))
      else:
        row_cells.append(cell.rjust(col_widths[i]))
    print("| " + " | ".join(row_cells) + " |")
  print(flush=True)
