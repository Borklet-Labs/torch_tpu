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

"""Tests collectives inside a compiled scan.

A scan op carries its body's device assignment, so the outer program runs in
the same device order as the collectives inside the scan body.
"""

from collections.abc import Sequence

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch._higher_order_ops import scan
from torch.distributed import _functional_collectives as funcol
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _assert_scan_all_gather(
    group: dist.ProcessGroup, expected_ranks: Sequence[int]
) -> None:
  """Asserts scanned all-gathers over `group` follow `expected_ranks` order."""
  rank = dist.get_rank()
  steps = 2

  def gather_step(carry: torch.Tensor, x: torch.Tensor):
    total = carry + funcol.all_gather_tensor(x, 0, group)
    return total, total.clone()

  compiled_scan = torch.compile(scan, backend="tpu", dynamic=False)
  xs = torch.full((steps, 1), float(rank), device="tpu")
  init = torch.zeros((len(expected_ranks),), device="tpu")
  carry, _ = compiled_scan(gather_step, init, xs, dim=0)

  # Block i holds the inputs of the rank expected_ranks[i], summed over steps.
  expected = steps * torch.tensor(expected_ranks, dtype=torch.float32)
  utils.assert_close(carry.cpu(), expected)


def run_scan_all_gather_world() -> None:
  """A scan of world all-gathers gathers the blocks in world rank order."""
  dist.init_process_group(backend="tpu_dist")
  _assert_scan_all_gather(dist.group.WORLD, list(range(dist.get_world_size())))


# Each column holds one core of each two-core chip, so on v7x the group rank
# order differs from the device order.
_COLUMNS = [[0, 2, 4, 6], [1, 3, 5, 7]]


def run_scan_all_gather_group() -> None:
  """A scan of subgroup all-gathers gathers the blocks in group rank order."""
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  # Every rank creates every group, in the same order.
  groups = [dist.new_group(ranks=ranks) for ranks in _COLUMNS]
  my_ranks = next(ranks for ranks in _COLUMNS if rank in ranks)
  _assert_scan_all_gather(groups[_COLUMNS.index(my_ranks)], my_ranks)


def run_scan_without_collectives() -> None:
  """A scan without collectives still runs correctly under a process group."""
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  steps = 3

  def local_step(carry: torch.Tensor, x: torch.Tensor):
    carry = carry * 2 + x
    return carry, carry.clone()

  local_scan = torch.compile(scan, backend="tpu", dynamic=False)
  xs_cpu = torch.stack(
      [torch.full((8, 128), float(rank + 10 * s)) for s in range(steps)]
  )
  init_cpu = torch.full((8, 128), float(rank))
  carry, ys = local_scan(
      local_step, init_cpu.to("tpu"), xs_cpu.to("tpu"), dim=0
  )

  expected_carry = init_cpu
  expected_ys = []
  for s in range(steps):
    expected_carry, y = local_step(expected_carry, xs_cpu[s])
    expected_ys.append(y)
  utils.assert_close(carry.cpu(), expected_carry)
  utils.assert_close(ys.cpu(), torch.stack(expected_ys))


class ScanCollectivesTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_scan_gathers_in_world_rank_order(self):
    distributed_utils.dist_run(nproc_per_node=8, fn=run_scan_all_gather_world)

  def test_scan_gathers_in_group_rank_order(self):
    distributed_utils.dist_run(nproc_per_node=8, fn=run_scan_all_gather_group)

  def test_scan_without_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=8, fn=run_scan_without_collectives
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
