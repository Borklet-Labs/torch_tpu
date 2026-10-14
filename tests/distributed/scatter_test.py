# Copyright 2025 Google LLC
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

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils


class ScatterTest(continuous_test_utils.TpuMultiProcContinuousTest):

  @property
  def world_size(self) -> int:
    return 8

  def test_scatter_tensor(self) -> None:
    """Tests scatter functionality."""
    src_rank = 0
    size = 5
    inputs = None
    if self.rank == src_rank:
      inputs = [
          torch.full((size,), i, device="tpu", dtype=torch.float32)
          for i in range(self.world_size)
      ]
    output = torch.empty(size, device="tpu", dtype=torch.float32)

    torch.distributed.scatter(output, inputs, src=src_rank)
    expected = torch.full((size,), fill_value=self.rank, dtype=torch.float32)
    utils.assert_close(output.cpu(), expected)

  def test_scatter_scalar(self) -> None:
    """Tests scatter on scalar input."""
    src_rank = 0
    inputs = None
    if self.rank == src_rank:
      inputs = [
          torch.tensor(i, device="tpu", dtype=torch.float32)
          for i in range(self.world_size)
      ]
    output = torch.tensor(-1, device="tpu", dtype=torch.float32)

    torch.distributed.scatter(output, inputs, src=src_rank)
    expected = torch.tensor(self.rank, dtype=torch.float32)
    utils.assert_close(output.cpu(), expected)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
