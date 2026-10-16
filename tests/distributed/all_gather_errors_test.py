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

"""Tests PyTorch collective all_gather errors on multiple TPUs."""

import re

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils


class AllGatherErrorsTest(
    continuous_test_utils.TpuMultiProcContinuousTest,
    et.TpuOnlyDistributedErrorTestBase,
):

  world_size = distributed_utils.get_tpu_device_count()

  def test_all_gather_dtype_error(self) -> None:
    input_tensor = torch.ones((1, 2), dtype=torch.float64, device="tpu")
    output_tensors = [
        torch.empty((1, 2), dtype=torch.int64, device="tpu")
        for _ in range(self.world_size)
    ]
    # This message comes from native PyTorch.
    expected_msg = re.compile(
        "Invalid usage of tensors with different dtypes.*"
    )
    with et.assert_raises_message(ValueError, tpu=expected_msg):
      dist.all_gather(output_tensors, input_tensor)

  def test_all_gather_wrong_num_tensors(self) -> None:
    input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
    output_tensors = [torch.empty((1, 2), dtype=torch.float32, device="tpu")]
    # This message comes from native PyTorch.
    expected_msg = re.compile(
        r"distributed\.all_gather\(\): expected output tensor list to have one"
        rf" tensor per process, got 1 tensor and {self.world_size} processes.*"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.all_gather(output_tensors, input_tensor)

  def test_all_gather_dispatch_failure(self) -> None:
    input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
    output_tensors = [
        torch.empty((1, 2), dtype=torch.float32, device="tpu")
        for _ in range(self.world_size)
    ]
    tt_testing.set_op_dispatch_failure(
        "distributed.all_gather", "forced dispatch failure"
    )
    try:
      expected_msg = re.compile(
          r"distributed\.all_gather\(\): forced dispatch failure.*"
      )
      with et.assert_raises_message(RuntimeError, tpu=expected_msg):
        dist.all_gather(output_tensors, input_tensor)
    finally:
      tt_testing.set_op_dispatch_failure("", "")


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
