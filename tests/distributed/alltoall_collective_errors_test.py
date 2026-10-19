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

"""Tests PyTorch distributed collective all_to_all errors on multiple TPUs."""

import os
import re

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import continuous_test_utils


def run_all_to_all_single_dtype_error() -> None:
  input_tensor = torch.ones((1, 2), dtype=torch.float64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  # This message comes from native PyTorch.
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_tensor_size_error() -> None:
  input_tensor = torch.ones((1, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  expected_msg = (
      "distributed.all_to_all_single(): expected tensor first dimension to be "
      "divisible by process group size, got 1 is not divisible by 8"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_split_sizes_size_error() -> None:
  input_tensor = torch.ones((8, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((8, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [2] * 9
  output_split_sizes = [1] * 8
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes to have the same "
      "size as process group size, got 9 for split sizes and 8 for process "
      "group size"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_invalid_split_sizes_sum_error() -> None:
  input_tensor = torch.ones((8, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((8, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [1, 3, 1, 1, 1, 1, 1, 1]
  output_split_sizes = [1] * 8
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes sum to be equal to"
      " tensor first dimension, got 10 for split sizes sum and 8 for tensor"
      " shape [8, 2] dim 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_uneven_trailing_dims_mismatch_error() -> None:
  input_tensor = torch.ones((10, 4), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty((10, 8), dtype=torch.float32, device="tpu")
  input_split_sizes = [2, 1, 1, 1, 1, 1, 1, 2]
  output_split_sizes = [1, 2, 1, 1, 1, 1, 2, 1]
  expected_msg = (
      "distributed.all_to_all_single(): expected trailing dimensions of input"
      " and output to match, got [10, 4] and [10, 8]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_uneven_dim_count_mismatch_error() -> None:
  input_tensor = torch.ones((10, 4), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(10, dtype=torch.float32, device="tpu")
  input_split_sizes = [2, 1, 1, 1, 1, 1, 1, 2]
  output_split_sizes = [1, 2, 1, 1, 1, 1, 2, 1]
  expected_msg = (
      "distributed.all_to_all_single(): expected input and output tensors to"
      " have the same number of dimensions, got 2 and 1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_negative_split_size_error() -> None:
  input_tensor = torch.ones((8, 2), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty((8, 2), dtype=torch.float32, device="tpu")
  input_split_sizes = [-1, 3, 1, 1, 1, 1, 1, 1]
  output_split_sizes = [1] * 8
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes to be"
      " non-negative, got -1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_send_recv_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  input_tensor = torch.ones((9, 2), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty((10, 2), dtype=torch.float32, device="tpu")
  input_split_sizes = [2] + [1] * (world_size - 1)
  output_split_sizes = [2, 2] + [1] * (world_size - 2)
  expected_msg = re.compile(
      ".*send and recv slice size mismatch between rank.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_asymmetric_cross_rank_buffer_error() -> None:
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  if rank == 0:
    local_size = 9
    input_split_sizes = [2] + [1] * (world_size - 1)
    output_split_sizes = [2] + [1] * (world_size - 1)
  else:
    local_size = 10
    input_split_sizes = [2, 2] + [1] * (world_size - 2)
    output_split_sizes = [2, 2] + [1] * (world_size - 2)
  input_tensor = torch.ones((local_size, 2), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(
      (local_size, 2), dtype=torch.float32, device="tpu"
  )
  expected_msg = re.compile(
      ".*asymmetric .*buffer sizes across ranks are not supported on TPU.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_asymmetric_cross_rank_output_buffer_error() -> None:
  """Keeps input buffers symmetric so the output buffer check is reached."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  # Every rank allocates the same input capacity (9), so the input buffer check
  # passes and validation proceeds to the output buffer check.
  input_split_sizes = [2] + [1] * (world_size - 1)
  if rank == 0:
    output_split_sizes = [2] + [1] * (world_size - 1)
  else:
    output_split_sizes = [3] + [1] * (world_size - 1)
  input_tensor = torch.ones(
      (sum(input_split_sizes), 2), dtype=torch.float32, device="tpu"
  )
  output_tensor = torch.empty(
      (sum(output_split_sizes), 2), dtype=torch.float32, device="tpu"
  )
  expected_msg = re.compile(
      ".*asymmetric output buffer sizes across ranks are not supported on TPU.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_unequal_tensor_list_size_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected output and input tensors to have the"
      " same number of tensors, got 9 for output and 8 for input"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_wrong_num_input_tensors_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected input tensors to have the same"
      " number of tensors as the process group size, got 9 for input and 8"
      " for process group size"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_input_tensor_trailing_dims_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones((1, 4), device=device, dtype=torch.int64)
  output_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected trailing dimensions of input tensors"
      " to match, got [1, 4] and [1, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_output_tensor_trailing_dims_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  output_tensors[0] = torch.ones((1, 4), device=device, dtype=torch.int64)
  expected_msg = (
      "distributed.all_to_all(): expected trailing dimensions of output"
      " tensors to match, got [1, 4] and [1, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_dim_count_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones((1, 2), device=device, dtype=torch.int64)
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected all input tensors to have the same"
      " number of dimensions, got 2 and 1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_dtype_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones(1, device=device, dtype=torch.float32)
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def _make_ragged_all_to_all_metadata(
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Builds valid uniform offsets and slice sizes for ragged_all_to_all."""
  offsets = torch.arange(world_size, dtype=torch.int32).tpu()
  sizes = torch.ones(world_size, dtype=torch.int32).tpu()
  return offsets, sizes, offsets, sizes


def run_ragged_all_to_all_dim_count_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones(world_size, dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 2), dtype=torch.float32, device="tpu")
  expected_msg = re.compile(
      ".*expected operand and output to have the same number of dimensions.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


def run_ragged_all_to_all_trailing_dims_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones((world_size, 2), dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 3), dtype=torch.float32, device="tpu")
  expected_msg = re.compile(
      ".*expected trailing dimensions of operand and output to match.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


def run_ragged_all_to_all_dtype_mismatch_error() -> None:
  world_size = int(os.environ["WORLD_SIZE"])
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones((world_size, 2), dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 2), dtype=torch.bfloat16, device="tpu")
  expected_msg = re.compile(
      ".*expected operand and output to have the same dtype.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


class AllToAllCollectiveErrorsTest(
    continuous_test_utils.TpuMultiProcContinuousTest,
    et.TpuOnlyDistributedErrorTestBase,
):

  world_size = 8

  def test_all_to_all_single_send_recv_mismatch_error(self) -> None:
    run_all_to_all_single_send_recv_mismatch_error()

  def test_all_to_all_single_asymmetric_cross_rank_buffer_error(self) -> None:
    run_all_to_all_single_asymmetric_cross_rank_buffer_error()

  def test_all_to_all_single_asymmetric_cross_rank_output_buffer_error(
      self,
  ) -> None:
    run_all_to_all_single_asymmetric_cross_rank_output_buffer_error()

  def test_all_to_all_single_negative_split_size_error(self) -> None:
    run_all_to_all_single_negative_split_size_error()

  def test_all_to_all_single_dtype_error(self) -> None:
    run_all_to_all_single_dtype_error()

  def test_all_to_all_single_invalid_tensor_size_error(self) -> None:
    run_all_to_all_single_invalid_tensor_size_error()

  def test_all_to_all_single_invalid_split_sizes_size_error(self) -> None:
    run_all_to_all_single_invalid_split_sizes_size_error()

  def test_all_to_all_single_invalid_split_sizes_sum_error(self) -> None:
    run_all_to_all_single_invalid_split_sizes_sum_error()

  def test_all_to_all_single_uneven_trailing_dims_mismatch_error(self) -> None:
    run_all_to_all_single_uneven_trailing_dims_mismatch_error()

  def test_all_to_all_single_uneven_dim_count_mismatch_error(self) -> None:
    run_all_to_all_single_uneven_dim_count_mismatch_error()

  def test_all_to_all_unequal_tensor_list_size_error(self) -> None:
    run_all_to_all_unequal_tensor_list_size_error()

  def test_all_to_all_wrong_num_input_tensors_error(self) -> None:
    run_all_to_all_wrong_num_input_tensors_error()

  def test_all_to_all_unequal_input_tensor_trailing_dims_error(self) -> None:
    run_all_to_all_unequal_input_tensor_trailing_dims_error()

  def test_all_to_all_unequal_output_tensor_trailing_dims_error(self) -> None:
    run_all_to_all_unequal_output_tensor_trailing_dims_error()

  def test_all_to_all_dim_count_mismatch_error(self) -> None:
    run_all_to_all_dim_count_mismatch_error()

  def test_all_to_all_dtype_mismatch_error(self) -> None:
    run_all_to_all_dtype_mismatch_error()

  def test_ragged_all_to_all_dim_count_mismatch_error(self) -> None:
    run_ragged_all_to_all_dim_count_mismatch_error()

  def test_ragged_all_to_all_trailing_dims_mismatch_error(self) -> None:
    run_ragged_all_to_all_trailing_dims_mismatch_error()

  def test_ragged_all_to_all_dtype_mismatch_error(self) -> None:
    run_ragged_all_to_all_dtype_mismatch_error()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
