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

"""Tests for ragged_dot in a distributed environment on TPU."""

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils


def manual_ragged_dot(x, w, gs):
  """Reference implementation of ragged_dot."""
  out = torch.zeros(x.shape[0], w.shape[-1])
  idx = 0
  for i, g in enumerate(gs):
    if g == 0:
      continue
    out[idx : (idx + g), :] = x[idx : (idx + g), :] @ w[i]
    idx += g
  return out


class RaggedDotDistributedTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):
  """Tests for ragged_dot in a distributed environment on TPU.

  Note that this test is not testing the distributed version of ragged_dot.
  Instead it is testing that the local version of ragged_dot works correctly in
  a distributed environment.
  """

  world_size = 8

  def _run_ragged_dot_local_test(self, compile_test: bool = False) -> None:
    """Runs ragged_dot independently on each rank."""
    rank = self.rank

    # Different shapes/inputs for each rank based on rank.
    b_fused = (rank + 1) * 8
    d_m = 16
    d_f = 32
    num_experts = 4

    torch.manual_seed(42 + rank)
    h = torch.randn(b_fused, d_m, device="tpu")
    weights = torch.randn(num_experts, d_m, d_f, device="tpu")

    # Distribute h across experts.
    # Simple case: each expert gets 1/num_experts of the tokens.
    gs_list = [b_fused // num_experts] * num_experts
    group_sizes = torch.tensor(gs_list, dtype=torch.int32, device="tpu")

    def op_fn(x, w, gs):
      return torch.tpu.ops.ragged_dot(x, w, gs)

    if compile_test:
      compiled_op = torch.compile(
          op_fn, backend="tpu", options={"serializable": False}
      )
      out = compiled_op(h, weights, group_sizes)
    else:
      out = op_fn(h, weights, group_sizes)

    # Compare against CPU reference
    expected = manual_ragged_dot(
        h.cpu(), weights.cpu(), group_sizes.cpu().tolist()
    )
    utils.assert_close(
        out.cpu(),
        expected,
        atol=5e-2,
        rtol=1e-1,
    )

  def test_ragged_dot_independent_eager(self):
    self._run_ragged_dot_local_test(compile_test=False)

  def test_ragged_dot_independent_compiled(self):
    self._run_ragged_dot_local_test(compile_test=True)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
