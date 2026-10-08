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

"""Unit tests for the synthetic distributed host dispatch benchmarks."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.distributed_models import synthetic
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD_SIZE = 2
_NUM_PARAMS = 16
_NUM_OUTPUTS = 16


def _worker_cpu_training_step() -> None:
  """One SimpleFSDP training step of the synthetic model on gloo / CPU."""
  torch.distributed.init_process_group(backend="gloo", init_method="env://")
  try:
    model = synthetic.build_fsdp_model(
        _NUM_PARAMS, "cpu", _WORLD_SIZE, torch.float32
    )
    assert (
        len(list(model.parameters())) == _NUM_PARAMS
    ), f"Expected {_NUM_PARAMS} parameters"
    inputs = synthetic.make_input(
        torch.distributed.get_rank(), torch.float32, "cpu"
    )
    # A plain optimizer: `capturable=True` (used by the benchmark) is TPU / GPU
    # only.
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = [p.detach().to_local().clone() for p in model.parameters()]

    loss = model(inputs)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss), f"Non-finite loss {loss}"
    after = [p.detach().to_local() for p in model.parameters()]
    assert all(
        not torch.equal(b, a) for b, a in zip(before, after)
    ), "Not every parameter changed after the optimizer step"
  finally:
    torch.distributed.destroy_process_group()


def _worker_cpu_replicated_forward() -> None:
  """One forward of the replicated model on gloo / CPU."""
  torch.distributed.init_process_group(backend="gloo", init_method="env://")
  try:
    model = synthetic.build_replicated_model(_NUM_OUTPUTS, "cpu", torch.float32)
    rank = torch.distributed.get_rank()

    with torch.no_grad():
      outputs = model(synthetic.make_input(rank, torch.float32, "cpu"))

    assert (
        len(outputs) == _NUM_OUTPUTS + 1
    ), f"Expected {_NUM_OUTPUTS + 1} outputs, got {len(outputs)}"
    expected_sync = sum(
        synthetic.make_input(r, torch.float32, "cpu").sum()
        for r in range(_WORLD_SIZE)
    )
    test_utils.assert_close(outputs[-1], expected_sync)
  finally:
    torch.distributed.destroy_process_group()


def _ctx(
    platform: target_lib.Platform, run_scope: context_lib.RunScope
) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(platform), run_scope=run_scope
  )


class DistributedSyntheticBenchmarkTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def test_registers_all_benchmarks(self):
    for n in (16, 256, 1024):
      self.assertIn(
          f"synthetic_replicated_{n}_outputs_forward", registry_lib.REGISTRY
      )
      self.assertIn(
          f"synthetic_fsdp_{n}_params_training", registry_lib.REGISTRY
      )

  @parameterized.parameters(
      "synthetic_replicated_16_outputs_forward",
      "synthetic_fsdp_16_params_training",
  )
  def test_rejects_unsupported_platforms(self, name):
    factory = registry_lib.REGISTRY[name].factory
    for platform in (
        target_lib.Platform.V7_1X1X1,
        target_lib.Platform.V7_2X2X2,
        target_lib.Platform.B200_2,
    ):
      with self.assertRaisesRegex(
          target_lib.UnsupportedBenchmark, "do not support"
      ):
        factory(_ctx(platform, context_lib.RunScope.PRESUBMIT))

  @parameterized.parameters(
      "synthetic_replicated_256_outputs_forward",
      "synthetic_replicated_1024_outputs_forward",
      "synthetic_fsdp_256_params_training",
      "synthetic_fsdp_1024_params_training",
  )
  def test_presubmit_skips_large_sweeps(self, name):
    factory = registry_lib.REGISTRY[name].factory
    with self.assertRaisesRegex(
        target_lib.UnsupportedBenchmark, "only run in the full scope"
    ):
      factory(
          _ctx(target_lib.Platform.V7_1X1X1_2TC, context_lib.RunScope.PRESUBMIT)
      )

  @parameterized.parameters(
      (
          "synthetic_replicated_16_outputs_forward",
          context_lib.RunScope.PRESUBMIT,
      ),
      ("synthetic_fsdp_16_params_training", context_lib.RunScope.PRESUBMIT),
      ("synthetic_fsdp_1024_params_training", context_lib.RunScope.FULL),
  )
  def test_requires_initialized_process_group(self, name, run_scope):
    factory = registry_lib.REGISTRY[name].factory
    with self.assertRaisesRegex(
        target_lib.UnsupportedBenchmark, "initialized torch.distributed"
    ):
      factory(_ctx(target_lib.Platform.V7_1X1X1_2TC, run_scope))

  def test_simple_fsdp_training_step_ws2_cpu(self):
    distributed_utils.dist_run(_WORLD_SIZE, _worker_cpu_training_step)

  def test_replicated_forward_ws2_cpu(self):
    distributed_utils.dist_run(_WORLD_SIZE, _worker_cpu_replicated_forward)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
