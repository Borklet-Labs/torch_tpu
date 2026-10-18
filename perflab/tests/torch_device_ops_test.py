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

"""Tests for torch_device_ops."""

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab import device_utils
from perflab import mode as mode_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from tests import seed_test_utils


class TorchDeviceOpsTest(seed_test_utils.RepeatableTest):

  def test_init_success(self):
    target = target_lib.make_target(
        platform=target_lib.Platform.V5E_1X1,
        dtype=target_lib.DType.BF16,
    )
    ops = torch_device_ops.TorchDeviceOps(target)
    self.assertEqual(ops.device.type, "tpu")
    self.assertEqual(ops.dtype, torch.bfloat16)

  @mock.patch("torch.accelerator")
  def test_init_device_count_mismatch(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 0
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        torch_device_ops.DeviceCountMismatch,
        r"expects 1 device\(s\) but the host has 0",
    ):
      torch_device_ops.TorchDeviceOps(target)

  @mock.patch("torch.accelerator")
  def test_await_result(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.await_result(None)

    mock_accelerator.synchronize.assert_called_once()

  @mock.patch.object(device_utils, "synchronize", autospec=True)
  def test_check_result_raises_if_output_execution_failed(
      self, mock_synchronize
  ):
    # A failed asynchronous execution (e.g. an HBM OOM) leaves the device
    # barrier successful and only marks the outputs as failed.
    mock_synchronize.side_effect = RuntimeError("RESOURCE_EXHAUSTED")
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "RESOURCE_EXHAUSTED"
    ):
      ops.check_result({"logits": torch.ones(4, device=ops.device)})

  @mock.patch.object(device_utils, "synchronize", autospec=True)
  def test_check_result_raises_without_tpu_outputs(self, mock_synchronize):
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    for out in (None, {"n": 3}, torch.ones(4)):
      with self.subTest(out=out):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
            ValueError,
            "returned no TPU tensors, .* a failed TorchTPU execution only"
            " marks its outputs as failed",
        ):
          ops.check_result(out)
    mock_synchronize.assert_not_called()

  @mock.patch.object(device_utils, "synchronize", autospec=True)
  def test_check_result_validates_cpu_outputs_without_tpu_sync(
      self, mock_synchronize
  ):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.check_result({"logits": torch.ones(4, device=ops.device)})
    # As in the FakeTensorMode CPU smoke test.
    with mode_lib.fake_tensor_context():
      ops.check_result({"logits": torch.ones(4, device=ops.device)})

    for bad_out in (None, {"n": 3}):
      with self.subTest(bad_out=bad_out):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
            ValueError,
            "returned no CPU tensors, .* the TPU run checks the step's outputs"
            " to detect failures",
        ):
          ops.check_result(bad_out)

    mock_synchronize.assert_not_called()

  @mock.patch.object(device_utils, "synchronize", autospec=True)
  @mock.patch("torch.accelerator")
  def test_check_result_ignores_cuda_targets(
      self, mock_accelerator, mock_synchronize
  ):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.check_result({"logits": mock.sentinel.logits})
    ops.check_result(None)

    mock_synchronize.assert_not_called()

  def test_check_result_succeeds_for_real_tpu_outputs(self):
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)
    # Non-TPU leaves (a CPU tensor, an int) are ignored.
    out = {
        "x": torch.ones(4, device=ops.device) + 1,
        "cpu": torch.ones(2),
        "n": 3,
    }
    ops.await_result(out)

    ops.check_result(out)

    test_utils.assert_close(out["x"].cpu(), torch.full((4,), 2.0))

  @mock.patch("torch.accelerator")
  def test_reset_peak_memory(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.reset_peak_memory()

    mock_accelerator.reset_peak_memory_stats.assert_called_once()

  @mock.patch("torch.accelerator")
  def test_peak_memory_mb(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    mock_accelerator.max_memory_allocated.return_value = 5242880  # 5 * 2**20

    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)
    peak_mb = ops.peak_memory_mb()

    self.assertEqual(peak_mb, 5.0)
    mock_accelerator.max_memory_allocated.assert_called_once()

  @mock.patch("torch.accelerator")
  def test_compile_count_dynamo(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(
        ops, "_dynamo_counters", return_value={"frames": {"ok": 42}}
    ):
      count = ops.compile_count()

    self.assertEqual(count, 42)

  def test_compile_count_dynamo_real_compile(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)

    ops = torch_device_ops.TorchDeviceOps(target)
    model = torch.nn.Linear(2, 2)
    compiled_model = torch.compile(model, backend="inductor")
    _ = compiled_model(torch.randn(2, 2))

    self.assertGreater(ops.compile_count(), 0)

  def test_compile_count_dynamo_none(self):
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(ops, "_dynamo_counters", return_value=None):
      count = ops.compile_count()

    self.assertEqual(count, 0)

  @mock.patch("torch.accelerator")
  def test_compile_count_tpu(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(torch, "tpu", create=True) as mock_tpu:
      mock_tpu._get_cache_misses.return_value = 7
      count = ops.compile_count()

    self.assertEqual(count, 7)

  @mock.patch("torch.accelerator")
  def test_measure_step_device_time_cuda_merges_intervals_and_excludes_gaps(
      self, mock_accelerator
  ):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)
    ops = torch_device_ops.TorchDeviceOps(target)

    self.assertTrue(ops.supports_event_device_time())
    call_order = []
    mock_accelerator.synchronize.side_effect = lambda: call_order.append("sync")

    def _make_event(
        start_ns,
        end_ns,
        *,
        device_type=torch.autograd.DeviceType.CUDA,
        device_index=0,
        name="kernel",
        activity_type="kernel",
        is_user_annotation=False,
    ):
      ev = mock.MagicMock()
      ev.start_ns.return_value = start_ns
      ev.end_ns.return_value = end_ns
      ev.device_type.return_value = device_type
      ev.device_index.return_value = device_index
      ev.name.return_value = name
      ev.activity_type.return_value = activity_type
      ev.is_user_annotation.return_value = is_user_annotation
      ev.is_hidden_event.return_value = False
      return ev

    # Kernel 1: [1ms, 5ms] (4ms)
    # Overlapping stream kernel 2: [3ms, 7ms] -> merged [1ms, 7ms] = 6ms
    # Host idle gap: [7ms, 15ms] (8ms gap excluded!)
    # Kernel 3: [15ms, 19ms] (4ms) -> total active GPU time = 6ms + 4ms = 10ms
    events = [
        _make_event(1_000_000, 5_000_000),
        _make_event(3_000_000, 7_000_000),
        _make_event(15_000_000, 19_000_000),
        # Non-execution annotations/CPU events must be ignored:
        _make_event(0, 50_000_000, activity_type="gpu_user_annotation"),
        _make_event(0, 50_000_000, name="Context Sync"),
        _make_event(0, 50_000_000, device_type=torch.autograd.DeviceType.CPU),
    ]
    fake_prof = mock.MagicMock()
    fake_prof.__enter__.side_effect = lambda: (
        call_order.append("prof_enter") or fake_prof
    )
    fake_prof.__exit__.side_effect = lambda *_: call_order.append("prof_exit")
    fake_prof.profiler.kineto_results.events.return_value = events

    def step_fn():
      call_order.append("step")
      return "step_out"

    with mock.patch(
        "torch.profiler.profile", return_value=fake_prof
    ) as mock_profile:
      elapsed_s = ops.measure_step_device_time(step_fn)

    self.assertAlmostEqual(elapsed_s, 0.010)
    mock_profile.assert_called_once_with(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    )
    self.assertEqual(
        call_order, ["sync", "prof_enter", "step", "sync", "prof_exit"]
    )

  @mock.patch("torch.accelerator")
  def test_measure_step_device_time_cuda_empty_events_returns_negative_one(
      self, mock_accelerator
  ):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)
    ops = torch_device_ops.TorchDeviceOps(target)

    fake_prof = mock.MagicMock()
    fake_prof.__enter__.return_value = fake_prof
    fake_prof.profiler.kineto_results.events.return_value = []

    with mock.patch("torch.profiler.profile", return_value=fake_prof):
      self.assertEqual(ops.measure_step_device_time(lambda: "out"), -1.0)

  def test_measure_step_device_time_non_cuda(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torch_device_ops.TorchDeviceOps(target)

    self.assertFalse(ops.supports_event_device_time())
    with mock.patch.object(ops, "check_result") as mock_check:
      elapsed_s = ops.measure_step_device_time(lambda: "out")
    self.assertEqual(elapsed_s, -1.0)
    mock_check.assert_called_once_with("out")


if __name__ == "__main__":
  absltest.main()
