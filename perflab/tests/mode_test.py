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

"""Tests for harness/mode.py."""

from absl.testing import absltest
import torch
from torch._subclasses import fake_tensor
from perflab import common
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils


class ModeEnumTest(seed_test_utils.RepeatableTest):

  def test_run_mode_values(self):
    self.assertEqual(common.RunMode.EAGER_DEFAULT.value, "eager_default")
    self.assertEqual(common.RunMode.EAGER_OPTIMIZED.value, "eager_optimized")
    self.assertEqual(
        common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING.value,
        "eager_defer_never_and_launch_blocking",
    )
    self.assertEqual(common.RunMode.COMPILED.value, "compiled")

  def test_bad_mode_string_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      common.RunMode("compiled_max_autotune")

  def test_torch_tpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(mode_lib.Framework.TORCH, target_lib.DeviceKind.TPU),
        [
            common.RunMode.EAGER_DEFAULT,
            common.RunMode.EAGER_OPTIMIZED,
            common.RunMode.COMPILED,
        ],
    )

  def test_torch_cuda_modes(self):
    self.assertEqual(
        mode_lib.modes_for(
            mode_lib.Framework.TORCH, target_lib.DeviceKind.CUDA
        ),
        [common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED],
    )

  def test_torch_cpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(mode_lib.Framework.TORCH, target_lib.DeviceKind.CPU),
        [common.RunMode.EAGER_DEFAULT],
    )

  def test_torchax_tpu_modes(self):
    self.assertEqual(
        mode_lib.modes_for(
            mode_lib.Framework.TORCHAX, target_lib.DeviceKind.TPU
        ),
        [common.RunMode.COMPILED],
    )

  def test_torchax_non_tpu_raises(self):
    for kind in target_lib.DeviceKind:
      if kind is not target_lib.DeviceKind.TPU:
        with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
          mode_lib.modes_for(mode_lib.Framework.TORCHAX, kind)

  def test_eager_optimized_is_torch_tpu_only(self):
    """Verifies eager_optimized is supported only for torch on TPU."""
    for fw in mode_lib.Framework:
      for kind in target_lib.DeviceKind:
        if (
            fw is mode_lib.Framework.TORCHAX
            and kind is not target_lib.DeviceKind.TPU
        ):
          continue
        has_eo = common.RunMode.EAGER_OPTIMIZED in mode_lib.modes_for(fw, kind)
        expected = (
            fw is mode_lib.Framework.TORCH and kind is target_lib.DeviceKind.TPU
        )
        self.assertEqual(has_eo, expected, f"{fw.value} x {kind.value}")

  def test_modes_for_accepts_strings(self):
    self.assertEqual(
        mode_lib.modes_for("torch", "cpu"),
        [common.RunMode.EAGER_DEFAULT],
    )
    self.assertEqual(
        mode_lib.modes_for("torchax", "tpu"), [common.RunMode.COMPILED]
    )

  def test_bad_framework_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      mode_lib.modes_for("framework", target_lib.DeviceKind.CPU)

  def test_bad_device_kind_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      mode_lib.modes_for(mode_lib.Framework.TORCH, "invalid_device")

  def test_framework_values(self):
    self.assertEqual(mode_lib.Framework.TORCH.value, "torch")
    self.assertEqual(mode_lib.Framework.TORCHAX.value, "torchax")


class RunModeContextTest(seed_test_utils.RepeatableTest):

  def test_non_tpu_target_is_noop(self):
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mode_lib.run_mode_context(common.RunMode.EAGER_DEFAULT, target):
      pass

  def test_tpu_target_sets_eager_mode(self):
    from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top

    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    with mode_lib.run_mode_context(common.RunMode.EAGER_DEFAULT, target):
      self.assertEqual(
          execution_mode.eager_mode, execution_mode.EagerMode.DEFER_NEVER
      )
    with mode_lib.run_mode_context(common.RunMode.EAGER_OPTIMIZED, target):
      self.assertEqual(
          execution_mode.eager_mode, execution_mode.EagerMode.DEFER_AND_FUSE
      )
    with mode_lib.run_mode_context(
        common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING, target
    ):
      self.assertEqual(
          execution_mode.eager_mode,
          execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING,
      )

  def test_invalid_run_mode_raises(self):
    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      with mode_lib.run_mode_context("invalid_mode", target):
        pass

  def test_cuda_target_sets_float32_matmul_precision(self):
    from unittest import mock  # pylint: disable=g-import-not-at-top

    target = target_lib.make_target(target_lib.Platform.B200_1)
    with mock.patch("torch.set_float32_matmul_precision") as mock_prec:
      with mode_lib.run_mode_context(common.RunMode.COMPILED, target):
        pass
    mock_prec.assert_called_once_with("high")


class FakeTensorContextTest(seed_test_utils.RepeatableTest):

  def test_enters_fake_tensor_mode(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4)
      layer = layer.to(dtype=torch.bfloat16)
      x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
      y = layer(x).sum()
      y.backward()
      self.assertIsInstance(x, fake_tensor.FakeTensor)
      self.assertIsInstance(x.grad, fake_tensor.FakeTensor)
      self.assertEqual(x.grad.dtype, torch.bfloat16)
      self.assertEqual(layer.weight.dtype, torch.bfloat16)

  def test_converts_constant_parameters(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4)
      # A constant like CLIP's logit scale, converted with its own storage.
      layer.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))
      layer = layer.to(dtype=torch.bfloat16)
      self.assertIsInstance(layer.logit_scale, fake_tensor.FakeTensor)
      self.assertEqual(layer.logit_scale.dtype, torch.bfloat16)

  def test_restores_real_tensors_on_exit(self):
    with mode_lib.fake_tensor_context():
      pass
    self.assertNotIsInstance(torch.zeros(4, 4), fake_tensor.FakeTensor)

  def test_trunc_normal_is_a_no_op_inside(self):
    with mode_lib.fake_tensor_context():
      layer = torch.nn.Linear(8, 4).to(dtype=torch.bfloat16)
      # As in HuggingFace ViT's position embedding init. Without the patch, this
      # would raise DataDependentOutputException.
      torch.nn.init.trunc_normal_(layer.weight)
      self.assertIsInstance(layer.weight, fake_tensor.FakeTensor)

  def test_restores_trunc_normal_on_exit(self):
    with mode_lib.fake_tensor_context():
      pass
    x = torch.zeros(4, 4)
    torch.nn.init.trunc_normal_(x, a=1.0, b=2.0)
    self.assertTrue(bool((x >= 1.0).all()))

  def test_drop_path_rates_are_readable_inside(self):
    with mode_lib.fake_tensor_context():
      rates = torch.linspace(0, 0.3, 4)
      self.assertIsInstance(rates, fake_tensor.FakeTensor)
      # As in HuggingFace Swin2SR and timm. Without the constants, `.item()`
      # would raise DataDependentOutputException.
      layer_rates = [x.item() for x in rates]
      # As in HuggingFace ConvNeXt V2.
      stage_rates = [x.tolist() for x in rates.split([1, 3])]
      # As in HuggingFace BiT, which builds them with NumPy.
      tensor_stage_rates = [
          x.tolist() for x in torch.tensor([0.0, 0.1, 0.2, 0.3]).split([1, 3])
      ]
    self.assertSequenceAlmostEqual(layer_rates, [0.0, 0.1, 0.2, 0.3], places=6)
    self.assertLen(stage_rates, 2)
    self.assertSequenceAlmostEqual(stage_rates[1], [0.1, 0.2, 0.3], places=6)
    self.assertSequenceAlmostEqual(
        tensor_stage_rates[1], [0.1, 0.2, 0.3], places=6
    )

  def test_large_linspace_stays_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.linspace(0, 1, 4096)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_restores_constant_numel_limit_on_exit(self):
    limit = fake_tensor.CONSTANT_NUMEL_LIMIT
    with mode_lib.fake_tensor_context():
      pass
    self.assertEqual(fake_tensor.CONSTANT_NUMEL_LIMIT, limit)

  def test_layer_drop_draw_is_readable_inside(self):
    with mode_lib.fake_tensor_context():
      draw = torch.rand([])
      self.assertIsInstance(draw, fake_tensor.FakeTensor)
      # As in the LayerDrop of HuggingFace BART in training. Without the
      # constants, the comparison would raise DataDependentOutputException.
      self.assertTrue(bool(draw < 1.0))
      self.assertFalse(bool(draw < 0.0))

  def test_large_rand_stays_fake(self):
    with mode_lib.fake_tensor_context():
      x = torch.rand(4096)
      with self.assertRaises(fake_tensor.DataDependentOutputException):
        x[0].item()

  def test_distribution_arguments_are_not_validated_inside(self):
    validate_args = torch.distributions.Distribution._validate_args  # pylint: disable=protected-access
    with mode_lib.fake_tensor_context():
      # Too large for a constant, as the logits of the TAPAS cell selection.
      logits = torch.randn(2, 2048)
      distribution = torch.distributions.Categorical(logits=logits)
      self.assertIsInstance(distribution.logits, fake_tensor.FakeTensor)
    self.assertEqual(
        torch.distributions.Distribution._validate_args,  # pylint: disable=protected-access
        validate_args,
    )


if __name__ == "__main__":
  absltest.main()
