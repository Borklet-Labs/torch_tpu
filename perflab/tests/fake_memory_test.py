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

"""Tests for perflab/fake_memory.py."""

from absl.testing import absltest
import torch
from torch._subclasses import fake_tensor
from perflab import fake_memory
from perflab import mode as mode_lib
from tests import seed_test_utils

_MIB = 1 << 20
_GIB = 1 << 30


def _build_model(dtype: torch.dtype | None) -> torch.nn.Module:
  """Builds 4 linear layers of 12 GiB in bfloat16, in `dtype` and then cast."""
  model = torch.nn.Sequential(*(
      torch.nn.Linear(65536, 98304, bias=False, dtype=dtype) for _ in range(4)
  ))
  return model.to(dtype=torch.bfloat16)


class TensorMemoryTrackerTest(seed_test_utils.RepeatableTest):

  def test_counts_live_and_peak_bytes(self):
    with fake_memory.TensorMemoryTracker() as tracker:
      x = torch.empty(1024, 1024, dtype=torch.float32)
      y = torch.empty(512, 1024, dtype=torch.float32)
      del x
    self.assertEqual(tracker.live_bytes, 2 * _MIB)
    self.assertEqual(tracker.peak_bytes, 6 * _MIB)
    del y
    self.assertEqual(tracker.live_bytes, 0)
    self.assertEqual(tracker.peak_bytes, 6 * _MIB)

  def test_counts_a_storage_once_for_all_of_its_views(self):
    with fake_memory.TensorMemoryTracker() as tracker:
      x = torch.empty(1024, 1024, dtype=torch.float32)
      views = [x.view(-1), x[0], x.t(), x.add_(1.0)]
    self.assertLen(views, 4)
    self.assertEqual(tracker.live_bytes, 4 * _MIB)
    self.assertEqual(tracker.peak_bytes, 4 * _MIB)

  def test_tracks_an_older_storage_once_an_op_returns_a_view_of_it(self):
    x = torch.empty(1024, 1024, dtype=torch.float32)
    with fake_memory.TensorMemoryTracker() as tracker:
      z = torch.empty(256, 1024, dtype=torch.float32)
      # `x` predates the tracker, so its 4 MiB count only from this view on.
      self.assertEqual(tracker.live_bytes, _MIB)
      y = x[0]
    del y, z
    self.assertEqual(tracker.live_bytes, 4 * _MIB)
    del x
    self.assertEqual(tracker.live_bytes, 0)

  def test_tracks_fake_tensors_through_a_dtype_cast(self):
    with (
        mode_lib.fake_tensor_context(),
        fake_memory.TensorMemoryTracker() as tracker,
    ):
      model = torch.nn.Linear(1024, 1024, bias=False)
      model = model.to(dtype=torch.bfloat16)
      self.assertIsInstance(model.weight, fake_tensor.FakeTensor)
    # The float32 weight (4 MiB) is freed after the cast to bfloat16 (2 MiB).
    self.assertEqual(tracker.live_bytes, 2 * _MIB)
    self.assertEqual(tracker.peak_bytes, 6 * _MIB)


class CheckConstructionPeakTest(seed_test_utils.RepeatableTest):

  def test_raises_for_a_large_model_built_in_float32_then_cast(self):
    with (
        mode_lib.fake_tensor_context(),
        fake_memory.TensorMemoryTracker() as tracker,
    ):
      model = _build_model(dtype=None)
    self.assertEqual(model[0].weight.dtype, torch.bfloat16)
    # The 96 GiB of float32 weights and the first 12 GiB bfloat16 copy.
    self.assertEqual(tracker.peak_bytes, 108 * _GIB)
    self.assertEqual(tracker.live_bytes, 48 * _GIB)
    with self.assertRaisesRegex(
        AssertionError,
        "building model peaked at 108.00 GiB of tensors, more than the 94.74"
        " GiB",
    ):
      fake_memory.check_construction_peak(tracker, name="model")

  def test_accepts_a_large_model_built_in_its_dtype(self):
    with (
        mode_lib.fake_tensor_context(),
        fake_memory.TensorMemoryTracker() as tracker,
    ):
      model = _build_model(dtype=torch.bfloat16)
    self.assertEqual(model[0].weight.dtype, torch.bfloat16)
    self.assertEqual(tracker.peak_bytes, 48 * _GIB)
    fake_memory.check_construction_peak(tracker, name="model")

  def test_uses_the_given_limit(self):
    with fake_memory.TensorMemoryTracker() as tracker:
      x = torch.empty(1024, 1024, dtype=torch.float32)
    self.assertEqual(x.numel(), 1024 * 1024)
    fake_memory.check_construction_peak(
        tracker, name="x", max_peak_bytes=4 * _MIB
    )
    with self.assertRaisesRegex(AssertionError, "building x peaked"):
      fake_memory.check_construction_peak(
          tracker, name="x", max_peak_bytes=4 * _MIB - 1
      )


if __name__ == "__main__":
  absltest.main()
