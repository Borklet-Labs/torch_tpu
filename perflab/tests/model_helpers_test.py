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

"""Tests for perflab/model_helpers.py."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from perflab import context as context_lib
from perflab import model_helpers
from perflab import target as target_lib
from tests import seed_test_utils


def _context(run_scope: context_lib.RunScope) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(target_lib.Platform.CPU),
      run_scope=run_scope,
  )


class PresubmitSmokeOrRaiseTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )
    self.enter_context(
        flagsaver.flagsaver(run_scope=context_lib.RunScope.PRESUBMIT)
    )

  @flagsaver.flagsaver(fake_tensor_mode=False)
  def test_full_scope_returns_false(self):
    self.assertFalse(
        model_helpers.presubmit_smoke_or_raise(
            _context(context_lib.RunScope.FULL), unsupported_reason="unused"
        )
    )

  @flagsaver.flagsaver(fake_tensor_mode=False)
  def test_presubmit_without_fake_tensor_mode_raises(self):
    with self.assertRaisesRegex(target_lib.UnsupportedBenchmark, "too large"):
      model_helpers.presubmit_smoke_or_raise(
          _context(context_lib.RunScope.PRESUBMIT),
          unsupported_reason="too large",
      )

  @flagsaver.flagsaver(fake_tensor_mode=True)
  def test_presubmit_with_fake_tensor_mode_returns_true(self):
    self.assertTrue(
        model_helpers.presubmit_smoke_or_raise(
            _context(context_lib.RunScope.PRESUBMIT),
            unsupported_reason="unused",
        )
    )


if __name__ == "__main__":
  absltest.main()
