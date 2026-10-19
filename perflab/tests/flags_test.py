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

"""Tests for perflab/flags.py."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import target as target_lib
from tests import seed_test_utils


class FakeTensorModeValidatorTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      context_lib.RunScope.PRESUBMIT, context_lib.RunScope.FULL
  )
  def test_accepted_on_cpu(self, run_scope: context_lib.RunScope):
    with (
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"}),
        flagsaver.flagsaver(
            (context_lib.RUN_SCOPE, run_scope),
            (flags_lib.FAKE_TENSOR_MODE, True),
        ),
    ):
      self.assertTrue(flags_lib.FAKE_TENSOR_MODE.value)

  @parameterized.parameters("v5e_1x1", "b200_1", "v7_1x1x1")
  def test_rejected_outside_cpu(self, platform: str):
    with mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: platform}):
      with self.assertRaisesRegex(
          flags.IllegalFlagValueError, "fake_tensor_mode is only valid"
      ):
        with flagsaver.flagsaver((flags_lib.FAKE_TENSOR_MODE, True)):
          pass


if __name__ == "__main__":
  absltest.main()
