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

"""Tests that presubmit scope registers generated benchmarks only in the smoke.

Generated benchmarks are registered in presubmit scope only under the
FakeTensorMode CPU smoke (`--fake_tensor_mode`), and then only a small subset.
"""

import copy
import os
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.models import gen_benchmarks
from tests import seed_test_utils

_CPU_PRESUBMIT_MODEL_IDS = gen_benchmarks._CPU_PRESUBMIT_MODEL_IDS  # pylint: disable=protected-access


def _cpu_presubmit_names() -> set[str]:
  """Returns the generated benchmark names of the CPU presubmit models."""
  return {
      spec.benchmark_name(is_training=is_training, suffix="gen")
      for spec, is_training, _ in gen_benchmarks._iter_gen_benchmark_specs()  # pylint: disable=protected-access
      if spec.model_id in _CPU_PRESUBMIT_MODEL_IDS
  }


class GenBenchmarksPresubmitScopeTest(seed_test_utils.RepeatableTest):

  def _registered_names(
      self, *, platform: str, fake_tensor_mode: bool
  ) -> set[str]:
    with (
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: platform}),
        flagsaver.flagsaver(
            run_scope=context_lib.RunScope.PRESUBMIT,
            fake_tensor_mode=fake_tensor_mode,
        ),
        mock.patch.dict(registry_lib.REGISTRY, clear=True),
    ):
      gen_benchmarks._register_gen_benchmarks()  # pylint: disable=protected-access
      return set(registry_lib.REGISTRY)

  def _registered_names_at_import_time(self, argv: list[str]) -> set[str]:
    # Unparsed flags, as when production imports gen_benchmarks.
    import_time_flags = flags.FlagValues()
    for flag in (context_lib.RUN_SCOPE, flags_lib.FAKE_TENSOR_MODE):
      import_time_flags[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
    with (
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"}),
        mock.patch.object(flags, "FLAGS", import_time_flags),
        mock.patch.object(sys, "argv", argv),
        mock.patch.dict(registry_lib.REGISTRY, clear=True),
    ):
      gen_benchmarks._register_gen_benchmarks()  # pylint: disable=protected-access
      return set(registry_lib.REGISTRY)

  def test_cpu_presubmit_models_have_forward_and_training_benchmarks(self):
    self.assertLen(_cpu_presubmit_names(), 2 * len(_CPU_PRESUBMIT_MODEL_IDS))

  def test_cpu_presubmit_with_fake_tensor_mode_registers_subset(self):
    self.assertEqual(
        self._registered_names(platform="cpu", fake_tensor_mode=True),
        _cpu_presubmit_names(),
    )

  @parameterized.parameters(
      "cpu",
      "v5e_1x1",
      # The platform is only read with --fake_tensor_mode.
      "not_a_platform",
  )
  def test_presubmit_registers_nothing_without_fake_tensor_mode(
      self, platform: str
  ):
    self.assertEmpty(
        self._registered_names(platform=platform, fake_tensor_mode=False)
    )

  def test_import_time_cpu_presubmit_with_fake_tensor_mode_registers_subset(
      self,
  ):
    self.assertEqual(
        self._registered_names_at_import_time(
            ["prog", "--run_scope=presubmit", "--fake_tensor_mode"]
        ),
        _cpu_presubmit_names(),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "after_double_dash",
          "args": ["--run_scope=presubmit", "--", "--fake_tensor_mode"],
      },
      {
          "testcase_name": "negated",
          "args": [
              "--run_scope",
              "presubmit",
              "--fake_tensor_mode",
              "--nofake_tensor_mode",
          ],
      },
      {
          "testcase_name": "false",
          "args": ["--run_scope=presubmit", "--fake_tensor_mode=false"],
      },
  )
  def test_import_time_presubmit_without_fake_tensor_mode_registers_nothing(
      self, args: list[str]
  ):
    self.assertEmpty(self._registered_names_at_import_time(["prog", *args]))


if __name__ == "__main__":
  absltest.main()
