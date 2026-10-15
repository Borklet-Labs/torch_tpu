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

"""TorchTPU-specific configs for PyTorch device-generic tests."""

from collections.abc import Mapping, Sequence
from typing import Final

# Generic test classes or methods excluded from instantiation for TPU.
#
# Keyed by test module, tests are identified by their generic class and method
# names, e.g. `TestOps` instead of `TestOpsTPU`. In this way, one entry covers
# all instantiated variants of a test. Use "*" to exclude a whole class.
TEST_EXCLUSIONS: Final[Mapping[str, Mapping[str, str | Sequence[str]]]] = {
    # TODO(b/573758862): track per-test exclusion reasons and triage.
    # go/keep-sorted start block=yes
    "dynamo/test_debug_utils": {
        "TestInductorConfigOverrideIntegration": [
            "test_config_override_backward_propagation",
        ],
    },
    # go/keep-sorted end
}
