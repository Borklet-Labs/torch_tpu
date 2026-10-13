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

"""SDPA under torch.compile with a dynamic KV length and TT_KERNEL VLOG on.

The BUILD target passes `--vmodule=scaled_dot_product_attention_aten_kernels=1`,
which makes the SDPA TT_KERNELs log their arguments. `aten::_fused_sdp_choice`
is a DispatchStub selected by device type from inside
`scaled_dot_product_attention`, so it also runs on the FakeTensors Dynamo traces
with. Logging a tensor with symbolic sizes must neither specialize the dynamic
dimension (an `Eq(s, int)` guard that pins it to the traced size) nor throw.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from absl import flags
from absl.testing import absltest
import torch
from torch._dynamo.backends import registry
import torch.nn.functional as F
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils

# Set in the BUILD `args`; VLOG(1) is what makes TT_KERNEL log its arguments.
_VMODULE = "scaled_dot_product_attention_aten_kernels=1"


def _decode_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
  return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


@oss_utils.skip_if_libtpu_version(
    lambda v: v < (0, 0, 47),
    "Bounded dynamism requires libtpu >= 0.0.47",
)
class SdpaVlogTest(seed_test_utils.RepeatableTest):
  """Compiles a GQA decode step whose KV length becomes dynamic."""

  def setUp(self) -> None:
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    if "vmodule" not in flags.FLAGS:
      self.skipTest("The C++ --vmodule flag is not bridged into Python flags.")
    self.assertEqual(
        flags.FLAGS.vmodule,
        _VMODULE,
        "TT_KERNEL argument logging must be enabled through the BUILD args.",
    )
    tt_testing.reset_eager_state()
    torch._dynamo.reset()
    self.device = torch.accelerator.current_accelerator()

  def test_dynamic_kv_len_is_logged_without_specialization(self) -> None:
    tpu_backend = registry.lookup_backend("tpu")
    num_compilations = 0
    specializations: list[str] = []

    def backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
      # Dynamo has finished tracing when the backend runs, so any guard the
      # kernel logging installed on the symbolic KV length is visible here.
      nonlocal num_compilations
      num_compilations += 1
      shape_env = torch._guards.detect_fake_mode(example_inputs).shape_env
      specializations.extend(
          f"{symbol} = {value}"
          for symbol, value in shape_env.replacements.items()
          if value.is_number
      )
      return tpu_backend(graph_module, example_inputs, **kwargs)

    compiled = torch.compile(_decode_step, backend=backend, dynamic=None)
    with torch.no_grad():
      # The second KV length makes Dynamo recompile with a dynamic KV
      # dimension (automatic dynamic shapes); the third reuses that graph.
      for kv_len in (8, 13, 11):
        q = torch.randn(1, 8, 1, 32)
        k = torch.randn(1, 2, kv_len, 32)
        v = torch.randn(1, 2, kv_len, 32)
        expected = _decode_step(q, k, v)
        actual = compiled(
            q.to(self.device), k.to(self.device), v.to(self.device)
        )
        with self.subTest(kv_len=kv_len):
          utils.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    self.assertEmpty(specializations)
    self.assertEqual(num_compilations, 2)


if __name__ == "__main__":
  absltest.main()
