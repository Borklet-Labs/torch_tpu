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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import torchtitan_muse_glimmer
from tests import seed_test_utils


class TorchtitanMuseGlimmerTest(seed_test_utils.RepeatableTest):

  def test_text_config_and_build(self):
    config = torchtitan_muse_glimmer.text_config(
        "debugmodel", num_layers=2, seq_len=128
    )
    self.assertLen(config.layers, 2)
    self.assertIsNone(config.vision_encoder)

    torch.manual_seed(123)
    model = torchtitan_muse_glimmer.build_model(config, "cpu", torch.bfloat16)

    # Check that complex rope cache is preserved and matches fp32.
    torch.manual_seed(123)
    fp32_model = config.build()
    fp32_model.init_states(buffer_device=torch.device("cpu"))

    has_complex = False
    for (name, buf), (fp32_name, fp32_buf) in zip(
        model.named_buffers(), fp32_model.named_buffers()
    ):
      self.assertEqual(name, fp32_name)
      if buf.is_complex():
        has_complex = True
        test_utils.assert_close(buf, fp32_buf)
    self.assertTrue(has_complex)

  def test_sliding_window_validation(self):
    with self.assertRaisesRegex(ValueError, "exceeds the sliding window"):
      torchtitan_muse_glimmer.text_config(
          "debugmodel", num_layers=2, seq_len=256
      )


if __name__ == "__main__":
  absltest.main()
