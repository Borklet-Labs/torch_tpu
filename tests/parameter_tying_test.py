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

"""Tests that moving modules with tied parameters to TPU does not break ties."""

from absl.testing import absltest
import torch
from tests import seed_test_utils


class ParameterTyingTest(seed_test_utils.RepeatableTest):

  def test_tied_module_parameters_stay_tied_after_load(self):
    # Create a model with two layers on CPU.
    embedding = torch.nn.Embedding(5, 4, device="cpu")
    head = torch.nn.Linear(4, 5, bias=False, device="cpu")
    cpu_model = torch.nn.Sequential(embedding, head)

    # Tie the parameters; the two weights are now an identical Tensor on CPU.
    head.weight = embedding.weight
    self.assertEqual(head.weight.data_ptr(), embedding.weight.data_ptr())

    # Load the model onto TPU.
    tpu_device = torch.device("tpu:0")
    tpu_model = cpu_model.to(tpu_device)
    self.assertTrue(tpu_model[0].weight.device == tpu_device)
    self.assertTrue(tpu_model[1].weight.device == tpu_device)

    # Verify that the parameters are still tied.
    self.assertEqual(
        tpu_model[0].weight.data_ptr(), tpu_model[1].weight.data_ptr()
    )

  def test_tied_module_parameters_stay_tied_after_unload(self):
    # Create a model with two layers on TPU.
    embedding = torch.nn.Embedding(5, 4, device="tpu")
    head = torch.nn.Linear(4, 5, bias=False, device="tpu")
    tpu_model = torch.nn.Sequential(embedding, head)

    # Tie the parameters; the two weights are now an identical Tensor on TPU.
    head.weight = embedding.weight
    self.assertEqual(head.weight.data_ptr(), embedding.weight.data_ptr())

    # Load the model onto CPU.
    cpu_device = torch.device("cpu")
    cpu_model = tpu_model.to(cpu_device)
    self.assertTrue(cpu_model[0].weight.device == cpu_device)
    self.assertTrue(cpu_model[1].weight.device == cpu_device)

    # Verify that the parameters are still tied.
    self.assertEqual(
        cpu_model[0].weight.data_ptr(), cpu_model[1].weight.data_ptr()
    )


if __name__ == "__main__":
  absltest.main()
