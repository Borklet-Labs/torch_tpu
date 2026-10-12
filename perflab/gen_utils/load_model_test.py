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

"""Tests for load_model."""

import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from perflab import context as context_lib
from perflab import model_helpers
from perflab import target as target_lib
from perflab.gen_utils import load_model
from tests import seed_test_utils


class _FakeForSequenceClassification(nn.Module):
  """Stand-in for a HuggingFace `*ForSequenceClassification` model."""

  def __init__(self, config: types.SimpleNamespace):
    super().__init__()
    self.config = config
    self.linear = nn.Linear(4, 4)


def _deberta_config(
    num_labels: int, problem_type: str | None = None
) -> types.SimpleNamespace:
  """Returns a minimal DeBERTa-v2 sequence-classification config."""
  return types.SimpleNamespace(
      model_type="deberta-v2", num_labels=num_labels, problem_type=problem_type
  )


def _fake_registry(model: nn.Module, inputs: dict[str, Any]) -> Any:
  """Returns a module registry whose only module spec yields `model`."""
  module_spec = types.SimpleNamespace(
      module_factory=lambda: model,
      sample_inputs_factory=lambda shape, device: (None, dict(inputs)),
  )
  return types.SimpleNamespace(
      get_module_spec=lambda *args, **kwargs: module_spec
  )


class _FakeTimeSeriesModel(nn.Module):
  """Stand-in for a HuggingFace time series model such as TimesFM."""

  def __init__(self, config: types.SimpleNamespace):
    super().__init__()
    self.config = config


def _fake_time_series_registry(config: types.SimpleNamespace) -> Any:
  """Returns a module registry that sizes inputs by `context_length`.

  Like the TimesFM input builder, its inputs ignore the requested sequence
  length, and it applies the config hook of the caller.

  Args:
    config: The model config, passed to the config hook.
  """

  def get_module_spec(*args, modify_config_hook, **kwargs):
    del args, kwargs  # Unused.
    hooked_config = modify_config_hook(config)
    return types.SimpleNamespace(
        module_factory=lambda: _FakeTimeSeriesModel(hooked_config),
        sample_inputs_factory=lambda shape, device: (
            None,
            {
                "past_values": torch.randn(
                    shape[0], hooked_config.context_length
                )
            },
        ),
    )

  return types.SimpleNamespace(get_module_spec=get_module_spec)


class LoadModelTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    self.inputs = {
        "input_ids": torch.randint(0, 10, (1, 128)),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

  def _load(
      self, model: nn.Module, is_training: bool
  ) -> tuple[nn.Module, dict[str, Any]]:
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_registry(model, self.inputs),
    ):
      return load_model.default_load_model(
          self.ctx, "fake/model", is_training=is_training
      )

  def test_sequence_classification_training_sets_single_label_problem_type(
      self,
  ):
    config = _deberta_config(num_labels=3)
    model, inputs = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertEqual(model.config.problem_type, "single_label_classification")
    self.assertEqual(inputs["labels"].shape, (1,))
    self.assertEqual(inputs["labels"].dtype, torch.long)
    self.assertTrue(torch.all(inputs["labels"] < 3))

  def test_sequence_classification_regression_leaves_problem_type_unset(self):
    config = _deberta_config(num_labels=1)
    model, _ = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertIsNone(model.config.problem_type)

  def test_sequence_classification_preserves_preset_problem_type(self):
    config = _deberta_config(
        num_labels=3, problem_type="multi_label_classification"
    )
    model, _ = self._load(
        _FakeForSequenceClassification(config), is_training=True
    )

    self.assertEqual(model.config.problem_type, "multi_label_classification")

  def test_sequence_classification_inference_leaves_problem_type_unset(self):
    config = _deberta_config(num_labels=3)
    model, inputs = self._load(
        _FakeForSequenceClassification(config), is_training=False
    )

    self.assertIsNone(model.config.problem_type)
    self.assertNotIn("labels", inputs)

  def _load_time_series(
      self, context_length: int
  ) -> tuple[nn.Module, dict[str, Any]]:
    ctx = context_lib.Context(
        target=target_lib.make_target(platform=target_lib.Platform.CPU),
        run_scope=context_lib.RunScope.FULL,
    )
    config = types.SimpleNamespace(context_length=context_length)
    with mock.patch.object(
        model_helpers,
        "get_module_registry",
        return_value=_fake_time_series_registry(config),
    ):
      return load_model.default_load_model(
          ctx, "fake/time_series_model", is_training=False
      )

  def test_context_length_capped_at_seq_len(self):
    model, inputs = self._load_time_series(context_length=16384)

    self.assertEqual(model.config.context_length, 2048)
    self.assertEqual(inputs["past_values"].shape, (1, 2048))

  def test_shorter_context_length_kept(self):
    model, inputs = self._load_time_series(context_length=512)

    self.assertEqual(model.config.context_length, 512)
    self.assertEqual(inputs["past_values"].shape, (1, 512))


if __name__ == "__main__":
  absltest.main()
