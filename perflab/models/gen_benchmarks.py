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

"""Dynamic registration of benchmarks against static model list."""

import copy
import itertools
import sys

from absl import flags
import torch
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab.gen_utils import gen_utils

# Generated benchmarks that also run in presubmit scope in the FakeTensorMode
# CPU smoke test (`--fake_tensor_mode`). These are hand-picked representatives,
# one per architecture family that the hand-written (decoder-only LLM)
# benchmarks do not cover: a small sample of the models in
# `gen_utils/models.csv`, chosen to keep presubmit small. The list is not
# exhaustive and not driven by past breakages. Each has a checked-in config in
# the module registry's in-tree `model_configs` resources (all under
# `google/examples/huggingface_transformers/model_configs`), so the smoke test
# never falls back to GCS. Keep this list short: every entry adds a forward and
# a training case to presubmit.
_CPU_PRESUBMIT_MODEL_IDS = frozenset({
    "facebook/detr-resnet-50",  # Detection with a convolutional backbone.
    "google-bert/bert-base-uncased",  # Text encoder.
    "google-t5/t5-small",  # Text encoder-decoder.
    "google/vit-base-patch16-224-in21k",  # Vision transformer.
    "microsoft/resnet-50",  # Convolutional vision model.
    "openai/clip-vit-base-patch32",  # Text-image dual encoder.
    "openai/whisper-large-v3-turbo",  # Speech encoder-decoder.
})


def _make_factory(
    model_id: str,
    is_training: bool,
    model_load_fn: gen_utils.ModelLoadFn,
    full_scope_batch_size: int,
):
  """Creates a benchmark factory for a given model."""

  def factory(ctx: context_lib.Context):
    model, inputs = model_load_fn(
        ctx, model_id, is_training, full_scope_batch_size
    )
    if is_training:
      opt = torch.optim.AdamW(
          model.parameters(),
          lr=1e-4,
          capturable=True,
          fused=True,
      )
      return model, (), inputs, opt
    return model, (), inputs

  return factory


def _iter_gen_benchmark_specs():
  """Yields all combinations of models and execution steps for benchmarking."""
  for model_gen_spec, (is_training, step) in itertools.product(
      gen_utils.iter_models_csv(),
      [
          (False, step_lib.StepperType.FORWARD),
          (True, step_lib.StepperType.TRAINING),
      ],
  ):
    yield model_gen_spec, is_training, step


def _registration_flags() -> tuple[context_lib.RunScope, bool]:
  """Returns the values of `--run_scope` and `--fake_tensor_mode`.

  Generated benchmarks are registered at import time, usually before
  `flags.FLAGS` is parsed. These two flags are then parsed from `sys.argv` by
  absl's parser, so `--`, `--noflag`, `=false` and `--flagfile` work as in the
  full parse, but into a scratch `FlagValues` that holds copies of just them.
  Parsing `flags.FLAGS` itself here would mark it parsed with only the flags
  defined so far, and the full parse in `app.run` would then append the values
  of multi-value flags a second time. An invalid `--fake_tensor_mode` still
  fails the full parse.
  """
  if flags.FLAGS.is_parsed():
    return context_lib.RUN_SCOPE.value, mode_lib.fake_tensor_smoke_enabled()
  flag_values = flags.FlagValues()
  for flag in (context_lib.RUN_SCOPE, flags_lib.FAKE_TENSOR_MODE):
    flag_values[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
  flag_values(sys.argv, known_only=True)
  return (
      getattr(flag_values, context_lib.RUN_SCOPE.name),
      getattr(flag_values, flags_lib.FAKE_TENSOR_MODE.name),
  )


def _register_gen_benchmarks():
  """Registers generated benchmarks in the global registry."""
  run_scope, fake_tensor_mode = _registration_flags()

  model_ids = None
  if run_scope is context_lib.RunScope.PRESUBMIT:
    if not fake_tensor_mode:
      # We will need to split the target up to prevent timeouts.
      return
    model_ids = _CPU_PRESUBMIT_MODEL_IDS

  for model_gen_spec, is_training, step in _iter_gen_benchmark_specs():
    if model_ids is not None and model_gen_spec.model_id not in model_ids:
      continue
    extra_config = model_gen_spec.get_extra_config(is_training, suffix="gen")
    skipped_run_modes = extra_config.skipped_run_modes
    step_kwargs = extra_config.step_kwargs
    model_load_fn = extra_config.model_load_fn

    registry_lib.add_benchmark(
        factory=_make_factory(
            model_gen_spec.model_id,
            is_training,
            model_load_fn,
            extra_config.full_scope_batch_size,
        ),
        stepper=step,
        stepper_kwargs=step_kwargs,
        name=model_gen_spec.benchmark_name(
            is_training=is_training, suffix="gen"
        ),
        skipped_run_modes=skipped_run_modes,
    )


_register_gen_benchmarks()
