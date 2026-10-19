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

from collections.abc import Mapping
import copy
import functools
import itertools
import sys

from absl import flags
import torch
from perflab import common
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab.gen_utils import gen_utils

# The run modes that only TPUs and GPUs run. The FakeTensorMode CPU smoke test
# runs `eager_default`, the only run mode of CPU targets (`mode.modes_for`).
_ACCELERATOR_ONLY_RUN_MODES = frozenset({
    common.RunMode.COMPILED.value,
    common.RunMode.EAGER_OPTIMIZED.value,
})

_DATA_DEPENDENT_SCALAR = (
    "reads a Python scalar from a tensor (aten._local_scalar_dense), which"
    " FakeTensorMode cannot evaluate"
)
_DATA_DEPENDENT_SHAPE = (
    "uses an op whose output shape depends on tensor values (e.g."
    " aten.nonzero), which FakeTensorMode cannot evaluate"
)

# Generated benchmarks that the FakeTensorMode CPU smoke test
# (`--fake_tensor_mode`) cannot run, with the reason. Every other generated
# benchmark whose model config is in-tree runs in the smoke test, so that a
# change to the shared model loading (e.g. `tests/module_registry.py`) that
# breaks a generated benchmark fails in presubmit instead of in the nightly
# benchmark runs.
_FAKE_TENSOR_MODE_UNSUPPORTED: Mapping[str, str] = {
    # go/keep-sorted start
    "bart_large_mnli_forward_gen": _DATA_DEPENDENT_SHAPE,
    "bart_large_mnli_train_gen": _DATA_DEPENDENT_SHAPE,
    "clap_htsat_fused_forward_gen": _DATA_DEPENDENT_SHAPE,
    "gemma_4_e4b_it_forward_gen": _DATA_DEPENDENT_SHAPE,
    "gemma_4_e4b_it_train_gen": _DATA_DEPENDENT_SHAPE,
    "glm_ocr_forward_gen": _DATA_DEPENDENT_SCALAR,
    "glm_ocr_train_gen": _DATA_DEPENDENT_SCALAR,
    "kosmos_2_patch14_224_train_gen": _DATA_DEPENDENT_SHAPE,
    "lahja_sa_ahmad_v1_forward_gen": _DATA_DEPENDENT_SCALAR,
    "lightglue_superpoint_forward_gen": _DATA_DEPENDENT_SHAPE,
    "llava_1_5_7b_hf_forward_gen": _DATA_DEPENDENT_SCALAR,
    "llava_1_5_7b_hf_train_gen": _DATA_DEPENDENT_SCALAR,
    "qwen2_5_vl_7b_instruct_forward_gen": _DATA_DEPENDENT_SHAPE,
    "qwen2_5_vl_7b_instruct_train_gen": _DATA_DEPENDENT_SHAPE,
    "vilt_b32_finetuned_vqa_forward_gen": _DATA_DEPENDENT_SCALAR,
    "vjepa2_vitl_fpc64_256_forward_gen": _DATA_DEPENDENT_SCALAR,
    "wmt19_ru_en_train_gen": _DATA_DEPENDENT_SCALAR,
    "zoedepth_nyu_kitti_forward_gen": _DATA_DEPENDENT_SCALAR,
    # go/keep-sorted end
}


@functools.cache
def has_in_tree_config(model_id: str) -> bool:
  """Returns whether `module_registry` reads the config of `model_id` in-tree.

  The FakeTensorMode CPU smoke test has no network access, so it builds only
  the models whose configs `module_registry` reads from the in-tree resources
  instead of GCS.

  Args:
    model_id: The HuggingFace model ID of a `gen_utils/models.csv` entry.
  """
  # Imported here rather than at module scope: importing module_registry is
  # likely to probe the CUDA driver, which poisons fork() for CUDA, and only
  # the FakeTensorMode CPU smoke test needs it here.
  from tests import module_registry  # pylint: disable=g-import-not-at-top

  return (
      module_registry.TransformersProvider.in_tree_config(model_id) is not None
  )


def fake_tensor_smoke_skip_reason(model_id: str, name: str) -> str | None:
  """Returns why the FakeTensorMode smoke test skips a generated benchmark.

  Args:
    model_id: The HuggingFace model ID of the benchmark's model.
    name: The name of the generated benchmark.

  Returns:
    The reason, or None if the smoke test runs the benchmark.
  """
  if not has_in_tree_config(model_id):
    return f"the config of {model_id} is not in-tree"
  return _FAKE_TENSOR_MODE_UNSUPPORTED.get(name)


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

  if run_scope is context_lib.RunScope.PRESUBMIT and not fake_tensor_mode:
    # We will need to split the target up to prevent timeouts.
    return

  for model_gen_spec, is_training, step in _iter_gen_benchmark_specs():
    name = model_gen_spec.benchmark_name(is_training=is_training, suffix="gen")
    if fake_tensor_mode and fake_tensor_smoke_skip_reason(
        model_gen_spec.model_id, name
    ):
      continue
    extra_config = model_gen_spec.get_extra_config(is_training, suffix="gen")
    skipped_run_modes = extra_config.skipped_run_modes
    if fake_tensor_mode and _ACCELERATOR_ONLY_RUN_MODES - skipped_run_modes:
      # The smoke test only builds the benchmark and takes a step, so it also
      # checks the benchmarks that skip `eager_default` but run on accelerators
      # in another mode.
      skipped_run_modes = skipped_run_modes - {
          common.RunMode.EAGER_DEFAULT.value
      }
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
        name=name,
        skipped_run_modes=skipped_run_modes,
    )


_register_gen_benchmarks()
