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

"""Muse Glimmer (torchtitan) single-device benchmark definitions."""

import torch
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab.model_utils import torchtitan_muse_glimmer


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def muse_glimmer_30b_4_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Muse Glimmer (30B arch, 4 layers) forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    seq_len = 128
    num_layers = 2
  else:
    seq_len = 2048
    num_layers = 4

  device_str = ctx.device_kind.value
  config = torchtitan_muse_glimmer.text_config("30B", num_layers, seq_len)
  torch.manual_seed(12345)
  model = torchtitan_muse_glimmer.build_model(
      config, device_str, torch_device_ops.get_torch_dtype(ctx.dtype)
  )
  model.eval()
  tokens = torch.randint(
      0,
      config.vocab_size,
      (seq_len,),
      dtype=torch.long,
      generator=torch.Generator().manual_seed(1234),
  ).to(device_str)
  return model, (tokens,), {}


def _muse_glimmer_30b_4_layers_training_impl(ctx: context_lib.Context):
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    seq_len = 128
    num_layers = 2
  else:
    seq_len = 2048
    num_layers = 4

  device_str = ctx.device_kind.value
  config = torchtitan_muse_glimmer.text_config("30B", num_layers, seq_len)
  torch.manual_seed(12345)
  model = torchtitan_muse_glimmer.build_model(
      config, device_str, torch_device_ops.get_torch_dtype(ctx.dtype)
  )
  model.train()
  tokens, labels = torchtitan_muse_glimmer.make_inputs(
      config.vocab_size, seq_len, rank=0, device=device_str
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (tokens,), {"labels": labels}, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    stepper_kwargs={
        "compute_loss": torchtitan_muse_glimmer.cross_entropy_loss,
        "use_single_trace": False,
    },
    compile_config=compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    ),
)
def muse_glimmer_30b_4_layers_training_standard(ctx: context_lib.Context):
  """Benchmark factory for Muse Glimmer (30B arch, 4 layers) standard training."""
  return _muse_glimmer_30b_4_layers_training_impl(ctx)
