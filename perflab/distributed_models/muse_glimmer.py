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

"""Distributed Muse Glimmer benchmark definitions.

The model is the text-only 30B flavor truncated to its last 4 layers (of 52),
representing one full cycle of the model's sliding window attention pattern:
3 layers of SWA+RoPE followed by 1 layer of global NoPE.
Because seq_len (2048) equals the sliding window size, the upstream SDPA backend
is exact without modifications. It is trained with torchtitan's SimpleFSDP
fully_shard, per-layer activation checkpointing, and AdamW.
"""

import functools
from typing import Any

from absl import logging
import torch
from torch import nn
from torch.utils import checkpoint as checkpoint_utils
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab.model_utils import torchtitan_muse_glimmer
from torchtitan.experiments.graph_trainer import simple_fsdp

_SUPPORTED_PLATFORMS: frozenset[target_lib.Platform] = frozenset({
    target_lib.Platform.V7_1X1X1_2TC,
    target_lib.Platform.V7_1X2X1,
    target_lib.Platform.V7_2X2X1,
    target_lib.Platform.B200_2,
    target_lib.Platform.B200_4,
    target_lib.Platform.B200_8,
})

_FULL_SEQ_LEN = 2048


def build_model(
    config: torchtitan_muse_glimmer.muse_glimmer.MuseGlimmerModel.Config,
    device: str,
    world_size: int,
    dtype: torch.dtype,
) -> nn.Module:
  """Builds the torchtitan model, checkpoints and SimpleFSDP-shards it."""
  torch.manual_seed(12345)
  model = torchtitan_muse_glimmer.build_model(config, device, dtype)
  model.train()

  # Per-layer activation checkpointing, as SimpleFSDP expects.
  for layer in model.layers.values():
    layer.forward = functools.partial(
        checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
    )

  mesh = torch.distributed.device_mesh.init_device_mesh(
      device, (world_size,), mesh_dim_names=("fsdp",)
  )
  model = simple_fsdp.data_parallel(
      model,
      device_mesh=mesh,
      mode="fully_shard",
      mp_policy=simple_fsdp.MixedPrecisionPolicy(
          param_dtype=dtype,
          reduce_dtype=dtype,
      ),
  )
  if device == target_lib.DeviceKind.TPU.value:
    # Lets torch_tpu fuse the SimpleFSDP collectives with surrounding ops.
    model.forward = torch.tpu.distributed.spmd_safe(model.forward)
  return model


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    stepper_kwargs={
        "use_single_trace": False,
        "compute_loss": torchtitan_muse_glimmer.cross_entropy_loss,
    },
    compile_config=compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=False
    ),
    skipped_run_modes={
        "eager_default",
        "eager_optimized",
        "eager_defer_never_and_launch_blocking",
    },
)
def muse_glimmer_30b_4_layers_simple_fsdp_training(
    ctx: context_lib.Context,
    batch_size: int = 1,
) -> tuple[
    nn.Module,
    tuple[torch.Tensor, ...],
    dict[str, Any],
    torch.optim.Optimizer,
]:
  """Benchmark factory for Muse Glimmer (30B, 4 layers) SimpleFSDP training."""
  if batch_size != 1:
    raise NotImplementedError("Batch size > 1 is not implemented yet.")

  if ctx.target.platform not in _SUPPORTED_PLATFORMS:
    raise target_lib.UnsupportedBenchmark(
        "Muse Glimmer SimpleFSDP training is not supported on platform"
        f" {ctx.target.platform}"
    )
  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Muse Glimmer SimpleFSDP training requires an initialized "
        "torch.distributed process group."
    )
  world_size = torch.distributed.get_world_size()
  rank = torch.distributed.get_rank()

  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    seq_len = 128
    num_layers = 2
    flavor = "debugmodel"
  else:
    seq_len = _FULL_SEQ_LEN
    num_layers = 4
    flavor = "30B"

  device_str = ctx.device_kind.value
  config = torchtitan_muse_glimmer.text_config(flavor, num_layers, seq_len)
  logging.info(
      "Rank %d/%d: building torchtitan Muse Glimmer (%s arch) num_layers=%d"
      " seq_len=%d with SimpleFSDP fully_shard",
      rank,
      world_size,
      flavor,
      num_layers,
      seq_len,
  )
  model = build_model(
      config,
      device_str,
      world_size,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
  )
  tokens, labels = torchtitan_muse_glimmer.make_inputs(
      config.vocab_size, seq_len, rank, device_str
  )
  opt = torch.optim.AdamW(
      model.parameters(), lr=1e-4, fused=True, capturable=True
  )
  return model, (tokens,), {"labels": labels}, opt
