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

"""Tests SingleTraceTrainer on SimpleFSDP (DTensor) models with 2 ranks.

With `--device=cpu` (the default) the ranks use gloo and the traced step runs
uncompiled (`compile_fn` returns the FX graph). With `--device=tpu` they use
`tpu_dist` and StaticCompiler, with every forward wrapped in
`spmd_safe`. Either way the trained losses and parameter shards are compared
against eager SimpleFSDP plus the same functional optimizer applied to the
local gradient shards.
"""

from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.distributed import tensor as dtensor
import torch.multiprocessing as mp
from torch.utils import checkpoint as checkpoint_utils
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.optimizers import adamw
from torch_tpu._internal.optimizers import muon
from torch_tpu._internal.optimizers import sgd
from torch_tpu._internal.optimizers import single_trace_trainer
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils
from torchtitan.experiments.graph_trainer import simple_fsdp

_DEVICE = flags.DEFINE_enum(
    "device", "cpu", ["cpu", "tpu"], "Device the ranks train on."
)

_WORLD_SIZE = 2
_BATCH = 4
_IN_DIM = 16
_HIDDEN = 32
_NUM_STEPS = 3
# Compiled TPU steps fuse differently from eager ones.
_TOLERANCES = {"cpu": {}, "tpu": {"atol": 2e-3, "rtol": 2e-3}}


class _Mlp(nn.Module):
  """Two linear layers and a buffer, with the first block checkpointed.

  Under fully sharded SimpleFSDP with 2 ranks, `fc1` (dim 0 = 32) is sharded
  and `fc2` (dim 0 = 1 < world size) is replicated. Activation checkpointing
  recomputes `fc1` (and its all-gather) in backward, as in the torchtitan
  benchmarks. `unused` never contributes to the loss.
  """

  def __init__(self, with_unused: bool = False):
    super().__init__()
    self.fc1 = nn.Linear(_IN_DIM, _HIDDEN)
    self.fc2 = nn.Linear(_HIDDEN, 1)
    self.unused = nn.Linear(_HIDDEN, _HIDDEN) if with_unused else None
    self.register_buffer("scale", torch.full((_HIDDEN,), 0.5))

  def _block(self, x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.fc1(x)) * self.scale

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = checkpoint_utils.checkpoint(self._block, x, use_reentrant=False)
    return self.fc2(h)


class _DTensorBufferMlp(nn.Module):
  """A linear layer scaled by a replicated DTensor buffer."""

  def __init__(self, mesh: torch.distributed.device_mesh.DeviceMesh):
    super().__init__()
    self.fc = nn.Linear(_IN_DIM, 1)
    self.register_buffer(
        "scale",
        dtensor.distribute_tensor(
            torch.full((1,), 0.5), mesh, [dtensor.Replicate()]
        ),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x) * self.scale.to_local()


class _BufferOnly(nn.Module):
  """A parameterless module scaled by a replicated DTensor buffer."""

  def __init__(self, mesh: torch.distributed.device_mesh.DeviceMesh):
    super().__init__()
    self.register_buffer(
        "scale",
        dtensor.distribute_tensor(torch.ones(2), mesh, [dtensor.Replicate()]),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.scale.to_local()


def _init_process_group(device: str) -> None:
  torch.distributed.init_process_group(
      backend="tpu_dist" if device == "tpu" else "gloo", init_method="env://"
  )


def _mesh(device: str) -> torch.distributed.device_mesh.DeviceMesh:
  return torch.distributed.device_mesh.init_device_mesh(
      device, (_WORLD_SIZE,), mesh_dim_names=("fsdp",)
  )


def _build_fsdp_mlp(
    device: str, mode: str = "fully_shard", with_unused: bool = False
) -> nn.Module:
  torch.manual_seed(1234)  # Same weights on every rank and every call.
  with torch.device(device):
    model = _Mlp(with_unused=with_unused)
  return simple_fsdp.data_parallel(model, device_mesh=_mesh(device), mode=mode)


def _make_batch(step: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns a per-rank batch, as in data-parallel training."""
  generator = torch.Generator().manual_seed(
      100 * step + torch.distributed.get_rank()
  )
  inputs = torch.randn(_BATCH, _IN_DIM, generator=generator)
  targets = torch.randn(_BATCH, 1, generator=generator)
  return inputs.to(device), targets.to(device)


def _make_optimizer(name: str) -> Any:
  if name == "adamw":
    return adamw.ReferenceAdamW(lr=1e-2, weight_decay=1e-2)
  return sgd.ReferenceSgd(lr=1e-1, momentum=0.9)


def _make_trainer(
    model: nn.Module, device: str, optimizer: Any = None
) -> single_trace_trainer.SingleTraceTrainer:
  """SingleTraceTrainer: uncompiled on CPU, StaticCompiler + spmd_safe on TPU."""
  if device == "tpu":
    model.forward = torch.tpu.distributed.spmd_safe(model.forward)
    return single_trace_trainer.SingleTraceTrainer(
        model, optimizer=optimizer, compile_fn=compiler.StaticCompiler()
    )
  return single_trace_trainer.SingleTraceTrainer(
      model, optimizer=optimizer, compile_fn=lambda gm, _: gm
  )


def _safe_local_params(model: nn.Module) -> dict[str, torch.Tensor]:
  return {
      name.replace(".", "_"): p.detach().to_local().clone()
      for name, p in model.named_parameters()
  }


def _eager_reference(
    optimizer_name: str, device: str, mode: str
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
  """Eager SimpleFSDP fwd/bwd + the functional optimizer on local grads."""
  model = _build_fsdp_mlp(device, mode)
  optimizer = _make_optimizer(optimizer_name)
  param_group = optimizer.init_param_group(_safe_local_params(model))
  losses = []
  for step in range(_NUM_STEPS):
    # Run with the current local shards.
    with torch.no_grad():
      for name, p in model.named_parameters():
        p.to_local().copy_(param_group.params[name.replace(".", "_")])
    inputs, targets = _make_batch(step, device)
    model.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    grads = {
        name.replace(".", "_"): p.grad.to_local()
        for name, p in model.named_parameters()
    }
    param_group = optimizer(param_group, grads)
    losses.append(loss.detach())
  return losses, dict(param_group.params)


def _worker_matches_eager(optimizer_name: str, device: str, mode: str) -> None:
  _init_process_group(device)
  try:
    tolerances = _TOLERANCES[device]
    ref_losses, ref_params = _eager_reference(optimizer_name, device, mode)

    model = _build_fsdp_mlp(device, mode)
    placements = {
        name: tuple(p.placements) for name, p in model.named_parameters()
    }
    if mode == "fully_shard":
      assert placements["fc1.weight"] == (dtensor.Shard(0),), placements
    else:
      assert placements["fc1.weight"] == (dtensor.Replicate(),), placements
    assert placements["fc2.weight"] == (dtensor.Replicate(),), placements
    trainer = _make_trainer(model, device, _make_optimizer(optimizer_name))
    # The optimizer state holds local shards only.
    for name, p in model.named_parameters():
      local = trainer.param_group.params[name.replace(".", "_")]
      assert local.shape == p.to_local().shape, (name, local.shape)

    step_fn = trainer.make_compiled_train_step(*_make_batch(0, device))
    for step in range(_NUM_STEPS):
      loss = step_fn(*_make_batch(step, device))
      test_utils.assert_close(loss, ref_losses[step], **tolerances)

    for name, p in trainer.params.items():
      assert isinstance(p, dtensor.DTensor), name
      assert tuple(p.placements) == placements[name], name
      test_utils.assert_close(
          p.to_local(),
          ref_params[name.replace(".", "_")],
          preamble=name,
          **tolerances,
      )
  finally:
    torch.distributed.destroy_process_group()


def _worker_grads_without_optimizer(device: str) -> None:
  _init_process_group(device)
  try:
    tolerances = _TOLERANCES[device]
    ref_model = _build_fsdp_mlp(device, with_unused=True)
    inputs, targets = _make_batch(0, device)
    ref_loss = nn.functional.mse_loss(ref_model(inputs), targets)
    ref_loss.backward()

    model = _build_fsdp_mlp(device, with_unused=True)
    trainer = _make_trainer(model, device)
    loss = trainer.make_compiled_train_step(inputs, targets)(inputs, targets)
    test_utils.assert_close(loss, ref_loss.detach(), **tolerances)
    for (name, p), ref_p in zip(
        model.named_parameters(), ref_model.parameters()
    ):
      assert isinstance(p.grad, dtensor.DTensor), name
      if name.startswith("unused."):
        # Unused parameters get zero gradients rather than None.
        assert ref_p.grad is None, name
        test_utils.assert_close(
            p.grad.to_local(), torch.zeros_like(p.grad.to_local())
        )
        continue
      assert p.grad.placements == ref_p.grad.placements, name
      test_utils.assert_close(
          p.grad.to_local(), ref_p.grad.to_local(), preamble=name, **tolerances
      )
  finally:
    torch.distributed.destroy_process_group()


def _worker_dtensor_buffer(device: str) -> None:
  _init_process_group(device)
  try:
    tolerances = _TOLERANCES[device]
    torch.manual_seed(1234)
    with torch.device(device):
      model = _DTensorBufferMlp(_mesh(device))
    inputs, targets = _make_batch(0, device)
    ref_loss = nn.functional.mse_loss(model(inputs), targets)

    trainer = _make_trainer(model, device)
    loss = trainer.make_compiled_train_step(inputs, targets)(inputs, targets)
    test_utils.assert_close(loss, ref_loss.detach(), **tolerances)
    buffers = trainer._bufs  # pylint: disable=protected-access
    assert isinstance(buffers["scale"], dtensor.DTensor), buffers
    assert buffers["scale"].placements == (dtensor.Replicate(),), buffers
  finally:
    torch.distributed.destroy_process_group()


def _worker_muon_rejects_sharded_params(device: str) -> None:
  _init_process_group(device)
  try:
    try:
      single_trace_trainer.SingleTraceTrainer(
          _build_fsdp_mlp(device), optimizer=muon.ReferenceMuon()
      )
    except NotImplementedError as e:
      assert "does not support sharded parameters" in str(e), e
      assert "fc1.weight" in str(e), e
    else:
      raise AssertionError("Muon accepted sharded parameters.")
    # Replicated (DDP) parameters are full matrices, so Muon accepts them.
    single_trace_trainer.SingleTraceTrainer(
        _build_fsdp_mlp(device, mode="replicate"),
        optimizer=muon.ReferenceMuon(),
    )
  finally:
    torch.distributed.destroy_process_group()


def _worker_rejects_buffer_only_subclass_model(device: str) -> None:
  _init_process_group(device)
  try:
    with torch.device(device):
      model = _BufferOnly(_mesh(device))
    trainer = single_trace_trainer.SingleTraceTrainer(
        model, compile_fn=lambda gm, _: gm
    )
    inputs = torch.ones(2, device=device)
    try:
      trainer.make_compiled_train_step(inputs)
    except NotImplementedError as e:
      assert "without parameters" in str(e), e
    else:
      raise AssertionError("Traced a buffer-only model with DTensor buffers.")
  finally:
    torch.distributed.destroy_process_group()


class DistributedSingleTraceTrainerTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  @parameterized.product(
      optimizer_name=("adamw", "sgd"), mode=("fully_shard", "replicate")
  )
  def test_simple_fsdp_matches_eager(self, optimizer_name, mode):
    distributed_utils.dist_run(
        _WORLD_SIZE, _worker_matches_eager, optimizer_name, _DEVICE.value, mode
    )

  def test_simple_fsdp_grads_without_optimizer(self):
    distributed_utils.dist_run(
        _WORLD_SIZE, _worker_grads_without_optimizer, _DEVICE.value
    )

  def test_dtensor_buffer(self):
    distributed_utils.dist_run(
        _WORLD_SIZE, _worker_dtensor_buffer, _DEVICE.value
    )

  def test_muon_rejects_sharded_params(self):
    distributed_utils.dist_run(
        _WORLD_SIZE, _worker_muon_rejects_sharded_params, _DEVICE.value
    )

  def test_rejects_buffer_only_subclass_model(self):
    distributed_utils.dist_run(
        _WORLD_SIZE, _worker_rejects_buffer_only_subclass_model, _DEVICE.value
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
