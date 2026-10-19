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

"""Tracks the tensor memory that code would allocate, e.g. under FakeTensorMode.

FakeTensorMode allocates no storage, so the FakeTensorMode CPU smoke test
cannot see a change that only increases the memory a benchmark needs on a
device. `TensorMemoryTracker` counts the bytes of the tensor storages that are
alive while it is active, the same way for fake and real tensors.

The smoke test tracks only the construction of each benchmark (the model, its
inputs and its optimizer), not the gradients, optimizer state and activations
of its steps. It checks the construction peak against the HBM of a v7
TensorCore (`MAX_CONSTRUCTION_PEAK_BYTES`) at every run scope, although the
presubmit-scope benchmark runs, e.g. the v5e `benchmark-smoke` tests, have less.
So it catches benchmarks that cannot be built on any device, not steps that run
out of memory.
"""

import functools
from typing import Any
import weakref

from absl import logging
import torch
from torch.utils import _python_dispatch
from torch.utils import _pytree as pytree
from torch.utils import weak

_GIB = 1 << 30

# Benchmarks build their models on the device, so building one cannot need more
# memory than the device has: the 94.74 GiB of HBM of a v7 TensorCore, the
# smallest device that runs the full-size models nightly. The device needs some
# of it for programs and allocator overhead too, so this is a lower bound on
# what fails. Building a model in float32 and then casting it to bfloat16 needs
# about twice the memory of the bfloat16 model: cl/999488021 built the 52 GiB
# of bfloat16 weights of Qwen3.5-397B (4 layers) that way, which needs 104 GiB.
MAX_CONSTRUCTION_PEAK_BYTES = 101_724_453_888


class TensorMemoryTracker(_python_dispatch.TorchDispatchMode):
  """Tracks the live and peak bytes of the storages of the tensors created.

  Each storage that an op returns while the tracker is active counts once, from
  then until it is freed, however many tensors (e.g. views) share it. Storages
  are tracked through weak references, so the tracker neither keeps them alive
  nor adds weak references to tensors (`torch.utils.swap_tensors`, which
  `nn.Module.to` uses, refuses tensors that have them).

  Attributes:
    live_bytes: Bytes of the tracked storages that are still alive.
    peak_bytes: Largest `live_bytes` since the tracker was created.
  """

  def __init__(self):
    super().__init__()
    self._storages: weak.WeakIdKeyDictionary = weak.WeakIdKeyDictionary()
    self.live_bytes = 0
    self.peak_bytes = 0

  def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> Any:
    out = func(*args, **(kwargs or {}))
    for leaf in pytree.tree_leaves(out):
      if isinstance(leaf, torch.Tensor):
        self._track(leaf.untyped_storage())
    return out

  def _track(self, storage: torch.UntypedStorage) -> None:
    if storage in self._storages:
      return
    nbytes = storage.nbytes()
    # The weak reference calls `_release` when the storage is freed.
    self._storages[storage] = weakref.ref(
        storage, functools.partial(self._release, nbytes)
    )
    self.live_bytes += nbytes
    self.peak_bytes = max(self.peak_bytes, self.live_bytes)

  def _release(self, nbytes: int, unused_ref: weakref.ref) -> None:
    self.live_bytes -= nbytes


def check_construction_peak(
    tracker: TensorMemoryTracker,
    *,
    name: str,
    max_peak_bytes: int = MAX_CONSTRUCTION_PEAK_BYTES,
) -> None:
  """Raises if building `name` needed more tensor memory than a device has.

  Args:
    tracker: A tracker that was active while `name` was built, and whose tracked
      storages that `name` keeps are still alive.
    name: The name of what was built, for the messages.
    max_peak_bytes: The most tensor memory that building `name` may use.

  Raises:
    AssertionError: If the peak memory while building exceeded
      `max_peak_bytes`.
  """
  logging.info(
      "Building %s peaked at %.2f GiB of tensors and keeps %.2f GiB.",
      name,
      tracker.peak_bytes / _GIB,
      tracker.live_bytes / _GIB,
  )
  if tracker.peak_bytes > max_peak_bytes:
    raise AssertionError(
        f"building {name} peaked at {tracker.peak_bytes / _GIB:.2f} GiB of"
        f" tensors, more than the {max_peak_bytes / _GIB:.2f} GiB that fit on"
        f" the device, and keeps {tracker.live_bytes / _GIB:.2f} GiB (e.g. the"
        " model is built in float32 and then cast to a lower precision instead"
        " of being built in it)"
    )
