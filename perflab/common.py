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

"""Common definitions for benchmarks."""

import enum
import random
from typing import Any
import torch
from torch.utils import _pytree as pytree

_RANDOM_SEED = 0


def seed_rngs() -> None:
  """Seeds the Python and PyTorch RNGs with the given seed."""
  random.seed(_RANDOM_SEED)
  torch.manual_seed(_RANDOM_SEED)


def prepare_inputs(model: Any, inputs: Any) -> None:
  """Canonicalizes model parameters, buffers, and inputs in-place."""
  seen: set[int] = set()

  def _prepare_tensor(t: Any) -> Any:
    if isinstance(t, torch.Tensor) and not t.is_meta and id(t) not in seen:
      seen.add(id(t))
      with torch.no_grad():
        # TODO(basioli) Remove +0 once we fix the full reinterpret view path.
        t.data = (
            torch.clone(t.data, memory_format=torch.contiguous_format) + 0
        ).to(t.dtype)
    return t

  if isinstance(model, torch.nn.Module):
    for p in model.parameters():
      _prepare_tensor(p)
    for b in model.buffers():
      _prepare_tensor(b)
  else:
    pytree.tree_map(_prepare_tensor, model)
  pytree.tree_map(_prepare_tensor, inputs)
  if torch.accelerator.is_available():
    torch.accelerator.synchronize()


class RunMode(enum.Enum):
  """The mode to run the benchmark in.

  Make sure that no entry is a prefix of the other. Run mode name is appended
  to test names, and MLCompass runs tests based on a prefix match, it can lead
  to duplicate entries. For e.g., test_model_eager will match both
  test_model_eager and test_model_eager_optimized. Hence, we use
  eager_default and eager_optimized.
  """

  EAGER_DEFAULT = (  # Run the model in eager mode with DeferNever.
      "eager_default"
  )
  EAGER_OPTIMIZED = (  # Run the model in eager mode with DeferAndFuse.
      "eager_optimized"
  )
  # Run the model in eager mode with DeferNeverAndLaunchBlocking.
  EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING = (
      "eager_defer_never_and_launch_blocking"
  )
  COMPILED = "compiled"  # Run the model with torch.compile.


def is_torch_compile(run_mode: RunMode) -> bool:
  """Returns whether the given run mode uses torch.compile."""
  return run_mode == RunMode.COMPILED
