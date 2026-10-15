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

"""TPU target that the precompiled scaled_mm kernels are exported for."""

from collections.abc import Iterator
import contextlib
import inspect
from typing import Any

import jax

# Device kind the AOT export lowers for, independent of the build host. v7x is
# the FP8 training target.
AOT_DEVICE_KIND = "TPU7x"


def _abstract_tpu_device() -> Any:
  """Abstract TPU device for Google3 HEAD (with platform) and OSS jax==0.9.2."""
  cls = jax.sharding.AbstractDevice
  if "platform" in inspect.signature(cls).parameters:
    return cls(AOT_DEVICE_KIND, 1, "tpu")  # type: ignore[call-arg]
  return cls(AOT_DEVICE_KIND, 1)  # pyrefly: ignore[missing-argument]


@contextlib.contextmanager
def use_abstract_tpu_mesh() -> Iterator[None]:
  """Lowers for TPU `AOT_DEVICE_KIND`, whatever devices the host has."""
  abstract_mesh = jax.sharding.AbstractMesh(
      (1,),
      ("x",),
      axis_types=(jax.sharding.AxisType.Auto,),
      abstract_device=_abstract_tpu_device(),
  )
  with jax.sharding.use_abstract_mesh(abstract_mesh):
    yield
