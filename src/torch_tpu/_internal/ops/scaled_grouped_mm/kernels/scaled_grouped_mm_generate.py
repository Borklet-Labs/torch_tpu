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

"""Generate rowwise scaled grouped mm Pallas kernel MLIR module embedded C++ files."""

from collections.abc import Sequence
from typing import Any

from absl import app
from absl import flags
import jax.numpy as jnp
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import rowwise_scaled_grouped_mm_kernel as kernel
from torch_tpu._internal.pallas import pallas_kernel_generate_utils as kernel_utils

_HEADER = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "header",
    None,
    "Header file to include in the generated kernel string.",
)

_IMPLEMENTATION = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "implementation",
    None,
    "Implementation file to include in the generated kernel string.",
)

_KERNEL_NAME = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "kernel_name",
    None,
    "Name of the kernel to generate. Used as prefix for C++ symbols.",
)

_LHS_DTYPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "lhs_dtype",
    "float8_e4m3fn",
    "Input dtype of lhs (float8_e4m3fn or float8_e5m2).",
)

_RHS_DTYPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "rhs_dtype",
    "float8_e4m3fn",
    "Input dtype of rhs (float8_e4m3fn or float8_e5m2).",
)

_OUT_DTYPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "out_dtype",
    "bfloat16",
    "Output dtype of the kernel (float32 or bfloat16).",
)

_TRANSPOSE_RHS = flags.DEFINE_boolean(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "transpose_rhs",
    True,
    "Whether rhs is transposed ([G, N, K]).",
)

_TILE_M = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_m",
    512,
    "Tile size along M dimension.",
)

_TILE_K = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_k",
    2048,
    "Tile size along K dimension.",
)

_TILE_N = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_n",
    768,
    "Tile size along N dimension.",
)


def _get_dtype(name: str) -> Any:
  if name == "float8_e4m3fn":
    return jnp.float8_e4m3fn
  elif name == "float8_e5m2":
    return jnp.float8_e5m2
  elif name == "bfloat16":
    return jnp.bfloat16
  elif name == "float32":
    return jnp.float32
  else:
    raise ValueError(f"Unsupported dtype: {name}")


def generate_kernel_module(
    *,
    lhs_dtype_name: str,
    rhs_dtype_name: str,
    out_dtype_name: str,
    transpose_rhs: bool,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> str:
  """Generates a StableHLO MLIR module string for the rowwise scaled grouped mm kernel."""
  lhs_dtype = _get_dtype(lhs_dtype_name)
  rhs_dtype = _get_dtype(rhs_dtype_name)
  out_dtype = _get_dtype(out_dtype_name)

  return kernel.export_mlir_module(
      lhs_dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      out_dtype=out_dtype,
      transpose_rhs=transpose_rhs,
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
  )


def main(argv: Sequence[str]) -> None:
  del argv
  if not _HEADER.value or not _IMPLEMENTATION.value:
    raise ValueError("--header and --implementation must be specified.")
  if not _KERNEL_NAME.value:
    raise ValueError("--kernel_name must be specified.")

  mlir_str = generate_kernel_module(
      lhs_dtype_name=_LHS_DTYPE.value,
      rhs_dtype_name=_RHS_DTYPE.value,
      out_dtype_name=_OUT_DTYPE.value,
      transpose_rhs=_TRANSPOSE_RHS.value,
      tile_m=_TILE_M.value,
      tile_k=_TILE_K.value,
      tile_n=_TILE_N.value,
  )

  kernel_utils.generate_embedded_file(
      _HEADER.value,
      _IMPLEMENTATION.value,
      [(_KERNEL_NAME.value, mlir_str.encode("utf-8"))],
  )


if __name__ == "__main__":
  app.run(main)
