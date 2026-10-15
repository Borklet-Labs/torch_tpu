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

"""Grouped FP8 matmul Pallas kernel with rowwise scales applied in-kernel.

Implements `torch._scaled_grouped_mm` for a 2-D `a` and a 3-D `b` with rowwise
scales. Group `g` owns rows `[offs[g - 1], offs[g])` of `a` (with
`offs[-1] := 0`) and computes

  out[m, n] = scale_a[m] * scale_b[g, n] * sum_k a[m, k] * b[g, k, n]

Rows at or past `offs[G - 1]` are zero, as in `jax.lax.ragged_dot`. The scales
multiply the result (they are the dequantization factors, i.e. the reciprocals
of the quantization scales).

The FP8 operands feed the MXU directly and accumulate in f32. The scales are
applied once per output tile after the K reduction, so no scaled or upcast copy
of either operand is materialized.

Schedule: the grid is `(n tiles, visits, k tiles)`. A visit is one (group,
m-tile) pair; each group visits every m-tile its rows overlap, in row order, so
an m-tile shared by consecutive groups is visited once per group,
consecutively, and each visit writes only its group's rows. The number of
visits is computed on device from `offs`, so routing changes neither recompile
nor sync with the host. A virtual group `G` covers the rows past
`offs[G - 1]` and writes zeros there.
"""

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from torch_tpu._internal.ops.scaled_mm.kernels import aot_target

__all__ = [
    "default_tile",
    "export",
    "export_mlir_module",
    "get_shapes",
    "make_group_metadata",
    "rowwise_scaled_grouped_mm",
]

_DEFAULT_TILE_M = 512
# Largest tile picked by default along K and N. At tile_m=512 this keeps the
# double-buffered blocks plus the f32 accumulator under ~24 MiB of VMEM.
_MAX_DEFAULT_TILE = 2048
_LANES = 128
# v7x has 64 MiB of VMEM per TensorCore; v5e and v6e have 128 MiB.
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024

_FP8_DTYPES = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))


def _round_up(x: Any, multiple: int) -> Any:
  return pl.cdiv(x, multiple) * multiple


def default_tile(dim: int) -> int:
  """Largest multiple of 128 that divides `dim` (rounded up to 128), <= 2048."""
  padded = _round_up(dim, _LANES)
  for tile in range(min(padded, _MAX_DEFAULT_TILE), 0, -_LANES):
    if padded % tile == 0:
      return tile
  raise AssertionError("unreachable: 128 divides every multiple of 128")


def make_group_metadata(
    offs: jax.Array,
    *,
    num_rows: Any,
    tile_m: int,
    padded_rows: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds the visit schedule from torch-style cumulative group ends.

  Only shape-polymorphic ops are used, so `num_rows` and the number of groups
  may be symbolic at export time.

  Args:
    offs: `[G]` int32 cumulative group ends. Values are clamped to `[0,
      num_rows]` and made non-decreasing, so malformed input cannot index out of
      bounds.
    num_rows: rows of `a` (M).
    tile_m: row tile.
    padded_rows: M rounded up to a multiple of `tile_m`. Defaults to
      `_round_up(num_rows, tile_m)`.

  Returns:
    group_offsets: `[G + 2]` int32. Group `g` (`g == G` is the virtual group)
      owns rows `[group_offsets[g], group_offsets[g + 1])`.
    group_ids: `[padded_rows // tile_m + G]` int32 group of each visit.
    m_tile_ids: `[padded_rows // tile_m + G]` int32 m-tile of each visit.
    num_visits: int32 scalar, the number of valid leading visits.
  """
  if padded_rows is None:
    padded_rows = _round_up(num_rows, tile_m)
  num_groups = offs.shape[0]
  tiles_m = padded_rows // tile_m
  ends = lax.cummax(jnp.clip(offs.astype(jnp.int32), 0, num_rows), axis=0)
  group_offsets = jnp.concatenate([
      jnp.zeros((1,), jnp.int32),
      ends,
      jnp.full((1,), num_rows, jnp.int32),
  ])
  starts = group_offsets[:-1]
  stops = group_offsets[1:]
  first_tile = starts // tile_m
  last_tile = (stops - 1) // tile_m
  tiles_per_group = jnp.where(stops > starts, last_tile - first_tile + 1, 0)
  visits_end = jnp.cumsum(tiles_per_group)
  visits_start = visits_end - tiles_per_group
  visit = lax.iota(jnp.int32, tiles_m + num_groups)
  # Group of each visit: the number of groups whose visits end at or before it.
  # Visits past `num_visits` are clamped to the virtual group (`num_groups`).
  group_ids = jnp.sum(
      (visits_end[None, :] <= visit[:, None]).astype(jnp.int32), axis=1
  )
  group_ids = jnp.minimum(group_ids, num_groups)
  m_tile_ids = first_tile[group_ids] + visit - visits_start[group_ids]
  m_tile_ids = jnp.clip(m_tile_ids, 0, tiles_m - 1)
  return group_offsets, group_ids, m_tile_ids, visits_end[-1]


def _kernel(
    group_offsets_ref,  # SMEM [G + 2] int32
    group_ids_ref,  # SMEM [V] int32
    m_tile_ids_ref,  # SMEM [V] int32
    num_groups_ref,  # SMEM [1] int32
    m_probe_ref,  # SMEM [M] int8
    k_probe_ref,  # SMEM [K] int8
    n_probe_ref,  # SMEM [N] int8
    a_ref,  # [tile_m, tile_k] fp8
    b_ref,  # [tile_k, tile_n] or [tile_n, tile_k] (transpose_rhs) fp8
    sa_ref,  # [1, tile_m] f32
    sb_ref,  # [1, tile_n] f32
    out_ref,  # [tile_m, tile_n] out_dtype
    acc_ref,  # [tile_m, tile_n] f32 VMEM accumulator
    *,
    tile_m: int,
    tiles_k: int,
    transpose_rhs: bool,
):
  del m_probe_ref, k_probe_ref, n_probe_ref
  visit = pl.program_id(1)
  k_i = pl.program_id(2)
  group = group_ids_ref[visit]
  is_real_group = group < num_groups_ref[0]
  group_start = group_offsets_ref[group]
  group_end = group_offsets_ref[group + 1]
  row0 = m_tile_ids_ref[visit] * tile_m

  def store(tile):
    # Rows of other groups in a shared m-tile are written by their own visits,
    # which are consecutive, so the output block stays resident in between.
    whole_tile = (group_start <= row0) & (row0 + tile_m <= group_end)

    @pl.when(whole_tile)
    def _():
      out_ref[...] = tile.astype(out_ref.dtype)

    @pl.when(jnp.logical_not(whole_tile))
    def _():
      rows = row0 + lax.broadcasted_iota(jnp.int32, tile.shape, 0)
      in_group = (rows >= group_start) & (rows < group_end)
      out_ref[...] = jnp.where(
          in_group, tile, out_ref[...].astype(jnp.float32)
      ).astype(out_ref.dtype)

  def dot():
    contracting = (1,) if transpose_rhs else (0,)
    return lax.dot_general(
        a_ref[...],
        b_ref[...],
        (((1,), contracting), ((), ())),
        preferred_element_type=jnp.float32,
    )

  def scaled(acc):
    return acc * sa_ref[0][:, None] * sb_ref[...]

  @pl.when(is_real_group)
  def _():
    partial = dot()

    @pl.when(k_i == 0)
    def _():
      acc_ref[...] = partial

    @pl.when(k_i > 0)
    def _():
      acc_ref[...] += partial

    @pl.when(k_i == tiles_k - 1)
    def _():
      store(scaled(acc_ref[...]))

  @pl.when(jnp.logical_not(is_real_group) & (k_i == tiles_k - 1))
  def _():
    store(jnp.zeros(out_ref.shape, jnp.float32))


def _check_args(a, b, scale_a, scale_b, offs, out_dtype, transpose_rhs, tiles):
  if a.ndim != 2 or b.ndim != 3:
    raise ValueError(
        f"Expected a 2-D `a` and a 3-D `b`, got {a.shape} and {b.shape}."
    )
  for name, x in (("a", a), ("b", b)):
    if x.dtype not in _FP8_DTYPES:
      raise ValueError(f"`{name}` must be float8_e4m3fn or float8_e5m2.")
  if jnp.dtype(out_dtype) not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"out_dtype must be bfloat16 or float32, got {out_dtype}.")
  m, k = a.shape
  if transpose_rhs:
    num_groups, n, k_b = b.shape
  else:
    num_groups, k_b, n = b.shape
  if k_b != k:
    raise ValueError(f"Contraction mismatch: a is {a.shape}, b is {b.shape}.")
  if scale_a.shape != (m,):
    raise ValueError(f"scale_a must be [{m}], got {scale_a.shape}.")
  if scale_b.shape != (num_groups, n):
    raise ValueError(
        f"scale_b must be [{num_groups}, {n}], got {scale_b.shape}."
    )
  if offs.shape != (num_groups,) or offs.dtype != jnp.int32:
    raise ValueError(
        f"offs must be int32 [{num_groups}], got {offs.dtype} {offs.shape}."
    )
  for name, tile in tiles.items():
    if tile % _LANES:
      raise ValueError(f"{name}={tile} must be a multiple of {_LANES}.")
  return m, k, n, num_groups


def _cost_estimate(m, k, n, num_groups, out_dtype) -> pl.CostEstimate | None:
  try:
    return pl.CostEstimate(
        flops=2 * m * k * n,
        transcendentals=0,
        bytes_accessed=(
            m * k
            + num_groups * k * n
            + 4 * (m + num_groups * n)
            + jnp.dtype(out_dtype).itemsize * m * n
        ),
    )
  except (
      # Symbolic shapes (AOT export) have no concrete cost.
      jax.errors.ConcretizationTypeError,
      getattr(
          jax.core,
          "InconclusiveDimensionOperation",
          type("Dummy", (Exception,), {}),
      ),
      ValueError,
      TypeError,
  ):
    return None


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_dtype",
        "transpose_rhs",
        "tile_m",
        "tile_k",
        "tile_n",
        "interpret",
    ),
)
def rowwise_scaled_grouped_mm(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    offs: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    transpose_rhs: bool = False,
    tile_m: int = _DEFAULT_TILE_M,
    tile_k: int | None = None,
    tile_n: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Grouped FP8 matmul with rowwise scales; see the module docstring.

  Args:
    a: `[M, K]` float8_e4m3fn or float8_e5m2.
    b: `[G, K, N]`, or `[G, N, K]` if `transpose_rhs`; float8_e4m3fn or
      float8_e5m2.
    scale_a: `[M]` f32 row scales of `a`.
    scale_b: `[G, N]` f32 column scales of each `b[g]`.
    offs: `[G]` int32 cumulative group ends, non-decreasing, `offs[G-1] <= M`.
    out_dtype: output dtype, bfloat16 or float32.
    transpose_rhs: whether `b` is stored `[G, N, K]` (K-minor).
    tile_m: row tile, a multiple of 128.
    tile_k: K tile, a multiple of 128. Defaults to `default_tile(K)`.
    tile_n: N tile, a multiple of 128. Defaults to `default_tile(N)`.
    interpret: run the kernel in the Pallas interpreter (for CPU tests).

  Returns:
    `[M, N]` in `out_dtype`.
  """
  tile_k_val = (
      tile_k
      if tile_k is not None
      else (default_tile(a.shape[1]) if a.ndim == 2 else 128)
  )
  n_dim = (
      b.shape[1]
      if (b.ndim == 3 and transpose_rhs)
      else (b.shape[2] if b.ndim == 3 else 128)
  )
  tile_n_val = tile_n if tile_n is not None else default_tile(n_dim)
  m, k, n, num_groups = _check_args(
      a,
      b,
      scale_a,
      scale_b,
      offs,
      out_dtype,
      transpose_rhs,
      dict(tile_m=tile_m, tile_k=tile_k_val, tile_n=tile_n_val),
  )
  tile_k = tile_k_val
  tile_n = tile_n_val
  if (
      isinstance(m, int)
      and isinstance(num_groups, int)
      and not (m and num_groups)
  ):
    return jnp.zeros((m, n), out_dtype)

  # Zero-pad to tile multiples. Padded K adds zero products; padded rows and
  # columns are sliced off. Under symbolic shapes the pads are always emitted
  # and fold away when the shapes turn out aligned.
  padded_m = _round_up(m, tile_m)
  padded_k = _round_up(k, tile_k)
  padded_n = _round_up(n, tile_n)
  if padded_m != m or padded_k != k:
    a = jnp.pad(a, ((0, padded_m - m), (0, padded_k - k)))
  if padded_k != k or padded_n != n:
    if transpose_rhs:
      b = jnp.pad(b, ((0, 0), (0, padded_n - n), (0, padded_k - k)))
    else:
      b = jnp.pad(b, ((0, 0), (0, padded_k - k), (0, padded_n - n)))
  scale_a = scale_a.astype(jnp.float32)
  scale_b = scale_b.astype(jnp.float32)
  if padded_m != m:
    scale_a = jnp.pad(scale_a, (0, padded_m - m))
  if padded_n != n:
    scale_b = jnp.pad(scale_b, ((0, 0), (0, padded_n - n)))
  # Lane-major scale layouts: one contiguous row of tile_m (tile_n) values per
  # block.
  scale_a = scale_a.reshape(1, padded_m)
  scale_b = scale_b.reshape(num_groups, 1, padded_n)

  group_offsets, group_ids, m_tile_ids, _ = make_group_metadata(
      offs, num_rows=m, padded_rows=padded_m, tile_m=tile_m
  )
  num_groups_arr = jnp.full((1,), num_groups, jnp.int32)
  tiles_k = padded_k // tile_k

  def a_map(n_i, v, k_i, go, gid, mt, ng, mp, kp, np):
    del n_i, go, gid, ng, mp, kp, np
    return mt[v], k_i

  def weight_group(v, gid, ng):
    # The virtual group reads (and ignores) the last real group's blocks.
    return jnp.minimum(gid[v], ng[0] - 1)

  def b_map(n_i, v, k_i, go, gid, mt, ng, mp, kp, np):
    del go, mt, mp, kp, np
    g = weight_group(v, gid, ng)
    return (g, n_i, k_i) if transpose_rhs else (g, k_i, n_i)

  def sa_map(n_i, v, k_i, go, gid, mt, ng, mp, kp, np):
    del n_i, k_i, go, gid, ng, mp, kp, np
    return 0, mt[v]

  def sb_map(n_i, v, k_i, go, gid, mt, ng, mp, kp, np):
    del k_i, go, mt, mp, kp, np
    return weight_group(v, gid, ng), 0, n_i

  def out_map(n_i, v, k_i, go, gid, mt, ng, mp, kp, np):
    del k_i, go, gid, ng, mp, kp, np
    return mt[v], n_i

  # Provide raw 1-D symbolic dimensions to pallas_call so Mosaic's
  # lowering can locate dimvars (m, k, n) in lowering_context.avals_in
  # without allocating large buffers in SMEM (only 0-byte/trivial arrays).
  m_probe = jnp.zeros((m,), jnp.int8)
  k_probe = jnp.zeros((k,), jnp.int8)
  n_probe = jnp.zeros((n,), jnp.int8)

  b_block = (None, tile_n, tile_k) if transpose_rhs else (None, tile_k, tile_n)
  max_visits = padded_m // tile_m + num_groups
  out = pl.pallas_call(
      functools.partial(
          _kernel,
          tile_m=tile_m,
          tiles_k=tiles_k,
          transpose_rhs=transpose_rhs,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=7,
          grid=(padded_n // tile_n, max_visits, tiles_k),
          in_specs=[
              pl.BlockSpec((tile_m, tile_k), a_map),
              pl.BlockSpec(b_block, b_map),
              pl.BlockSpec((1, tile_m), sa_map),
              pl.BlockSpec((None, 1, tile_n), sb_map),
          ],
          out_specs=pl.BlockSpec((tile_m, tile_n), out_map),
          scratch_shapes=[pltpu.VMEM((tile_m, tile_n), jnp.float32)],
      ),
      out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), out_dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=_cost_estimate(m, k, n, num_groups, out_dtype),
      interpret=interpret,
      name="rowwise_scaled_grouped_mm",
  )(
      group_offsets,
      group_ids,
      m_tile_ids,
      num_groups_arr,
      m_probe,
      k_probe,
      n_probe,
      a,
      b,
      scale_a,
      scale_b,
  )
  if padded_m != m or padded_n != n:
    out = out[:m, :n]
  return out


def get_shapes(
    *, lhs_dtype: Any, rhs_dtype: Any, transpose_rhs: bool
) -> tuple[jax.ShapeDtypeStruct, ...]:
  """Symbolic `(a, b, scale_a, scale_b, offs)` for AOT export.

  M, G, K and N are all free: the kernel pads to tile multiples itself.
  """
  m, g, k, n = jax.export.symbolic_shape("m, g, k, n")
  return (
      jax.ShapeDtypeStruct((m, k), lhs_dtype),
      jax.ShapeDtypeStruct(
          (g, n, k) if transpose_rhs else (g, k, n), rhs_dtype
      ),
      jax.ShapeDtypeStruct((m,), jnp.float32),
      jax.ShapeDtypeStruct((g, n), jnp.float32),
      jax.ShapeDtypeStruct((g,), jnp.int32),
  )


def export(
    *,
    lhs_dtype: Any = jnp.float8_e4m3fn,
    rhs_dtype: Any = jnp.float8_e4m3fn,
    out_dtype: Any = jnp.bfloat16,
    transpose_rhs: bool = True,
    tile_m: int = _DEFAULT_TILE_M,
    tile_k: int,
    tile_n: int,
) -> jax.export.Exported:
  """Exports the kernel ahead of time for TPU `AOT_DEVICE_KIND`."""
  args = get_shapes(
      lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, transpose_rhs=transpose_rhs
  )
  with aot_target.use_abstract_tpu_mesh(), pl.pallas_export_experimental(
      dynamic_shapes=True
  ):
    return jax.export.export(rowwise_scaled_grouped_mm, platforms=["tpu"])(
        *args,
        out_dtype=out_dtype,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )


def export_mlir_module(
    *,
    lhs_dtype: Any = jnp.float8_e4m3fn,
    rhs_dtype: Any = jnp.float8_e4m3fn,
    out_dtype: Any = jnp.bfloat16,
    transpose_rhs: bool = True,
    tile_m: int = _DEFAULT_TILE_M,
    tile_k: int,
    tile_n: int,
) -> str:
  """Exports the kernel and returns the serialized MLIR module string."""
  return export(
      lhs_dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      out_dtype=out_dtype,
      transpose_rhs=transpose_rhs,
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
  ).mlir_module()
