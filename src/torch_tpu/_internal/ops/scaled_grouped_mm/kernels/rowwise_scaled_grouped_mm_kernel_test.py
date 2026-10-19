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

"""Numerics and bit-exact AOT regression tests for rowwise grouped FP8 GEMM.

Runs `RowwiseScaledGroupedMmKernelTest` on TPU accelerators (v5e, v6e, v7x):
- Layout coverage (even, uneven/unaligned, empty groups, single group)
- Dtype combinations (e4m3fn x e4m3fn, e5m2 x e4m3fn)
- Multi-tile K accumulation (`tiles_k >= 2`) with fractional FP8 values and
  non-dyadic scales
- Unaligned K and N padding and output slicing (`padded_k != k`, `padded_n !=
n`)
- Numerics vs CPU float32 reference, with chip-dependent tolerances for
  fractional FP8 inputs (see `scaled_mm_test_utils.fp8_tolerance`)
- AOT exported module bit-exact match against JIT execution
"""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import rowwise_scaled_grouped_mm_kernel as kernel
from torch_tpu._internal.ops.scaled_mm.kernels import scaled_mm_test_utils
from tests import seed_test_utils

_FP8_E4M3 = jnp.float8_e4m3fn.dtype
_FP8_E5M2 = jnp.float8_e5m2.dtype

_LAYOUTS = {
    "even": (1024, (256, 256, 256, 256)),
    "uneven_unaligned": (1000, (100, 300, 1, 512)),
    "empty_groups": (1000, (0, 500, 0, 300, 0)),
    "single_group": (640, (640,)),
}


def _offs(sizes) -> np.ndarray:
  return np.cumsum(sizes, dtype=np.int32)


def _reference(a, b, scale_a, scale_b, offs, *, transpose_rhs) -> np.ndarray:
  """f32 numpy reference: zeros past offs[-1]."""
  a = np.asarray(a).astype(np.float32)
  b = np.asarray(b).astype(np.float32)
  if transpose_rhs:
    b = b.transpose(0, 2, 1)
  scale_a = np.asarray(scale_a).astype(np.float32)
  scale_b = np.asarray(scale_b).astype(np.float32)
  out = np.zeros((a.shape[0], b.shape[2]), np.float32)
  start = 0
  for group, end in enumerate(np.asarray(offs)):
    if end > start:
      out[start:end] = (a[start:end] @ b[group]) * (
          scale_a[start:end, None] * scale_b[group]
      )
    start = end
  return out


@functools.cache
def _exported_module(
    lhs_dtype, rhs_dtype, out_dtype, transpose_rhs, tile_m, tile_k, tile_n
):
  return kernel.export(
      lhs_dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      out_dtype=out_dtype,
      transpose_rhs=transpose_rhs,
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
  )


class RowwiseScaledGroupedMmKernelTest(seed_test_utils.RepeatableTest):

  @parameterized.product(
      layout=("even", "uneven_unaligned", "empty_groups", "single_group"),
      transpose_rhs=(False, True),
      out_dtype=(jnp.bfloat16, jnp.float32),
      lhs_dtype=(_FP8_E4M3, _FP8_E5M2),
  )
  def test_numerics_vs_reference(
      self, layout, transpose_rhs, out_dtype, lhs_dtype
  ):
    num_rows, sizes = _LAYOUTS[layout]
    k, n, num_groups = 256, 384, len(sizes)
    rng = np.random.default_rng(self.test_random_seed)

    a_raw = rng.integers(-2, 3, size=(num_rows, k)).astype(np.float32)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b_raw = rng.integers(-2, 3, size=b_shape).astype(np.float32)

    a = jnp.asarray(a_raw).astype(lhs_dtype)
    b = jnp.asarray(b_raw).astype(_FP8_E4M3)
    scale_a = jnp.asarray(rng.uniform(0.5, 2.0, (num_rows,)), jnp.float32)
    scale_b = jnp.asarray(rng.uniform(0.5, 2.0, (num_groups, n)), jnp.float32)
    offs = jnp.asarray(_offs(sizes))

    out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=out_dtype,
        transpose_rhs=transpose_rhs,
        tile_m=256,
        tile_k=256,
        tile_n=128,
    )
    expected = _reference(
        a, b, scale_a, scale_b, offs, transpose_rhs=transpose_rhs
    ).astype(np.float32)

    # For integer inputs, the only difference is rounding to out_dtype
    expected_rounded = scaled_mm_test_utils.round_to_dtype(expected, out_dtype)
    out_np = np.asarray(out.astype(jnp.float32))
    np.testing.assert_allclose(out_np, expected_rounded, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      transpose_rhs=(False, True),
      out_dtype=(jnp.bfloat16, jnp.float32),
      lhs_dtype=(_FP8_E4M3, _FP8_E5M2),
  )
  def test_multi_tile_k_fractional_fp8(
      self, transpose_rhs, out_dtype, lhs_dtype
  ):
    sizes = (150, 0, 260, 92)
    num_rows, k, n, num_groups = 550, 512, 256, len(sizes)
    tile_m, tile_k, tile_n = 128, 256, 128  # tiles_k == 2
    rng = np.random.default_rng(self.test_random_seed)

    a = jnp.asarray(
        rng.normal(0.0, 0.5, size=(num_rows, k)), jnp.float32
    ).astype(lhs_dtype)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b = jnp.asarray(rng.normal(0.0, 0.5, size=b_shape), jnp.float32).astype(
        _FP8_E4M3
    )
    scale_a = jnp.asarray(rng.uniform(0.3, 1.7, (num_rows,)), jnp.float32)
    scale_b = jnp.asarray(rng.uniform(0.3, 1.7, (num_groups, n)), jnp.float32)
    offs = jnp.asarray(_offs(sizes))

    out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=out_dtype,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )
    expected = _reference(
        a, b, scale_a, scale_b, offs, transpose_rhs=transpose_rhs
    )
    expected_rounded = scaled_mm_test_utils.round_to_dtype(expected, out_dtype)
    out_np = np.asarray(out.astype(jnp.float32))
    scaled_mm_test_utils.assert_fp8_close(
        out_np, expected_rounded, **scaled_mm_test_utils.fp8_tolerance()
    )
    np.testing.assert_array_equal(out_np[sum(sizes) :], 0.0)

  @parameterized.product(
      transpose_rhs=(False, True),
      out_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_unaligned_k_and_n(self, transpose_rhs, out_dtype):
    sizes = (150, 0, 200, 100)
    num_rows, k, n, num_groups = 480, 300, 200, len(sizes)
    tile_m, tile_k, tile_n = (
        128,
        128,
        128,
    )  # padded_k=384 (tiles_k=3), padded_n=256 (tiles_n=2)
    rng = np.random.default_rng(self.test_random_seed)

    a = jnp.asarray(
        rng.normal(0.0, 0.5, size=(num_rows, k)), jnp.float32
    ).astype(_FP8_E4M3)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b = jnp.asarray(rng.normal(0.0, 0.5, size=b_shape), jnp.float32).astype(
        _FP8_E4M3
    )
    scale_a = jnp.asarray(rng.uniform(0.3, 1.7, (num_rows,)), jnp.float32)
    scale_b = jnp.asarray(rng.uniform(0.3, 1.7, (num_groups, n)), jnp.float32)
    offs = jnp.asarray(_offs(sizes))

    out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=out_dtype,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )
    self.assertEqual(out.shape, (num_rows, n))
    expected = _reference(
        a, b, scale_a, scale_b, offs, transpose_rhs=transpose_rhs
    )
    expected_rounded = scaled_mm_test_utils.round_to_dtype(expected, out_dtype)
    out_np = np.asarray(out.astype(jnp.float32))
    scaled_mm_test_utils.assert_fp8_close(
        out_np, expected_rounded, **scaled_mm_test_utils.fp8_tolerance()
    )
    np.testing.assert_array_equal(out_np[sum(sizes) :], 0.0)

  @parameterized.product(
      layout=("even", "uneven_unaligned", "empty_groups"),
      transpose_rhs=(False, True),
      out_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_aot_module_matches_jit(self, layout, transpose_rhs, out_dtype):
    num_rows, sizes = _LAYOUTS[layout]
    k, n, num_groups = 256, 256, len(sizes)
    tile_m, tile_k, tile_n = 256, 256, 256
    rng = np.random.default_rng(self.test_random_seed)

    a = jnp.asarray(
        rng.integers(-2, 3, size=(num_rows, k)), jnp.float32
    ).astype(_FP8_E4M3)
    b_shape = (num_groups, n, k) if transpose_rhs else (num_groups, k, n)
    b = jnp.asarray(rng.integers(-2, 3, size=b_shape), jnp.float32).astype(
        _FP8_E4M3
    )
    scale_a = jnp.asarray(rng.uniform(0.5, 2.0, (num_rows,)), jnp.float32)
    scale_b = jnp.asarray(rng.uniform(0.5, 2.0, (num_groups, n)), jnp.float32)
    offs = jnp.asarray(_offs(sizes))

    aot_mod = _exported_module(
        _FP8_E4M3, _FP8_E4M3, out_dtype, transpose_rhs, tile_m, tile_k, tile_n
    )
    aot_out = aot_mod.call(a, b, scale_a, scale_b, offs)
    jit_out = kernel.rowwise_scaled_grouped_mm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        out_dtype=out_dtype,
        transpose_rhs=transpose_rhs,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )
    np.testing.assert_array_equal(np.asarray(aot_out), np.asarray(jit_out))


if __name__ == "__main__":
  absltest.main()
