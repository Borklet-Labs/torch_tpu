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

"""Microbenchmark for rowwise scaled grouped GEMM on TPU v7x.

Measures latency at Qwen3-30B-A3B expert shapes:
- w1/w3: G=128, M=16384, K=2048, N=768
- w2:    G=128, M=16384, K=768,  N=2048

Gating thresholds (shipped bf16 _grouped_mm baseline):
- w1/w3: 470.0 µs
- w2:    510.9 µs
Evaluated under both uniform and random routing distributions.
"""

import functools
import time
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.ops.scaled_grouped_mm.kernels import rowwise_scaled_grouped_mm_kernel as kernel
from tests import seed_test_utils

_FP8_E4M3 = jnp.float8_e4m3fn.dtype

# Gating thresholds in microseconds
_GATE_THRESHOLDS_US = {
    "w1_w3": 470.0,
    "w2": 510.9,
}


def _make_routing_offsets(
    m: int, g: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
  if mode == "uniform":
    # Each expert gets exactly m // g rows
    rows_per_expert = m // g
    sizes = np.full(g, rows_per_expert, dtype=np.int32)
    sizes[-1] += m - int(sizes.sum())
  elif mode == "random":
    # Dirichlet-like random routing with empty and heavily loaded experts
    weights = rng.exponential(scale=1.0, size=g)
    weights /= weights.sum()
    sizes = rng.multinomial(m, weights).astype(np.int32)
  else:
    raise ValueError(f"Unknown routing mode: {mode}")
  return np.cumsum(sizes, dtype=np.int32)


def _benchmark_kernel(fn, num_warmup: int = 25, num_timed: int = 100) -> float:
  """Returns median execution latency in microseconds."""
  # Warmup
  for _ in range(num_warmup):
    res = fn()
    res.block_until_ready()

  times = []
  for _ in range(num_timed):
    start = time.perf_counter()
    res = fn()
    res.block_until_ready()
    times.append(time.perf_counter() - start)

  # Return median time in microseconds
  return float(np.median(times) * 1e6)


class RowwiseScaledGroupedMmBenchmarkTest(seed_test_utils.RepeatableTest):

  @parameterized.product(
      shape_name=("w1_w3", "w2"),
      routing=("uniform", "random"),
  )
  def test_qwen3_30b_a3b_benchmark(self, shape_name, routing):
    g = 128
    m = 16384
    if shape_name == "w1_w3":
      k, n = 2048, 768
    else:
      k, n = 768, 2048

    rng = np.random.default_rng(self.test_random_seed)
    offs = jnp.asarray(_make_routing_offsets(m, g, routing, rng))
    a = jnp.asarray(rng.integers(-2, 3, size=(m, k)), jnp.float32).astype(
        _FP8_E4M3
    )
    b = jnp.asarray(rng.integers(-2, 3, size=(g, n, k)), jnp.float32).astype(
        _FP8_E4M3
    )
    scale_a = jnp.asarray(rng.uniform(0.5, 2.0, (m,)), jnp.float32)
    scale_b = jnp.asarray(rng.uniform(0.5, 2.0, (g, n)), jnp.float32)

    # Precompile / JIT with optimal tile sizes per design doc
    # w1/w3: tm=512, tk=2048, tn=768
    # w2: tm=512, tk=768, tn=2048
    jitted_fn = jax.jit(
        functools.partial(
            kernel.rowwise_scaled_grouped_mm,
            transpose_rhs=True,
            out_dtype=jnp.bfloat16,
            tile_m=512,
            tile_k=k,
            tile_n=n,
        )
    )

    latency_us = _benchmark_kernel(
        lambda: jitted_fn(a, b, scale_a, scale_b, offs)
    )
    threshold_us = _GATE_THRESHOLDS_US[shape_name]

    logging.info(
        f"Benchmark {shape_name} ({routing}): latency={latency_us:.1f} µs, "
        f"gate={threshold_us:.1f} µs (speedup={threshold_us / latency_us:.2f}x)"
    )
    self.assertLess(
        latency_us,
        threshold_us,
        f"P3.3a gating failure: {shape_name} {routing} latency {latency_us:.1f}"
        f" µs exceeds bf16 threshold {threshold_us:.1f} µs",
    )


if __name__ == "__main__":
  absltest.main()
