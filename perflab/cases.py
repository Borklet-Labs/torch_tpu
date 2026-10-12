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

"""Provides logic to resolve benchmark cases."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
import hashlib
import os
import random
from typing import Any
import unittest

from absl import flags
from absl.testing import absltest
from perflab import common
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib

CaseParams = tuple[registry_lib.BenchmarkSpec, common.RunMode]

_MAX_ABSLTEST_SEED = 4294967295

_HEAVY_MODEL_PREFIXES = (
    "audio_flamingo",
    "bge_multilingual_gemma2",
    "colpali",
    "colqwen2",
    "deepseek",
    "dinov2_giant",
    "dse_qwen2",
    "e5_omni",
    "flux",
    "gemma",
    "glm",
    "gpt_oss",
    "granite_speech",
    "higgs_tts",
    "hunyuan",
    "if_i_xl",
    "janus",
    "llama",
    "llava",
    "metaclip",
    "mistral",
    "mixtral",
    "muse",
    "music_flamingo",
    "nemotron",
    "qwen",
    "sam2",
    "stable_diffusion",
    "swin_large",
    "vibevoice",
    "vit_giant",
    "vjepa2",
    "wan",
    "whisper",
)


def _case_order_key(name: str, seed: str) -> str:
  """Returns a deterministic hash key for ordering benchmark cases."""
  payload = f"{seed}:{name}" if seed else name
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_order_seed(env: Mapping[str, str] | None = None) -> str:
  """Resolves the benchmark ordering seed from environment variables."""
  if env is None:
    env = os.environ  # PY_ENVIRON_OK=benchmark
  seed = env.get("TORCH_TPU_BENCHMARK_ORDER_SEED", "").strip()
  if seed:
    return seed
  fallback = env.get("TEST_RANDOMIZE_ORDERING_SEED", "").strip()
  if fallback and fallback not in ("0", "random"):
    return fallback
  return ""


def seed_to_int(seed: str) -> int:
  """Converts an arbitrary seed string into a valid positive absltest seed."""
  cleaned = seed.strip()
  if not cleaned or cleaned == "0":
    return 0
  if cleaned.isascii() and cleaned.isdigit():
    value = int(cleaned)
    if 1 <= value <= _MAX_ABSLTEST_SEED:
      return value
  digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
  return 1 + (int(digest, 16) % _MAX_ABSLTEST_SEED)


def case_params(
    cases_iter: Iterable[
        tuple[str, registry_lib.BenchmarkSpec, common.RunMode]
    ],
    method_name: str = "test_benchmark",
) -> Mapping[str, CaseParams]:
  """Maps the names parameterized.named_parameters generates to (spec, mode)."""
  return {
      f"{method_name}_{name}": (spec, mode) for name, spec, mode in cases_iter
  }


def normalize_randomize_ordering_seed(
    env: MutableMapping[str, str] | None = None,
) -> int:
  """Normalizes test ordering seed env vars into a valid integer for absltest.

  `absltest` raises `ValueError` if `TEST_RANDOMIZE_ORDERING_SEED` is set to a
  non-decimal string (such as a Git commit SHA) or an integer outside
  `[1, 4294967295]`. This function converts any non-empty string seed in
  `TEST_RANDOMIZE_ORDERING_SEED` (or falls back to
  `TORCH_TPU_BENCHMARK_ORDER_SEED` when `TEST_RANDOMIZE_ORDERING_SEED` is
  unset) into a deterministic positive 32-bit integer in `env`.

  Args:
    env: Environment mapping to normalize in place; defaults to `os.environ`.

  Returns:
    The normalized integer seed (0 if randomization is disabled or 'random').
  """
  flag_seed = ""
  if env is None:
    env = os.environ  # PY_ENVIRON_OK=benchmark
    if (
        flags.FLAGS.is_parsed()
        and "test_randomize_ordering_seed" in flags.FLAGS
        and flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      flag_seed = str(flags.FLAGS.test_randomize_ordering_seed).strip()
      if flag_seed and flag_seed not in ("0", "random"):
        flags.FLAGS.test_randomize_ordering_seed = str(seed_to_int(flag_seed))
  raw_seed = env.get("TEST_RANDOMIZE_ORDERING_SEED", "").strip()
  if not raw_seed:
    raw_seed = env.get("TORCH_TPU_BENCHMARK_ORDER_SEED", "").strip()
  if not raw_seed and flag_seed:
    raw_seed = flag_seed
  if not raw_seed or raw_seed == "0":
    if "TEST_RANDOMIZE_ORDERING_SEED" in env and raw_seed == "0":
      env["TEST_RANDOMIZE_ORDERING_SEED"] = "0"
    return 0
  if raw_seed == "random":
    return 0
  normalized = seed_to_int(raw_seed)
  env["TEST_RANDOMIZE_ORDERING_SEED"] = str(normalized)
  return normalized


def estimate_case_cost(
    spec: registry_lib.BenchmarkSpec,
    mode: common.RunMode,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
    skip_behavior: str | None = None,
) -> float:
  """Estimates relative execution weight for LPT shard load balancing.

  Cases that are statically skipped in the parent process receive zero weight so
  they do not displace active benchmarks during shard assignment.

  Args:
    spec: Benchmark specification.
    mode: Execution mode (e.g. compiled or eager).
    framework: Execution framework (e.g. TORCH or TORCHAX).
    device_kind: Target device kind.
    skip_behavior: Skip behavior mode ('skip', 'run_skipped', 'ignore', or
      'assert_raise'). Defaults to `--skip_behavior` when parsed, else 'skip'.

  Returns:
    Estimated non-negative relative cost weight.
  """
  if (
      framework == mode_lib.Framework.TORCHAX
      and spec.stepper == step_lib.StepperType.DECODER_ONLY_DECODE
  ):
    return 0.0

  if skip_behavior is None:
    skip_behavior = (
        flags_lib.SKIP_BEHAVIOR.value if flags.FLAGS.is_parsed() else "skip"
    )

  skip_reason = get_skip_reason(
      spec, mode, framework=framework, device_kind=device_kind
  )
  if skip_behavior == "skip" and skip_reason is not None:
    return 0.0
  if skip_behavior == "run_skipped" and skip_reason is None:
    return 0.0

  cost = 3.0 if mode == common.RunMode.COMPILED else 1.0
  if spec.stepper == step_lib.StepperType.DECODER_ONLY_DECODE:
    cost *= 2.0
  elif spec.stepper in (
      step_lib.StepperType.TRAINING,
      step_lib.StepperType.SINGLE_TRACE_TRAINING,
  ):
    cost *= 1.5
  if spec.name.startswith(_HEAVY_MODEL_PREFIXES):
    cost *= 2.0
  return cost


class BenchmarkTestLoader(absltest.TestLoader):
  """TestLoader with deterministic hash ordering and LPT cost-balanced sharding."""

  _last_test_case_class: (
      type[unittest.TestCase] | None  # UNITTEST_OK=type annotation
  ) = None

  def __init__(self, *args: Any, **kwargs: Any):
    self._seed_initialized: bool = False
    self._randomize_ordering_seed: int | None = None
    self._random: random.Random | None = None
    normalize_randomize_ordering_seed()
    super().__init__(*args, **kwargs)
    self._ensure_randomize_ordering_seed()
    self._last_test_case_class = None

  def _ensure_randomize_ordering_seed(self) -> None:
    """Lazily resolves and normalizes randomize ordering seed after flag parsing."""
    if self._seed_initialized and (
        not flags.FLAGS.is_parsed()
        or not flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      return
    raw_test_seed = os.environ.get(  # PY_ENVIRON_OK=benchmark
        "TEST_RANDOMIZE_ORDERING_SEED", ""
    ).strip()
    raw_bench_seed = os.environ.get(  # PY_ENVIRON_OK=benchmark
        "TORCH_TPU_BENCHMARK_ORDER_SEED", ""
    ).strip()
    flag_seed = ""
    if (
        flags.FLAGS.is_parsed()
        and "test_randomize_ordering_seed" in flags.FLAGS
        and flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      flag_seed = str(flags.FLAGS.test_randomize_ordering_seed).strip()
    normalized = normalize_randomize_ordering_seed()
    if raw_test_seed == "0" or (
        flag_seed == "0" and not raw_test_seed and not raw_bench_seed
    ):
      self._randomize_ordering_seed = None
      self._random = None
    elif normalized > 0 and (raw_test_seed or raw_bench_seed or flag_seed):
      self._randomize_ordering_seed = normalized
      self._random = random.Random(normalized)
    else:
      self._randomize_ordering_seed = None
      self._random = None
    if flags.FLAGS.is_parsed():
      self._seed_initialized = True

  def getTestCaseNames(  # pylint: disable=invalid-name
      self,
      testCaseClass: type[unittest.TestCase],  # UNITTEST_OK=type annotation
  ) -> list[str]:
    """Returns test case names ordered by `case_params` before optional shuffle."""
    self._ensure_randomize_ordering_seed()
    self._last_test_case_class = testCaseClass
    names = list(super().getTestCaseNames(testCaseClass))
    class_case_params: Mapping[str, CaseParams] = getattr(
        testCaseClass, "case_params", {}
    )
    if class_case_params:
      if self._randomize_ordering_seed is None:
        case_rank = {
            name: idx for idx, name in enumerate(class_case_params.keys())
        }
        names.sort(
            key=lambda name: (1, case_rank[name])
            if name in case_rank
            else (0, name)
        )
      else:
        names.sort(key=lambda name: 1 if name in class_case_params else 0)
    return names

  def _estimate_method_cost(
      self,
      method_name: str,
      class_case_params: Mapping[str, CaseParams],
      framework: mode_lib.Framework | None,
      device_kind: target_lib.DeviceKind | None,
      skip_behavior: str,
  ) -> float:
    params = class_case_params.get(method_name)
    if params is None or framework is None or device_kind is None:
      return 0.1
    spec, mode = params
    return estimate_case_cost(
        spec,
        mode,
        framework=framework,
        device_kind=device_kind,
        skip_behavior=skip_behavior,
    )

  def shardTestCaseNames(  # pylint: disable=invalid-name
      self,
      iterator: Iterator[int],
      ordered_names: Sequence[str],
      shard_index: int,
  ) -> Sequence[str]:
    """Partitions `ordered_names` across shards using LPT greedy load balancing."""
    if not ordered_names:
      return []

    buckets: list[int] = []
    seen_buckets: set[int] = set()
    for _ in range(len(ordered_names)):
      bucket = next(iterator)
      if bucket in seen_buckets:
        break
      seen_buckets.add(bucket)
      buckets.append(bucket)

    if not buckets or shard_index not in seen_buckets:
      return []

    test_cls = self._last_test_case_class
    class_case_params: Mapping[str, CaseParams] = (
        getattr(test_cls, "case_params", {}) if test_cls is not None else {}
    )
    framework: mode_lib.Framework | None = (
        getattr(test_cls, "framework", None) if test_cls is not None else None
    )
    platform: target_lib.Platform | None = (
        getattr(test_cls, "platform", None) if test_cls is not None else None
    )
    device_kind = (
        target_lib.make_target(platform).device_kind
        if platform is not None
        else None
    )
    order_seed = _get_order_seed()
    skip_behavior = (
        flags_lib.SKIP_BEHAVIOR.value if flags.FLAGS.is_parsed() else "skip"
    )

    costs = {
        name: self._estimate_method_cost(
            name, class_case_params, framework, device_kind, skip_behavior
        )
        for name in ordered_names
    }
    lpt_ordered = sorted(
        ordered_names,
        key=lambda name: (
            -costs[name],
            _case_order_key(name, order_seed),
            name,
        ),
    )

    shard_load = {b: 0.0 for b in buckets}
    shard_count = {b: 0 for b in buckets}
    assigned_to_shard: set[str] = set()
    for testcase in lpt_ordered:
      cost = costs[testcase]
      if cost > 0.0:
        target_bucket = min(
            buckets,
            key=lambda b: (shard_load[b], shard_count[b], b),
        )
      else:
        target_bucket = min(
            buckets,
            key=lambda b: (shard_count[b], shard_load[b], b),
        )
      shard_load[target_bucket] += cost
      shard_count[target_bucket] += 1
      if target_bucket == shard_index:
        assigned_to_shard.add(testcase)

    return [x for x in ordered_names if x in assigned_to_shard]


def get_cases(
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    *,
    order_seed: str | None = None,
) -> Iterator[tuple[str, registry_lib.BenchmarkSpec, common.RunMode]]:
  """One case per (benchmark, applicable mode).

  The mode matrix is resolved here, at collection, so generated cases match the
  cell exactly: a CPU run never emits an eager_optimized case just to skip it.
  Ordering uses a deterministic hash (seeded by TORCH_TPU_BENCHMARK_ORDER_SEED
  when set) so all shards agree on the partition without alphabetical tail bias.

  Args:
    platform: Target hardware platform on which the benchmarks will run.
    framework: Execution framework (e.g. TORCH or TORCHAX) filtering modes.
    order_seed: Seed string used for deterministic benchmark ordering; falls
      back to the TORCH_TPU_BENCHMARK_ORDER_SEED environment variable if None.

  Yields:
    Tuples of (case_name, spec, run_mode) in deterministic hashed order.
  """
  if order_seed is None:
    order_seed = _get_order_seed()
  target_kind = target_lib.make_target(platform).device_kind
  ordered_names = sorted(
      registry_lib.REGISTRY,
      key=lambda name: _case_order_key(name, order_seed),
  )
  for name in ordered_names:
    spec = registry_lib.REGISTRY[name]
    for mode in mode_lib.modes_for(framework, target_kind):
      yield f"{spec.name}_{mode.value}", spec, mode


def get_skip_reason(
    spec: registry_lib.BenchmarkSpec,
    mode: common.RunMode,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
) -> str | None:
  """Returns why a case should be skipped, or None if it should run.

  A case is skipped if it matches an entry in `known_failures.KNOWN_FAILURES` or
  if `spec.skipped_run_modes` lists `mode`.
  """
  failure_reason = known_failures.get_known_failure_reason(
      spec.name, framework=framework, device_kind=device_kind, run_mode=mode
  )
  if failure_reason is not None:
    return f"Benchmark {spec.name} is known to fail: {failure_reason}"
  if mode.value in spec.skipped_run_modes:
    return f"Benchmark {spec.name} explicitly skips run mode {mode.value}"
  return None
