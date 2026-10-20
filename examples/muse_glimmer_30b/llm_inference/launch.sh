#!/bin/bash
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

set -e

# A launch script to run Muse Glimmer 30B LLM inference benchmarks on a single TPU device.
#
# Usage:
#   ./launch.sh [dynamic|static|all] [--random-weights] [extra flags...]
#
# Examples:
#   # Run both dynamic and static benchmarks:
#   ./launch.sh
#
#   # Run only dynamic inference benchmark with real weights:
#   ./launch.sh dynamic
#
#   # Run only static inference benchmark with random weights:
#   ./launch.sh static --random-weights
#
#   # Run with custom test filter:
#   ./launch.sh dynamic --test_filter=SingleTpuInferenceDynamicTest.test_muse_glimmer_bounded_dynamism_swa_prefill

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Add repo root to PYTHONPATH if running from a local checkout or container.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/..:${REPO_ROOT}/../..:${PYTHONPATH}"

# Configure TPU environment for single-device execution.
export PYTHONUNBUFFERED=1
export TPU_LAUNCH_BLOCKING=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_dvfs_p_state=7"

MODE="all"
USE_RANDOM_WEIGHTS=false
TEST_NAME=""
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    dynamic)
      MODE="dynamic"
      ;;
    static)
      MODE="static"
      ;;
    all)
      MODE="all"
      ;;
    --random-weights|--random_weights)
      USE_RANDOM_WEIGHTS=true
      ;;
    --test_filter=*)
      # Open-source absl-py has no --test_filter flag, so the test name is
      # passed as a positional argument instead, which both support.
      TEST_NAME="${arg#--test_filter=}"
      ;;
    *)
      PASSTHROUGH_ARGS+=("$arg")
      ;;
  esac
done

# A test name selects one test script, so run only the matching benchmark.
if [[ -n "$TEST_NAME" && "$MODE" == "all" ]]; then
  case "$TEST_NAME" in
    SingleTpuInferenceDynamicTest.*)
      MODE="dynamic"
      ;;
    SingleTpuInferenceStaticTest.*)
      MODE="static"
      ;;
    *)
      echo "Error: --test_filter=${TEST_NAME} must start with" \
        "SingleTpuInferenceDynamicTest. or SingleTpuInferenceStaticTest.," \
        "or pass 'dynamic' or 'static'." >&2
      exit 1
      ;;
  esac
fi

run_dynamic() {
  local test_name="${TEST_NAME}"
  if [[ -z "$test_name" ]]; then
    if [[ "$USE_RANDOM_WEIGHTS" == "true" ]]; then
      test_name="SingleTpuInferenceDynamicTest.test_muse_glimmer_random_weights_bounded_dynamism"
    else
      test_name="SingleTpuInferenceDynamicTest.test_muse_glimmer_bounded_dynamism"
    fi
  fi
  local args=("${PASSTHROUGH_ARGS[@]}" "$test_name")

  echo "============================================================"
  echo " Running Single TPU Dynamic Inference Benchmark"
  echo " Command: python3 ./single_tpu_inference_dynamic_test.py ${args[*]}"
  echo "============================================================"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Skipping command execution."
    return 0
  fi
  python3 "${SCRIPT_DIR}/single_tpu_inference_dynamic_test.py" "${args[@]}"
}

run_static() {
  local test_name="${TEST_NAME}"
  if [[ -z "$test_name" ]]; then
    if [[ "$USE_RANDOM_WEIGHTS" == "true" ]]; then
      test_name="SingleTpuInferenceStaticTest.test_muse_glimmer_random_weights_static"
    else
      test_name="SingleTpuInferenceStaticTest.test_muse_glimmer_static"
    fi
  fi
  local args=("${PASSTHROUGH_ARGS[@]}" "$test_name")

  echo "============================================================"
  echo " Running Single TPU Static Inference Benchmark"
  echo " Command: python3 ./single_tpu_inference_static_test.py ${args[*]}"
  echo "============================================================"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Skipping command execution."
    return 0
  fi
  python3 "${SCRIPT_DIR}/single_tpu_inference_static_test.py" "${args[@]}"
}

case "$MODE" in
  dynamic)
    run_dynamic
    ;;
  static)
    run_static
    ;;
  all)
    run_dynamic
    echo ""
    run_static
    ;;
esac
