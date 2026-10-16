/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_EAGER_TPU_HOOKS_H_
#define TORCH_TPU_CSRC_EAGER_TPU_HOOKS_H_

#include "absl/status/status.h"

namespace torch_tpu {

absl::Status AddTpuHooks();

// Logs an error if no TPU device node is detected (see HasTpuDeviceNodes() in
// tpu_hardware.h) and no TPU device paths are configured explicitly. Before the
// TPU runtime is initialized, TPU device guards report 0 devices in that case,
// assuming that initialization would fail. Call once the TPU runtime has
// initialized successfully, to detect device layouts that break that
// assumption. Does not initialize the TPU runtime.
void CheckTpuDeviceNodesAfterInitialization();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_TPU_HOOKS_H_
