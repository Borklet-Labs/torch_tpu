/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_
#define TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

// Given a list of nodes to execute, and a list of nodes to explicitly
// materialize, returns a list of traversals that will do this work.
//
// Every node that must be materialized will be an output of a traversal, and
// every node that must be executed will either be an output, or an input to a
// different output.
//
// The number of returned traversals will depend on the OpSplitMode of the nodes
// in the must_execute list. If there are no required splits, a single traversal
// will be returned, potentially with multiple outputs.
//
// The sequence of returned Traversals will be in a valid topological order and
// will be pinned to the appropriate core as specified by `core_pinning_mode`.
std::vector<absl_nonnull std::unique_ptr<Traversal>> PrepareTraversals(
    std::vector<SharedDeviceBufferList> must_execute,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    CorePinningMode core_pinning_mode);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_
