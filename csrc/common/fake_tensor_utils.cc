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

#include "csrc/common/fake_tensor_utils.h"

#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "csrc/common/const_tensor.h"

namespace torch_tpu {

bool IsFunctorchActive() {
  const c10::DispatchKeySet include_set =
      c10::impl::tls_local_dispatch_key_set().included_;
  return include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
         include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode);
}

bool IsFakeOrFunctorchTensor(const ConstTensor& tensor) {
  return IsFunctorchActive() || !tensor.has_storage_data();
}

}  // namespace torch_tpu
