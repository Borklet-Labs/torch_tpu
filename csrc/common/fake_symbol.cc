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

#include "csrc/common/fake_symbol.h"

#include <cstdint>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "c10/core/SymInt.h"

namespace torch_tpu {

int64_t FakeSymbol::guard_int(const char* /*file*/, int64_t /*line*/) {
  ++*num_specializations_;
  return int_();
}

at::Tensor MakeSymbolicTensor(int* num_specializations) {
  const c10::SymInt s0 = MakeSymbol(3, "s0", num_specializations);
  at::Tensor tensor = at::empty({2, 3});
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      std::vector<c10::SymInt>{c10::SymInt(2), s0},
      std::vector<c10::SymInt>{s0, c10::SymInt(1)});
  return tensor;
}

}  // namespace torch_tpu
