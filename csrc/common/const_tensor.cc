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

#include "csrc/common/const_tensor.h"

#include <cstdint>
#include <optional>

#include "ATen/SparseCsrTensorImpl.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/SymInt.h"
#include "c10/core/TensorImpl.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"

namespace torch_tpu {

std::optional<bool> ConstTensor::maybe_is_contiguous(
    c10::MemoryFormat memory_format) const {
  if (has_symbolic_sizes_strides() || layout() != c10::kStrided) {
    return std::nullopt;
  }
  return tensor_.is_contiguous(memory_format);
}

int64_t ConstTensor::sparse_dim() const {
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->sparse_dim();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->sparse_dim();
  }
  return tensor_.sparse_dim();
}

int64_t ConstTensor::dense_dim() const {
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->dense_dim();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->dense_dim();
  }
  return tensor_.dense_dim();
}

bool ConstTensor::is_coalesced() const {
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(
          tensor_.unsafeGetTensorImpl())) {
    return coo->coalesced();
  }
  return tensor_.is_coalesced();
}

std::optional<c10::SymInt> ConstTensor::sym_nnz() const {
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->sym_nnz();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->plain_indices().sym_size(-1);
  }
  return std::nullopt;
}

}  // namespace torch_tpu
