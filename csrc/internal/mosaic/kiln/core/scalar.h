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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_

#include <cstdint>

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"

namespace torch_tpu::kiln {

// A scalar SSA value (boolean, integer, or floating-point). Grid indices,
// loop induction variables, SMEM scalar values, constants, and scalar
// statistics are Scalars.
//
// Arithmetic, bitwise, comparison (yielding a bool `Scalar`), min / max /
// select and unary math operators come from `WithElementwiseOps`; overloads
// taking a `Literal` on either side come from `WithLiteralOps`.
class Scalar : public ValueBase,
               public WithElementwiseOps<Scalar>,
               public WithLiteralOps<Scalar> {
 public:
  Scalar MakeConstant(Literal value) const;

 private:
  friend struct internal::ValueAccess;
  Scalar(const void* impl, DType dtype);
};

Scalar I32(Literal value);
Scalar U32(Literal value);
Scalar F32(Literal value);

// Assumes that `value` is a multiple of `multiple` without emitting a runtime
// check (Pallas `pl.multiple_of` / `tpu.assume_multiple`).
Scalar MultipleOf(Scalar value, int32_t multiple);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_
