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

#include "csrc/internal/mosaic/kiln/core/memory.h"

#include <cstdint>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln {

Ref Alloca(const ScratchSpec& spec) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::MLIRContext* ctx = b.getContext();
  for (int64_t dim : spec.shape) {
    ABSL_CHECK_GT(dim, 0)  // CRASH_OK
        << "Alloca() requires static positive dimensions";
  }
  switch (spec.space) {
    case MemorySpace::kVmem: {
      ABSL_CHECK(!spec.shape.empty())  // CRASH_OK
          << "VMEM Alloca() requires at least a 1D shape";
      ABSL_CHECK(!spec.dtype.IsBool() &&  // CRASH_OK
                 spec.dtype.GetIntOrFloatBitWidth() <= 32u)
          << "VMEM Alloca() does not support bool (i1) or 64-bit element types";
      mlir::Type elem_type = internal::ToMlirType(spec.dtype, ctx);
      mlir::MemRefType type =
          mlir::torch_tpu::GetVmemMemRefType(ctx, spec.shape, elem_type);
      return internal::wrap<Ref>(mlir::memref::AllocaOp::create(b, type),
                                 spec.dtype);
    }
    case MemorySpace::kSmem: {
      ABSL_CHECK_EQ(spec.dtype.GetIntOrFloatBitWidth(), 32u)  // CRASH_OK
          << "SMEM Alloca() requires a 32-bit element type";
      mlir::Type elem_type = internal::ToMlirType(spec.dtype, ctx);
      mlir::MemRefType type =
          mlir::torch_tpu::GetSmemMemRefType(ctx, spec.shape, elem_type);
      return internal::wrap<Ref>(mlir::memref::AllocaOp::create(b, type),
                                 spec.dtype);
    }
    default:
      ABSL_CHECK(false)  // CRASH_OK
          << "Unsupported memory space for Alloca()";
      return internal::wrap<Ref>(mlir::Value(), spec.dtype);
  }
}

}  // namespace torch_tpu::kiln
