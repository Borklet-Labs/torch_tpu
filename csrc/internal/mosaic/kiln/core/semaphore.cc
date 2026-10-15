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

#include "csrc/internal/mosaic/kiln/core/semaphore.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

namespace tpu = mlir::tpu;

template <typename T>
mlir::MemRefType MemRefTypeOf(const T& sem) {
  return llvm::cast<mlir::MemRefType>(internal::unwrap(sem).getType());
}

void ValidateSemaphoreMemRefType(mlir::Type type, std::string_view type_name) {
  auto memref_type = llvm::dyn_cast<mlir::MemRefType>(type);
  ABSL_CHECK(memref_type != nullptr)  // CRASH_OK
      << type_name << " must be a memref value";
  auto space = llvm::dyn_cast_or_null<tpu::MemorySpaceAttr>(
      memref_type.getMemorySpace());
  ABSL_CHECK(space != nullptr &&  // CRASH_OK
             space.getValue() == tpu::MemorySpace::kSemaphoreMem)
      << type_name << " must be in semaphore_mem memory space";
  ABSL_CHECK((llvm::isa<tpu::SemaphoreType, tpu::DMASemaphoreType>(  // CRASH_OK
      memref_type.getElementType())))
      << type_name << " must have a TPU semaphore element type";
}

void ValidateConstantBound(const llvm::APInt& val, bool is_unsigned,
                           bool require_positive, std::string_view name) {
  if (require_positive) {
    const bool is_positive =
        is_unsigned ? !val.isZero() : val.isStrictlyPositive();
    ABSL_CHECK(is_positive) << name << " must be positive";  // CRASH_OK
  } else {
    const bool is_non_negative = is_unsigned || !val.isNegative();
    ABSL_CHECK(is_non_negative) << name << " must be non-negative";  // CRASH_OK
  }
  const bool fits_i32 = is_unsigned
                            ? val.ule(std::numeric_limits<int32_t>::max())
                            : val.sle(std::numeric_limits<int32_t>::max());
  ABSL_CHECK(fits_i32) << name << " exceeds int32_t max";  // CRASH_OK
}

mlir::Value ToValidatedI32Value(const IndexArg& arg, bool require_positive,
                                std::string_view name) {
  if (arg.IsInt()) {
    ValidateConstantBound(llvm::APInt(64, arg.AsInt(), /*isSigned=*/true),
                          /*is_unsigned=*/false, require_positive, name);
  } else {
    Scalar s = arg.ToScalar();
    ABSL_CHECK_LE(s.dtype().GetIntOrFloatBitWidth(), 32u)  // CRASH_OK
        << name << " scalar bitwidth must be <= 32";
    if (auto const_op =
            internal::unwrap(s).getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto attr = llvm::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
        ValidateConstantBound(attr.getValue(), s.dtype().IsUnsigned(),
                              require_positive, name);
      }
    }
  }
  return internal::unwrap(arg.ToScalar().ToI32());
}

mlir::Value OptionalI32Value(const std::optional<IndexArg>& arg,
                             std::string_view name) {
  if (!arg.has_value()) return nullptr;
  return ToValidatedI32Value(*arg, /*require_positive=*/false, name);
}

}  // namespace

Semaphore::Semaphore(const void* impl) : impl_(impl) {
  mlir::Type type = internal::unwrap(*this).getType();
  ValidateSemaphoreMemRefType(type, "Semaphore");
  ABSL_CHECK_EQ(llvm::cast<mlir::MemRefType>(type).getRank(), 0)  // CRASH_OK
      << "Semaphore must be 0-D (use SemaphoreArray for multi-dimensional "
         "semaphores)";
}

bool Semaphore::is_dma() const {
  return llvm::isa<tpu::DMASemaphoreType>(MemRefTypeOf(*this).getElementType());
}

Scalar Semaphore::Read() const {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      tpu::SemaphoreReadOp::create(b, b.getI32Type(), internal::unwrap(*this)),
      Int32Type());
}

void Semaphore::Signal(IndexArg amount, SemaphoreSignalOptions options) const {
  ABSL_CHECK(!is_dma())  // CRASH_OK
      << "Semaphore::Signal() requires a regular semaphore (AllocaSemaphore())";
  mlir::Value amt = ToValidatedI32Value(amount, /*require_positive=*/true,
                                        "Semaphore::Signal() amount");
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::SemaphoreSignalOp::create(
      b, internal::unwrap(*this), amt,
      OptionalI32Value(options.device_id, "SemaphoreSignalOptions::device_id"),
      OptionalI32Value(options.core_id, "SemaphoreSignalOptions::core_id"),
      /*subcore_id=*/nullptr);
}

void Semaphore::Wait(IndexArg amount) const {
  ABSL_CHECK(!is_dma())  // CRASH_OK
      << "Semaphore::Wait() requires a regular semaphore (AllocaSemaphore())";
  mlir::Value amt = ToValidatedI32Value(amount, /*require_positive=*/true,
                                        "Semaphore::Wait() amount");
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::SemaphoreWaitOp::create(b, internal::unwrap(*this), amt);
}

SemaphoreArray::SemaphoreArray(const void* impl) : impl_(impl) {
  mlir::Type type = internal::unwrap(*this).getType();
  ValidateSemaphoreMemRefType(type, "SemaphoreArray");
  ABSL_CHECK_GE(llvm::cast<mlir::MemRefType>(type).getRank(), 1)  // CRASH_OK
      << "SemaphoreArray must have rank >= 1";
}

llvm::ArrayRef<int64_t> SemaphoreArray::shape() const {
  return MemRefTypeOf(*this).getShape();
}

int64_t SemaphoreArray::rank() const { return MemRefTypeOf(*this).getRank(); }

bool SemaphoreArray::is_dma() const {
  return llvm::isa<tpu::DMASemaphoreType>(MemRefTypeOf(*this).getElementType());
}

Semaphore SemaphoreArray::AtImpl(llvm::ArrayRef<IndexArg> indices) const {
  ABSL_CHECK_EQ(static_cast<int64_t>(indices.size()), rank())  // CRASH_OK
      << "SemaphoreArray indexing requires exactly " << rank()
      << " indices, got " << indices.size();
  llvm::SmallVector<internal::DimDesc, 4> dims;
  dims.reserve(indices.size());
  for (const IndexArg& idx : indices) {
    dims.push_back(internal::DimDesc::From(idx));
  }
  const void* sliced = internal::SliceMemRefImpl(impl_, dims);
  return internal::wrap<Semaphore>(mlir::Value::getFromOpaquePointer(sliced));
}

}  // namespace torch_tpu::kiln
