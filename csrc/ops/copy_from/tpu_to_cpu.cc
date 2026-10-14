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

#include "csrc/ops/copy_from/tpu_to_cpu.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/as_strided.h"
#include "ATen/ops/empty.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/reroute_warnings.h"
#include "csrc/common/status_builder.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/as_strided/as_strided_aten_kernels.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Translates a low-level XLA OOM error to an error that is easier to understand
// by a pytorch user. If the status is not an XLA OOM error, returns it as is.
//
// `elem_type` is the element type of the tensor that caused the XLA OOM error.
// `dims` is the dimensions of the tensor that caused the XLA OOM error.
static absl::Status TranslateXlaTensorOomError(absl::Status status,
                                               at::ScalarType elem_type,
                                               absl::Span<const int64_t> dims) {
  if (status.ok()) {
    return status;
  }

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=All PyTorch dtypes are supported
                        // by mlir::ElementType.
      const auto dtype, ConvertTo<mlir::ElementType>(elem_type));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=IsXlaOomError doesn't match format of
                 // compile-time OOM errors
      !IsXlaOomError(status), error::kResourceExhausted)
      << "the TPU ran out of memory while awaiting the materialization of "
         "value "
      << ToString(dtype) << "[" << absl::StrJoin(dims, ", ") << "]:\n"
      << status.message();

  // From this point, we know that the given status is not an XLA OOM error.
  return status;
}

// Ideal case: both src and dest are contiguous, tensors of the same shape and
// dtype, and src's storage buffer has exactly the right amount of data to copy
// into dest.
bool CanCopyStorageDirect(const at::Tensor& src, const at::Tensor& dest,
                          mlir::ElementType src_base_element_type) {
  if (src.scalar_type() != dest.scalar_type()) {
    // If the dtypes are different, we need to do a type conversion (not a
    // bitcast).
    // If the src's storage currently holds FP4 data, we either need to do a
    // non-blocking bitcast on device (e.g. bitcast FP4 to UI8 and transfer UI8)
    // or a blocking "packing" operation on host after transfer.
    return false;
  }

  if (src.sizes() != dest.sizes()) {
    // src needs to be broadcasted (either on TPU or CPU) to supply enough data
    // for dest.
    return false;
  }

  if (src.storage_offset() != 0) {
    // A storage copy will always start from an offset of 0 on the TPU side;
    // if src's storage offset is non-zero, a storage copy would move
    // out-of-scope data incorrectly, so we need a slice view.
    // Note that we don't care about dest's storage offset; CPU tensors will
    // automatically update Tensor::data_ptr() to reflect the offset.
    return false;
  }
  if (!src.is_contiguous() || !dest.is_contiguous()) {
    // dest must be contiguous, so that we can copy contiguous bytes into it
    // and ensure they are read properly.
    return false;
  }

  if (src_base_element_type == mlir::ElementType::F4E2M1FN) {
    // We can't directly copy FP4 data to CPU without a blocking packing
    // operation.
    return false;
  }

  if (src_base_element_type == mlir::ElementType::PRED) {
    // Boolean data data is held in a 1-bit representation on TPU, but an 8-bit
    // representation on CPU. However, PJRT will copy booleans as bytes, in the
    // same representation as PyTorch.
    // So as long as we have 1 boolean per byte to write, we can do a direct
    // copy.
    return src.storage().nbytes() * 8 == dest.nbytes();
  }

  // For all element types of byte size or larger, the PjRt and PyTorch
  // representations are the same.
  // This means that we can do a direct storage copy if the number of bytes
  // sent matches the number of bytes needed.
  return src.storage().nbytes() == dest.nbytes();
}

// Materializes the base buffer of src, and copies it byte-for-byte into dest's
// storage. This can be done asynchronously and does not materialize any
// ephemeral buffers.
absl::Status CopyTpuToCpuStorageDirect(const at::Tensor& src,
                                       const at::Tensor& dest,
                                       bool non_blocking) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buf, GetBaseBuffer(src));
  ABSL_DCHECK(CanCopyStorageDirect(src, dest, base_buf.element_type()));
  // Materialize the contiguous base buffer of the source tensor.
  const MaterializationBlockMode block_mode =
      non_blocking ? MaterializationBlockMode::kNonBlocking
                   : MaterializationBlockMode::kBlockUntilStarted;
  {
    tsl::profiler::TraceMe trace_mat("CopyTpuToCpu::MaterializeBase");
    TT_RETURN_IF_ERROR(TranslateXlaTensorOomError(
        Materialize(base_buf, MaterializationReason::kCpuTransfer,
                    MaterializationMode::kSplitGraph, block_mode),
        dest.scalar_type(), src.sizes()));
  }
  // Copy directly into dest's storage.
  return TpuMemcpyDtoHDirect(base_buf, dest.data_ptr(), non_blocking);
}

// If dest is overlapping or non-dense, then we cannot write a contiguous byte
// stream into it directly, and a blocking TPU->CPU->CPU conversion is required.
//
// If dest is FP4, then we can't transfer directly into its storage (even with a
// TPU-side conversion); a CPU conversion is required to pack the data.
bool RequiresCpuConversion(const at::Tensor& src, const at::Tensor& dest) {
  return !dest.is_non_overlapping_and_dense() ||
         dest.scalar_type() == at::kFloat4_e2m1fn_x2;
}

// Using metadata-only views (as_strided), broadcasts src to the shape of dest.
// Returns an error if this is not possible.
absl::StatusOr<at::Tensor> BroadcastToDest(const at::Tensor& src,
                                           const at::Tensor& dest) {
  if (src.sizes() == dest.sizes()) {
    // No broadcasting is needed.
    return src;
  }
  // Broadcast src to match the sizes of dest.
  auto [new_src_sizes, new_src_strides] =
      at::inferExpandGeometry(src.sizes(), src.strides(), dest.sizes());
  return AsStrided(src, new_src_sizes, new_src_strides, src.storage_offset());
}

// The source and destination tensors for a copy, after ensuring that the dest
// tensor is contiguous.
struct ContiguouslyTransposedCopy {
  at::Tensor src;   // UNINITIALIZED_TENSOR_OK=named output value
  at::Tensor dest;  // UNINITIALIZED_TENSOR_OK=named output value
};

// Using metadata-only views (as_strided), transposes dest so that it is
// row-major (possibly with leading size-1 dimensions), and applies the same
// transpose to src.
absl::StatusOr<ContiguouslyTransposedCopy> TransposeToContiguousDest(
    const at::Tensor& src, const at::Tensor& dest) {
  ABSL_CHECK_EQ(  // CRASH_OK=precondition established by BroadcastToDest
      src.dim(), dest.dim())
      << "cannot create a matching transpose for tensors of different "
         "dimensions";
  if (dest.is_contiguous()) {
    return ContiguouslyTransposedCopy{src, dest};
  }

  // Determine a transpose that can be applied to both src and dest to make
  // dest contiguous (even if it makes src non-contiguous).
  struct TransposeDim {
    int64_t original_index;
    int64_t size;
    int64_t src_stride;
    int64_t dest_stride;
  };

  std::vector<TransposeDim> transpose_dims;
  transpose_dims.reserve(dest.dim());
  for (int64_t original_index = 0; original_index < dest.dim();
       ++original_index) {
    transpose_dims.push_back({original_index, dest.size(original_index),
                              src.stride(original_index),
                              dest.stride(original_index)});
  }
  // Sort transpose_dims by dest.stride descending, except we put all size=1
  // dimensions first. If there are multiple size-1 dims, do a stable sort
  // (lower original_index first).
  std::sort(transpose_dims.begin(), transpose_dims.end(),
            [](const TransposeDim& a, const TransposeDim& b) {
              if (a.size == 1 && b.size != 1) {
                return true;
              }
              if (a.size != 1 && b.size == 1) {
                return false;
              }
              if (a.size == 1 && b.size == 1) {
                return a.original_index < b.original_index;
              }
              return a.dest_stride > b.dest_stride;
            });
  Dimensions new_sizes;
  Strides new_src_strides;
  Strides new_dest_strides;
  new_sizes.reserve(dest.dim());
  new_src_strides.reserve(dest.dim());
  new_dest_strides.reserve(dest.dim());
  for (const auto& dim : transpose_dims) {
    new_sizes.push_back(dim.size);
    new_src_strides.push_back(dim.src_stride);
    new_dest_strides.push_back(dim.dest_stride);
  }
  TT_ASSIGN_OR_RETURN(
      at::Tensor restrided_src,
      AsStrided(src, new_sizes, new_src_strides, src.storage_offset()));
  at::Tensor restrided_dest =
      at::as_strided(dest, new_sizes, new_dest_strides, dest.storage_offset());
  return ContiguouslyTransposedCopy{restrided_src, restrided_dest};
}

// Converts a TPU source tensor into a contiguous DeviceBufferRef with the same
// size, shape, and dtype as a CPU destination tensor, and then copies it to the
// destination's storage.
absl::Status CopyTpuToCpuWithTpuConversion(
    const at::Tensor& src, const at::Tensor& dest,
    mlir::ElementType src_base_element_type, bool non_blocking) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=checked in "to" or "copy_" first
      src.dim() <= dest.dim(), error::kInvalidArgument)
      << "source tensor is not broadcastable to destination tensor; source "
         "tensor has "
      << src.dim() << " dimensions, destination tensor has " << dest.dim()
      << " dimensions";
  // Broadcast src to dest, if needed.
  TT_ASSIGN_OR_RETURN(at::Tensor broadcasted_src, BroadcastToDest(src, dest));

  // Transpose src and dest so that dest is contiguous, if needed.
  TT_ASSIGN_OR_RETURN(ContiguouslyTransposedCopy after_transpose,
                      TransposeToContiguousDest(broadcasted_src, dest));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check of logic in
                 // aten::is_non_overlapping_and_dense
      after_transpose.dest.is_contiguous(), error::kInternal)
      << "expected a non-overlapping, dense, tensor to be contiguous after "
         "row-major transpose";

  if (CanCopyStorageDirect(after_transpose.src, after_transpose.dest,
                           src_base_element_type)) {
    // In the case where src and dest are identical transposes of a contiguous
    // storage (no broadcast or type conversion) then after the transpose, we
    // can copy directly from src to dest.
    return CopyTpuToCpuStorageDirect(after_transpose.src, after_transpose.dest,
                                     non_blocking);
  }

  // Convert the source tensor to the destination tensor's dtype, if needed.
  at::Tensor converted_src = after_transpose.src;
  if (after_transpose.src.scalar_type() != dest.scalar_type()) {
    TT_ASSIGN_OR_RETURN(
        converted_src,
        MakeEmptyMemoryFormat(
            after_transpose.dest.sizes(), dest.scalar_type(),
            /*layout_opt=*/std::nullopt, after_transpose.src.device(),
            /*pin_memory_opt=*/false, at::MemoryFormat::Contiguous));
    TT_RETURN_IF_ERROR(CopyTpuToTpu(after_transpose.src, converted_src));
  }

  // Read the view data from src and materialize it as an ephemeral contiguous
  // buffer.
  MaterializationBlockMode block_mode =
      non_blocking ? MaterializationBlockMode::kNonBlocking
                   : MaterializationBlockMode::kBlockUntilStarted;
  TT_ASSIGN_OR_RETURN(
      const DeviceBufferRef materialized_src_buf,
      [&]() {
        tsl::profiler::TraceMe trace_mat("CopyTpuToCpu::MaterializeAndReturn");
        return MaterializeAndReturn(
            converted_src, MaterializationReason::kCpuTransfer, block_mode);
      }(),
      TranslateXlaTensorOomError(_, dest.scalar_type(), src.sizes()));

  // Start or schedule a copy from the TPU buffer into the destination CPU
  // storage. Because we only applied as_strided to dest, the storage data_ptr
  // is unchanged.
  return TpuMemcpyDtoHDirect(materialized_src_buf, dest.data_ptr(),
                             non_blocking);
}

// PJRT can only asynchronously copy from a TPU tensor into contiguous CPU
// storage. Therefore, if the destination tensor is not contiguous, we must do
// a two step blocking copy:
//  1. Copy from the source TPU tensor into a temporary contiguous CPU buffer
//  2. Copy from the contiguous CPU buffer into the destination CPU buffer.
//
// We also use this case for blocking copies that require broadcasting or
// type conversion; doing the broadcast after the copy minimizes the amount of
// data transferred, and does conversions (both bitcasting from the base buffer
// to the src dtype, and non-bitcast conversions from src to dest dtypes) on
// host where they tend to be faster/more efficient.
absl::Status CopyTpuToCpuWithCpuConversion(const at::Tensor& src,
                                           const at::Tensor& dest) {
  // Materialize the contiguous base buffer of the source tensor.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buf, GetBaseBuffer(src));
  {
    tsl::profiler::TraceMe trace_mat("CopyTpuToCpu::MaterializeBase");
    TT_RETURN_IF_ERROR(TranslateXlaTensorOomError(
        Materialize(base_buf, MaterializationReason::kCpuTransfer,
                    MaterializationMode::kSplitGraph,
                    MaterializationBlockMode::kBlockUntilStarted),
        dest.scalar_type(), src.sizes()));
  }

  // Copy exactly these contiguous elements into a contiguous CPU storage.
  Dimensions unpacked_base_dims;
  at::IntArrayRef base_dims_array = base_buf.dimensions();
  if (base_buf.element_type() == mlir::ElementType::F4E2M1FN) {
    // XLA will copy to an "unpacked" representation, with one byte per 4-bit
    // element. This means we need twice as much storage in the empty CPU temp
    // tensor.
    unpacked_base_dims = CopyIntVector(base_buf.dimensions());
    unpacked_base_dims.push_back(2);
    base_dims_array = unpacked_base_dims;
  }
  at::ScalarType base_buf_at_type =
      ConvertTo<at::ScalarType>(base_buf.element_type());
  at::TensorOptions cpu_temp_options =
      at::TensorOptions()
          .dtype(base_buf_at_type)
          .device(dest.device())
          .memory_format(at::MemoryFormat::Contiguous);
  at::Tensor cpu_temp = at::empty(base_dims_array, cpu_temp_options);
  TT_RETURN_IF_ERROR(TpuMemcpyDtoHDirect(base_buf, cpu_temp.data_ptr(),
                                         /*non_blocking=*/false));

  // Pack FP4 data if necessary.
  if (base_buf.element_type() == mlir::ElementType::F4E2M1FN) {
    at::Tensor packed_cpu_temp =
        at::empty(base_buf.dimensions(), cpu_temp_options);
    PackFp4(static_cast<const uint8_t*>(cpu_temp.data_ptr()),
            static_cast<uint8_t*>(packed_cpu_temp.data_ptr()),
            packed_cpu_temp.numel());
    cpu_temp = std::move(packed_cpu_temp);
  }

  // View the source Storage on host in the same way as on TPU.
  auto* cpu_temp_impl = cpu_temp.unsafeGetTensorImpl();
  cpu_temp_impl->set_storage_and_dtype(cpu_temp.storage(), src.dtype());
  cpu_temp_impl->set_sizes_and_strides(src.sizes(), src.strides(),
                                       src.storage_offset());

  // Then do a CPU-to-CPU copy from our temp tensor to the destination.
  // This will do three things:
  //   1. If necessary, broadcast from src.sizes() to dest.sizes().
  //   2. If necessary, convert (not a bitcast) from src.scalar_type() to
  //      dest.scalar_type().
  //   3. If dest is not contiguous and dense (e.g. it is a slice or has a
  //      non-row-major layout like ChannelsLast3D), this will copy it into
  //      the correct positions in dest's existing Storage.
  // This must be a blocking copy because as soon as this function returns,
  // cpu_temp will be deallocated.
  dest.copy_(cpu_temp, /*non_blocking=*/false);
  return absl::OkStatus();
}

}  // namespace

absl::Status CopyTpuToCpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking) {
  tsl::profiler::TraceMe trace("CopyTpuToCpu");
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> CPU copy path for "
               << ToString(src, "src");
  // Surface warnings rerouted from background threads that the copy waited on,
  // also on errors.
  absl::Cleanup cleanup = [] { SurfaceReroutedWarnings(); };

  if (src.numel() == 0) {
    // Shortcut: no data to move. Create a new storage for dest if it doesn't
    // have one, otherwise leave it as is.
    if (!dest.has_storage()) {
      at::Tensor empty_dest = at::empty(dest.sizes(), dest.options());
      dest.unsafeGetTensorImpl()->set_storage_keep_dtype(empty_dest.storage());
    }
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef src_buf, GetBaseBuffer(src));

  if (CanCopyStorageDirect(src, dest, src_buf.element_type())) {
    // Ideal case: copy directly from TPU to CPU storage with no materialized
    // view, with async capability.
    return CopyTpuToCpuStorageDirect(src, dest, non_blocking);
  }

  if (RequiresCpuConversion(src, dest)) {
    // If we need to write non-consecutive bytes or do FP4 packing, we must do
    // a blocking copy with CPU conversion, even if the user requested a
    // non-blocking copy.
    return CopyTpuToCpuWithCpuConversion(src, dest);
  }

  if (non_blocking) {
    // If we can do a non-blocking copy and the user requested it, we do so.
    // This will materialize a view tensor with the size, shape, and dtype of
    // dest, and copy into its storage.
    return CopyTpuToCpuWithTpuConversion(src, dest, src_buf.element_type(),
                                         non_blocking);
  }

  // This is a blocking copy which does not require a CPU conversion, but we
  // could still do one if it would be more efficient.
  // If we materialize the view on TPU, then the transfer cost is the byte size
  // of the destination tensor we write into.
  // Note: src.nbytes() uses PyTorch's element size.
  int64_t tpu_conversion_transfer_cost = dest.nbytes();
  if (dest.scalar_type() == at::ScalarType::Float4_e2m1fn_x2) {
    // Even though both PyTorch and XLA use 4 bits per FP4 element, during
    // transfer they are moved as 1 byte per element, doubling the cost.
    tpu_conversion_transfer_cost *= 2;
  }

  // If we copy the entire storage buffer to host, then the transfer cost is the
  // byte size of the underlying storage. This is more efficient for broadcasts
  // of small tensors.
  // Note: src_buf.size_bytes() uses XLA's element size.
  int64_t cpu_conversion_transfer_cost = src_buf.size_bytes();
  if (src_buf.element_type() == mlir::ElementType::F4E2M1FN) {
    // Even though both PyTorch and XLA use 4 bits per FP4 element, during
    // transfer they are moved as 1 byte per element, doubling the cost.
    cpu_conversion_transfer_cost *= 2;
  } else if (src_buf.element_type() == mlir::ElementType::PRED) {
    // XLA uses 1 bit per PRED element, but transfers them as 1 byte per
    // element.
    cpu_conversion_transfer_cost *= 8;
  }

  // In case of a tie, we prefer to use a CPU conversion, as this requires less
  // work on the TPU; view operations are usually free on CPU.
  if (tpu_conversion_transfer_cost < cpu_conversion_transfer_cost) {
    return CopyTpuToCpuWithTpuConversion(src, dest, src_buf.element_type(),
                                         non_blocking);
  }
  return CopyTpuToCpuWithCpuConversion(src, dest);
}

}  // namespace torch_tpu
