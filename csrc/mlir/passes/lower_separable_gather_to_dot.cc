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

#include "csrc/mlir/passes/lower_separable_gather_to_dot.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

struct SingleAxisCoord {
  int64_t batch_axis;
  mlir::Value raw_value;
};

struct InitialCoordResult {
  llvm::SmallVector<int64_t> dim_to_batch;
  std::optional<SingleAxisCoord> immediate_coord;
};

struct StepResult {
  mlir::Value value;
  llvm::SmallVector<int64_t> dim_to_batch;
};

std::optional<InitialCoordResult> ComputeInitialDimToBatch(
    mlir::RankedTensorType cur_type, llvm::ArrayRef<int64_t> batch_shape,
    mlir::Value v) {
  const int64_t batch_rank = batch_shape.size();
  InitialCoordResult result;
  if (cur_type.getRank() == batch_rank + 1 &&
      cur_type.getDimSize(batch_rank) == 1) {
    for (int64_t b = 0; b < batch_rank; ++b) {
      if (cur_type.getDimSize(b) != batch_shape[b]) {
        return std::nullopt;
      }
      result.dim_to_batch.push_back(b);
    }
    result.dim_to_batch.push_back(-1);
    return result;
  }
  if (cur_type.getRank() == batch_rank) {
    for (int64_t b = 0; b < batch_rank; ++b) {
      if (cur_type.getDimSize(b) != batch_shape[b]) {
        return std::nullopt;
      }
      result.dim_to_batch.push_back(b);
    }
    return result;
  }
  if (cur_type.getRank() == 1) {
    int64_t matched_axis = -1;
    for (int64_t b = 0; b < batch_rank; ++b) {
      if (cur_type.getDimSize(0) == batch_shape[b]) {
        if (matched_axis != -1) {
          return std::nullopt;
        }
        matched_axis = b;
      }
    }
    if (matched_axis == -1) {
      return std::nullopt;
    }
    result.immediate_coord = SingleAxisCoord{matched_axis, v};
    return result;
  }
  return std::nullopt;
}

mlir::Value PeelIntegerConvert(mlir::Value v) {
  if (auto conv_op = v.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto opnd_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv_op.getOperand().getType());
    if (opnd_type && opnd_type.hasStaticShape() &&
        mlir::isa<mlir::IntegerType>(opnd_type.getElementType())) {
      return conv_op.getOperand();
    }
  }
  return v;
}

std::optional<int64_t> GetSingleActiveAxis(
    mlir::RankedTensorType type, llvm::ArrayRef<int64_t> dim_to_batch) {
  llvm::SmallVector<int64_t> active_axes;
  for (int64_t j = 0; j < type.getRank(); ++j) {
    if (dim_to_batch[j] != -1) {
      active_axes.push_back(dim_to_batch[j]);
    }
  }
  if (active_axes.size() == 1) {
    return active_axes[0];
  }
  return std::nullopt;
}

std::optional<StepResult> StepThroughBroadcast(
    mlir::stablehlo::BroadcastInDimOp bcast_op, mlir::RankedTensorType cur_type,
    llvm::ArrayRef<int64_t> dim_to_batch) {
  mlir::Value opnd = bcast_op.getOperand();
  auto opnd_type = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
  if (!opnd_type || !opnd_type.hasStaticShape()) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> bcast_dims = bcast_op.getBroadcastDimensions();
  llvm::SmallVector<int64_t> next_map(opnd_type.getRank(), -1);
  for (int64_t i = 0; i < opnd_type.getRank(); ++i) {
    const int64_t d = bcast_dims[i];
    if (opnd_type.getDimSize(i) == 1 && cur_type.getDimSize(d) > 1) {
      next_map[i] = -1;
    } else {
      next_map[i] = dim_to_batch[d];
    }
  }
  return StepResult{opnd, std::move(next_map)};
}

llvm::SmallVector<int64_t> GetNonUnitDimensions(mlir::RankedTensorType type) {
  llvm::SmallVector<int64_t> dims;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.getDimSize(i) > 1) {
      dims.push_back(i);
    }
  }
  return dims;
}

bool NonUnitDimensionsMatch(mlir::RankedTensorType opnd_type,
                            mlir::RankedTensorType cur_type,
                            llvm::ArrayRef<int64_t> opnd_nu,
                            llvm::ArrayRef<int64_t> cur_nu) {
  if (opnd_nu.size() != cur_nu.size()) {
    return false;
  }
  for (size_t t = 0; t < opnd_nu.size(); ++t) {
    if (opnd_type.getDimSize(opnd_nu[t]) != cur_type.getDimSize(cur_nu[t])) {
      return false;
    }
  }
  return true;
}

std::optional<StepResult> StepThroughReshape(
    mlir::stablehlo::ReshapeOp reshape_op, mlir::RankedTensorType cur_type,
    llvm::ArrayRef<int64_t> dim_to_batch) {
  mlir::Value opnd = reshape_op.getOperand();
  auto opnd_type = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
  if (!opnd_type || !opnd_type.hasStaticShape()) {
    return std::nullopt;
  }

  const llvm::SmallVector<int64_t> opnd_nu = GetNonUnitDimensions(opnd_type);
  const llvm::SmallVector<int64_t> cur_nu = GetNonUnitDimensions(cur_type);
  if (!NonUnitDimensionsMatch(opnd_type, cur_type, opnd_nu, cur_nu)) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> next_map(opnd_type.getRank(), -1);
  const size_t num_anchors = opnd_nu.size();
  for (size_t t = 0; t <= num_anchors; ++t) {
    const int64_t opnd_lo = (t == 0) ? 0 : opnd_nu[t - 1] + 1;
    const int64_t opnd_hi =
        (t == num_anchors) ? opnd_type.getRank() : opnd_nu[t];
    const int64_t cur_lo = (t == 0) ? 0 : cur_nu[t - 1] + 1;
    const int64_t cur_hi = (t == num_anchors) ? cur_type.getRank() : cur_nu[t];

    int64_t i = opnd_lo;
    for (int64_t j = cur_lo; j < cur_hi; ++j) {
      if (dim_to_batch[j] != -1 && i < opnd_hi) {
        next_map[i++] = dim_to_batch[j];
      }
    }
    if (t < num_anchors) {
      next_map[opnd_nu[t]] = dim_to_batch[cur_nu[t]];
    }
  }
  return StepResult{opnd, std::move(next_map)};
}

std::optional<StepResult> StepThroughSlice(
    mlir::stablehlo::SliceOp slice_op, mlir::RankedTensorType cur_type,
    llvm::ArrayRef<int64_t> dim_to_batch) {
  mlir::Value opnd = slice_op.getOperand();
  auto opnd_type = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
  if (!opnd_type || !opnd_type.hasStaticShape() ||
      !llvm::all_of(slice_op.getStrides(), [](int64_t s) { return s == 1; })) {
    return std::nullopt;
  }
  if (opnd_type.getShape() == cur_type.getShape()) {
    return StepResult{opnd, llvm::SmallVector<int64_t>(dim_to_batch)};
  }

  auto bcast_op = opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op) {
    return std::nullopt;
  }
  mlir::Value inner = bcast_op.getOperand();
  auto inner_type = mlir::dyn_cast<mlir::RankedTensorType>(inner.getType());
  if (!inner_type || !inner_type.hasStaticShape()) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> bcast_dims = bcast_op.getBroadcastDimensions();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::SmallVector<int64_t> next_map(inner_type.getRank(), -1);
  for (int64_t i = 0; i < inner_type.getRank(); ++i) {
    const int64_t d = bcast_dims[i];
    if (inner_type.getDimSize(i) > 1) {
      if (starts[d] != 0 || limits[d] != inner_type.getDimSize(i)) {
        return std::nullopt;
      }
      next_map[i] = dim_to_batch[d];
    } else if (opnd_type.getDimSize(d) == 1 && cur_type.getDimSize(d) == 1) {
      next_map[i] = dim_to_batch[d];
    } else {
      next_map[i] = -1;
    }
  }
  return StepResult{inner, std::move(next_map)};
}

std::optional<SingleAxisCoord> ResolveFromRemainingDimensions(
    mlir::RankedTensorType cur_type, mlir::Value cur,
    llvm::ArrayRef<int64_t> dim_to_batch) {
  const llvm::SmallVector<int64_t> non_unit_dims =
      GetNonUnitDimensions(cur_type);
  if (non_unit_dims.size() == 1 && dim_to_batch[non_unit_dims[0]] != -1) {
    return SingleAxisCoord{dim_to_batch[non_unit_dims[0]], cur};
  }
  if (non_unit_dims.empty()) {
    for (int64_t j = 0; j < cur_type.getRank(); ++j) {
      if (dim_to_batch[j] != -1) {
        return SingleAxisCoord{dim_to_batch[j], cur};
      }
    }
  }
  return std::nullopt;
}

// Walks backward through ConvertOp, ReshapeOp, BroadcastInDimOp, and unit
// stride SliceOp to verify that coordinate tensor `v` depends on at most one
// batch dimension of `batch_shape`, returning `(batch_axis, raw_value)` where
// `raw_value` has element count `batch_shape[batch_axis]`.
std::optional<SingleAxisCoord> ExtractSingleAxisCoord(
    mlir::Value v, llvm::ArrayRef<int64_t> batch_shape) {
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!cur_type || !cur_type.hasStaticShape()) {
    return std::nullopt;
  }

  std::optional<InitialCoordResult> initial_state =
      ComputeInitialDimToBatch(cur_type, batch_shape, v);
  if (!initial_state) {
    return std::nullopt;
  }
  if (initial_state->immediate_coord) {
    return initial_state->immediate_coord;
  }

  mlir::Value cur = v;
  llvm::SmallVector<int64_t> dim_to_batch =
      std::move(initial_state->dim_to_batch);

  while (cur) {
    cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
    if (!cur_type || !cur_type.hasStaticShape()) {
      return std::nullopt;
    }

    if (auto peeled = PeelIntegerConvert(cur); peeled != cur) {
      cur = peeled;
      continue;
    }

    if (auto active_axis = GetSingleActiveAxis(cur_type, dim_to_batch)) {
      return SingleAxisCoord{*active_axis, cur};
    }

    if (auto conv_op = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto opnd_type = mlir::dyn_cast<mlir::RankedTensorType>(
          conv_op.getOperand().getType());
      if (opnd_type && opnd_type.hasStaticShape()) {
        cur = conv_op.getOperand();
        continue;
      }
    }

    if (auto bcast_op =
            cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (auto step = StepThroughBroadcast(bcast_op, cur_type, dim_to_batch)) {
        cur = step->value;
        dim_to_batch = std::move(step->dim_to_batch);
        continue;
      }
    }

    if (auto reshape_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (auto step = StepThroughReshape(reshape_op, cur_type, dim_to_batch)) {
        cur = step->value;
        dim_to_batch = std::move(step->dim_to_batch);
        continue;
      }
    }

    if (auto slice_op = cur.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      if (auto step = StepThroughSlice(slice_op, cur_type, dim_to_batch)) {
        cur = step->value;
        dim_to_batch = std::move(step->dim_to_batch);
        continue;
      }
    }

    break;
  }

  return ResolveFromRemainingDimensions(cur_type, cur, dim_to_batch);
}

struct CoordStep {
  int64_t map_idx;
  int64_t batch_axis;
  mlir::Value raw_coord;
};

struct SeparableGatherToDotPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern<mlir::stablehlo::GatherOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::GatherOp gather_op,
      mlir::PatternRewriter& rewriter) const override {
    auto operand_type = mlir::dyn_cast<mlir::RankedTensorType>(
        gather_op.getOperand().getType());
    auto result_type =
        mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());
    auto start_indices_type = mlir::dyn_cast<mlir::RankedTensorType>(
        gather_op.getStartIndices().getType());
    if (!operand_type || !operand_type.hasStaticShape() || !result_type ||
        !result_type.hasStaticShape() || !start_indices_type ||
        !start_indices_type.hasStaticShape()) {
      return mlir::failure();
    }

    mlir::Type elem_type = operand_type.getElementType();
    if ((!elem_type.isBF16() && !elem_type.isF32()) ||
        result_type.getElementType() != elem_type) {
      return mlir::failure();
    }

    mlir::stablehlo::GatherDimensionNumbersAttr dnums =
        gather_op.getDimensionNumbers();
    if (!dnums.getOperandBatchingDims().empty() ||
        !dnums.getStartIndicesBatchingDims().empty()) {
      return mlir::failure();
    }

    llvm::ArrayRef<int64_t> start_index_map = dnums.getStartIndexMap();
    llvm::ArrayRef<int64_t> collapsed_slice_dims =
        dnums.getCollapsedSliceDims();
    llvm::ArrayRef<int64_t> offset_dims = dnums.getOffsetDims();
    llvm::ArrayRef<int64_t> slice_sizes = gather_op.getSliceSizes();

    const int64_t num_coords = start_index_map.size();
    if (num_coords < 2 ||
        static_cast<int64_t>(collapsed_slice_dims.size()) != num_coords) {
      return mlir::failure();
    }

    llvm::SmallVector<int64_t> sorted_collapsed =
        llvm::to_vector(collapsed_slice_dims);
    llvm::SmallVector<int64_t> sorted_map = llvm::to_vector(start_index_map);
    llvm::sort(sorted_collapsed);
    llvm::sort(sorted_map);
    if (sorted_collapsed != sorted_map) {
      return mlir::failure();
    }

    const int64_t operand_rank = operand_type.getRank();
    if (static_cast<int64_t>(slice_sizes.size()) != operand_rank) {
      return mlir::failure();
    }

    llvm::DenseSet<int64_t> collapsed_set(collapsed_slice_dims.begin(),
                                          collapsed_slice_dims.end());
    if (static_cast<int64_t>(collapsed_set.size()) != num_coords) {
      return mlir::failure();
    }

    int64_t unindexed_elems = 1;
    for (int64_t a = 0; a < operand_rank; ++a) {
      if (collapsed_set.contains(a)) {
        if (slice_sizes[a] != 1) {
          return mlir::failure();
        }
      } else {
        if (slice_sizes[a] != operand_type.getDimSize(a)) {
          return mlir::failure();
        }
        unindexed_elems *= operand_type.getDimSize(a);
      }
    }
    if (unindexed_elems < 32) {
      return mlir::failure();
    }

    const int64_t index_vector_dim = dnums.getIndexVectorDim();
    if (index_vector_dim != start_indices_type.getRank() - 1 ||
        start_indices_type.getDimSize(index_vector_dim) != num_coords) {
      return mlir::failure();
    }
    llvm::ArrayRef<int64_t> batch_shape =
        start_indices_type.getShape().drop_back(1);

    const int64_t batch_rank = batch_shape.size();
    llvm::SmallVector<CoordStep> steps;
    steps.reserve(num_coords);

    if (batch_rank != num_coords) {
      return mlir::failure();
    }
    auto concat_op = gather_op.getStartIndices()
                         .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!concat_op || concat_op.getDimension() != index_vector_dim ||
        static_cast<int64_t>(concat_op.getInputs().size()) != num_coords) {
      return mlir::failure();
    }

    llvm::DenseSet<int64_t> seen_batch_axes;
    for (int64_t k = 0; k < num_coords; ++k) {
      mlir::Value c_k = concat_op.getInputs()[k];
      auto c_type = mlir::dyn_cast<mlir::RankedTensorType>(c_k.getType());
      if (!c_type || !c_type.hasStaticShape() ||
          c_type.getRank() != batch_rank + 1 ||
          c_type.getDimSize(batch_rank) != 1) {
        return mlir::failure();
      }
      std::optional<SingleAxisCoord> extracted =
          ExtractSingleAxisCoord(c_k, batch_shape);
      if (!extracted.has_value()) {
        return mlir::failure();
      }
      const int64_t b_k = extracted->batch_axis;
      const int64_t out_dim_size = batch_shape[b_k];
      const int64_t in_dim_size = operand_type.getDimSize(start_index_map[k]);
      if (b_k < 0 || b_k >= num_coords || out_dim_size < 1 ||
          out_dim_size > 4096 || in_dim_size < 1 || in_dim_size > 4096 ||
          !seen_batch_axes.insert(b_k).second) {
        return mlir::failure();
      }
      steps.push_back(CoordStep{k, b_k, extracted->raw_value});
    }

    // Contract outer spatial coordinates first so shared outer contractions
    // across multiple corners deduplicate under CSE.
    llvm::sort(steps, [](const CoordStep& a, const CoordStep& b) {
      return a.batch_axis < b.batch_axis;
    });

    const int64_t result_rank = result_type.getRank();
    llvm::DenseSet<int64_t> offset_dim_set(offset_dims.begin(),
                                           offset_dims.end());
    llvm::SmallVector<int64_t> batch_out_dims;
    batch_out_dims.reserve(batch_rank);
    for (int64_t d = 0; d < result_rank; ++d) {
      if (!offset_dim_set.contains(d)) {
        batch_out_dims.push_back(d);
      }
    }
    if (static_cast<int64_t>(batch_out_dims.size()) != batch_rank) {
      return mlir::failure();
    }

    llvm::SmallVector<int64_t> cur_axes(operand_rank, -1);
    for (int64_t k = 0; k < num_coords; ++k) {
      cur_axes[start_index_map[k]] = -(k + 1);
    }
    int64_t unindexed_idx = 0;
    for (int64_t a = 0; a < operand_rank; ++a) {
      if (cur_axes[a] >= -1 && !collapsed_set.contains(a)) {
        cur_axes[a] = offset_dims[unindexed_idx++];
      }
    }

    mlir::MLIRContext* context = rewriter.getContext();
    const mlir::Location loc = gather_op.getLoc();
    const mlir::Type i32_type = rewriter.getI32Type();
    const mlir::Type i1_type = rewriter.getI1Type();

    mlir::ArrayAttr precision_config = nullptr;
    if (elem_type.isF32()) {
      precision_config = rewriter.getArrayAttr({
          mlir::stablehlo::PrecisionAttr::get(
              context,
              mlir::stablehlo::Precision::
                  HIGHEST),  // EXPLICIT_PRECISION_OK=f32_gather
          mlir::stablehlo::PrecisionAttr::get(
              context,
              mlir::stablehlo::Precision::
                  HIGHEST),  // EXPLICIT_PRECISION_OK=f32_gather
      });
    }

    mlir::Value cur = gather_op.getOperand();
    for (const CoordStep& step : steps) {
      const int64_t k = step.map_idx;
      const int64_t b_k = step.batch_axis;
      const int64_t in_dim_size = operand_type.getDimSize(start_index_map[k]);
      const int64_t out_dim_size = batch_shape[b_k];

      int64_t lhs_contract_dim = -1;
      for (int64_t i = 0; i < static_cast<int64_t>(cur_axes.size()); ++i) {
        if (cur_axes[i] == -(k + 1)) {
          lhs_contract_dim = i;
          break;
        }
      }
      if (lhs_contract_dim == -1) {
        return mlir::failure();
      }

      mlir::Value coord_1d = step.raw_coord;
      auto coord_raw_type =
          mlir::cast<mlir::RankedTensorType>(coord_1d.getType());
      if (coord_raw_type.getRank() != 1 ||
          coord_raw_type.getDimSize(0) != out_dim_size) {
        coord_1d = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc,
            mlir::RankedTensorType::get({out_dim_size},
                                        coord_raw_type.getElementType()),
            coord_1d);
      }

      mlir::Value coord_i32 = coord_1d;
      if (mlir::cast<mlir::RankedTensorType>(coord_i32.getType())
              .getElementType() != i32_type) {
        coord_i32 = rewriter.create<mlir::stablehlo::ConvertOp>(
            loc, mlir::RankedTensorType::get({out_dim_size}, i32_type),
            coord_i32);
      }

      auto mat_i32_type =
          mlir::RankedTensorType::get({out_dim_size, in_dim_size}, i32_type);
      mlir::Value coord_bcast =
          rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, mat_i32_type, coord_i32, rewriter.getDenseI64ArrayAttr({0}));
      mlir::Value iota = rewriter.create<mlir::stablehlo::IotaOp>(
          loc, mat_i32_type, rewriter.getI64IntegerAttr(1));
      mlir::Value indicator_i1 = rewriter.create<mlir::stablehlo::CompareOp>(
          loc,
          mlir::RankedTensorType::get({out_dim_size, in_dim_size}, i1_type),
          coord_bcast, iota,
          mlir::stablehlo::ComparisonDirectionAttr::get(
              context, mlir::stablehlo::ComparisonDirection::EQ),
          mlir::stablehlo::ComparisonTypeAttr());
      mlir::Value weight = rewriter.create<mlir::stablehlo::ConvertOp>(
          loc,
          mlir::RankedTensorType::get({out_dim_size, in_dim_size}, elem_type),
          indicator_i1);

      auto cur_tensor_type = mlir::cast<mlir::RankedTensorType>(cur.getType());
      llvm::SmallVector<int64_t> dot_out_shape;
      dot_out_shape.reserve(cur_tensor_type.getRank());
      for (int64_t i = 0; i < cur_tensor_type.getRank(); ++i) {
        if (i != lhs_contract_dim) {
          dot_out_shape.push_back(cur_tensor_type.getDimSize(i));
        }
      }
      dot_out_shape.push_back(out_dim_size);

      auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
          context, /*lhsBatchingDimensions=*/{}, /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{lhs_contract_dim},
          /*rhsContractingDimensions=*/{1});
      cur = rewriter.create<mlir::stablehlo::DotGeneralOp>(
          loc, mlir::RankedTensorType::get(dot_out_shape, elem_type), cur,
          weight, dot_dims, precision_config, /*algorithm=*/nullptr);

      cur_axes.erase(cur_axes.begin() + lhs_contract_dim);
      cur_axes.push_back(batch_out_dims[b_k]);
    }

    llvm::SmallVector<int64_t> perm(result_rank, 0);
    for (int64_t i = 0; i < result_rank; ++i) {
      perm[cur_axes[i]] = i;
    }

    bool is_identity_perm = true;
    for (int64_t i = 0; i < result_rank; ++i) {
      if (perm[i] != i) {
        is_identity_perm = false;
        break;
      }
    }
    if (!is_identity_perm) {
      cur = rewriter.create<mlir::stablehlo::TransposeOp>(
          loc, result_type, cur, rewriter.getDenseI64ArrayAttr(perm));
    }

    rewriter.replaceOp(gather_op, cur);
    return mlir::success();
  }
};

struct LowerSeparableGatherToDotPass
    : public mlir::PassWrapper<LowerSeparableGatherToDotPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSeparableGatherToDotPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-lower-separable-gather-to-dot";
  }
  llvm::StringRef getDescription() const override {
    return "Lowers separable multi axis gathers into sequential unbatched dot "
           "products.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<SeparableGatherToDotPattern>(context);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);
    if (mlir::failed(
            mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLowerSeparableGatherToDotPass() {
  return std::make_unique<LowerSeparableGatherToDotPass>();
}

}  // namespace torch_tpu
