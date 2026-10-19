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

#include "csrc/ops/reductions/sum.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildSumShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    const ReductionMode reduction_mode,
    const c10::optional<mlir::ElementType> element_type_opt) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::ElementType out_elem_type =
      element_type_opt.value_or(GetElementTypeOrDie(input_op));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(out_elem_type));
  const mlir::Type mlir_type =
      mlir::getElementType(builder.getContext(), compute_elem_type);
  mlir::MlirOp zero_val = MakeScalarConstant(builder, 0.0, mlir_type);

  const auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<stablehlo::AddOp>(
        mlir_type, rb.getRegion(), rb.getOpBuilder());
  };

  mlir::MlirOp result = BuildReductionShlo(input_op, reduce_dims, mlir_type,
                                           zero_val, reduce_fn, reduction_mode);
  return CastIfNeeded(result, out_elem_type);
}

}  // namespace torch_tpu
