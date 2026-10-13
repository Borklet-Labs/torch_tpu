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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.125f -
              static_cast<float>((i * 7) % 13) * 0.5f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = static_cast<float>((i * 13 + 3) % 64) * 0.25f - 4.0f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeI64Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int64_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 64));
  return mlir::DenseElementsAttr::get(type, vals);
}

TEST(LowerSeparableGatherToDotTest, Separable2DBilinearCornerGatherBf16AndF32) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIrBf16 = R"mlir(
func.func @separable_2d_bf16(
    %operand: tensor<1x64x148x148xbf16>,
    %idx_y: tensor<296x1xi64>,
    %idx_x: tensor<1x296xi64>) -> tensor<1x64x296x296xbf16> {
  %bcast_y = stablehlo.broadcast_in_dim %idx_y, dims = [0, 1]
      : (tensor<296x1xi64>) -> tensor<296x296xi64>
  %bcast_x = stablehlo.broadcast_in_dim %idx_x, dims = [0, 1]
      : (tensor<1x296xi64>) -> tensor<296x296xi64>
  %res_y = stablehlo.reshape %bcast_y
      : (tensor<296x296xi64>) -> tensor<296x296x1xi64>
  %res_x = stablehlo.reshape %bcast_x
      : (tensor<296x296xi64>) -> tensor<296x296x1xi64>
  %start_indices = stablehlo.concatenate %res_y, %res_x, dim = 2
      : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [0, 1],
          collapsed_slice_dims = [2, 3],
          start_index_map = [2, 3],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xbf16>, tensor<296x296x2xi64>)
      -> tensor<1x64x296x296xbf16>
  return %out : tensor<1x64x296x296xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_bf16 =
      mlir::parseSourceString<mlir::ModuleOp>(kIrBf16, &context);
  ASSERT_TRUE(module_bf16);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module_bf16, &context),
              absl_testing::IsOk());
  const std::string printed_bf16 = PrintModule(*module_bf16);

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module_bf16), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_bf16), 2);
  EXPECT_THAT(printed_bf16,
              testing::Not(testing::HasSubstr("stablehlo.gather")));
  EXPECT_THAT(printed_bf16, testing::HasSubstr("stablehlo.dot_general"));

  const char* kIrF32 = R"mlir(
func.func @separable_2d_f32(
    %operand: tensor<1x64x148x148xf32>,
    %idx_y: tensor<296x1xi64>,
    %idx_x: tensor<1x296xi64>) -> tensor<1x64x296x296xf32> {
  %bcast_y = stablehlo.broadcast_in_dim %idx_y, dims = [0, 1]
      : (tensor<296x1xi64>) -> tensor<296x296xi64>
  %bcast_x = stablehlo.broadcast_in_dim %idx_x, dims = [0, 1]
      : (tensor<1x296xi64>) -> tensor<296x296xi64>
  %res_y = stablehlo.reshape %bcast_y
      : (tensor<296x296xi64>) -> tensor<296x296x1xi64>
  %res_x = stablehlo.reshape %bcast_x
      : (tensor<296x296xi64>) -> tensor<296x296x1xi64>
  %start_indices = stablehlo.concatenate %res_y, %res_x, dim = 2
      : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [0, 1],
          collapsed_slice_dims = [2, 3],
          start_index_map = [2, 3],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xf32>, tensor<296x296x2xi64>)
      -> tensor<1x64x296x296xf32>
  return %out : tensor<1x64x296x296xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_f32 =
      mlir::parseSourceString<mlir::ModuleOp>(kIrF32, &context);
  ASSERT_TRUE(module_f32);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module_f32, &context),
              absl_testing::IsOk());
  const std::string printed_f32 = PrintModule(*module_f32);

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module_f32), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_f32), 2);
  EXPECT_THAT(printed_f32,
              testing::HasSubstr("precision = [HIGHEST, HIGHEST]"));
}

TEST(LowerSeparableGatherToDotTest, FourCornerBilinearSharesVerticalDotViaCSE) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kFourCornersIr = R"mlir(
func.func @four_corners(
    %operand: tensor<1x64x148x148xbf16>,
    %y0: tensor<296x1xi64>,
    %y1: tensor<296x1xi64>,
    %x0: tensor<1x296xi64>,
    %x1: tensor<1x296xi64>) -> tensor<1x64x296x296xbf16> {
  %by0 = stablehlo.broadcast_in_dim %y0, dims = [0, 1] : (tensor<296x1xi64>) -> tensor<296x296x1xi64>
  %by1 = stablehlo.broadcast_in_dim %y1, dims = [0, 1] : (tensor<296x1xi64>) -> tensor<296x296x1xi64>
  %bx0 = stablehlo.broadcast_in_dim %x0, dims = [0, 1] : (tensor<1x296xi64>) -> tensor<296x296x1xi64>
  %bx1 = stablehlo.broadcast_in_dim %x1, dims = [0, 1] : (tensor<1x296xi64>) -> tensor<296x296x1xi64>
  %c00 = stablehlo.concatenate %by0, %bx0, dim = 2 : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %c01 = stablehlo.concatenate %by0, %bx1, dim = 2 : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %c10 = stablehlo.concatenate %by1, %bx0, dim = 2 : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %c11 = stablehlo.concatenate %by1, %bx1, dim = 2 : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %g00 = "stablehlo.gather"(%operand, %c00) <{
      dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xbf16>, tensor<296x296x2xi64>) -> tensor<1x64x296x296xbf16>
  %g01 = "stablehlo.gather"(%operand, %c01) <{
      dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xbf16>, tensor<296x296x2xi64>) -> tensor<1x64x296x296xbf16>
  %g10 = "stablehlo.gather"(%operand, %c10) <{
      dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xbf16>, tensor<296x296x2xi64>) -> tensor<1x64x296x296xbf16>
  %g11 = "stablehlo.gather"(%operand, %c11) <{
      dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xbf16>, tensor<296x296x2xi64>) -> tensor<1x64x296x296xbf16>
  %s0 = stablehlo.add %g00, %g01 : tensor<1x64x296x296xbf16>
  %s1 = stablehlo.add %g10, %g11 : tensor<1x64x296x296xbf16>
  %sum = stablehlo.add %s0, %s1 : tensor<1x64x296x296xbf16>
  return %sum : tensor<1x64x296x296xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kFourCornersIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module), 0);
  // 2 shared vertical contractions (y0, y1) + 4 horizontal contractions = 6.
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module), 6);
}

TEST(LowerSeparableGatherToDotTest, NonSeparable2DGatherPreserved) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kNonSeparableIr = R"mlir(
func.func @non_separable_2d(
    %operand: tensor<1x64x148x148xf32>,
    %idx_y_2d: tensor<296x296xi64>,
    %idx_x_1d: tensor<1x296xi64>) -> tensor<1x64x296x296xf32> {
  %res_y = stablehlo.reshape %idx_y_2d
      : (tensor<296x296xi64>) -> tensor<296x296x1xi64>
  %bcast_x = stablehlo.broadcast_in_dim %idx_x_1d, dims = [0, 1]
      : (tensor<1x296xi64>) -> tensor<296x296x1xi64>
  %start_indices = stablehlo.concatenate %res_y, %bcast_x, dim = 2
      : (tensor<296x296x1xi64>, tensor<296x296x1xi64>) -> tensor<296x296x2xi64>
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [0, 1],
          collapsed_slice_dims = [2, 3],
          start_index_map = [2, 3],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 64, 1, 1>
  }> : (tensor<1x64x148x148xf32>, tensor<296x296x2xi64>)
      -> tensor<1x64x296x296xf32>
  return %out : tensor<1x64x296x296xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kNonSeparableIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module), 0);
}

TEST(LowerSeparableGatherToDotTest, ExactNumericalMatchAsymmetricShapesF32) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kAsymmetricIr = R"mlir(
func.func @main(
    %operand: tensor<2x16x5x7xf32>,
    %idx_y: tensor<4x1xi64>,
    %idx_x: tensor<1x6xi64>) -> tensor<2x16x4x6xf32> {
  %bcast_y = stablehlo.broadcast_in_dim %idx_y, dims = [0, 1]
      : (tensor<4x1xi64>) -> tensor<4x6xi64>
  %bcast_x = stablehlo.broadcast_in_dim %idx_x, dims = [0, 1]
      : (tensor<1x6xi64>) -> tensor<4x6xi64>
  %res_y = stablehlo.reshape %bcast_y : (tensor<4x6xi64>) -> tensor<4x6x1xi64>
  %res_x = stablehlo.reshape %bcast_x : (tensor<4x6xi64>) -> tensor<4x6x1xi64>
  %start_indices = stablehlo.concatenate %res_y, %res_x, dim = 2
      : (tensor<4x6x1xi64>, tensor<4x6x1xi64>) -> tensor<4x6x2xi64>
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [0, 1],
          collapsed_slice_dims = [2, 3],
          start_index_map = [2, 3],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 2, 16, 1, 1>
  }> : (tensor<2x16x5x7xf32>, tensor<4x6x2xi64>) -> tensor<2x16x4x6xf32>
  return %out : tensor<2x16x4x6xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kAsymmetricIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kAsymmetricIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeF32Tensor(context, {2, 16, 5, 7}, 1.0f);
  mlir::DenseElementsAttr idx_y = MakeI64Tensor(context, {4, 1}, {4, 1, 3, 0});
  mlir::DenseElementsAttr idx_x =
      MakeI64Tensor(context, {1, 6}, {6, 2, 5, 0, 3, 1});

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {operand, idx_y, idx_x}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {operand, idx_y, idx_x},
                                            interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  EXPECT_EQ((*expected)[0].getValues<float>().size(),
            (*actual)[0].getValues<float>().size());

  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(LowerSeparableGatherToDotTest,
     ExactNumericalMatchNonConsecutiveDimsAndSwappedCoordsBf16) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kNonConsecutiveIr = R"mlir(
func.func @main(
    %operand: tensor<2x5x16x7xbf16>,
    %idx_dim1_from_b1: tensor<1x6xi64>,
    %idx_dim3_from_b0: tensor<4x1xi64>) -> tensor<4x6x2x16xbf16> {
  %b0 = stablehlo.broadcast_in_dim %idx_dim1_from_b1, dims = [0, 1]
      : (tensor<1x6xi64>) -> tensor<4x6x1xi64>
  %b1 = stablehlo.broadcast_in_dim %idx_dim3_from_b0, dims = [0, 1]
      : (tensor<4x1xi64>) -> tensor<4x6x1xi64>
  %start_indices = stablehlo.concatenate %b0, %b1, dim = 2
      : (tensor<4x6x1xi64>, tensor<4x6x1xi64>) -> tensor<4x6x2xi64>
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [2, 3],
          collapsed_slice_dims = [1, 3],
          start_index_map = [1, 3],
          index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 2, 1, 16, 1>
  }> : (tensor<2x5x16x7xbf16>, tensor<4x6x2xi64>) -> tensor<4x6x2x16xbf16>
  return %out : tensor<4x6x2x16xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kNonConsecutiveIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kNonConsecutiveIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeBF16Tensor(context, {2, 5, 16, 7});
  mlir::DenseElementsAttr idx_0 =
      MakeI64Tensor(context, {1, 6}, {3, 1, 4, 0, 2, 3});
  mlir::DenseElementsAttr idx_1 = MakeI64Tensor(context, {4, 1}, {6, 2, 5, 1});

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {operand, idx_0, idx_1}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {operand, idx_0, idx_1},
                                            interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));

  auto exp_it = (*expected)[0].value_begin<llvm::APFloat>();
  auto act_it = (*actual)[0].value_begin<llvm::APFloat>();
  for (; exp_it != (*expected)[0].value_end<llvm::APFloat>();
       ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ((*exp_it).convertToFloat(), (*act_it).convertToFloat());
  }
}

TEST(LowerSeparableGatherToDotTest, ExactNumericalMatch1DGatherWithUnitBatch) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kGather1DIr = R"mlir(
func.func @main(
    %operand: tensor<2x16x7xf32>,
    %start_indices: tensor<1x6x1x1xi64>) -> tensor<2x16x1x6x1xf32> {
  %out = "stablehlo.gather"(%operand, %start_indices) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [0, 1],
          collapsed_slice_dims = [2],
          start_index_map = [2],
          index_vector_dim = 3>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 2, 16, 1>
  }> : (tensor<2x16x7xf32>, tensor<1x6x1x1xi64>) -> tensor<2x16x1x6x1xf32>
  return %out : tensor<2x16x1x6x1xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kGather1DIr, &context);
  ASSERT_TRUE(opt_mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 0);
}

}  // namespace
}  // namespace torch_tpu
