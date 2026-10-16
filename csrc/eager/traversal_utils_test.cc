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

#include "csrc/eager/traversal_utils.h"

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(PrepareTraversalsTest, NothingToExecute) {
  EXPECT_THAT(PrepareTraversals({}, {}, PrepareTraversalsOptions()),
              testing::IsEmpty());
}

TEST(PrepareTraversalsTest, SingleNode) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();

  auto actual = PrepareTraversals({list_a}, {}, PrepareTraversalsOptions());
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
}

TEST(PrepareTraversalsTest, MultiNodesFullGraph) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, a -> c
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  auto actual = PrepareTraversals(
      {list_a, list_b, list_c}, {},
      PrepareTraversalsOptions{.graph_split_mode = GraphSplitMode::kFullGraph});
  // Compiles as a single traversal with multiple outputs: b and c.
  // a is used within the traversal and is not "must materialize" so is not
  // an output.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_a, list_b, list_c));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_b, ref_c));
}

TEST(PrepareTraversalsTest, MultiNodesFullGraphWithMustMaterialize) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, a -> c
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  // List a as "must_materialize"
  auto actual = PrepareTraversals(
      {list_a, list_b, list_c}, {list_a},
      PrepareTraversalsOptions{.graph_split_mode = GraphSplitMode::kFullGraph});
  // Compiles as a single traversal with multiple outputs: a, b, and c.
  // Even though a is used within the traversal, it is also marked as
  // must_materialize, so it is included as an output.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_a, list_b, list_c));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a, ref_b, ref_c));
}

TEST(PrepareTraversalsTest, MultiNodesSplitGraph) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, a -> c
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  auto actual =
      PrepareTraversals({list_a, list_b, list_c}, {},
                        PrepareTraversalsOptions{
                            .graph_split_mode = GraphSplitMode::kSplitGraph});
  // Compiles as three separate traversals, one each for a, b, and c.
  // b and c are split into single outputs, and a is materialized to avoid
  // re-execution.
  ASSERT_EQ(actual.size(), 3);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
  EXPECT_THAT(actual[1]->execution_order(), testing::ElementsAre(list_b));
  EXPECT_THAT(actual[1]->outputs(), testing::ElementsAre(ref_b));
  EXPECT_THAT(actual[2]->execution_order(), testing::ElementsAre(list_c));
  EXPECT_THAT(actual[2]->outputs(), testing::ElementsAre(ref_c));
}

TEST(PrepareTraversalsTest, MultiNodesSplitGraphAllAlreadyOutputs) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, a -> c
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  // Force everything to materialize but still specify kSplitGraph.
  auto actual =
      PrepareTraversals({list_a, list_b, list_c}, {list_a, list_b, list_c},
                        PrepareTraversalsOptions{
                            .graph_split_mode = GraphSplitMode::kSplitGraph});
  // Because there are no additional materialization points necessary to
  // prevent re-execution, we can compile as a single traversal, even though
  // kSplitGraph is specified.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_a, list_b, list_c));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a, ref_b, ref_c));
}

TEST(PrepareTraversalsTest, OpSplitMode) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, c, but `a` has kSplitAfter and `c` is kSplitBefore.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape},
                                           OpSplitMode::kSplitAfter));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape},
                                           OpSplitMode::kSplitBefore));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  auto actual = PrepareTraversals(
      {list_a, list_b, list_c}, {},
      PrepareTraversalsOptions{.graph_split_mode = GraphSplitMode::kFullGraph});
  // Compiles as a three separate traversals, one each for a, b, and c,
  // even though kFullGraph is specified; OpSplitMode takes precedence.
  ASSERT_EQ(actual.size(), 3);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
  EXPECT_THAT(actual[1]->execution_order(), testing::ElementsAre(list_b));
  EXPECT_THAT(actual[1]->outputs(), testing::ElementsAre(ref_b));
  EXPECT_THAT(actual[2]->execution_order(), testing::ElementsAre(list_c));
  EXPECT_THAT(actual[2]->outputs(), testing::ElementsAre(ref_c));
}

TEST(PrepareTraversalsTest, MustMaterializeConstant) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Three ops: a, b -> c where a and b are constants.
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a, DeviceBufferList::CreateDeferred(
                       OpName::kTorchTpuInternalConstant, DummyBuilder, {},
                       OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b, DeviceBufferList::CreateDeferred(
                       OpName::kTorchTpuInternalConstant, DummyBuilder, {},
                       OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_b},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  // Prepare traversals specifying c as must-execute and a as must-materialize.
  // Leave b unspecified.
  auto actual =
      PrepareTraversals({list_c}, {list_a}, PrepareTraversalsOptions());
  // Compiles as a single traversal.
  // b is prepended before c as it is a required input to a must-execute node.
  // a is appended at the end as it is a must-materialize node but is otherwise
  // unused.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_b, list_c, list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_c, ref_a));
}

TEST(PrepareTraversalsTest, ExplicitCorePinning) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();

  auto actual =
      PrepareTraversals({list_a}, {},
                        PrepareTraversalsOptions{
                            .core_pinning_mode = CorePinningMode::kSparseCore});
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
  EXPECT_EQ(actual[0]->core_pinning_mode(), CorePinningMode::kSparseCore);
}

TEST(PrepareTraversalsTest, AutoSkinnySparseCoreOffload) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Dispatch a skinny collective op and a regular op.
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(
          OpName::kDistributedAllReduce, DummyBuilder, {},
          OpParamCacheKeys::Empty(), {shape}, OpSplitMode::kSplitBoth));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();

  auto actual = PrepareTraversals(
      {list_a, list_b}, {},
      PrepareTraversalsOptions{.skinny_sparse_core_offload_mode =
                                   SkinnySparseCoreOffload::kEnabled});
  // The skinny collective is split out and pinned to the SparseCore.
  // The regular op is left unpinned.
  ASSERT_EQ(actual.size(), 2);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
  EXPECT_EQ(actual[0]->core_pinning_mode(), CorePinningMode::kSparseCore);
  EXPECT_THAT(actual[1]->execution_order(), testing::ElementsAre(list_b));
  EXPECT_THAT(actual[1]->outputs(), testing::ElementsAre(ref_b));
  EXPECT_EQ(actual[1]->core_pinning_mode(), CorePinningMode::kUnpinned);
}

TEST(PrepareTraversalsTest,
     AutoSkinnySparseCoreOffloadDoesNotOverrideExplicitPinning) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Dispatch a skinny collective op and a regular op.
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a,
      DeviceBufferList::CreateDeferred(
          OpName::kDistributedAllReduce, DummyBuilder, {},
          OpParamCacheKeys::Empty(), {shape}, OpSplitMode::kSplitBoth));
  auto ref_a = refs_a[0];
  auto list_a = refs_a[0].device_buffer_list();

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = refs_b[0].device_buffer_list();

  // Explicitly pin to TensorCore.
  auto actual =
      PrepareTraversals({list_a, list_b}, {},
                        PrepareTraversalsOptions{
                            .core_pinning_mode = CorePinningMode::kTensorCore,
                            .skinny_sparse_core_offload_mode =
                                SkinnySparseCoreOffload::kEnabled});
  // The skinny collective is split out, but both traversals are pinned to the
  // TensorCore; the auto-offload is ignored.
  ASSERT_EQ(actual.size(), 2);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
  EXPECT_EQ(actual[0]->core_pinning_mode(), CorePinningMode::kTensorCore);
  EXPECT_THAT(actual[1]->execution_order(), testing::ElementsAre(list_b));
  EXPECT_THAT(actual[1]->outputs(), testing::ElementsAre(ref_b));
  EXPECT_EQ(actual[1]->core_pinning_mode(), CorePinningMode::kTensorCore);
}

}  // namespace
}  // namespace torch_tpu
