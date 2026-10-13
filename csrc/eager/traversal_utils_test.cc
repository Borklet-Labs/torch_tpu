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
  EXPECT_THAT(PrepareTraversals({}, {}, CorePinningMode::kUnpinned),
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

  auto actual = PrepareTraversals({list_a}, {}, CorePinningMode::kUnpinned);
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_a));
}

TEST(PrepareTraversalsTest, MultiNodes) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  // Three ops: a -> b, c
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
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = refs_c[0].device_buffer_list();

  auto actual = PrepareTraversals({list_a, list_b, list_c}, {},
                                  CorePinningMode::kUnpinned);
  // Compiles as a single traversal with multiple outputs: b and c.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_a, list_b, list_c));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_b, ref_c));
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

  auto actual = PrepareTraversals({list_a, list_b, list_c}, {},
                                  CorePinningMode::kUnpinned);
  // Compiles as a three separate traversals, one each for a, b, and c.
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
      PrepareTraversals({list_c}, {list_a}, CorePinningMode::kUnpinned);
  // Compiles as a single traversal.
  // b is prepended before c as it is a required input to a must-execute node.
  // a is appended at the end as it is a must-materialize node but is otherwise
  // unused.
  ASSERT_EQ(actual.size(), 1);
  EXPECT_THAT(actual[0]->execution_order(),
              testing::ElementsAre(list_b, list_c, list_a));
  EXPECT_THAT(actual[0]->outputs(), testing::ElementsAre(ref_c, ref_a));
}

}  // namespace
}  // namespace torch_tpu
