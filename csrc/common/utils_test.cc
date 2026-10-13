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

#include "csrc/common/utils.h"

#include <cstddef>

#include "csrc/common/dimension_types.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(PercAsStr, ComputesPercentage) {
  EXPECT_EQ(PercAsStr(50, 100), "50.0%");
  EXPECT_EQ(PercAsStr(1, 3), "33.3%");
  EXPECT_EQ(PercAsStr(0, 100), "0.0%");
  EXPECT_EQ(PercAsStr(10, 0), "NA");
}

TEST(FilterIndices, FiltersCorrectly) {
  Indices even = FilterIndices(6, [](size_t i) { return i % 2 == 0; });
  EXPECT_EQ(even, (Indices{0, 2, 4}));
}

}  // namespace
}  // namespace torch_tpu
