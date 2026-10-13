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

#include "csrc/common/to_string.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "absl/strings/str_cat.h"
#include "c10/core/ConstantSymNodeImpl.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymNodeImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;
using testing::Not;

// A SymInt node that behaves like a Dynamo size symbol: it reports no constant
// value, so TensorImpl treats sizes built from it as symbolic, it prints as its
// symbol name, and it counts the guard_int() calls that would specialize it.
class FakeSymbol final : public c10::ConstantSymNodeImpl<int64_t> {
 public:
  FakeSymbol(int64_t hint, std::string name, int* num_specializations)
      : ConstantSymNodeImpl<int64_t>(hint),
        name_(std::move(name)),
        num_specializations_(num_specializations) {}

  std::optional<int64_t> constant_int() override { return std::nullopt; }
  std::optional<int64_t> maybe_as_int() override { return std::nullopt; }
  bool is_constant() override { return false; }
  bool is_symbolic() override { return true; }
  std::string str() override { return name_; }

  int64_t guard_int(const char* /*file*/, int64_t /*line*/) override {
    ++*num_specializations_;
    return int_();
  }

  c10::SymNode clone() override {
    return c10::make_intrusive<FakeSymbol>(int_(), name_, num_specializations_);
  }

  c10::SymNode wrap_int(int64_t num) override {
    return c10::make_intrusive<FakeSymbol>(num, absl::StrCat(num),
                                           num_specializations_);
  }

  c10::SymNode mul(const c10::SymNode& other) override {
    return c10::make_intrusive<FakeSymbol>(
        int_() * other->int_(), absl::StrCat(name_, "*", other->str()),
        num_specializations_);
  }

 private:
  std::string name_;
  int* num_specializations_;
};

c10::SymInt MakeSymbol(int64_t hint, std::string name,
                       int* num_specializations) {
  return c10::SymInt(c10::SymNode(c10::make_intrusive<FakeSymbol>(
      hint, std::move(name), num_specializations)));
}

TEST(ToStringTensorTest, UndefinedTensor) {
  EXPECT_EQ(ToString(at::Tensor()), "undefined");
  EXPECT_EQ(ToString(at::Tensor(), "x"), "x: undefined");
}

TEST(ToStringTensorTest, StaticShape) {
  const std::string result = ToString(at::empty({2, 3}), "x");

  EXPECT_THAT(result, HasSubstr("x: shape=[2, 3], strides=[3, 1], numel=6, "));
  EXPECT_THAT(result, HasSubstr("device=cpu, is_contiguous=1, is_cpu=1, "));
  EXPECT_THAT(result, HasSubstr("storage_offset=0, storage_use_count=1, "));
}

TEST(ToStringTensorTest, SymbolicShapeIsPrintedWithoutSpecializing) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const c10::SymInt s0 = MakeSymbol(3, "s0", &num_specializations);
  at::Tensor tensor = at::empty({2, 3});
  // Mirrors what torch.compile does for a FakeTensor with a dynamic dim 1.
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      std::vector<c10::SymInt>{c10::SymInt(2), s0},
      std::vector<c10::SymInt>{s0, c10::SymInt(1)});
  ASSERT_TRUE(tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides());

  const std::string result = ToString(tensor);

  EXPECT_THAT(result,
              HasSubstr("shape=[2, s0], strides=[s0, 1], numel=2*s0, "));
  EXPECT_THAT(result, HasSubstr("is_cpu=1, is_cuda=0, is_meta=0, "
                                "storage_offset=0, storage_use_count=1, "));
  // Contiguity of a symbolic tensor would be a guarded expression.
  EXPECT_THAT(result, Not(HasSubstr("is_contiguous")));
  EXPECT_EQ(num_specializations, 0);
}

}  // namespace
}  // namespace torch_tpu
