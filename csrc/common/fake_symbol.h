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

#ifndef TORCH_TPU_CSRC_COMMON_FAKE_SYMBOL_H_
#define TORCH_TPU_CSRC_COMMON_FAKE_SYMBOL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/strings/str_cat.h"
#include "c10/core/ConstantSymNodeImpl.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymNodeImpl.h"
#include "c10/util/intrusive_ptr.h"

namespace torch_tpu {

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

  int64_t guard_int(const char* file, int64_t line) override;

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

// Returns a symbolic SymInt with the given hint and name that increments
// `*num_specializations` whenever it is specialized.
[[nodiscard]] inline c10::SymInt MakeSymbol(int64_t hint, std::string name,
                                            int* num_specializations) {
  return c10::SymInt(c10::SymNode(c10::make_intrusive<FakeSymbol>(
      hint, std::move(name), num_specializations)));
}

// Returns a CPU tensor whose sizes/strides are {2, s0}/{s0, 1}, the way
// torch.compile sees a FakeTensor with a dynamic dim 1.
[[nodiscard]] at::Tensor MakeSymbolicTensor(int* num_specializations);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_FAKE_SYMBOL_H_
