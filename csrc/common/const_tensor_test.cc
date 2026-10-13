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

#include "csrc/common/const_tensor.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/SparseTensorImpl.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/sparse_coo_tensor.h"
#include "ATen/ops/sparse_csr_tensor.h"
#include "ATen/ops/zeros.h"
#include "absl/strings/str_cat.h"
#include "c10/core/ConstantSymNodeImpl.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymNodeImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/typeid.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"

namespace torch_tpu {
namespace {

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

// Prints a SymInt the way at::Tensor's operator<< does, without guarding.
std::string SymIntToString(const c10::SymInt& value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

// Returns a CPU tensor whose sizes/strides are {2, s0}/{s0, 1}, the way
// torch.compile sees a FakeTensor with a dynamic dim 1.
at::Tensor MakeSymbolicTensor(int* num_specializations) {
  const c10::SymInt s0 = MakeSymbol(3, "s0", num_specializations);
  at::Tensor tensor = at::empty({2, 3});
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      std::vector<c10::SymInt>{c10::SymInt(2), s0},
      std::vector<c10::SymInt>{s0, c10::SymInt(1)});
  return tensor;
}

// Concepts used to check at compile time which at::Tensor methods ConstTensor
// does not expose, because they may specialize or throw on symbolic shapes.
template <typename T>
concept HasSizes = requires(const T& t) { t.sizes(); };
template <typename T>
concept HasStrides = requires(const T& t) { t.strides(); };
template <typename T>
concept HasSize = requires(const T& t) { t.size(0); };
template <typename T>
concept HasStride = requires(const T& t) { t.stride(0); };
template <typename T>
concept HasNumel = requires(const T& t) { t.numel(); };
template <typename T>
concept HasStorageOffset = requires(const T& t) { t.storage_offset(); };
template <typename T>
concept HasIsContiguous = requires(const T& t) { t.is_contiguous(); };
template <typename T>
concept HasSymIsContiguous = requires(const T& t) { t.sym_is_contiguous(); };
template <typename T>
concept HasStorage = requires(const T& t) { t.storage(); };
template <typename T>
concept HasDataPtr = requires(const T& t) { t.data_ptr(); };
template <typename T>
concept HasUseCount = requires(const T& t) { t.use_count(); };
template <typename T>
concept HasNnz = requires(const T& t) { t._nnz(); };
template <typename T>
concept HasIsSameSize =
    requires(const T& t, const at::Tensor& other) { t.is_same_size(other); };
template <typename T>
concept HasIsContiguousOrFalse =
    requires(const T& t) { t.is_contiguous_or_false(); };
template <typename T>
concept HasSuggestMemoryFormat =
    requires(const T& t) { t.suggest_memory_format(); };
template <typename T>
concept HasIsNonOverlappingAndDense =
    requires(const T& t) { t.is_non_overlapping_and_dense(); };
template <typename T>
concept HasGradFn = requires(const T& t) { t.grad_fn(); };
template <typename T>
concept HasConstDataPtr = requires(const T& t) { t.const_data_ptr(); };
template <typename T>
concept HasUnsafeGetTensorImpl =
    requires(const T& t) { t.unsafeGetTensorImpl(); };
template <typename T>
concept HasItem = requires(const T& t) { t.item(); };

TEST(ConstTensorTest, UndefinedTensor) {
  const ConstTensor tensor{at::Tensor()};

  EXPECT_FALSE(tensor.defined());
  EXPECT_FALSE(tensor.has_storage());
  EXPECT_FALSE(tensor.has_symbolic_sizes_strides());
  EXPECT_FALSE(tensor.ToTensor().defined());
}

TEST(ConstTensorTest, RoundTrip) {
  const at::Tensor tensor = at::empty({2, 3});
  const ConstTensor const_tensor = tensor;

  EXPECT_TRUE(const_tensor.ToTensor().is_same(tensor));
  EXPECT_TRUE(const_tensor.is_same(tensor));
  EXPECT_TRUE(const_tensor.is_same(ConstTensor(tensor)));
  EXPECT_FALSE(const_tensor.is_same(at::empty({2, 3})));
  EXPECT_FALSE(const_tensor.is_same(ConstTensor(at::empty({2, 3}))));
}

TEST(ConstTensorTest, Conversions) {
  static_assert(!std::is_default_constructible_v<ConstTensor>);
  static_assert(std::is_convertible_v<at::Tensor, ConstTensor>);
  static_assert(std::is_convertible_v<const at::Tensor&, ConstTensor>);
  static_assert(!std::is_convertible_v<ConstTensor, at::Tensor>);
  static_assert(!std::is_convertible_v<ConstTensor, const at::Tensor&>);
  static_assert(std::is_copy_constructible_v<ConstTensor>);
  static_assert(std::is_copy_assignable_v<ConstTensor>);
  static_assert(std::is_nothrow_move_constructible_v<ConstTensor>);
  static_assert(std::is_nothrow_move_assignable_v<ConstTensor>);
}

TEST(ConstTensorTest, UnsafeMethodsNotExposed) {
  // Sanity check that the concepts detect the methods on at::Tensor.
  static_assert(
      HasSizes<at::Tensor> && HasStrides<at::Tensor> && HasSize<at::Tensor> &&
      HasStride<at::Tensor> && HasNumel<at::Tensor> &&
      HasStorageOffset<at::Tensor> && HasIsContiguous<at::Tensor> &&
      HasSymIsContiguous<at::Tensor> && HasStorage<at::Tensor> &&
      HasDataPtr<at::Tensor> && HasUseCount<at::Tensor> && HasNnz<at::Tensor> &&
      HasIsSameSize<at::Tensor> && HasIsContiguousOrFalse<at::Tensor> &&
      HasSuggestMemoryFormat<at::Tensor> &&
      HasIsNonOverlappingAndDense<at::Tensor> && HasGradFn<at::Tensor> &&
      HasConstDataPtr<at::Tensor> && HasUnsafeGetTensorImpl<at::Tensor> &&
      HasItem<at::Tensor>);

  static_assert(!HasSizes<ConstTensor>);
  static_assert(!HasStrides<ConstTensor>);
  static_assert(!HasSize<ConstTensor>);
  static_assert(!HasStride<ConstTensor>);
  static_assert(!HasNumel<ConstTensor>);
  static_assert(!HasStorageOffset<ConstTensor>);
  static_assert(!HasIsContiguous<ConstTensor>);
  static_assert(!HasSymIsContiguous<ConstTensor>);
  static_assert(!HasStorage<ConstTensor>);
  static_assert(!HasDataPtr<ConstTensor>);
  static_assert(!HasUseCount<ConstTensor>);
  static_assert(!HasNnz<ConstTensor>);
  static_assert(!HasIsSameSize<ConstTensor>);
  static_assert(!HasIsContiguousOrFalse<ConstTensor>);
  static_assert(!HasSuggestMemoryFormat<ConstTensor>);
  static_assert(!HasIsNonOverlappingAndDense<ConstTensor>);
  static_assert(!HasGradFn<ConstTensor>);
  static_assert(!HasConstDataPtr<ConstTensor>);
  static_assert(!HasUnsafeGetTensorImpl<ConstTensor>);
  static_assert(!HasItem<ConstTensor>);
}

TEST(ConstTensorTest, StaticTensorAccessorsMatchTensor) {
  const at::Tensor base = at::empty({2, 3});
  const at::Tensor transposed = base.t();
  const at::Tensor offset = base.select(/*dim=*/0, /*index=*/1);

  for (const at::Tensor& tensor : {base, transposed, offset}) {
    SCOPED_TRACE(tensor.sizes());
    const ConstTensor const_tensor(tensor);

    EXPECT_TRUE(const_tensor.defined());
    EXPECT_FALSE(const_tensor.has_symbolic_sizes_strides());
    EXPECT_EQ(const_tensor.scalar_type(), tensor.scalar_type());
    EXPECT_EQ(const_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(const_tensor.device(), tensor.device());
    EXPECT_EQ(const_tensor.layout(), tensor.layout());
    EXPECT_EQ(const_tensor.key_set(), tensor.key_set());
    EXPECT_EQ(const_tensor.is_cpu(), tensor.is_cpu());
    EXPECT_EQ(const_tensor.is_cuda(), tensor.is_cuda());
    EXPECT_EQ(const_tensor.is_meta(), tensor.is_meta());
    EXPECT_EQ(const_tensor.is_privateuseone(), tensor.is_privateuseone());
    EXPECT_EQ(const_tensor.is_sparse(), tensor.is_sparse());
    EXPECT_EQ(const_tensor.is_nested(), tensor.is_nested());
    EXPECT_EQ(const_tensor.is_quantized(), tensor.is_quantized());
    EXPECT_FALSE(const_tensor.is_python_dispatch());
    EXPECT_EQ(const_tensor.requires_grad(), tensor.requires_grad());
    EXPECT_EQ(const_tensor.has_storage(), tensor.has_storage());
    EXPECT_EQ(const_tensor.dim(), tensor.dim());
    EXPECT_EQ(const_tensor.sym_sizes(), tensor.sym_sizes());
    EXPECT_EQ(const_tensor.sym_strides(), tensor.sym_strides());
    for (int64_t d = 0; d < tensor.dim(); ++d) {
      EXPECT_EQ(const_tensor.sym_size(d), tensor.size(d));
      EXPECT_EQ(const_tensor.sym_stride(d), tensor.stride(d));
    }
    EXPECT_EQ(const_tensor.sym_numel(), tensor.numel());
    EXPECT_EQ(const_tensor.sym_storage_offset(), tensor.storage_offset());
    EXPECT_EQ(const_tensor.maybe_is_contiguous(), tensor.is_contiguous());
    EXPECT_EQ(const_tensor.maybe_is_contiguous(at::MemoryFormat::ChannelsLast),
              tensor.is_contiguous(at::MemoryFormat::ChannelsLast));
    EXPECT_EQ(const_tensor.is_xpu(), tensor.is_xpu());
    EXPECT_EQ(const_tensor.is_ipu(), tensor.is_ipu());
    EXPECT_EQ(const_tensor.is_xla(), tensor.is_xla());
    EXPECT_EQ(const_tensor.is_mtia(), tensor.is_mtia());
    EXPECT_EQ(const_tensor.is_hpu(), tensor.is_hpu());
    EXPECT_EQ(const_tensor.is_lazy(), tensor.is_lazy());
    EXPECT_EQ(const_tensor.is_hip(), tensor.is_hip());
    EXPECT_EQ(const_tensor.is_ve(), tensor.is_ve());
    EXPECT_EQ(const_tensor.is_mps(), tensor.is_mps());
    EXPECT_EQ(const_tensor.is_maia(), tensor.is_maia());
    EXPECT_EQ(const_tensor.is_vulkan(), tensor.is_vulkan());
    EXPECT_EQ(const_tensor.is_metal(), tensor.is_metal());
    EXPECT_EQ(const_tensor.is_mkldnn(), tensor.is_mkldnn());
    EXPECT_EQ(const_tensor.is_sparse_csr(), tensor.is_sparse_csr());
    EXPECT_FALSE(const_tensor.is_sparse_compressed());
    EXPECT_EQ(const_tensor.is_inference(), tensor.is_inference());
    EXPECT_EQ(const_tensor.is_conj(), tensor.is_conj());
    EXPECT_EQ(const_tensor.is_neg(), tensor.is_neg());
    EXPECT_EQ(const_tensor.is_zerotensor(), tensor._is_zerotensor());
    EXPECT_EQ(const_tensor.is_floating_point(), tensor.is_floating_point());
    EXPECT_EQ(const_tensor.is_complex(), tensor.is_complex());
    EXPECT_EQ(const_tensor.is_signed(), tensor.is_signed());
    EXPECT_EQ(const_tensor.get_device(), tensor.get_device());
    EXPECT_EQ(const_tensor.options().dtype(), tensor.options().dtype());
    EXPECT_EQ(const_tensor.options().device(), tensor.options().device());
    EXPECT_EQ(const_tensor.options().layout(), tensor.options().layout());
    EXPECT_EQ(const_tensor.itemsize(), tensor.itemsize());
    EXPECT_EQ(const_tensor.element_size(), tensor.element_size());
    EXPECT_EQ(const_tensor.toString(), tensor.toString());
    EXPECT_EQ(const_tensor.sym_nbytes(), tensor.nbytes());
    EXPECT_EQ(const_tensor.is_leaf(), tensor.is_leaf());
    EXPECT_EQ(const_tensor.output_nr(), tensor.output_nr());
    EXPECT_EQ(const_tensor.retains_grad(), tensor.retains_grad());
    EXPECT_EQ(const_tensor.is_view(), tensor.is_view());
    EXPECT_EQ(const_tensor.grad_dtype(), tensor.grad_dtype());
    EXPECT_FALSE(const_tensor.grad().defined());
    EXPECT_FALSE(const_tensor.is_wrapped_number());
    EXPECT_EQ(const_tensor.sparse_dim(), 0);
    EXPECT_EQ(const_tensor.dense_dim(), tensor.dim());
  }
  ASSERT_TRUE(ConstTensor(transposed).is_view());
  EXPECT_TRUE(ConstTensor(transposed).base().is_same(base));
  EXPECT_ANY_THROW({ (void)ConstTensor(base).base(); });
  EXPECT_EQ(ConstTensor(base).sym_nnz(), std::nullopt);
  EXPECT_EQ(ConstTensor(base).maybe_is_contiguous(), true);
  EXPECT_EQ(ConstTensor(transposed).maybe_is_contiguous(), false);
  EXPECT_EQ(ConstTensor(offset).sym_storage_offset(), 3);
}

TEST(ConstTensorTest, SymbolicTensorAccessorsDoNotSpecialize) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const ConstTensor tensor(MakeSymbolicTensor(&num_specializations));

  EXPECT_TRUE(tensor.defined());
  EXPECT_TRUE(tensor.has_symbolic_sizes_strides());
  EXPECT_EQ(tensor.dim(), 2);
  ASSERT_EQ(tensor.sym_sizes().size(), 2);
  EXPECT_EQ(tensor.sym_sizes()[0], 2);
  EXPECT_EQ(SymIntToString(tensor.sym_sizes()[1]), "s0");
  EXPECT_EQ(SymIntToString(tensor.sym_strides()[0]), "s0");
  EXPECT_EQ(tensor.sym_strides()[1], 1);
  EXPECT_EQ(SymIntToString(tensor.sym_size(1)), "s0");
  EXPECT_EQ(SymIntToString(tensor.sym_size(-1)), "s0");
  EXPECT_EQ(SymIntToString(tensor.sym_stride(0)), "s0");
  EXPECT_EQ(SymIntToString(tensor.sym_numel()), "2*s0");
  EXPECT_EQ(tensor.sym_storage_offset(), 0);
  EXPECT_EQ(tensor.maybe_is_contiguous(), std::nullopt);
  EXPECT_EQ(tensor.scalar_type(), at::kFloat);
  EXPECT_EQ(tensor.device(), c10::Device(c10::kCPU));
  EXPECT_EQ(tensor.layout(), c10::kStrided);
  EXPECT_TRUE(tensor.is_cpu());
  EXPECT_FALSE(tensor.is_meta());
  EXPECT_FALSE(tensor.requires_grad());
  EXPECT_TRUE(tensor.has_storage());
  EXPECT_FALSE(tensor.is_xla());
  EXPECT_FALSE(tensor.is_mkldnn());
  EXPECT_FALSE(tensor.is_sparse_csr());
  EXPECT_FALSE(tensor.is_sparse_compressed());
  EXPECT_FALSE(tensor.is_inference());
  EXPECT_FALSE(tensor.is_conj());
  EXPECT_FALSE(tensor.is_neg());
  EXPECT_FALSE(tensor.is_zerotensor());
  EXPECT_TRUE(tensor.is_floating_point());
  EXPECT_FALSE(tensor.is_complex());
  EXPECT_TRUE(tensor.is_signed());
  EXPECT_EQ(tensor.get_device(), -1);
  EXPECT_EQ(tensor.options().dtype(), caffe2::TypeMeta::Make<float>());
  EXPECT_EQ(tensor.itemsize(), sizeof(float));
  EXPECT_EQ(tensor.element_size(), sizeof(float));
  EXPECT_EQ(tensor.toString(), "CPUFloatType");
  EXPECT_EQ(SymIntToString(tensor.sym_nbytes()), "2*s0*4");
  EXPECT_TRUE(tensor.is_leaf());
  EXPECT_EQ(tensor.output_nr(), 0);
  EXPECT_FALSE(tensor.retains_grad());
  EXPECT_FALSE(tensor.is_view());
  EXPECT_EQ(tensor.grad_dtype(), at::kFloat);
  EXPECT_FALSE(tensor.grad().defined());
  EXPECT_FALSE(tensor.is_wrapped_number());
  EXPECT_EQ(tensor.sparse_dim(), 0);
  EXPECT_EQ(tensor.dense_dim(), 2);

  EXPECT_EQ(num_specializations, 0);
}

// Guards the test harness itself: FakeSymbol must count the guard_int() calls
// that specialize a symbol. This calls guard_int() directly rather than through
// at::Tensor::sizes(), whose handling of symbolic shapes differs between
// PyTorch builds (e.g. it may need a Python interpreter).
TEST(ConstTensorTest, SymbolicSanityCheck) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const ConstTensor tensor(MakeSymbolicTensor(&num_specializations));

  EXPECT_EQ(tensor.sym_size(1).guard_int(__FILE__, __LINE__), 3);
  EXPECT_EQ(num_specializations, 1);
}

TEST(ConstTensorTest, Grad) {
  at::Tensor tensor = at::ones({2});
  tensor.set_requires_grad(true);
  EXPECT_FALSE(ConstTensor(tensor).grad().defined());

  tensor.mutable_grad() = at::ones({2});
  const ConstTensor const_tensor(tensor);
  EXPECT_TRUE(const_tensor.grad().is_same(tensor.grad()));
  EXPECT_TRUE(const_tensor.is_leaf());
  EXPECT_TRUE(const_tensor.requires_grad());
}

TEST(ConstTensorTest, SparseCooTensor) {
  // Three uncoalesced entries at (0, 0) of a 2x2 matrix with 4-element values.
  const at::Tensor tensor = at::sparse_coo_tensor(at::zeros({2, 3}, at::kLong),
                                                  at::ones({3, 4}), {2, 2, 4});
  const ConstTensor const_tensor(tensor);

  EXPECT_TRUE(const_tensor.is_sparse());
  EXPECT_EQ(const_tensor.layout(), c10::kSparse);
  EXPECT_EQ(const_tensor.sparse_dim(), 2);
  EXPECT_EQ(const_tensor.dense_dim(), 1);
  EXPECT_EQ(const_tensor.sparse_dim(), tensor.sparse_dim());
  EXPECT_EQ(const_tensor.dense_dim(), tensor.dense_dim());
  ASSERT_TRUE(const_tensor.sym_nnz().has_value());
  EXPECT_EQ(*const_tensor.sym_nnz(), 3);
  EXPECT_EQ(*const_tensor.sym_nnz(), tensor._nnz());
  EXPECT_FALSE(const_tensor.is_coalesced());
  EXPECT_EQ(const_tensor.is_coalesced(), tensor.is_coalesced());
  EXPECT_EQ(const_tensor.maybe_is_contiguous(), std::nullopt);
  EXPECT_FALSE(const_tensor.is_sparse_compressed());
}

TEST(ConstTensorTest, SparseCsrTensor) {
  // A 2x2 matrix with one specified element per row, both in column 0.
  const at::Tensor tensor =
      at::sparse_csr_tensor(at::arange(3, at::kLong), at::zeros({2}, at::kLong),
                            at::ones({2}), {2, 2}, at::TensorOptions());
  const ConstTensor const_tensor(tensor);

  EXPECT_TRUE(const_tensor.is_sparse_csr());
  EXPECT_TRUE(const_tensor.is_sparse_compressed());
  EXPECT_EQ(const_tensor.layout(), c10::kSparseCsr);
  EXPECT_EQ(const_tensor.sparse_dim(), 2);
  EXPECT_EQ(const_tensor.dense_dim(), 0);
  EXPECT_EQ(const_tensor.sparse_dim(), tensor.sparse_dim());
  EXPECT_EQ(const_tensor.dense_dim(), tensor.dense_dim());
  ASSERT_TRUE(const_tensor.sym_nnz().has_value());
  EXPECT_EQ(*const_tensor.sym_nnz(), 2);
  EXPECT_EQ(*const_tensor.sym_nnz(), tensor._nnz());
  // is_coalesced() is only defined for sparse COO tensors.
  EXPECT_ANY_THROW({ (void)const_tensor.is_coalesced(); });
  EXPECT_EQ(const_tensor.maybe_is_contiguous(), std::nullopt);
}

TEST(ConstTensorTest, SparseCooTensorWithSymbolicNnzDoesNotSpecialize) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  GTEST_SKIP() << "c10::SymInt pointer packing is incompatible with HWASAN.";
#endif
  int num_specializations = 0;
  const at::Tensor tensor = at::sparse_coo_tensor(at::zeros({2, 3}, at::kLong),
                                                  at::ones({3}), {2, 2});
  // Mirrors a FakeTensor whose number of specified elements is dynamic.
  // at::Tensor::_values() returns an alias, so modify the impl's own values.
  const auto* sparse_impl =
      dynamic_cast<const at::SparseTensorImpl*>(tensor.unsafeGetTensorImpl());
  ASSERT_NE(sparse_impl, nullptr);
  sparse_impl->values().unsafeGetTensorImpl()->set_sizes_and_strides(
      std::vector<c10::SymInt>{MakeSymbol(3, "u0", &num_specializations)},
      std::vector<c10::SymInt>{c10::SymInt(1)});
  const ConstTensor const_tensor(tensor);

  ASSERT_TRUE(const_tensor.sym_nnz().has_value());
  EXPECT_EQ(SymIntToString(*const_tensor.sym_nnz()), "u0");
  EXPECT_EQ(const_tensor.sparse_dim(), 2);
  EXPECT_EQ(const_tensor.dense_dim(), 0);
  EXPECT_FALSE(const_tensor.is_coalesced());
  EXPECT_EQ(const_tensor.maybe_is_contiguous(), std::nullopt);
  EXPECT_EQ(num_specializations, 0);
}

TEST(ConstTensorTest, MetaDeviceTensor) {
  const ConstTensor tensor(at::empty({2}, at::kMeta));

  EXPECT_TRUE(tensor.is_meta());
  EXPECT_FALSE(tensor.is_cpu());
  EXPECT_EQ(tensor.device(), c10::Device(c10::kMeta));
  EXPECT_TRUE(tensor.has_storage());
  EXPECT_EQ(tensor.sym_numel(), 2);
  EXPECT_EQ(tensor.maybe_is_contiguous(), true);
}

}  // namespace
}  // namespace torch_tpu
