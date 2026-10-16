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

#include "csrc/common/fake_tensor_utils.h"

#include <optional>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/ones.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/StorageImpl.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/util/Exception.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

TEST(IsFakeOrFunctorchTensorTest, RealTensor) {
  EXPECT_FALSE(IsFakeOrFunctorchTensor(at::ones({2, 3})));
}

TEST(IsFakeOrFunctorchTensorTest, MetaTensor) {
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::empty({2, 3}, at::kMeta)));
}

TEST(IsFakeOrFunctorchTensorTest, UndefinedTensor) {
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::Tensor()));
}

TEST(IsFakeOrFunctorchTensorTest, ZeroElementTensor) {
  // Real tensors without any bytes have a null data pointer.
  EXPECT_TRUE(IsFakeOrFunctorchTensor(at::empty({0})));
}

TEST(IsFakeOrFunctorchTensorTest, DataPtrAccessThrows) {
  // A storage whose data pointer access throws used to make this throw. Such
  // storages (e.g. of real tensors whose memory was reclaimed by CUDA graph
  // trees) still hold real data, so they are not reported as fake.
  const at::Tensor tensor = at::ones({2, 3});
  tensor.unsafeGetTensorImpl()
      ->unsafe_storage()
      .unsafeGetStorageImpl()
      ->release_data_and_set_meta_custom_data_ptr_error_msg_(std::nullopt);
  ASSERT_THROW(tensor.storage().data_ptr(), c10::Error);

  EXPECT_FALSE(IsFakeOrFunctorchTensor(tensor));
}

TEST(IsFakeOrFunctorchTensorTest, FunctorchActive) {
  // Create the tensor first: running ops with the functorch keys included but
  // no functorch interpreter on the stack fails.
  const at::Tensor tensor = at::ones({2, 3});
  EXPECT_FALSE(IsFunctorchActive());
  {
    const c10::impl::IncludeDispatchKeyGuard guard(
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
    EXPECT_TRUE(IsFunctorchActive());
    EXPECT_TRUE(IsFakeOrFunctorchTensor(tensor));
  }
  {
    const c10::impl::IncludeDispatchKeyGuard guard(
        c10::DispatchKey::FuncTorchDynamicLayerBackMode);
    EXPECT_TRUE(IsFunctorchActive());
    EXPECT_TRUE(IsFakeOrFunctorchTensor(tensor));
  }
}

}  // namespace
}  // namespace torch_tpu
