# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TorchTPU-specific configs for PyTorch device-generic tests.

For official documentation and guidance, see
https://github.com/pytorch/pytorch/blob/main/docs/source/accelerator/testing.md.
"""

from collections.abc import Mapping, Sequence
from typing import Final

# Generic test classes or methods excluded from instantiation for TPU.
#
# Keyed by test module, tests are identified by their generic class and method
# names, e.g. `TestOps` instead of `TestOpsTPU`. In this way, one entry covers
# all instantiated variants of a test. Use "*" to exclude a whole class.
TEST_EXCLUSIONS: Final[Mapping[str, Mapping[str, str | Sequence[str]]]] = {
    # TODO(b/573758862): track per-test exclusion reasons and triage.
    # go/keep-sorted start block=yes
    "distributed/fsdp/test_utils": {
        "TestUtils": ["test_apply_to_tensors"],
    },
    "dynamo/test_debug_utils": {
        "TestInductorConfigOverrideIntegration": [
            "test_config_override_backward_propagation",
        ],
    },
    "dynamo/test_fx_annotate": {
        "AnnotateTestsDevice": [
            "test_flex_attention_backward_tag_does_not_leak"
        ],
    },
    "dynamo/test_repros": {
        "ReproTestsDevice": [
            "test_custom_op_mutation_with_noop_add",
            "test_data_attr_mutation_with_noop_add",
            "test_layer_norm_mixed_dtype_aot_eager_decomp_partition_errors",
            "test_mem_leak_guards",
            "test_memleak_when_graph_input_has_tensor_attr",
            "test_tensor_set_data_cross_device",
            "test_tensor_set_data_cross_device_placeholder_metadata",
            "test_torch_device_is_initialized",
        ],
    },
    "inductor/test_slice_scatter_chunking": {
        "TestSliceScatterChunking": [
            "test_functionalized_copy_chain_compile",
            "test_nested_region_output_fusibility",
        ],
    },
    "nn/test_convolution": {
        "TestConvolutionNNDevice": [
            "test_Conv2d_depthwise_naive_groups",
            "test_Conv2d_inconsistent_types_on_accelerator_without_cudnn",
            "test_Conv2d_size_1_kernel",
            "test_Conv3d_depthwise_naive_groups",
            "test_ConvTranspose2d_size_1_kernel",
            "test_conv1d_vs_scipy",
            "test_conv2d_vs_scipy",
            "test_conv3d_same_padding_backward",
            "test_conv3d_vs_scipy",
            "test_conv_backend",
            "test_conv_empty_channel",
            "test_conv_mismatch_memory_format",
            "test_convert_conv2d_weight_memory_format",
            "test_convert_conv3d_weight_memory_format",
            "test_slow_conv_dilated2d_unbatched",
            "test_slow_conv_dilated3d_unbatched",
            "test_slow_conv_transpose3d_kernel_size_mismatch",
            "test_thnn_conv_strided_padded_dilated",
        ],
    },
    "nn/test_dropout": {
        "TestDropoutNNDeviceType": [
            "test_Dropout",
        ],
    },
    "nn/test_embedding": {
        "TestEmbeddingNNDeviceType": [
            "test_EmbeddingBag_empty_per_sample_weights_and_offsets",
            "test_EmbeddingBag_per_sample_weights_and_new_offsets",
            "test_EmbeddingBag_per_sample_weights_and_no_offsets",
            "test_EmbeddingBag_per_sample_weights_and_offsets",
            "test_embedding_backward_atomic_accumulate_kernel",
            "test_embedding_bag_device",
            "test_embedding_bag_out_of_bounds_idx",
            "test_embedding_dense_grad",
            "test_embedding_float_indices_error",
            "test_embedding_scalar_weight_error",
        ],
    },
    "nn/test_init": {
        "TestNNInitDevice": [
            "test_trunc_normal_narrow_interval",
            "test_trunc_normal_unique_values",
        ],
    },
    "nn/test_multihead_attention": {
        "TestMultiheadAttentionNNDeviceType": [
            "test_fast_path_check_with_mask_does_not_break_in_compile",
            "test_multihead_attn_fast_path_query_and_bias_have_different_dtypes",
            "test_multihead_self_attn_two_masks_fast_path",
        ],
    },
    "nn/test_pooling": {
        "TestPoolingNNDevice": [
            "test_AdaptiveMaxPool1d_indices",
            "test_AdaptiveMaxPool2d_indices",
            "test_AdaptiveMaxPool3d_indices",
            "test_AdaptiveMaxPool_zero_batch_dim",
            "test_FractionalMaxPool2d_zero_batch",
            "test_FractionalMaxPool2d_zero_out_size",
            "test_FractionalMaxPool2d_zero_samples",
            "test_FractionalMaxPool3d_zero_batch",
            "test_FractionalMaxPool3d_zero_out_size",
            "test_FractionalMaxPool3d_zero_samples",
            "test_MaxPool_zero_batch_dim",
            "test_MaxUnpool_zero_batch_dim",
            "test_adaptive_avg_pooling_backward_fails",
            "test_adaptive_avg_pooling_nhwc_overflow",
            "test_adaptive_avg_pooling_overflow",
            "test_adaptive_pooling_avg_nhwc",
            "test_adaptive_pooling_avg_nhwc_non_contiguous",
            "test_adaptive_pooling_empty_output_size",
            "test_adaptive_pooling_max_nhwc",
            "test_adaptive_pooling_no_suppot_input",
            "test_fractional_max_pool2d",
            "test_fractional_max_pool3d",
            "test_fractional_max_pool_nan_inf",
            "test_max_pool3d_ndhwc",
            "test_max_pool_nan_inf",
            "test_maxpool_indices_no_batch_dim",
            "test_pool_invalid_size",
            "test_pooling_bfloat16",
            "test_pooling_zero_stride",
        ],
    },
    "test_custom_ops": {
        "TestCustomOpTesting": [
            "test_autograd_registration_check_autograd_kernel",
            "test_autograd_registration_check_compositeimplicitautograd",
            "test_autograd_registration_check_incorrect",
            "test_autograd_registration_check_incorrect_composite",
            "test_incorrect_schema_mutation",
        ],
    },
    "test_fake_tensor": {
        "PropagateRealTensorsTestFakeTensorDevice": "*",
        "TestFakeTensorDevice": [
            "test_aten_set_data_multi_device",
            "test_conv_mismatched_device_error_matches_eager",
            "test_index_put_error",
        ],
    },
    "test_out_dtype_op": {
        "TestOutDtypeOpDevice": [
            "test_out_dtype_inductor_decomp",
            "test_out_dtype_inductor_decomp_trace",
        ],
    },
    # go/keep-sorted end
}
