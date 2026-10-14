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

"""Runs PyTorch flex_attention on TPU tensors.

Upstream flex_attention rejects TPU tensors and has no TPU kernel for its
higher-order op. `patch()` lets TPU tensors through and redirects the common
case (no score_mod, no LSE requested) to scaled_dot_product_attention with an
additive bias built from the BlockMask, which TorchTPU lowers to its flash
attention kernel. Everything else falls back to upstream's eager math path.
"""

# pylint: disable=protected-access

import typing
import weakref

import torch
from torch._higher_order_ops.flex_attention import flex_attention_backward
from torch.nn.attention import flex_attention
import torch.nn.functional as F


def patch() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  original_validate = flex_attention._validate_device

  def _validate_device(query, key, value):
    if query.device.type != "tpu":
      original_validate(query, key, value)

  flex_attention._validate_device = _validate_device

  dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
  original_bwd = flex_attention_backward.py_kernels.pop(dispatch_key)

  @flex_attention_backward.py_impl(dispatch_key)
  def _sdpa_dense_backward(query, key, value, *args, **kwargs):
    with query.device:
      return original_bwd(query, key, value, *args, **kwargs)

  original_flex_attention = flex_attention.flex_attention

  _causal_mask_cache = weakref.WeakKeyDictionary()

  @torch.compiler.assume_constant_result
  def _get_causal_window_size(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      device: torch.device,
  ) -> int | None:
    """Returns the causal window size in [1, q_len] for `block_mask`, or None."""
    cache = _causal_mask_cache.setdefault(block_mask, {})
    key = (q_len, kv_len)
    if key in cache:
      return cache[key]

    b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
    if q_len <= 0 or q_len != kv_len or b_mask != 1 or h_mask != 1:
      return cache.setdefault(key, None)

    bool_mask = flex_attention.create_mask(
        block_mask.mask_mod, 1, 1, q_len, kv_len, device=device
    )
    # In a causal mask, the last query row attends to the last
    # `candidate_window` keys (`q_len` for full causal, `< q_len` for sliding
    # window).
    candidate_window = int(bool_mask[0, 0, -1].sum().item())
    if candidate_window <= 0:
      return cache.setdefault(key, None)

    # Reference causal band mask (`0 <= row - col < candidate_window`).
    expected = (
        torch.ones((q_len, kv_len), dtype=torch.bool, device=device)
        .tril_()
        .triu_(1 - candidate_window)
    )
    window_size = (
        candidate_window if torch.equal(bool_mask[0, 0], expected) else None
    )
    return cache.setdefault(key, window_size)

  def _get_sdpa_attn_bias(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    # Cached on the BlockMask. Under torch.compile Dynamo records the cache
    # write as a side effect, so the bias is built in-graph once and later
    # graphs read it as an input.
    cache = block_mask.__dict__.setdefault("_tpu_sdpa_mask_cache", {})
    key = (q_len, kv_len, dtype, device)
    if key not in cache:
      b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
      with torch.no_grad():
        bool_mask = flex_attention.create_mask(
            block_mask.mask_mod,
            b_mask,
            h_mask,
            q_len,
            kv_len,
            device=device,
        )
        attn_bias = torch.zeros(
            bool_mask.shape, dtype=dtype, device=device
        ).masked_fill_(~bool_mask, float("-inf"))
      cache[key] = attn_bias
    return cache[key]

  class _SdpaMaskArgs(typing.NamedTuple):
    attn_mask: torch.Tensor | None = None
    is_causal: bool = False

  def _block_mask_to_sdpa_args(
      block_mask: flex_attention.BlockMask | None,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> _SdpaMaskArgs:
    """Returns SDPA mask arguments for lowering `block_mask`."""
    if block_mask is None:
      return _SdpaMaskArgs()

    window_size = _get_causal_window_size(block_mask, q_len, kv_len, device)
    if window_size is None:
      return _SdpaMaskArgs(
          attn_mask=_get_sdpa_attn_bias(
              block_mask, q_len, kv_len, dtype, device
          )
      )

    if window_size == q_len:
      return _SdpaMaskArgs(is_causal=True)

    return _SdpaMaskArgs(
        attn_mask=torch.empty((window_size,), dtype=dtype, device=device)
    )

  def _flex_attention_tpu(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      score_mod=None,
      block_mask=None,
      scale=None,
      enable_gqa: bool = False,
      return_lse: bool = False,
      kernel_options=None,
      *,
      return_aux=None,
  ):
    need_aux = return_lse or (
        return_aux is not None
        and (
            getattr(return_aux, "lse", False)
            or getattr(return_aux, "max_scores", False)
        )
    )
    if (
        query.device.type == "tpu"
        and score_mod is None
        and not need_aux
        and query.dtype in (torch.bfloat16, torch.float32)
    ):
      q_len = query.shape[-2]
      kv_len = key.shape[-2]
      # TODO(b/561865555): `_tpu_doc_len` is set by the Muse Glimmer benchmark on
      # masks that are block-diagonal over equal-length documents, so attention
      # can run per document. Replace with a general mechanism (e.g. deriving the
      # document layout from the BlockMask's block sparsity) if other models need
      # it.
      doc_len = getattr(block_mask, "_tpu_doc_len", None)
      num_docs = 1
      if doc_len and query.shape[0] == 1:
        if q_len != kv_len or q_len % doc_len:
          raise ValueError(
              f"_tpu_doc_len={doc_len} must divide q_len=kv_len"
              f" ({q_len}, {kv_len})."
          )
        num_docs = q_len // doc_len

      mask_args = _block_mask_to_sdpa_args(
          block_mask,
          q_len // num_docs,
          kv_len // num_docs,
          query.dtype,
          query.device,
      )
      if num_docs > 1:

        def fold(
            tensor: torch.Tensor,
        ) -> torch.Tensor:  # [1,H,T,D] -> [N,H,T/N,D]
          return tensor.reshape(
              tensor.shape[1], num_docs, doc_len, -1
          ).transpose(0, 1)

        query, key, value = fold(query), fold(key), fold(value)
      out = F.scaled_dot_product_attention(
          query,
          key,
          value,
          attn_mask=mask_args.attn_mask,
          dropout_p=0.0,
          is_causal=mask_args.is_causal,
          scale=scale,
          enable_gqa=enable_gqa,
      )
      if num_docs > 1:
        out = out.transpose(0, 1).reshape(1, -1, q_len, out.shape[-1])
      if return_aux is not None:
        return out, flex_attention.AuxOutput()
      return out
    return original_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        return_aux=return_aux,
    )

  flex_attention.flex_attention = _flex_attention_tpu
