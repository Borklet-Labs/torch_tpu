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

"""Simple RoPE implementation using sin/cos + rotate_half."""

import torch


def apply_rope(x, sin, cos):
  """Applies RoPE to tensor `x` using pre-computed sin/cos.

  Args:
    x: The input tensor to apply RoPE to, shaped [batch, heads, seq_len, dh].
    sin: The pre-computed sine values for RoPE.
    cos: The pre-computed cosine values for RoPE.

  Returns:
    A tensor of the same shape as `x` with RoPE applied.
  """
  dtype = x.dtype
  x = x.to(dtype=torch.float32)
  return ((x * cos) + (rotate_half(x) * sin)).to(dtype=dtype)


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  dh = x.shape[-1]
  x1 = x[..., : dh // 2]
  x2 = x[..., dh // 2 :]
  return torch.cat([-x2, x1], dim=-1)


def precompute_rope_sin_cos(seq_len, dim_head, rope_theta):
  """Pre-comptue sin/cos values for all positions of given seq_len."""
  inv_freq = 1.0 / (
      rope_theta
      ** (torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
  )
  inv_freq = inv_freq.repeat(2).view(1, 1, 1, dim_head)

  pos_idx = torch.arange(0, seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)

  sin = torch.sin(pos_idx * inv_freq)  # [1, 1, seq_len, dim_head]
  cos = torch.cos(pos_idx * inv_freq)  # [1, 1, seq_len, dim_head]

  return sin, cos
