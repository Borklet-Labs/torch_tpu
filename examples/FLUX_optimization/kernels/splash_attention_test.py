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

"""Tests for Splash Attention on TPU matching TorchTitan."""

from collections.abc import Callable
import functools

from absl import logging
from absl.testing import absltest
import torch
from torch.nn import attention
import torch.nn.functional as F
from splash_attention import splash_sdpa


class SplashAttentionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.device = torch.device("tpu")

  def check_forward_parity(
      self,
      splash_fn: Callable[
          [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
      ],
      b: int,
      n_heads: int,
      n_kv_heads: int,
      seq_len: int,
      head_dim: int,
      is_causal: bool = False,
  ):
    # Generates inputs in standard (B, H, S, D) format matching PyTorch SDPA & torchtitan
    torch.manual_seed(42)
    q = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=self.device
    )
    k = torch.randn(
        b,
        n_kv_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
        device=self.device,
    )
    v = torch.randn(
        b,
        n_kv_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
        device=self.device,
    )

    # Computes expected result using Math SDPA on TPU
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q,
          k,
          v,
          is_causal=is_causal,
      )

    actual_out = splash_fn(q, k, v)
    # TPU MXU matmuls default to bfloat16 operand rounding (~7.8e-3 eps) even
    # for float32 inputs, yielding ~2.4e-3 to 6.3e-3 max abs diff vs math SDPA.
    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=1e-2, atol=1e-2
    )

  def test_flux_bidirectional_mha_parity(self):
    """FLUX uses bidirectional (non-causal) multi-head attention."""
    splash_fn = functools.partial(
        splash_sdpa, is_causal=False, enable_gqa=False
    )
    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
        is_causal=False,
    )
    logging.info("FLUX bidirectional MHA forward parity passed.")

  def test_causal_mha_parity(self):
    """Causal multi-head attention parity matching torchtitan."""
    splash_fn = functools.partial(splash_sdpa, is_causal=True, enable_gqa=False)
    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
        is_causal=True,
    )
    logging.info("Causal MHA forward parity passed.")

  def test_mha_backward_parity(self):
    """Backward gradients from splash attention match reference SDPA."""
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    torch.manual_seed(42)
    q = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=self.device
    )
    k = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=self.device
    )
    v = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=self.device
    )

    # Reference backward via math SDPA
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_ref, v_ref, is_causal=False
      )
    out_ref.sum().backward()

    # Splash backward
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = splash_sdpa(
        q_test, k_test, v_test, is_causal=False, enable_gqa=False
    )
    out_test.sum().backward()

    self.assertIsNotNone(q_test.grad, "grad_q is None")
    self.assertIsNotNone(k_test.grad, "grad_k is None")
    self.assertIsNotNone(v_test.grad, "grad_v is None")
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")
    self.assertGreater(k_test.grad.abs().sum().item(), 0, "grad_k is all zeros")
    self.assertGreater(v_test.grad.abs().sum().item(), 0, "grad_v is all zeros")

    torch.testing.assert_close(
        q_test.grad.cpu(),
        q_ref.grad.cpu(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        k_test.grad.cpu(),
        k_ref.grad.cpu(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        v_test.grad.cpu(),
        v_ref.grad.cpu(),
        rtol=1e-2,
        atol=1e-2,
    )
    logging.info("MHA backward gradient parity passed.")

  def test_compiled_forward_backward_matches_eager(self):
    """splash_sdpa works under torch.compile and matches eager fwd + bwd."""
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    torch.manual_seed(42)
    q, k, v = (
        torch.randn(
            b,
            n_heads,
            seq_len,
            head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(3)
    )

    def attention_loss(q, k, v):
      return splash_sdpa(q, k, v, is_causal=False, enable_gqa=False).sum()

    compiled_attention_loss = torch.compile(attention_loss, backend="tpu")

    results = []
    for fn in (attention_loss, compiled_attention_loss):
      q_i, k_i, v_i = (t.clone().requires_grad_(True) for t in (q, k, v))
      loss = fn(q_i, k_i, v_i)
      loss.backward()
      results.append((loss.detach(), q_i.grad, k_i.grad, v_i.grad))

    for eager, compiled in zip(*results):
      torch.testing.assert_close(
          compiled.cpu(), eager.cpu(), rtol=1e-2, atol=1e-2
      )
    logging.info("Compiled splash attention fwd + bwd matches eager.")


if __name__ == "__main__":
  absltest.main()
