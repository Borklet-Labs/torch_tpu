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

# pylint: skip-file
"""Bounded dynamism for LLM inference with TorchTPU."""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Bounded dynamism

    This guide explains how **bounded dynamism** works in TorchTPU, how it leverages OpenXLA's bounded dynamic shape representation under the hood, and how to apply it to an autoregressive LLM decode loop using the **Muse Glimmer** architecture with a dynamic KV cache.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. Why autoregressive decoding needs bounded dynamism

    The XLA compiler is traditionally designed around **statically known tensor bounds** so that it can lay out High Bandwidth Memory (HBM) buffers ahead of time, fuse operator pipelines, and tile matrix multiplications for the TPU's systolic arrays (MXUs).

    In an autoregressive LLM decode loop using a standard dynamically growing KV cache, each decode step appends the new token's key and value states to the cached history via `torch.cat`:

    * **Prefill:** Input prompt has shape `[B, S]` (batch size `B`, prompt sequence length `S`), producing a KV cache of shape `[B, H_kv, S, D]` (`H_kv` key/value heads, head dimension `D`).
    * **Decode step 1:** Input token has shape `[B, 1]`, concatenating with the cache to produce shape `[B, H_kv, S + 1, D]`.
    * **Decode step `t`:** The KV cache grows to length `S + t`.

    When you compile with static shapes (`torch.compile(model, backend="tpu")`), TorchDynamo guards on the exact sequence length of the KV cache. Every single decode step changes the tensor shape, invalidates the guard, and triggers a **full XLA recompilation on every generated token**—turning millisecond decode steps into multi-second compilations unless you rewrite the model to use preallocated static buffers and do in-place mutations.

    **Bounded dynamism** solves this without requiring you to rewrite your model's dynamic KV cache or attention masking logic.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. How bounded dynamism works in TorchTPU and XLA

    When you compile a model with `torch.compile(model, backend="tpu", dynamic=None)` (as input shapes vary across steps) or explicitly mark dynamic dimensions with `torch._dynamo.mark_dynamic`, TorchDynamo traces the model and produces an FX graph where changing dimensions are represented as symbolic integer variables (`torch.SymInt`, such as `s0`) instead of fixed constants. TorchTPU handles these symbolic dimensions as follows:

    ### Compilation

    * TorchTPU intercepts the traced FX graph, determines an overall `[lower_bound, upper_bound]` range for each symbolic dimension (starting from the initial runtime size `S` up to the user-provided `max` from `mark_dynamic` or a default bound derived from the runtime shape hint), and divides that range into contiguous buckets aligned to multiples of `bucket_size` (default `128`, optimally sized to fit within a TPU's MXU), up to `num_dynamic_buckets` (default `1`, maximum `8`):
        * The first bucket covers from `S` up to the next multiple of `128` (`U_1`), the second bucket covers from `U_1 + 1` up to the next multiple of `128` (`U_2`), and so on (for example, `[S, 128]`, `[129, 256]`, ...).
        * TorchTPU registers a single range guard (`lower_bound <= s0 <= upper_bound`) with TorchDynamo across the full span of all buckets, so TorchDynamo only triggers recompilation if the dimension grows beyond the overall `upper_bound`.
    * For each bucket, TorchTPU transforms the FX graph by replacing the symbolic dimension with the bucket's static upper bound `U`, passing the actual runtime dimension size as a scalar `int32` input, and inserting two key operations at the graph boundary during StableHLO lowering:
        * **`stablehlo.custom_call @PadRealToStatic`:** Pads the physical input tensor up to the static upper bound `U`.
        * **`stablehlo.set_dimension_size`:** Attaches the runtime `int32` logical size to the padded dimension, producing a StableHLO bounded tensor type:

    ```mlir
    func.func @main(%input: tensor<1x2x?x256xbf16, #stablehlo.bounds<?, ?, 128, ?>>, %runtime_size: tensor<i32>) {
      %padded_input = stablehlo.custom_call @PadRealToStatic(%input, ...) -> tensor<1x2x128x256xbf16>
      %bounded_dynamic_tensor = stablehlo.set_dimension_size %padded_input, %runtime_size, dim=2 -> tensor<1x2x?x256xbf16, #stablehlo.bounds<?, ?, 128, ?>>
      ...
    }
    ```

    * As the graph is lowered to StableHLO, both the static upper bound (`#stablehlo.bounds`) and the runtime logical size are propagated through downstream operations so every intermediate and output tensor tracks its active logical size.
    * XLA automatically resets values in the padded region beyond the logical size to algebraic identity values, ensuring that padded elements do not affect the output result.
    * The resulting module is compiled by XLA once per bucket upper bound `U`.

    ### Execution

    * At runtime, based on the actual size of the dynamic dimension, TorchTPU selects the corresponding compiled bucket executable.
    * When executing the compiled module, XLA pads the dynamic input tensor to the bucket's static upper bound `U`.
    * The model executes on the padded buffers, while the compiled XLA instructions ensure that padded values do not affect the output result.
    * For dynamic output tensors (such as the updated KV cache), padding remains at the end of the buffer, allowing TorchTPU to return the buffer to PyTorch with a lightweight shape metadata update rather than physically slicing the tensor.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Practical example: Adding a decode loop to Muse Glimmer

    Below, we take the base **Muse Glimmer** model implementation and extend it for end-to-end autoregressive generation by adding:

    1. **DynamicKVCache**: Appends new key and value states along the sequence dimension (`dim=-2`) via `torch.cat`.
    2. **Autoregressive decode loop**: Runs prefill on the prompt followed by greedy token-by-token decoding, recording per-step latencies.
    """)
  return


@app.cell
def _():
  import dataclasses
  import time
  import warnings
  import torch
  from torch._functorch._aot_autograd import autograd_cache
  import torch.nn as nn
  import torch.nn.functional as F

  torch._logging.set_logs()  # pylint: disable=protected-access
  warnings.filterwarnings(
      "ignore",
      category=UserWarning,
      message=r".*FLASH_ATTENTION SDPBackend.*",
  )
  device = torch.device("tpu")
  print(f"Using device: {device}")
  return F, autograd_cache, dataclasses, device, nn, time, torch


@app.cell
def _(F, dataclasses, nn, torch):
  def rotate_half(x: torch.Tensor) -> torch.Tensor:
    dh = x.shape[-1]
    x1 = x[..., : dh // 2]
    x2 = x[..., dh // 2 :]
    return torch.cat([-x2, x1], dim=-1)

  def apply_rope(
      x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
  ) -> torch.Tensor:
    dtype = x.dtype
    x_f32 = x.to(dtype=torch.float32)
    return ((x_f32 * cos) + (rotate_half(x_f32) * sin)).to(dtype=dtype)

  def precompute_rope_sin_cos(
      seq_len: int, dim_head: int, rope_theta: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
    )
    inv_freq = inv_freq.repeat(2).view(1, 1, 1, dim_head)
    pos_idx = torch.arange(0, seq_len, dtype=torch.float32).view(
        1, 1, seq_len, 1
    )
    return torch.sin(pos_idx * inv_freq), torch.cos(pos_idx * inv_freq)

  class DynamicKVCache:
    """KV cache that grows dynamically along the sequence dimension via torch.cat."""

    def __init__(self):
      self.keys: list[torch.Tensor] = []
      self.values: list[torch.Tensor] = []

    def get_seq_length(self) -> int:
      if not self.keys:
        return 0
      return self.keys[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      if layer_idx >= len(self.keys):
        self.keys.append(key_states)
        self.values.append(value_states)
      else:
        self.keys[layer_idx] = torch.cat(
            [self.keys[layer_idx], key_states], dim=-2
        )
        self.values[layer_idx] = torch.cat(
            [self.values[layer_idx], value_states], dim=-2
        )
      return self.keys[layer_idx], self.values[layer_idx]

  @dataclasses.dataclass
  class Config:
    # General model config
    vocab_size: int = 202048
    num_layers: int = 4  # Reduced from 52 for faster notebook execution
    dmodel: int = 6656
    dff: int = 19968
    dhead: int = 128
    num_q_heads: int = 32
    num_kv_heads: int = 2

    swa_size: int = 2048
    rope_freq_base: float = 500000.0
    rmsnorm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    qk_scale_factor: float = 3.87
    final_logit_softcapping: float = 20.0
    output_multiplier: float = 0.19611613513818404

    prefill_seq_len: int = 128
    num_decode_steps: int = 10

  class Attention(nn.Module):
    """Grouped-query Attention with DynamicKVCache support."""

    def __init__(self, config: Config, is_swa: bool, layer_idx: int = 0):
      super().__init__()
      self.config = config
      self.is_swa = is_swa
      self.layer_idx = layer_idx

      dm, dh = config.dmodel, config.dhead
      num_q_heads = config.num_q_heads
      num_kv_heads = config.num_kv_heads

      self.q_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
      self.k_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
      self.v_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
      self.gate_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
      self.o_proj = nn.Linear(num_q_heads * dh, dm, bias=False)

      self.qk_norm = nn.RMSNorm(
          normalized_shape=(dh,),
          eps=config.rmsnorm_eps,
          dtype=torch.float32,
          elementwise_affine=False,
      )

    def forward(
        self,
        h: torch.Tensor,
        rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DynamicKVCache,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
      local_bsz, seq_len, _ = h.shape
      dh = self.config.dhead
      num_q_heads = self.config.num_q_heads
      num_kv_heads = self.config.num_kv_heads

      q = self.q_proj(h).view(local_bsz, seq_len, num_q_heads, dh)
      k = self.k_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)
      v = self.v_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)

      qk_scale = self.config.qk_scale_factor
      q = self.qk_norm(q.transpose(1, 2).contiguous()) * qk_scale
      k = self.qk_norm(k.transpose(1, 2).contiguous())
      v = v.transpose(1, 2).contiguous()

      if self.is_swa:
        # In Muse Glimmer 30B, RoPE is only applied in SWA layers; full attention
        # layers use NoPE.
        rs, rc = rope_sin_cos
        q = apply_rope(q, rs, rc)
        k = apply_rope(k, rs, rc)

      k, v = past_key_values.update(k, v, self.layer_idx)
      is_causal = attn_mask is None and seq_len > 1

      attn_out = F.scaled_dot_product_attention(
          q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True
      )
      attn_out = attn_out.transpose(1, 2).reshape(
          local_bsz, seq_len, num_q_heads * dh
      )

      gate = F.sigmoid(self.gate_proj(h))
      return self.o_proj(attn_out * gate)

  class FeedForwardSwiGLU(nn.Module):
    """Simple FFN with SwiGLU activation."""

    def __init__(self, config: Config):
      super().__init__()
      dm, df = config.dmodel, config.dff
      self.up_proj = nn.Linear(dm, df, bias=False)
      self.gate_proj = nn.Linear(dm, df, bias=False)
      self.down_proj = nn.Linear(df, dm, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
      return self.down_proj(self.up_proj(h) * F.silu(self.gate_proj(h)))

  class DecoderBlock(nn.Module):

    def __init__(self, config: Config, layer_idx: int):
      super().__init__()
      self.self_attn = Attention(
          config, is_swa=((layer_idx % 4) != 3), layer_idx=layer_idx
      )
      self.mlp = FeedForwardSwiGLU(config)

      dm = config.dmodel
      f32 = torch.float32
      eps_pre, eps_post = config.rmsnorm_eps, config.post_norm_eps
      self.input_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
      self.post_attention_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)
      self.pre_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
      self.post_feedforward_layernorm = nn.RMSNorm(
          (dm,), eps=eps_post, dtype=f32
      )

    def forward(
        self,
        h: torch.Tensor,
        rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DynamicKVCache,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
      h = h + self.post_attention_layernorm(
          self.self_attn(
              self.input_layernorm(h),
              rope_sin_cos,
              past_key_values=past_key_values,
              attn_mask=attn_mask,
          )
      )
      h = h + self.post_feedforward_layernorm(
          self.mlp(self.pre_feedforward_layernorm(h))
      )
      return h

  class MuseGlimmer30B(nn.Module):

    def __init__(self, config: Config):
      super().__init__()
      self.config = config
      self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
      self.embed_norm = nn.RMSNorm(
          (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
      )
      self.layers = nn.ModuleList(
          DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
      )
      self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
      self.lm_head = nn.Linear(config.dmodel, config.vocab_size, bias=False)

      max_len = config.prefill_seq_len + config.num_decode_steps
      rope_sin, rope_cos = precompute_rope_sin_cos(
          seq_len=max_len,
          dim_head=config.dhead,
          rope_theta=config.rope_freq_base,
      )
      self.rope_sin = nn.Buffer(rope_sin, persistent=False)
      self.rope_cos = nn.Buffer(rope_cos, persistent=False)

      causal_mask = torch.ones(1, 1, max_len, max_len).tril()
      swa_mask = causal_mask.triu(-config.swa_size + 1)
      self.swa_mask = nn.Buffer(
          torch.where(swa_mask == 1, 0.0, float("-inf")), persistent=False
      )

    def forward(
        self,
        tok: torch.Tensor,
        past_key_values: DynamicKVCache | None = None,
    ) -> tuple[torch.Tensor, DynamicKVCache]:
      if past_key_values is None:
        past_key_values = DynamicKVCache()
      past_len = past_key_values.get_seq_length()
      seq_len = tok.shape[1]
      # SDPA with is_causal=True aligns the causal mask to the top-left corner,
      # which is only valid when the cache is empty (past_len == 0).
      assert past_len == 0 or seq_len == 1
      kv_len = past_len + seq_len
      rope_sin_cos = (
          self.rope_sin[:, :, past_len:kv_len, :],
          self.rope_cos[:, :, past_len:kv_len, :],
      )
      swa_mask = self.swa_mask[:, :, past_len:kv_len, :kv_len]

      h = self.embed_norm(self.embed_tokens(tok))
      for layer in self.layers:
        attn_mask = swa_mask if layer.self_attn.is_swa else None
        h = layer(
            h,
            rope_sin_cos,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
        )
      h = self.norm(h)

      alpha = self.config.output_multiplier
      c = self.config.final_logit_softcapping
      logits = F.tanh(self.lm_head(h[:, -1:, :]) * (alpha / c)) * c
      return logits, past_key_values

  @torch.no_grad()
  def init_nn_module(m: nn.Module) -> None:
    for p in m.parameters():
      p.data.copy_(torch.randn_like(p.data) * 0.05)

  return Config, DynamicKVCache, MuseGlimmer30B, init_nn_module


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 4. Prefill and decode loop with performance comparison

    Next, we define `model_generate` to run the prefill pass and greedy decode loop, and compare the performance of the two runs below on the TPU using the `MuseGlimmer30B` model and `DynamicKVCache`:

    1. **Static Compilation (`dynamic=False`)**: Every decode step changes the KV cache sequence length (`kv_len`), forcing XLA to recompile the entire model on every generated token.
    2. **Bounded Dynamism (`dynamic=None`)**: Only the initial prefill and warmup decode steps compile. From step 4 onward, **zero recompilations** occur and decode latency drops by orders of magnitude.
    """)
  return


@app.cell
def _(Config, MuseGlimmer30B, device, init_nn_module, time, torch):

  def model_generate(
      compiled_model,
      initial_inputs: torch.Tensor,
      num_decode_steps: int,
      prefix: str = "",
  ) -> list[int]:
    """Runs prefill and greedy decode loop, returning generated token IDs."""
    with torch.inference_mode():
      prefill_start = time.perf_counter()
      logits, past_key_values = compiled_model(initial_inputs)
      torch.accelerator.synchronize()
      prefill_time_ms = (time.perf_counter() - prefill_start) * 1000
      print(f"{prefix}Prefill done in {prefill_time_ms:.2f} ms", flush=True)

      generated_tokens = [initial_inputs]

      for step in range(num_decode_steps):
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(next_token)

        step_start = time.perf_counter()
        logits, past_key_values = compiled_model(
            next_token,
            past_key_values=past_key_values,
        )
        torch.accelerator.synchronize()
        step_time_ms = (time.perf_counter() - step_start) * 1000

        print(
            f"{prefix}Decode step {step + 1}/{num_decode_steps}:"
            f" {step_time_ms:.2f} ms",
            flush=True,
        )

      output_tokens = torch.cat([t.to("cpu") for t in generated_tokens], dim=1)

    return output_tokens[0, initial_inputs.shape[1] :].tolist()

  config = Config()

  torch.manual_seed(42)
  torch.set_default_dtype(torch.bfloat16)

  inputs = torch.randint(
      0,
      config.vocab_size,
      (1, config.prefill_seq_len),
      dtype=torch.long,
      device=device,
  )
  with torch.device(device):
    base_model = MuseGlimmer30B(config)
  init_nn_module(base_model)
  torch.accelerator.synchronize()
  return base_model, config, inputs, model_generate


@app.cell
def _(autograd_cache, base_model, config, inputs, model_generate, torch):
  print("--- 1. Running with dynamic=False (recompiles every decode step) ---")
  torch._dynamo.reset()
  autograd_cache.AOTAutogradCache.clear()
  # Allow enough recompilations for the static (dynamic=False) baseline to finish all decode steps
  torch._dynamo.config.recompile_limit = 64
  model_no_bd = torch.compile(base_model, backend="tpu", dynamic=False)
  tokens_no_bd = model_generate(
      model_no_bd,
      inputs,
      config.num_decode_steps,
      prefix="[dynamic=False] ",
  )
  print(f"[dynamic=False] Generated token IDs: {tokens_no_bd}")
  del model_no_bd
  return


@app.cell
def _(autograd_cache, base_model, config, inputs, model_generate, torch):
  print("--- 2. Running with dynamic=None (BD) ---")
  torch._dynamo.reset()
  autograd_cache.AOTAutogradCache.clear()
  model_bd = torch.compile(base_model, backend="tpu", dynamic=None)
  tokens_bd = model_generate(
      model_bd,
      inputs,
      config.num_decode_steps,
      prefix="[dynamic=None] ",
  )
  print(f"[dynamic=None] Generated token IDs: {tokens_bd}")
  del model_bd
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### Benchmark results

    Running the benchmark above with the full 52-layer `MuseGlimmer30B` model (`num_layers=52`) on a single TPU v7x chip produces the following latency comparison:

    | Metric | `dynamic=False` | `dynamic=None` (BD) |
    | --- | --- | --- |
    | **Prefill time (ms)** | 21620.00 | 21207.66 |
    | **Decode step 1 (ms)** | 21851.85 | 21505.71 |
    | **Decode step 2 (ms)** | 21598.38 | 29786.60 |
    | **Decode step 3 (ms)** | 21786.57 | 5832.60 |
    | **Decode step 4 (ms)** | 21837.68 | 32.41 |
    | **Decode step 5 (ms)** | 21831.57 | 31.63 |
    | **Decode step 6 (ms)** | 21914.96 | 30.68 |
    | **Decode step 7 (ms)** | 21979.15 | 30.73 |
    | **Decode step 8 (ms)** | 22044.13 | 30.77 |
    | **Decode step 9 (ms)** | 22206.13 | 30.53 |
    | **Decode step 10 (ms)** | 21985.66 | 30.69 |
    | **Avg decode time / token (ms)** | **21903.61** | **4334.24** |
    | **Avg decode time / token (no warmup) (ms)** | **21971.33** | **31.06** |

    * **With `dynamic=False`:** Every decode step changes `kv_len` and triggers a full XLA recompilation (~21.9 seconds per token).
    * **With `dynamic=None` (Bounded Dynamism):** After the initial warmup compilations on steps 1–3, **Decode steps 4–10 execute with zero recompilations at ~31.06 ms per token** (~700x faster per step than recompiling).
    """)
  return


if __name__ == "__main__":
  app.run()
