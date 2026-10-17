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

"""Simplified Dynamic and Static KV cache implementations for LLM inference."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class Cache:
  """Base KV cache interface.

  Attributes:
    swa_size: Sliding window size of the SWA layers, or None if there are none.
    swa_layers: `swa_layers[layer_idx]` is True for sliding-window layers.
  """

  def __init__(
      self, swa_size: int | None = None, swa_layers: Sequence[bool] = ()
  ):
    if swa_size is not None and swa_size < 2:
      raise ValueError(f"swa_size must be at least 2, got {swa_size}")
    if swa_size is None and any(swa_layers):
      raise ValueError("swa_size is required when there are SWA layers")
    self.swa_size = swa_size
    self.swa_layers = tuple(swa_layers)

  def window(self, layer_idx: int) -> int | None:
    """Returns the sliding window size of `layer_idx`, or None if it is full."""
    if layer_idx < len(self.swa_layers) and self.swa_layers[layer_idx]:
      return self.swa_size
    return None

  def get_seq_length(self) -> int | torch.Tensor:
    raise NotImplementedError()

  def swa_key_positions(self, seq_len: int) -> torch.Tensor:
    """Returns the absolute positions the SWA slots hold after `seq_len` tokens."""
    raise NotImplementedError()

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()

  def mark_dynamic(self) -> None:
    """Marks dynamic dimensions on cache tensors (no-op for static caches)."""


def _window_key_positions(
    past_len: torch.Tensor, seq_len: int, num_slots: int
) -> torch.Tensor:
  """Returns the absolute positions the window slots hold after `seq_len` tokens.

  Slot `i` holds position `kv_offset + i`, where the offset stays 0 until the
  window is full and then advances with every token. Slots that are still
  empty map to positions after the newest token, which the causal mask hides.
  """
  kv_offset = (past_len + seq_len - num_slots).clamp(min=0)
  return torch.arange(num_slots, device=past_len.device) + kv_offset


class DynamicCache(Cache):
  """KV cache whose full-attention layers grow along the sequence via torch.cat.

  Sliding-window layers hold a fixed window of `num_swa_slots` slots instead,
  with the same slot bookkeeping as `StaticCache`: the prefill stores the last
  `num_swa_slots` prompt tokens (while attention still sees the whole prompt
  through the band mask) and every decode step writes one token. Only the
  full-attention entries change shape, so under `torch.compile` they are the
  only dynamic inputs (and the only ones `torch._dynamo.mark_dynamic` applies
  to); `max_cache_len` caps the window at the model's maximum length.
  """

  def __init__(
      self,
      swa_size: int | None = None,
      swa_layers: Sequence[bool] = (),
      max_cache_len: int | None = None,
  ):
    super().__init__(swa_size, swa_layers)
    if swa_size is None:
      self.num_swa_slots = 0
    elif max_cache_len is None:
      self.num_swa_slots = swa_size
    else:
      self.num_swa_slots = min(swa_size, max_cache_len)
    # A window that is not smaller than `max_cache_len` never evicts a token.
    self.swa_rolls = max_cache_len is None or self.num_swa_slots < max_cache_len
    self.keys: list[torch.Tensor] = []
    self.values: list[torch.Tensor] = []
    # The sequence length is read from the last layer, which must use full
    # attention: it holds every token and, as the layers update in order, it
    # still holds the previous length while the sliding-window layers update.
    # Reading it from a tensor dimension is free under torch.compile, whereas
    # a counter kept per sliding-window layer adds a graph input and output
    # per layer to every compiled step.
    if self.swa_layers and self.swa_layers[-1]:
      raise ValueError(
          "DynamicCache needs the last layer to use full attention"
      )
    self._length_layer = max(len(self.swa_layers) - 1, 0)

  def get_seq_length(self) -> int:
    if self._length_layer >= len(self.keys):
      return 0
    return self.keys[self._length_layer].shape[-2]

  def _past_len_tensor(self, device: torch.device) -> torch.Tensor:
    """Returns the sequence length as a 0-D tensor for branch-free slot math.

    Under torch.compile the length is a SymInt, which this turns into a
    tensor inside the graph instead of a Python branch on it.
    """
    return (
        torch.zeros((), dtype=torch.long, device=device) + self.get_seq_length()
    )

  def swa_key_positions(self, seq_len: int) -> torch.Tensor:
    past_len = self._past_len_tensor(self.keys[self._length_layer].device)
    return _window_key_positions(past_len, seq_len, self.num_swa_slots)

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.window(layer_idx) is None:
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

    new_len = key_states.shape[-2]
    if layer_idx >= len(self.keys):
      # Prefill: keep the newest tokens that fit in the window, but return the
      # whole prompt so that its early queries can attend tokens that do not.
      kept = min(new_len, self.num_swa_slots)
      padding = (0, 0, 0, self.num_swa_slots - kept)
      self.keys.append(F.pad(key_states[:, :, new_len - kept :, :], padding))
      self.values.append(
          F.pad(value_states[:, :, new_len - kept :, :], padding)
      )
      return key_states, value_states
    if new_len > 1:
      raise ValueError(
          f"sliding-window layer {layer_idx} already holds tokens and only"
          " accepts single-token updates after the prefill; chunked decoding"
          " is not supported"
      )
    keys, values = self.keys[layer_idx], self.values[layer_idx]
    num_slots = self.num_swa_slots
    filled = self._past_len_tensor(keys.device).clamp(max=num_slots)
    if self.swa_rolls:
      # A full window rolls by exactly one slot per token, so a static rotation
      # selected by the 0-D flag replaces the gather of `StaticCache`.
      evicted = (filled + 1 - num_slots).clamp(min=0)
      rolled_keys = torch.cat([keys[:, :, 1:], keys[:, :, :1]], dim=2)
      rolled_values = torch.cat([values[:, :, 1:], values[:, :, :1]], dim=2)
      keys = torch.where(evicted > 0, rolled_keys, keys)
      values = torch.where(evicted > 0, rolled_values, values)
      slot = filled - evicted
    else:
      slot = filled
    # Out of place, like the torch.cat growth: a compiled step returns the new
    # window as a graph output. A select over the window is cheaper on TPU
    # than a scatter into it.
    write = torch.arange(num_slots, device=keys.device) == slot
    write = write.view(1, 1, num_slots, 1)
    self.keys[layer_idx] = torch.where(write, key_states, keys)
    self.values[layer_idx] = torch.where(write, value_states, values)
    return self.keys[layer_idx], self.values[layer_idx]

  def mark_dynamic(self) -> None:
    """Marks key and value tensors of full-attention layers as dynamic along dimension 2.

    Sliding-window attention (SWA) layers keep a fixed number of slots.
    The shapes of SWA tensors do not change during decoding.
    Do not mark SWA tensors as dynamic.

    Full-attention layers append new tokens on each decode step.
    The sequence length of full-attention tensors increases continuously.
    Mark only full-attention key and value tensors as dynamic along dimension 2.
    This prevents graph recompilation in Dynamo during decoding.
    """
    if len(self.keys) != len(self.values):
      raise ValueError(
          f"Mismatched keys ({len(self.keys)}) and values ({len(self.values)})"
          " length"
      )
    for layer_idx in range(len(self.keys)):
      # SWA layers have a non-None window. Their tensor shapes are static.
      # Skip SWA layers.
      if self.window(layer_idx) is not None:
        continue

      # Full-attention layers have a window of None.
      # These tensors grow during decoding and require dynamic shape tracking.
      key, value = self.keys[layer_idx], self.values[layer_idx]
      if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected 4D key and value tensors, got key ndim={key.ndim}, value"
            f" ndim={value.ndim}"
        )
      # Mark dimension 2 (sequence length) as dynamic.
      torch._dynamo.mark_dynamic(key, 2)
      torch._dynamo.mark_dynamic(value, 2)


class StaticCache(Cache):
  """Fixed-size preallocated KV cache mutated in-place for static compilation.

  Sliding-window layers hold only `min(swa_size, max_cache_len)` slots with
  the newest tokens in position order (see `swa_key_positions()`). They accept
  one multi-token update of at most that many tokens (the prefill) followed by
  single-token updates; longer prefills and chunked decoding raise a
  `ValueError` because the evicted tokens would be needed by the earlier
  queries of the chunk (use `DynamicCache` for those). When the window is not
  smaller than `max_cache_len`, nothing is ever evicted and such layers behave
  exactly like full-attention layers.
  """

  def __init__(
      self,
      num_layers: int,
      batch_size: int,
      num_kv_heads: int,
      max_cache_len: int,
      head_dim: int,
      dtype: torch.dtype,
      device: torch.device | str,
      swa_size: int | None = None,
      swa_layers: Sequence[bool] = (),
  ):
    super().__init__(swa_size, swa_layers)
    self.num_swa_slots = (
        max_cache_len if swa_size is None else min(swa_size, max_cache_len)
    )
    # Layers whose buffer is smaller than `max_cache_len` evict tokens.
    self.rolls = [
        self.window(layer_idx) is not None
        and self.num_swa_slots < max_cache_len
        for layer_idx in range(num_layers)
    ]
    self.keys: list[torch.Tensor] = []
    self.values: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
      num_slots = self.num_swa_slots if self.rolls[layer_idx] else max_cache_len
      shape = (batch_size, num_kv_heads, num_slots, head_dim)
      self.keys.append(torch.zeros(shape, dtype=dtype, device=device))
      self.values.append(torch.zeros(shape, dtype=dtype, device=device))
    # Stored as a 0-D tensor on device so in-place updates do not trigger
    # torch.compile(dynamic=False) recompilations across decode steps.
    self.cumulative_length = [
        torch.zeros((), dtype=torch.long, device=device)
        for _ in range(num_layers)
    ]
    # Python flags are fine here: they flip together with the prefill-to-decode
    # shape change, so they do not cause extra recompilations.
    self.has_tokens = [False] * num_layers

  def get_seq_length(self) -> torch.Tensor:
    return self.cumulative_length[0].clone()

  def swa_key_positions(self, seq_len: int) -> torch.Tensor:
    return _window_key_positions(
        self.cumulative_length[0], seq_len, self.num_swa_slots
    )

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    new_len = key_states.shape[-2]
    keys, values = self.keys[layer_idx], self.values[layer_idx]
    past_len = self.cumulative_length[layer_idx]
    slots = torch.arange(new_len, device=keys.device)
    if not self.rolls[layer_idx]:
      slots = slots + past_len
    else:
      num_slots = keys.shape[-2]
      if new_len > num_slots:
        raise ValueError(
            f"sliding-window layer {layer_idx} holds {num_slots} slots but"
            f" received {new_len} tokens at once; use DynamicCache for longer"
            " prefills"
        )
      if new_len > 1 and self.has_tokens[layer_idx]:
        raise ValueError(
            f"sliding-window layer {layer_idx} already holds tokens and only"
            " accepts single-token updates after the prefill; use DynamicCache"
            " for chunked decoding"
        )
      # Roll the window so that the oldest token that no longer fits wraps
      # around to the end, where the new token overwrites it. Tensor math on
      # the 0-D length keeps the compiled graph free of data-dependent
      # branches, at the cost of rewriting the whole window every step.
      filled = past_len.clamp(max=num_slots)
      evicted = (filled + new_len - num_slots).clamp(min=0)
      # Wrap without `%`: TPU has no native 64-bit integer division, and
      # 0 <= evicted < num_slots holds here.
      order = torch.arange(num_slots, device=keys.device) + evicted
      order = torch.where(order >= num_slots, order - num_slots, order)
      keys.copy_(keys.index_select(2, order))
      values.copy_(values.index_select(2, order))
      slots = slots + (filled - evicted)
    keys.index_copy_(2, slots, key_states)
    values.index_copy_(2, slots, value_states)
    past_len.add_(new_len)
    self.has_tokens[layer_idx] = True
    return keys, values
