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

"""Tests compiling graphs that mix CPU and TPU ops with the TPU backend."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo import exc as dynamo_exc
from torch._dynamo import testing as dynamo_testing
from torch._dynamo import utils as dynamo_utils
from torch._inductor.runtime import cache_dir_utils
from torch.utils import _pytree as pytree
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _tpu(*values: float, requires_grad: bool = False) -> torch.Tensor:
  return torch.tensor(
      values, dtype=torch.float32, device="tpu", requires_grad=requires_grad
  )


def _constant_used_on_cpu_and_tpu(x: torch.Tensor) -> torch.Tensor:
  c = torch.tensor([1.0, 2.0, 3.0])
  return x * c.to(x.device) + (torch.ones(3) * c).to(x.device)


def _interleaved_cpu_masks(x: torch.Tensor) -> torch.Tensor:
  # Like Swin Transformer layers, which build attention masks on CPU between
  # TPU ops that do not depend on them.
  for i in range(3):
    x = torch.sin(x) * 2.0
    mask = torch.arange(3) >= i
    x = x.masked_fill(mask.to(x.device), 0.0)
  return x


def _size_expression_used_around_cpu_stage(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  n = x.shape[0] * 2
  y = ((x * 2.0).cpu() + 1.0).to(x.device)
  return torch.ones(n, device=x.device), y.repeat(2).view(n, 1)


class HeterogeneousDeviceTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()  # pylint: disable=protected-access

  def _assert_same(self, actual, expected):
    """Asserts that outputs have the same structure, devices and values."""
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    self.assertEqual(actual_spec, expected_spec)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
      if not isinstance(expected_leaf, torch.Tensor):
        self.assertEqual(actual_leaf, expected_leaf)
        continue
      self.assertEqual(actual_leaf.device.type, expected_leaf.device.type)
      utils.assert_close(actual_leaf.cpu(), expected_leaf.cpu())

  def _check(self, fn, *args, partitions=None, seed=None):
    """Checks that compiling `fn` matches eager.

    Args:
      fn: The function to compile.
      *args: The arguments to call `fn` with.
      partitions: If set, the expected number of TPU and CPU partitions of the
        compiled graph.
      seed: If set, the seed to draw random numbers from in both modes.

    Returns:
      The compiled function.
    """
    debugs = []
    compiled = torch.compile(
        fn,
        backend="tpu",
        fullgraph=True,
        # Avoid AOTAutograd cache hits so that debug info is always recorded.
        options={"debug_callback": debugs.append, "serializable": False},
    )
    if seed is not None:
      torch.manual_seed(seed)
    actual = compiled(*args)
    if seed is not None:
      torch.manual_seed(seed)
    self._assert_same(actual, fn(*args))
    if partitions is not None:
      (executable,) = debugs[0].compiled_executables
      split_gm = executable._split_gm  # pylint: disable=protected-access
      num_cpu_partitions = sum(
          isinstance(submod, torch.fx.GraphModule)
          for submod in split_gm.children()
      )
      self.assertEqual(
          (len(executable.compiled_executables), num_cpu_partitions),
          partitions,
          msg=split_gm.print_readable(print_output=False),
      )
    return compiled

  def test_tpu_only_graph(self):
    self._check(lambda x: x * 2.0 + 1.0, _tpu(1, 2, 3), partitions=(1, 0))

  def test_cpu_only_graph(self):
    self._check(
        lambda x: x * 2.0 + 1.0, torch.tensor([1.0, 2.0]), partitions=(0, 1)
    )

  def test_cpu_factory_feeds_tpu(self):
    linear = torch.nn.Linear(4, 4).to("tpu")

    def fn(x):
      mask = torch.zeros((1, 4), dtype=torch.bool)
      mask[0, 1] = True
      return linear(x) * mask.to(x.device)

    self._check(fn, torch.randn(2, 4, device="tpu"), partitions=(1, 1))

  def test_cpu_computation_feeds_tpu(self):
    def fn(x, y):
      return x + (y * 2.0).to(x.device)

    self._check(
        fn, _tpu(1, 2, 3), torch.tensor([4.0, 5.0, 6.0]), partitions=(1, 1)
    )

  def test_tpu_result_moved_to_cpu(self):
    self._check(
        lambda x: (x + 1.0).cpu() * 2.0, _tpu(1, 2, 3), partitions=(1, 1)
    )

  def test_round_trip_through_cpu(self):
    def fn(x):
      y = (x + 1.0).cpu() * 2.0
      return y.to(x.device) + 3.0

    self._check(fn, _tpu(1, 2, 3), partitions=(2, 1))

  def test_independent_cpu_work_does_not_split_tpu_program(self):
    self._check(_interleaved_cpu_masks, _tpu(1, 2, 3), partitions=(1, 1))

  @parameterized.named_parameters(
      dict(
          testcase_name="to",
          fn=lambda x, y: x + y.to(x.device),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="to_with_dtype",
          fn=lambda x, y: x + y.to(x.device, torch.bfloat16),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="copy",
          fn=lambda x, y: x + torch.empty_like(x).copy_(y),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="scalar_operand",
          fn=lambda x, y: x * y[0],
          partitions=(1, 1),
      ),
      dict(
          testcase_name="index_operand",
          fn=lambda x, y: x[y.long()],
          partitions=(1, 1),
      ),
  )
  def test_cpu_input_used_on_tpu(self, fn, partitions):
    self._check(
        fn, _tpu(1, 2, 3), torch.tensor([2.0, 0.0, 1.0]), partitions=partitions
    )

  def test_non_contiguous_cpu_tensor_used_on_tpu(self):
    def fn(x, y):
      return x + y.t().to(x.device)

    self._check(
        fn,
        torch.randn(3, 4, device="tpu"),
        torch.randn(4, 3),
        partitions=(1, 1),
    )

  def test_cpu_scalar_computed_on_cpu(self):
    def fn(x, s):
      return x * (s * 3.0)

    self._check(fn, _tpu(1, 2, 3), torch.tensor(2.0), partitions=(1, 1))

  @parameterized.named_parameters(
      dict(
          testcase_name="moved_to_tpu",
          fn=lambda x: x + torch.tensor([1.0, 2.0, 3.0]).to(x.device),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="expression_moved_to_tpu",
          fn=lambda x: x + (torch.tensor([1.0, 2.0, 3.0]) * 2.0).to(x.device),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="scalar_operand",
          fn=lambda x: x * torch.tensor(2.0),
          partitions=(1, 0),
      ),
      dict(
          testcase_name="combined_with_cpu_factory",
          fn=lambda x: x
          + (torch.zeros(3) + torch.tensor([1.0, 2.0, 3.0])).to(x.device),
          partitions=(1, 1),
      ),
      dict(
          testcase_name="used_on_cpu_and_tpu",
          fn=_constant_used_on_cpu_and_tpu,
          partitions=(1, 1),
      ),
  )
  def test_cpu_constants(self, fn, partitions):
    # Constants that are only used on TPU stay embedded in the TPU program.
    self._check(fn, _tpu(1, 2, 3), partitions=partitions)

  def test_outputs_on_both_devices(self):
    def fn(x, y):
      return x + 1.0, x.sum().cpu(), y * 2.0, torch.tensor([1.0, 2.0]) * 3.0

    self._check(fn, _tpu(1, 2, 3), torch.tensor([4.0, 5.0]), partitions=(1, 2))

  def test_module_with_cpu_buffer(self):
    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.register_buffer("scale", torch.tensor([1.0, 2.0, 3.0]))

      def forward(self, x):
        return self.linear(x) * self.scale.to(x.device)

    model = Model()
    model.linear.to("tpu")
    x = torch.randn(2, 3, device="tpu")
    model(x).sum().backward()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    compiled = torch.compile(model, backend="tpu", fullgraph=True)
    self._assert_same(compiled(x), model(x))
    compiled(x).sum().backward()
    self._assert_same([p.grad for p in model.parameters()], expected_grads)

  def test_cpu_input_mutation(self):
    def fn(x, counter):
      counter.add_(1.0)
      return x * counter.to(x.device)

    counter_eager = torch.zeros(3)
    expected = fn(_tpu(1, 2, 3), counter_eager)
    counter = torch.zeros(3)
    compiled = torch.compile(fn, backend="tpu", fullgraph=True)
    self._assert_same(compiled(_tpu(1, 2, 3), counter), expected)
    utils.assert_close(counter, counter_eager)

  def test_cpu_values_change_without_recompiling(self):
    def fn(x, y):
      return x * (y + 1.0).to(x.device)

    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(fn, backend=counter, fullgraph=True)
    for y in (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])):
      self._assert_same(compiled(_tpu(1, 2, 3), y), fn(_tpu(1, 2, 3), y))
    self.assertEqual(counter.frame_count, 1)

  @parameterized.named_parameters(
      ("factory", lambda x: x + torch.rand(3).to(x.device)),
      (
          "constant_input",
          lambda x: x + torch.bernoulli(torch.tensor([0.5] * 3)).to(x.device),
      ),
      # The first random op depends on a TPU result, so it runs in a later
      # stage than the second could on its own.
      (
          "dependent_first",
          lambda x: (torch.rand_like((x + 1.0).cpu()), torch.rand(3)),
      ),
      # The partition of the first random op does not feed the second one's.
      (
          "independent_first",
          lambda x: (torch.rand(3), torch.rand_like((x + 1.0).cpu())),
      ),
  )
  def test_cpu_random_ops_match_eager(self, fn):
    compiled = torch.compile(fn, backend="tpu", fullgraph=True)
    for seed in (0, 1, 2):
      torch.manual_seed(seed)
      expected = fn(_tpu(1, 2, 3))
      torch.manual_seed(seed)
      self._assert_same(compiled(_tpu(1, 2, 3)), expected)

  def test_cpu_random_tensor_consumed_on_tpu(self):
    # The random tensor is drawn on CPU on every call rather than embedded in
    # the TPU program as a constant.
    def fn(x):
      return x @ torch.randn(3, 3).to(x.device)

    x = torch.eye(3, device="tpu")
    compiled = self._check(fn, x, partitions=(1, 1), seed=0)
    torch.manual_seed(1)
    expected = fn(x)
    torch.manual_seed(1)
    self._assert_same(compiled(x), expected)

  def test_training_with_cpu_mask(self):
    def fn(x):
      mask = torch.ones(3)
      mask[0] = 0.0
      return (x * mask.to(x.device)).sum()

    x_eager = _tpu(1, 2, 3, requires_grad=True)
    fn(x_eager).backward()
    x = _tpu(1, 2, 3, requires_grad=True)
    torch.compile(fn, backend="tpu", fullgraph=True)(x).backward()
    self._assert_same(x.grad, x_eager.grad)

  def test_training_with_cpu_leaf(self):
    def fn(x, y):
      return (x * (y * 2.0).to(x.device)).sum()

    x_eager = _tpu(1, 2, 3, requires_grad=True)
    y_eager = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    fn(x_eager, y_eager).backward()
    x = _tpu(1, 2, 3, requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    torch.compile(fn, backend="tpu", fullgraph=True)(x, y).backward()
    self._assert_same((x.grad, y.grad), (x_eager.grad, y_eager.grad))

  @parameterized.named_parameters(
      dict(
          testcase_name="cpu_tensor_sized_by_tpu_input",
          fn=lambda x: x + torch.ones(x.shape[0]).to(x.device),
          device="tpu",
      ),
      dict(
          testcase_name="cpu_input",
          fn=lambda x: (x * 2.0).to("tpu") + 1.0,
          device="cpu",
      ),
      dict(
          testcase_name="size_used_after_cpu_stage",
          fn=lambda x: ((x * 2.0).cpu() + 1.0).to(x.device).view(x.shape[0], 1),
          device="tpu",
      ),
      dict(
          testcase_name="size_expression_used_around_cpu_stage",
          fn=_size_expression_used_around_cpu_stage,
          device="tpu",
      ),
  )
  def test_dynamic_shapes(self, fn, device):
    compiled = torch.compile(fn, backend="tpu", fullgraph=True)
    for size in (8, 12):
      x = torch.arange(size, dtype=torch.float32, device=device)
      torch._dynamo.mark_dynamic(x, 0, min=2, max=64)  # pylint: disable=protected-access
      self._assert_same(compiled(x), fn(x))

  def test_async_compile(self):
    def fn(x):
      mask = torch.zeros(3)
      mask[1] = 1.0
      return (x + mask.to(x.device)).cpu() * 2.0

    compiled = tpu_compile.async_compile(fn, [(_tpu(1, 2, 3),)])
    self._assert_same(compiled(_tpu(4, 5, 6)), fn(_tpu(4, 5, 6)))

  def test_aot_autograd_cache_round_trip(self):
    def fn(x, y):
      return ((x + 1.0).cpu() * y + torch.ones(3)).to(x.device) * 2.0

    y = torch.tensor([4.0, 5.0, 6.0])
    with tempfile.TemporaryDirectory() as cache_dir:
      with cache_dir_utils.temporary_cache_dir(cache_dir):
        for expected_event in ("autograd_cache_miss", "autograd_cache_hit"):
          torch._dynamo.reset()  # pylint: disable=protected-access
          dynamo_utils.counters.clear()
          compiled = torch.compile(fn, backend="tpu", fullgraph=True)
          self._assert_same(compiled(_tpu(1, 2, 3), y), fn(_tpu(1, 2, 3), y))
          self.assertEqual(
              dynamo_utils.counters["aot_autograd"][expected_event], 1
          )

  def test_data_dependent_cpu_shape(self):
    # Dynamo graph breaks on data-dependent ops by default, so the TPU graph
    # receives the CPU result with a concrete shape.
    def fn(x, mask):
      return x[torch.nonzero(mask).squeeze(1).to(x.device)] * 2.0

    compiled = torch.compile(fn, backend="tpu")
    for mask in ([True, False, True], [False, True, False]):
      mask = torch.tensor(mask)
      self._assert_same(compiled(_tpu(1, 2, 3), mask), fn(_tpu(1, 2, 3), mask))

  @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)  # pylint: disable=protected-access
  def test_data_dependent_cpu_shape_in_graph_raises(self):
    def fn(x, mask):
      return x[torch.nonzero(mask).squeeze(1).to(x.device)] * 2.0

    compiled = torch.compile(fn, backend="tpu", fullgraph=True)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python-side compile check
        dynamo_exc.BackendCompilerFailed, "CPU tensors with data-dependent"
    ):
      compiled(_tpu(1, 2, 3), torch.tensor([True, False, True]))

  @torch._dynamo.config.patch(capture_scalar_outputs=True)  # pylint: disable=protected-access
  def test_data_dependent_cpu_scalar_in_graph(self):
    def fn(x, y):
      return x * y.sum().item()

    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(fn, backend=counter, fullgraph=True)
    for y in (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0])):
      self._assert_same(compiled(_tpu(1, 2, 3), y), fn(_tpu(1, 2, 3), y))
    self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
  absltest.main()
