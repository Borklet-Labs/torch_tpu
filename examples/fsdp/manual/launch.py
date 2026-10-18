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

"""A test runner for the manual FSDP example."""

import torch
from torch.distributed import launcher
from torch.tpu import distributed  # pyrefly: ignore[missing-import]
import manual_fsdp_worker  # from fsdp.manual


def main():
  # Launch one worker per TPU chip on the host, which is also what
  # `set_tpu_launch_env()` sizes the slice for when `nproc_per_node` is omitted.
  # Fewer workers than chips would fail on 8-chip TPU v5e hosts, where libtpu
  # cannot reliably build a 4-chip sub-slice. The TPU runtime is not
  # initialized in this process, so `device_count()` counts all the chips.
  distributed.set_tpu_launch_env()
  world_size = torch.accelerator.device_count()
  config = launcher.LaunchConfig(
      min_nodes=1,
      max_nodes=1,
      nproc_per_node=world_size,
      rdzv_backend="c10d",
      rdzv_endpoint="localhost:0",
      max_restarts=0,
  )
  launcher.elastic_launch(config, manual_fsdp_worker.worker_fn)()


if __name__ == "__main__":
  main()
