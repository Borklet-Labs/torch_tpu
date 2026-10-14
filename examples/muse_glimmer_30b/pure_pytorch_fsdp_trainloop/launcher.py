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


import os

import torch
import torch.distributed.launcher as launcher
import torch.multiprocessing as mp

from . import worker


def worker_entrypoint():
  rank = int(os.getenv("RANK"))
  print(f"Rank {rank} started.", flush=True)

  torch.distributed.init_process_group("tpu_dist")
  worker.run_worker_trainloop(device="tpu")

  print(f"Rank {rank} done.", flush=True)


if __name__ == "__main__":
  mp.set_start_method("spawn")

  # We are running on a single host with 4 Ironwood TPUs (topology 2x2x1).
  # Because Ironwood TPUs each have two TensorCores, each with their own HBM,
  # they show up in PyTorch as 8 devices, hence world size is 8.
  world_size = 8

  torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=world_size)
  config = launcher.LaunchConfig(
      min_nodes=1,
      max_nodes=1,
      nproc_per_node=world_size,
      rdzv_backend="c10d",
      rdzv_endpoint="localhost:0",
      max_restarts=0,
  )
  launcher.elastic_launch(config, worker_entrypoint)()
