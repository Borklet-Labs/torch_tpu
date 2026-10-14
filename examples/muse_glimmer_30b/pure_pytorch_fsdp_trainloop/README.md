# Pure PyTorch trainloop for Muse Glimmer 30B

This is a fully self-contained, without external dependencies (other than
`torch` and `torch_tpu`), example that demonstrates a basic LLM trainloop for
the full-size
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) model.

It runs distributed via FSDP on 4x Ironwood TPUs (i.e. 8 TensorCores).

This example is intentionally using only standard PyTorch features, ops, and
modules like:

-   nn.Linear, nn.Embedding, nn.RMSNorm
-   F.silu, F.scaled_dot_product_attention, F.linear, etc
-   native PyTorch functional collectives
-   torch.compile
-   torch.utils.checkpoint for activation checkpointing

It does *not* use any custom kernels, or TPU-specific APIs.

Notably, the model implementation itself ([model_fsdp.py](model_fsdp.py)) is
completely device-agnostic, and can run on both TPUs and GPUs/CUDA without any
code changes.

All TPU-specific code is limited to the [launcher.py](launcher.py):

```python
# This is a helpful utility that configures some environment
# variables, necessary for correct TPU slice initialization.
torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=world_size)
```

```python
# For TPUs, we muse use `tpu_dist` backend, instead of `nccl`.
# NOTE: in future TorchTPU versions, this will become automatic.
torch.distributed.init_process_group("tpu_dist")
```

```python
# The worker entrypoint is written to accept "tpu" or "cuda" device arg.
# This is used only to determine torch.compile backend ("tpu" or "inductor").
worker.run_worker_trainloop(device="tpu")
```

This example is designed for simplicity and readability:

-   it focuses on the LLM backbone only,
-   illustrates FSDP forward, loss backward, and optimizer step running on TPUs,
-   does *not* include data loading, tokenization, checkpoint loading/saving.

For more details about this example, see our Muse Glimmer case study:

TODO(vladbelous): link to case study, once ready.

## How to run

TODO(vladbelous): review/update once the example lands in nightly

The included [batchv1_job.yaml](./batchv1_job.yaml) provides a working starting
point for running this on Kubernetes, using the standard `batch/v1.Job` API.

For your own Kubernetes environment, you will need to update the nodepool
selector:

```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: torchtpu-v7x-4chip-pool
```

You can run the job with `kubectl apply -f batchv1_job.yaml`. Once the job
starts running, you will see logs like this:

```
...
Run 0: 51.387 seconds, 797.09 TPS (per rank)
Run 1: 17.040 seconds, 2403.78 TPS (per rank)
Run 2: 17.202 seconds, 2381.06 TPS (per rank)
Run 3: 17.101 seconds, 2395.16 TPS (per rank)
Run 4: 17.073 seconds, 2399.05 TPS (per rank)
```

Run 0 is slower than subsequent runs, because this is when `torch.compile` does
the compilation work. Starting with the 2nd train step (i.e. Run 1), the
trainloop runs at full speed.
