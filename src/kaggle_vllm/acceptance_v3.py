"""Fresh Kaggle NCCL world-size-four smoke test; run only on the target GPU host."""
from __future__ import annotations

import os


def main() -> None:
    import torch
    import torch.distributed as dist

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if world != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("acceptance requires world_size=4 and four visible L4 GPUs")
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_name(rank) != "NVIDIA L4" or torch.cuda.get_device_capability(rank) != (8, 9):
        raise RuntimeError("GPU is not exact NVIDIA L4 / SM89")
    dist.init_process_group("nccl")
    try:
        value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(value)
        if value.item() != 10.0:
            raise RuntimeError(f"NCCL all_reduce mismatch: {value.item()}")
        print(f"PASS rank={rank} world_size={world} sum={value.item()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
