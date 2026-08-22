import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    x = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(x)
    print(f"rank={rank} all_reduce={x.item()}")
    assert x.item() == 3.0
    dist.destroy_process_group()


if __name__ == "__main__":
    assert torch.cuda.device_count() == 2
    mp.spawn(worker, args=(2, 29591), nprocs=2, join=True)
    print("NCCL all-reduce PASS")
