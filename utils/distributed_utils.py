import os
import subprocess

import torch
import torch.distributed as dist


def dist_init():
    proc_id = int(os.environ['SLURM_PROCID'])
    num_gpus = torch.cuda.device_count()
    local_id = proc_id % num_gpus
    torch.cuda.set_device(local_id)

    world_size = get_world_size()
    rank = get_rank()

    node_list = os.environ["SLURM_NODELIST"]
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = '5671'
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    return local_id, rank, world_size


def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))


def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))


def print_debug(*args, **kw):
    # return
    print(*args, **kw)
