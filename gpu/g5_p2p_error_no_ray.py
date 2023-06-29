"""
$ export NUM_GPU=8
$ python3 -m torch.distributed.launch --use-env --nproc-per-node $NUM_GPU g5_p2p_error_noray.py
"""

import torch
import torch.distributed as dist

assert torch.cuda.is_available()
dist.init_process_group()

rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
avg = sum(range(world_size)) / world_size

expected = torch.tensor([avg], dtype=torch.float16, device='cuda')

actual = torch.tensor([rank], dtype=torch.float16, device='cuda')
dist.all_reduce(actual, op=dist.ReduceOp.AVG)

assert torch.allclose(actual, expected), f"incorrect values {actual=} {expected=}"

if rank == 0:
    print('Success!')
