"""
$ export NUM_GPU=8
$ python3 -m torch.distributed.launch --use-env --nproc-per-node $NUM_GPU g5_p2p_error_noray.py
$ mpirun -n $NUM_GPU  python3 g5_p2p_error_noray.py
"""

import os
if 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ['WORLD_RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1024'

import torch
import torch.distributed as dist

assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')

rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
avg = sum(range(world_size)) / world_size

if rank == 0:
    print('Creating data')

numel = 2**30
expected = torch.ones(numel, dtype=torch.float16, device='cuda') * avg
actual = torch.ones(numel, dtype=torch.float16, device='cuda') * rank

# CUDA buffers more than 511
num_iters = 10

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

torch.distributed.barrier()
if rank == 0:
    print('Starting')

def enqueue(tensor, start, end):
    start.record()
    handle = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, async_op=True)
    handle.wait()
    end.record()

for start, end in zip(start_events, end_events):
    enqueue(actual, start, end)

dur_s = 0
sent_bytes = 0
sent_gb = 0
iteration = 0

while True:
    for start, end in zip(start_events, end_events):
        end.synchronize()

        dur_s += start.elapsed_time(end) / 1000

        sizeof_float16 = 2
        message_size_bytes = numel * sizeof_float16

        # allreduce communication is O(2N)
        num_comms = 2

        sent_bytes += message_size_bytes * num_comms
        sent_gb = sent_bytes / 2**30

        if rank == 0:
            print(f'{iteration=} {dur_s=:.02f} {sent_gb=:.02f} throughput {sent_gb/dur_s:.02f} GB/s')

        enqueue(actual, start, end)
        iteration += 1
