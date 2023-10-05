"""
$ export NUM_GPU=8
$ python3 -m torch.distributed.launch --use-env --nproc-per-node $NUM_GPU g5_p2p_noray_nccl_env.py
$ mpirun -n $NUM_GPU  python3 g5_p2p_noray_nccl_env.py
"""

import os
if 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ['WORLD_RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1024'

#os.environ['NCCL_SINGLE_RING_THRESHOLD'] = 
#os.environ['NCCL_ALGO'] = 'Tree'
#os.environ['NCCL_ALGO'] = 'Ring'
#os.environ['NCCL_PROTO'] = 'LL,LL128'
#os.environ['NCCL_PROTO'] = 'LL128,Simple'
#os.environ['NCCL_PROTO'] = 'Simple'
os.environ['NCCL_CHECKS_DISABLE'] = '1'

#if True:
#    os.environ['NCCL_P2P_LEVEL'] = 'LOC'
#    os.environ['NCCL_SHM_USE_CUDA_MEMCPY'] = '1'
#    os.environ['SHM_MEMCPY_MODE'] = '3'
#    os.environ['NCCL_TOPO_FILE'] = 'nccl-topo.xml'

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

d_model = 8192
num_layers = 80
dtype = torch.float16
batch_size = 200

numel = d_model * batch_size
expected = torch.ones(numel, dtype=dtype, device='cuda') * avg
actual = torch.ones(numel, dtype=dtype, device='cuda') * rank

# CUDA buffers more than 511
num_iters = 511

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

torch.distributed.barrier()
if rank == 0:
    print('Starting')

def enqueue(tensor, start, end):
    start.record()
    handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    #handle = dist.all_gather(tensor, async_op=True)
    handle.wait()
    end.record()

for start, end in zip(start_events, end_events):
    enqueue(actual, start, end)

dur_s = 0
sent_bytes = 0
sent_gb = 0
iteration = 0

all_latency_s = []
for start, end in zip(start_events, end_events):
    end.synchronize()
    
    latency_s = start.elapsed_time(end) / 1000
    all_latency_s.append(latency_s)
    dur_s += latency_s

    sizeof_float16 = 2
    message_size_bytes = numel * sizeof_float16

    # allreduce ring communication is O(2N)
    num_comms = 2

    sent_bytes += message_size_bytes * num_comms
    sent_gb = sent_bytes / 2**30

    if rank == 0:
        print(f'{iteration=} {dur_s=:.02f} latency_ms={latency_s*1000:.02f} {sent_gb=:.02f} throughput {sent_gb/dur_s:.02f} GB/s')

    enqueue(actual, start, end)
    iteration += 1

if rank == 0:
    print(f'{d_model=} {num_layers=} {batch_size=} tp={world_size}')
    
    # Two comms per layer
    min_layer_comms_latency_ms = 2 * 1000 * min(all_latency_s)

    print(f'min latency_ms: {min_layer_comms_latency_ms:.03f}')

    model_weights_size = 130 * 2**30
    single_gpu_memory_io = 2039 * 2**30
    aggregate_memory_io = single_gpu_memory_io * world_size

    memory_io_time_ms = 1000 * model_weights_size / aggregate_memory_io
    memory_io_time_single_layer_ms = memory_io_time_ms / num_layers
    print(f'{memory_io_time_single_layer_ms=:.03f}')

    total_time_ms_single_layer = min_layer_comms_latency_ms + memory_io_time_single_layer_ms
    print(f'ms_per_layer {total_time_ms_single_layer:.03f} nccl_time {100 * min_layer_comms_latency_ms/total_time_ms_single_layer:.1f}%')
