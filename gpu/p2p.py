import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import ray

ray.init()

@ray.remote
class Actor:
    def __init__(self, rank, world_size, rank_dev_map):
        self.rank = rank
        self.dev = rank_dev_map[rank]
        self.devs = [rank_dev_map[r] for r in range(world_size)]

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.dev)
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1025'

    def comm(self):
        import torch
        
        torch.distributed.init_process_group()
        numel = 5 * 2**30
        device_tensor = torch.ones(numel, dtype=torch.uint8, device='cuda')

        num_iters = 100
        warmup_runs = 5
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for start, end in zip(start_events, end_events):
            start.record()
            if self.rank == 0:
                torch.distributed.send(device_tensor, 1)
                torch.distributed.recv(device_tensor, 1)
            else:
                torch.distributed.recv(device_tensor, 0)
                torch.distributed.send(device_tensor, 0)
            end.record()
        
        dur_s = 0
        sent_bytes = 0
        for i, (start, end) in enumerate(zip(start_events, end_events)):
            if i < warmup_runs:
                continue
            end.synchronize()

            dur_s += start.elapsed_time(end) / 1000
            sent_bytes += 2 * numel
        
        if self.rank == 0:
            print(f'[{self.devs[0]} -> {self.devs[1]}] {dur_s=:.02f} rate {(sent_bytes/2**30)/dur_s:.02f} GB/s')


def test_speed(a, b):
    world_size = 2
    
    rank_dev_map = {
        0: a,
        1: b,
    }
    
    actors = [Actor.remote(i, world_size, rank_dev_map) for i in range(world_size)]
    
    ray.get([a.comm.remote() for a in actors])

from collections import defaultdict
seen = defaultdict(list)
num_gpus = 8
for a in range(num_gpus):
    for b in range(num_gpus):
        if a == b:
            continue
        if b in seen[a] or a in seen[b]:
            continue
        test_speed(a, b)
        seen[a].append(b)
        seen[b].append(a)
