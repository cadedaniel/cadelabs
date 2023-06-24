import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import ray
from collections import defaultdict

ray.init()


@ray.remote
class AllreduceActor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.rank)
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1025'

    def _prev_rank(self):
        return (self.rank - 1 + self.world_size) % self.world_size

    def _next_rank(self):
        return (self.rank + 1) % self.world_size

    def comm(self):
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(0)
        dist.init_process_group()

        numel = 10 * 2**30
        dev_send_tensor = torch.ones(numel, dtype=torch.uint8, device='cuda')
        #dev_recv_tensor = torch.ones(numel, dtype=torch.uint8, device='cuda')

        num_iters = 5
        warmup_runs = 1
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for start, end in zip(start_events, end_events):

            start.record()
            handle = dist.all_reduce(dev_send_tensor, async_op=True)
            handle.wait()
            
            end.record()

        dur_s = 0
        sent_bytes = 0
        for i, (start, end) in enumerate(zip(start_events, end_events)):
            if i < warmup_runs:
                continue
            end.synchronize()

            dur_s += start.elapsed_time(end) / 1000
            # Ring allreduce comms is O(2N)
            sent_bytes += 2*numel
        
        print(f'[{self.rank}] {dur_s=:.02f} rate {(sent_bytes/2**30)/dur_s:.02f} GB/s')

def test_allreduce():

    world_size = 8
    actors = [AllreduceActor.remote(i, world_size) for i in range(world_size)]
    print('Created actors')
    ray.get([a.comm.remote() for a in actors])

if __name__ == '__main__':
    test_allreduce()
