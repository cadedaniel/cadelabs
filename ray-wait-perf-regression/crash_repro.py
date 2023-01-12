#!/usr/bin/env python3

import ray
import time
import numpy as np

# This appears to have a bug -- I can't deserialize the dataset across nodes.
# My workspace is somehow borked. I need to start from scratch.
# I am going to restart the workspace and see if I can repro.

ray.init()

@ray.remote(num_cpus=36)
def source_task(latch):
    print('src task')

    ray.get(latch.count_down.remote())
    while not ray.get(latch.is_ready.remote()):
        time.sleep(1)

    ds = ray.data.range_tensor(10)
    return ds

@ray.remote(num_cpus=36)
def dest_task(latch):

    future = source_task.options(num_cpus=36).remote(latch)

    print('dest task')

    ray.get(latch.count_down.remote())
    while not ray.get(latch.is_ready.remote()):
        time.sleep(1)

    ds = ray.get(future)
    print(ds.size_bytes() / 2**30, "GB")
    
    context = ray.data.context.DatasetContext.get_current()
    context.actor_prefetcher_enabled = False
    
    start = time.time()
    latencies = []
    for _ in ds.iter_batches(batch_size=None, prefetch_blocks=10):
        latencies.append(time.time() - start)
        start = time.time()
        if len(latencies) % 10 == 0:
                print("Latencies (mean/50/90)", np.mean(latencies), np.median(latencies), np.quantile(latencies, 0.9))
    
    print(ds.stats())

def cpu_bundle(num_cpu):
    return {"CPU": num_cpu}

@ray.remote(num_cpus=0)
class Latch:
    def __init__(self, count):
        self.count = count
        self.original_count = count

    def is_ready(self):
        return self.count == 0

    def count_down(self):
        self.count = max(self.count - 1, 0)

    def reset(self):
        self.count = self.original_count

if __name__ == '__main__':

    latch = Latch.remote(3)

    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    #pg = placement_group([cpu_bundle(36), cpu_bundle(36)], strategy='STRICT_SPREAD')
    #ray.get(pg.ready())
    #strategy = PlacementGroupSchedulingStrategy(placement_group=pg)

    #ref = source_task.options(
    #    #scheduling_strategy=strategy,
    #    num_cpus=1,
    #).remote(latch)

    future = dest_task.options(
        #scheduling_strategy=strategy,
        num_cpus=36,
    ).remote(latch)

    input('waiting')
    latch.count_down.remote()
    ray.get(future)
