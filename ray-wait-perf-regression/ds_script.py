#!/usr/bin/env python3

import ray
import time
import numpy as np

@ray.remote
def create_ds():
    # 15 GB
    ds = ray.data.range_tensor(100000, shape=(80, 80, 3), parallelism=200)

    return ray.put(ds)

# Create ds on worker node
#ref = ray.get(create_ds.options(resources={'node:172.31.199.37': 0.001}).remote())
ref = ray.get(create_ds.remote())
ds = ray.get(ref)
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
