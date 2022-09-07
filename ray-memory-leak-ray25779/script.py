#!/usr/bin/env python3

import gc
import ray

if ray.is_initialized():
    ray.shutdown()
ray.init(address=None, object_store_memory=200_000_000)

@ray.remote
def work(x):
    import random
    import time
    time.sleep(5)
    return random.randbytes(10_000_000)

def launch_ray():
    import random
    x = random.randbytes(10_000_000)
    return work.remote(x)

###############################################################################

# This attempts to pace job submissions

idx = 0
N = 50
pending_tasks = []
while True:
    gc.collect()
    completed_tasks, pending_tasks = ray.wait(pending_tasks)

    while len(pending_tasks) < 8 and idx < N:
        idx += 1

        r = launch_ray()
        pending_tasks.append(r)

    for t in completed_tasks:
        t = ray.get(t)
    del completed_tasks

    if not pending_tasks:
        break
del pending_tasks
