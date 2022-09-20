#!/usr/bin/env python3
import ray
ray.init()

@ray.remote(num_gpus=1)
def task():

    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax import random
    
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    print(x)
    
    size = 3000
    x = random.normal(key, (size, size), dtype=jnp.float32)
    
    import time
    start = time.time()
    jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
    print(f'Done in {(time.time() - start) * 1000:.2f}ms')

@ray.remote(num_gpus=1)
def nccl_test_task(world_rank):
    import cupy as cp
    import numpy as np
    import os
    print(f'cupy import worked, LD_LIBRARY_PATH={os.environ.get("LD_LIBRARY_PATH")}')

    x_cpu = np.array([1, 2, 3])
    with cp.cuda.Device(0):
        x_gpu = cp.asarray(x_cpu)
    
    if world_rank == 0:
        print('nccl version', cp.cuda.nccl.get_version())
    #print(f'unique nccl id: {cupy.cuda.nccl.get_unique_id()}')


#ray.get([task.remote() for _ in range(2)])
ray.get([nccl_test_task.remote(world_rank) for world_rank in range(2)])
