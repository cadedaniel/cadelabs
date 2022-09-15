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


ray.get([task.remote() for _ in range(2)])
