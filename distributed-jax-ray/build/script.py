#!/usr/bin/env python3

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
