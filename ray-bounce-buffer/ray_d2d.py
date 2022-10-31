#!/usr/bin/env python3

import ray

import time
from dataclasses import dataclass
import cupy as cp
import numpy as np
import os

@dataclass
class GpuMeasurement:
    t: float
    data: cp.ndarray

@ray.remote(num_gpus=0.1)
class GpuActor:

    def __init__(self):
        pass
        #cp.cuda.set_allocator(None)

    def create_gpu_tensor(self, index, pass_gpu_tensor):
        cpu_tensor = np.array([10])
        #gpu_tensor = cp.asarray(cpu_tensor)
        gpu_tensor = cp.random.rand(100)
        #print('create_gpu_tensor')

        tensor_to_pass = gpu_tensor if pass_gpu_tensor else cpu_tensor
        return GpuMeasurement(time.time(), tensor_to_pass)

    def consume_gpu_tensor(self, measurement):
        end_time = time.time()
        gpu_tensor = measurement.data
        start_time = measurement.t
        duration_s = end_time - start_time
        #print(f'{duration_s:0.2f} seconds')
        return duration_s

a = GpuActor.remote()
b = GpuActor.remote()

def measure(index):
    gpu_tensor = a.create_gpu_tensor.remote(index, True)
    return ray.get(b.consume_gpu_tensor.remote(gpu_tensor))

for measurement in sorted([measure(i) for i in range(1)]):
    print(f'{measurement * 1000:0.2f} ms')
