#!/usr/bin/env python3

import requests
import sys
import time
from ray import serve
from ray.serve.drivers import DAGDriver
from ray.serve.dag import InputNode
from ray.serve.http_adapters import json_request

@serve.deployment
class CpuDeployedModel:
    def __init__(self):
        pass

    def get_time(self, _):
        return time.time()

    def get_duration(self, start_time):
        return time.time() - start_time

from dataclasses import dataclass
import cupy as cp
import numpy as np
import os
import uuid

@dataclass
class GpuMeasurement:
    t: float
    data: cp.ndarray

@serve.deployment(ray_actor_options=dict(num_gpus=0.1))
class GpuDeployedModel:
    """
    This appears to be running all tasks in the same raylet.
    They are in different processes..

    Is cupy automatically mapping memory?
    I can check by measuring the first access time in the second process.
    """
    def __init__(self):
        pass

    def get_time(self, data):
        cp.cuda.nvtx.RangePop()
        output_gpu_tensor = True
        size = 1 << 10
        tag = ('gpu' if output_gpu_tensor else 'cpu') + f':{size*4}B'
        cp.cuda.nvtx.RangePush(f'get_time_{tag}')
        tensors = [cp.random.rand(size, dtype=cp.float32) for _ in range(1)]

        if output_gpu_tensor:
            output = tensors[0]
        else:
            # Approximate size of handle
            output = str(uuid.uuid4())

        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePush(f'between_tasks_{tag}')
        return output

    def get_duration(self, tensor):
        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePush(f'dag_to_response')
        return 0

def main():
    try:
        #measure_cpu_inter_model_latency()
        measure_gpu_inter_model_latency()
    finally:
        serve.shutdown()

    cp.cuda.nvtx.RangePop()
    
    import ray
    ray.timeline(filename="timeline.json")


def measure_gpu_inter_model_latency():
    with InputNode() as input_node:
        model_1 = GpuDeployedModel.bind()
        #model_2 = GpuDeployedModel.bind()
        dag = model_1.get_duration.bind(model_1.get_time.bind(input_node))

    cp.cuda.nvtx.RangePush('serve.run')
    serve.run(DAGDriver.bind(dag, http_adapter=json_request), port=1025)

    def measure(index):
        cp.cuda.nvtx.RangePush('measure')
        cp.cuda.nvtx.RangePush(f'request_to_dag')
        rval = requests.post("http://localhost:1025/", json=index).json()
        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePop()
        return rval
    
    cp.cuda.nvtx.RangePush('measure_loop')
    [measure(i) for i in range(128)]
    cp.cuda.nvtx.RangePop()

    #durations_ms = sorted(
    #    [(i, 1000 * measure(i)) for i in range(15)], key=lambda x: x[1]
    #)
    #durations_fmt = [f"{i:03}: {duration:0.2f}" for i, duration in durations_ms]
    #print("\n".join(durations_fmt))

def measure_cpu_inter_model_latency():
    with InputNode() as input_node:
        model_1 = CpuDeployedModel.bind()
        model_2 = CpuDeployedModel.bind()
        dag = model_2.get_duration.bind(model_1.get_time.bind(input_node))

    serve.run(DAGDriver.bind(dag, http_adapter=json_request), port=1025)

    def measure():
        return requests.post("http://localhost:1025/", json="").json()

    durations_ms = sorted(
        [(i, 1000 * measure()) for i in range(15)], key=lambda x: x[1]
    )
    durations_fmt = [f"{i:03}: {duration:0.2f}" for i, duration in durations_ms]
    print("\n".join(durations_fmt))
    serve.shutdown()

if __name__ == "__main__":
    sys.exit(main())
