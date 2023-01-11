#!/usr/bin/env python3

import ray
import time
import numpy as np

ray.init()

@ray.remote
def source_task():
    # 15 GB
    #ds = ray.data.range_tensor(100000, shape=(80, 80, 3), parallelism=200)
    #return ray.put(ds)

    import socket
    print(socket.gethostname())

@ray.remote
def dest_task(ds):
    import socket
    print(socket.gethostname())

def create_node_resource(node):
    return {'node:{}'.format(node['NodeName']): 0.001}

def wait_for_total_node_count(count):
    import time
    while True:
        nodes = ray.nodes()
        if len(nodes) >= count:
            return
        print(f'Waiting for total node count {count}, have {len(nodes)}')
        time.sleep(1)

def get_nodes_with_cpu_gpu(cpu_count, gpu_count):
    nodes = ray.nodes()
    matching = []
    for node in nodes:
        if node['Resources'].get('CPU', 0) != cpu_count:
            continue

        if node['Resources'].get('GPU', 0) != gpu_count:
            continue

        matching.append(node)
    return matching

if __name__ == '__main__':
    wait_for_total_node_count(2)
    nodes = get_nodes_with_cpu_gpu(cpu_count=36, gpu_count=0)
    assert len(nodes) == 2
    resources = [create_node_resource(node) for node in nodes]

    source_task.options(resources={**resources[0]}).remote()
    dest_task.options(resources={**resources[1]}).remote(1)

#@ray.remote
#def create_ds():
#    # 15 GB
#    ds = ray.data.range_tensor(100000, shape=(80, 80, 3), parallelism=200)
#
#    return ray.put(ds)
#
## Create ds on worker node
##ref = ray.get(create_ds.options(resources={'node:172.31.199.37': 0.001}).remote())
#ref = ray.get(create_ds.remote())
#ds = ray.get(ref)
#print(ds.size_bytes() / 2**30, "GB")
#
#context = ray.data.context.DatasetContext.get_current()
#context.actor_prefetcher_enabled = False
#
#start = time.time()
#latencies = []
#for _ in ds.iter_batches(batch_size=None, prefetch_blocks=10):
#    latencies.append(time.time() - start)
#    start = time.time()
#    if len(latencies) % 10 == 0:
#            print("Latencies (mean/50/90)", np.mean(latencies), np.median(latencies), np.quantile(latencies, 0.9))
#
#print(ds.stats())
