#!/usr/bin/env python3

import ray
import time
import numpy as np

ray.init()

@ray.remote
def source_task():
    # 15 GB
    ds = ray.data.range_tensor(100000, shape=(80, 80, 3), parallelism=200)
    return ds

@ray.remote
def dest_task(ds):
    ds = ds
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

    ds = ray.get(source_task.options(
        resources={**resources[0]}
    ).remote())
    ray.get(dest_task.options(
        resources={**resources[1]}
    ).remote(ds))

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


"""
Traceback (most recent call last):
  File "/home/ray/cadelabs/ray-wait-perf-regression/./ds_script.py", line 67, in <module>
    ray.get(dest_task.options(
  File "/data/pypi/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 105, in wrapper
    return func(*args, **kwargs)
  File "/data/pypi/lib/python3.10/site-packages/ray/_private/worker.py", line 2309, in get
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError: ray::dest_task() (pid=5175, ip=172.31.121.0)
  File "/home/ray/cadelabs/ray-wait-perf-regression/./ds_script.py", line 25, in dest_task
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/dataset.py", line 2720, in iter_batches
    yield from batch_block_refs(
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 101, in batch_block_refs
    yield from batch_blocks(
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 145, in batch_blocks
    for formatted_batch in batch_iter:
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 305, in _format_batches
    for block in block_iter:
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 268, in _blocks_to_batches
    for block in block_iter:
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 169, in _resolve_blocks
    for block_ref in block_ref_iter:
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/block_batching.py", line 210, in _prefetch_blocks
    sliding_window = collections.deque(
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/lazy_block_list.py", line 414, in __next__
    ref, meta = next(self._base_iter)
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/data/_internal/lazy_block_list.py", line 459, in __next__
    generator = ray.get(generator_ref)
ray.exceptions.RayTaskError: ray::_execute_read_task_split() (pid=76752, ip=172.31.98.252)
  At least one of the input arguments for this task could not be computed:
ray.exceptions.RaySystemError: System error: Can't get attribute '_make_function' on <module 'ray.cloudpickle.cloudpickle' from '/data/pypi/lib/python3.10/site-packages/ray/cloudpickle/cloudpickle.py'>
traceback: Traceback (most recent call last):
AttributeError: Can't get attribute '_make_function' on <module 'ray.cloudpickle.cloudpickle' from '/data/pypi/lib/python3.10/site-packages/ray/cloudpickle/cloudpickle.py'>
(_execute_read_task_split pid=76752) 2023-01-10 17:37:46,969    ERROR serialization.py:371 -- Can't get attribute '_make_function' on <module 'ray.cloudpickle.cloudpickle' from '/data/pypi/lib/python3.10/site-packages/ray/cloudpickle/cloudpickle.py'>
(_execute_read_task_split pid=76752) Traceback (most recent call last):
(_execute_read_task_split pid=76752)   File "/data/pypi/lib/python3.10/site-packages/ray/_private/serialization.py", line 369, in deserialize_objects
(_execute_read_task_split pid=76752)     obj = self._deserialize_object(data, metadata, object_ref)
(_execute_read_task_split pid=76752)   File "/data/pypi/lib/python3.10/site-packages/ray/_private/serialization.py", line 252, in _deserialize_object
(_execute_read_task_split pid=76752)     return self._deserialize_msgpack_data(data, metadata_fields)
(_execute_read_task_split pid=76752)   File "/data/pypi/lib/python3.10/site-packages/ray/_private/serialization.py", line 207, in _deserialize_msgpack_data
(_execute_read_task_split pid=76752)     python_objects = self._deserialize_pickle5_data(pickle5_data)
(_execute_read_task_split pid=76752)   File "/data/pypi/lib/python3.10/site-packages/ray/_private/serialization.py", line 197, in _deserialize_pickle5_data
(_execute_read_task_split pid=76752)     obj = pickle.loads(in_band)
(_execute_read_task_split pid=76752) AttributeError: Can't get attribute '_make_function' on <module 'ray.cloudpickle.cloudpickle' from '/data/pypi/lib/python3.10/site-packages/ray/cloudpickle/cloudpickle.py'>
(dest_task pid=5175, ip=172.31.121.0) 14.30511474609375 GB
"""
