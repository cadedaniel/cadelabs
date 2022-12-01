#!/usr/bin/env python3
"""
Goal: Reproduce the issue in https://github.com/ray-project/ray/issues/30694 locally.
Notes: https://docs.google.com/document/d/1f1QpFaw-hoz1WcYwgu9lPLy43NeOKXnD6-CgwGK39ic/edit
Current hypothesis: There are potentially scaling issues in PlasmaStore::Wait

I should see scaling limits in PlasmaStore::Wait by waiting on a large number of objects with
fetch_local=True.
"""

import ray
import numpy as np
import time
import os

@ray.remote
class Waiter:
    
    def __init__(self):
        self._should_go = False

    def signal(self):
        self._should_go = True

    def should_go(self):
        return self._should_go
        

@ray.remote(num_cpus=0.001)
class SourceActor:
    def __init__(self, size_bytes):
        self.returnval = np.empty(int(size_bytes), dtype=np.int8)

    def is_started(self):
        return True

    def get_large_object(self, waiter):
        while not ray.get(waiter.should_go.remote()):
            time.sleep(0.1)
        return self.returnval

@ray.remote
class DestinationActor:
    def __init__(self):
        pass

    def get_many_large_objects(self):
        size_mb = 0.01
        num_tasks = int(os.cpu_count() * 2)
        
        print(f'Starting {num_tasks} actors')
        actors = [SourceActor.remote(2**20 * size_mb) for _ in range(num_tasks)]
        ray.get([actor.is_started.remote() for actor in actors])
        waiter = Waiter.remote()

        print(f'Running {num_tasks} tasks, each returning {size_mb} MB objects')
        refs = [actor.get_large_object.remote(waiter) for actor in actors]

        print('Unblocking tasks')
        waiter.signal.remote()
        
        print('Waiting for results')
        remaining = refs
        start_time = time.time()
        while remaining:
            print('wait iteration. Remaining:', len(remaining))
            _, remaining = ray.wait(remaining, fetch_local=False, timeout=0.1)
        end_time = time.time()
        print(f'Duration {end_time - start_time:.02f}')

a = DestinationActor.remote()
ray.get(a.get_many_large_objects.remote())
