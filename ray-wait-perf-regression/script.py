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
    def __init__(self):
        pass

    def is_started(self):
        return True

    def create_returnval(self, size_bytes):
        self.returnval = np.empty(int(size_bytes), dtype=np.int8)

    def get_large_object(self):
        #while not ray.get(waiter.should_go.remote()):
        #    time.sleep(0.1)
        return self.returnval

@ray.remote
class DestinationActor:
    def __init__(self):
        pass

    def get_many_large_objects(self):
        #size_mb = 0.125
        #size_mb = 1.5 # 1200
        #size_mb = '0.001'
        size_mb = 0.125
        num_tasks = int(10 * 1e3)
        fetch_local = True

        total_num_cpus = int(ray.cluster_resources()["CPU"])
        
        print(f'Starting {total_num_cpus} actors')
        actors = [SourceActor.options(resources={'node:172.31.199.37': 0.0001}).remote() for _ in range(total_num_cpus)]
        ray.get([actor.is_started.remote() for actor in actors])

        print(f'Creating returnvals')
        ray.get([actor.create_returnval.remote(1) for actor in actors])

        durations = []
        for _ in range(1):
            print(f'Running {num_tasks} tasks, each returning {size_mb} MB objects')
            #waiter = Waiter.remote()
            refs = []
            for i in range(int(num_tasks / len(actors))):
                refs += [actor.get_large_object.remote() for actor in actors]

            #print('Unblocking tasks')
            #waiter.signal.remote()
        
            print('Waiting for results')
            remaining = refs
            start_time = time.time()
            while remaining:
                done, remaining = ray.wait(remaining, fetch_local=fetch_local, timeout=0.1)
                #ray.get(done)
            end_time = time.time()
            duration = end_time - start_time
            durations.append(duration)
            print(f'Duration {end_time - start_time:.02f}')
        print('Durations', ' '.join([f'{d:.02f}' for d in durations]))

a = DestinationActor.remote()
ray.get(a.get_many_large_objects.remote())

"""
Without our change:
32 actors, 125KB each, fetch_local=True
2.18s
2.2s

With our change:
32 actors, 125KB each, fetch_local=True
2.15s
2.32s

64 actors, 1.5MB each, fetch_local=True
with our change: 3.47, 3.49 2.36 2.23
without our change: 3.47, 2.26, 3.55 2.32 2.31

14 actors, 1 byte, 20000 tasks
Without our change: 46.30
With our change: 113.03

12->25
"""
