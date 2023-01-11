#!/usr/bin/env python3

import ray
import os
import time

def main():
    num_tasks = 10_000
    run_test(num_tasks)

def run_test(num_tasks):
    actors = create_actors(num=os.cpu_count())

    durations = []
    for i in range(5):
        dur_s = run_single_iteration(num_tasks, actors)
        print(f'Trial {i}: {dur_s:.02f}s')
        durations.append(dur_s)

    print('Dropping warmup trial')
    durations = durations[1:]

    print(f'RAY_core_worker_new_path={os.environ.get("RAY_core_worker_new_path", None)}')
    print(', '.join(f'{dur_s:.02f}' for dur_s in sorted(durations)))
    print(f'Average duration: {sum(durations)/len(durations):.02f}s')

def create_actors(num):
    print(f'Creating {num} actors')

    @ray.remote
    class SourceActor:
        def get_result(self):
            pass

    actors = []
    for _ in range(num):
        actors.append(SourceActor.remote())
    return actors

def run_single_iteration(num_tasks, actors):
    num_tasks_per_actor = num_tasks // len(actors)
    remainder = num_tasks % len(actors)

    futures = []

    print(f'Scheduling {num_tasks} tasks')
    while len(futures) < num_tasks:
        for a in actors:
            futures.append(a.get_result.remote())
            if len(futures) == num_tasks:
                break
    
    print(f'Waiting {len(futures)} results')
    remaining = futures
    start_time = time.time()
    while remaining:
        _,  remaining = ray.wait(remaining, fetch_local=True, timeout=0.1)
    end_time = time.time()

    duration_s = end_time - start_time
    return duration_s

if __name__ == '__main__':
    main()
