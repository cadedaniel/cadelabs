#!/usr/bin/env python3

import time
import math
import ray


@ray.remote(num_cpus=1)
def time_very_slow_create_list(num_elem: int) -> float:
    start_time = time.time()
    a = []
    for _ in range(num_elem):
        a.append(None)
    stop_time = time.time()
    return 1000 * (stop_time - start_time)


@ray.remote(num_cpus=1)
def time_slow_create_list(num_elem: int) -> float:
    start_time = time.time()
    a = [None for _ in range(num_elem)]
    stop_time = time.time()
    return 1000 * (stop_time - start_time)


@ray.remote(num_cpus=1)
def time_prepend(num_elem: int) -> float:
    a = [None] * num_elem
    start_time = time.time()
    c = [None] + a
    stop_time = time.time()
    return 1000 * (stop_time - start_time)


@ray.remote(num_cpus=1)
def time_append(num_elem: int) -> float:
    a = [None] * num_elem
    start_time = time.time()
    a.append(None)
    stop_time = time.time()
    return 1000 * (stop_time - start_time)


@ray.remote(num_cpus=1)
def time_create_list(num_elem: int) -> float:
    start_time = time.time()
    a = [None] * num_elem
    stop_time = time.time()
    return 1000 * (stop_time - start_time)


@ray.remote(num_cpus=1)
def test_strategy(strategy: str, trial_set: str = "fast"):

    if trial_set == "fast":
        powers = (21, 23, 25)
    elif trial_set == "lowres":
        powers = (21, 25, 28)
    elif trial_set == "highres":
        powers = range(21, 29, 1)
    elif trial_set == "wide_highres":
        powers = range(1, 31, 1)
    else:
        raise ValueError(f"unknown trial set {trial_set}")
    trials = [1 << power for power in powers]

    if strategy == "prepend":
        f = time_prepend
    elif strategy == "append":
        f = time_append
    elif strategy == "create_list":
        f = time_create_list
    elif strategy == "slow_create_list":
        f = time_slow_create_list
    elif strategy == "very_slow_create_list":
        f = time_very_slow_create_list
    else:
        raise ValueError(f"unknown strategy {strategy}")

    times = ray.get([f.remote(num_elem) for num_elem in trials])
    return times, trials


ray.init()

strategies = [
    "append",
    "prepend",
    "create_list",
    "slow_create_list",
    "very_slow_create_list",
]
trial_set = "fast"
results = ray.get(
    [test_strategy.remote(strategy, trial_set) for strategy in strategies]
)

for strategy, test_result in zip(strategies, results):
    print(f"==={strategy}")
    print(f"num_elements total_time time_per_element")
    for time, num_elem in zip(*test_result):
        time_per_element_ns = 10e6 * time / num_elem
        power = int(math.log(num_elem, 2))
        print(f"2^{power} {time:0.2f}ms {time_per_element_ns:0.2f}ns")
