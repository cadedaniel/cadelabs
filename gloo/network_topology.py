#!/usr/bin/env python3

import os

os.environ['RAY_DEDUP_LOGS'] = '0'

import ray
from random import shuffle

ray.init()

@ray.remote
class MeasureActor:
    def get_ip(self):
        import socket
        return socket.gethostname()

    def measure(self, ips):
        
        try:
            import ping3
        except ImportError:
            import subprocess
            subprocess.check_call("pip install ping3", shell=True)
            import ping3

        ip_to_p50_latency = {}
        shuffle(ips)

        for ip in ips:
            latencies = [ping3.ping(ip) for _ in range(10)]
            latencies.sort()
            p50_latency_ms = latencies[len(latencies)//2] * 1000
            ip_to_p50_latency[ip] = p50_latency_ms

        return self.get_ip(), ip_to_p50_latency

def schedule_on_node(node, soft=False):
    return schedule_on_node_id(node['NodeID'], soft)

def schedule_on_node_id(node_id, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=soft
    )

nodes = ray.nodes()

actors = [
    MeasureActor.options(
        scheduling_strategy=schedule_on_node(node)
    ).remote()
    for node in nodes
]

ips = ray.get([actor.get_ip.remote() for actor in actors])

all_latencies = ray.get([actor.measure.remote(ips) for actor in actors])
top_bot = {}

for ip, latencies_to_others in all_latencies:
    latencies = list(sorted(latencies_to_others.values()))
    #sorted_latencies = {k: v for k, v in sorted(latencies_to_others.items(), key=lambda item: item[1])}

    top_avg_difference_ms = sum(latencies)/len(latencies) - latencies[0]
    top_bot_difference_ms = latencies[-1] - latencies[0]
    top_bot[ip] = top_avg_difference_ms

sorted_top_bot = {k: v for k, v in reversed(sorted(latencies_to_others.items(), key=lambda item: item[1]))}
for ip, top_bot_difference_ms in sorted_top_bot.items():
    print(f'{ip=} {top_bot_difference_ms=}')
