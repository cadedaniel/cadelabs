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
        ping_size_bytes = 9000 - 1

        for ip in ips:
            latencies = [ping3.ping(ip, size=ping_size_bytes) for _ in range(10)]
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
info_per_ip = {}

for ip, latencies_to_others in all_latencies:
    latencies = list(sorted(latencies_to_others.values()))
    
    avg_lat = sum(latencies)/len(latencies)
    top_lat = latencies[0]
    bot_lat = latencies[-1]
    top_avg_difference_ms = avg_lat - top_lat
    top_avg_difference_ms = bot_lat - top_lat
    info_per_ip[ip] = (top_avg_difference_ms, avg_lat)

sorted_info_per_ip = {k: v for k, v in reversed(sorted(info_per_ip.items(), key=lambda item: item[1][0]))}
for ip, (top_avg_difference_ms, avg_lat) in sorted_info_per_ip.items():
    print(f'{ip=} {top_avg_difference_ms=} {avg_lat=}')
