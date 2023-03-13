#!/usr/bin/env python3

import ray
import subprocess
import time

ray.init()

@ray.remote
class NodeActor:
    def __init__(self, rank, world_size, redis_hostname, redis_port):
        self.rank = rank
        self.world_size = world_size
        self.file_store_dir = '/mnt/cluster_storage/gloo/'
        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        print(self.redis_hostname, self.redis_port)

    def get_hostname(self):
        import socket
        return socket.gethostname()

    def install_gloo(self):
        pass
        #subprocess.run("pip install pygloo", shell=True)
    
    def test_gloo(self, prefix):
        import pygloo

        context = pygloo.rendezvous.Context(self.rank, self.world_size)

        attr = pygloo.transport.tcp.attr(self.get_hostname())
        #attr = pygloo.transport.tcp.attr("localhost")
        dev = pygloo.transport.tcp.CreateDevice(attr)
        #file_store = pygloo.rendezvous.FileStore(self.file_store_dir)
        store = pygloo.rendezvous.RedisStore(self.redis_hostname, self.redis_port)
        store = pygloo.rendezvous.PrefixStore(str(prefix), store)

        print('connectFullMesh start')
        # Getting segfault here
        context.connectFullMesh(store, dev)
        print('connectFullMesh done')
    
def schedule_on_node(node, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node['NodeID'], soft=soft
    )

def alive(nodes):
    for node in nodes:
        if node['Alive']:
            yield node

redis_port = 7777
redis_hostname = "ip-10-0-24-230"

alive_nodes = list(alive(ray.nodes()))

actors = [
    NodeActor.options(
        scheduling_strategy=schedule_on_node(node)
    ).remote(
        rank=rank,
        world_size=len(alive_nodes),
        redis_hostname=redis_hostname,
        redis_port=redis_port,
    ) for rank, node in enumerate(alive_nodes)
]

prefix = f"{time.time()}"

ray.get([actor.test_gloo.remote(prefix) for actor in actors])

while True:
    import time
    time.sleep(1)

#ray.get([actor.install_gloo.remote() for actor in actors])
#ray.get([actor.test_redis.remote() for actor in actors])

"""
import os
import ray
import pygloo
import numpy as np

@ray.remote(num_cpus=1)
def test_allreduce(rank, world_size, fileStore_path):
    '''
    rank  # Rank of this process within list of participating processes
    world_size  # Number of participating processes
    fileStore_path # The path to create filestore
    '''
    context = pygloo.rendezvous.Context(rank, world_size)
    # Prepare device and store for rendezvous
    attr = pygloo.transport.tcp.attr("localhost")
    dev = pygloo.transport.tcp.CreateDevice(attr)
    fileStore = pygloo.rendezvous.FileStore(fileStore_path)
    store = pygloo.rendezvous.PrefixStore(str(world_size), fileStore)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    pygloo.allreduce(context, sendptr, recvptr,
                    sendbuf.size, pygloo.glooDataType_t.glooFloat32,
                    pygloo.ReduceOp.SUM, pygloo.allreduceAlgorithm.RING)

if __name__ == "__main__":
    ray.init(num_cpus=6)
    world_size = 2
    fileStore_path = f"{ray.worker._global_node.get_session_dir_path()}" + "/collective/gloo/rendezvous"
    os.makedirs(fileStore_path)
    ray.get([test_allreduce.remote(rank, world_size, fileStore_path) for rank in range(world_size)])
"""
