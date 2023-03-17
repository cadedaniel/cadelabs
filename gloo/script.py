#!/usr/bin/env python3

import ray
import subprocess
import time
import numpy as np

ray.init()

@ray.remote(num_cpus=0)
class NodeStateManager:
    def setup(self):
        cache_dir = "/home/ray/.cache/gloo-deps-src"

        if subprocess.run(f"ls {cache_dir} > /dev/null", shell=True).returncode != 0:
            subprocess.check_call(f"mkdir {cache_dir}", shell=True)
        
        if subprocess.run(f"ls {cache_dir}/ray > /dev/null", shell=True).returncode != 0:
            print('Shallow clone ray')
            subprocess.check_call(
                f"git clone --depth 1 https://github.com/ray-project/ray.git "
                f"{cache_dir}/ray",
                shell=True
            )

        if subprocess.run(f"ls {cache_dir}/pygloo > /dev/null", shell=True).returncode != 0:
            print('Shallow clone pygloo')
            subprocess.check_call(
                f"git clone --depth 1 https://github.com/ray-project/pygloo.git "
                f"{cache_dir}/pygloo",
                shell=True
            )

        if subprocess.run("ls ~/bin/bazel > /dev/null", shell=True).returncode != 0:
            # install bazel
            subprocess.check_call("sudo apt-get install curl", shell=True)
            subprocess.check_call(f"cd {cache_dir}/ray; bash ./ci/env/install-bazel.sh", shell=True)
            subprocess.check_call("ls ~/bin/bazel", shell=True)
        
        if subprocess.run("pip show pygloo > /dev/null", shell=True).returncode != 0:
            subprocess.check_call(f"cd {cache_dir}/pygloo; python setup.py install", shell=True)

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

    def get_hostname_with_rank(self):
        return (self.rank, self.get_hostname())

    def set_hostnames(self, hostnames_by_rank):
        self.hostnames = hostnames_by_rank
        print(self.hostnames)
    
    def clear_redis(self):
        print('clearing redis')
        import redis
        redis_client = redis.Redis(host=self.redis_hostname, port=self.redis_port)
        redis_client.flushall()

    def test_gloo(self, prefix):
        import pygloo

        context = pygloo.rendezvous.Context(self.rank, self.world_size)

        attr = pygloo.transport.tcp.attr(self.get_hostname())
        dev = pygloo.transport.tcp.CreateDevice(attr)
        print('device:', dev)
        store = pygloo.rendezvous.RedisStore(self.redis_hostname, self.redis_port)

        print('connectFullMesh start')
        context.connectFullMesh(store, dev)

        self.test_small_p2p_comm(context)
        self.test_network_bandwidth_without_glue()
        self.test_large_p2p_comm(context)

    def test_small_p2p_comm(self, context):
        import pygloo
        print('Starting small P2P communication')
        tag = 0
        
        value_to_send = np.ones(5, dtype=np.float32)
        if self.rank == 0:
            send_buf = value_to_send
            sendptr = send_buf.ctypes.data
            pygloo.send(context, sendptr, size=5, datatype=pygloo.glooDataType_t.glooFloat32, peer=1, tag=tag)
        else:
            recv_buf = np.zeros(5, dtype=np.float32)
            recvptr = recv_buf.ctypes.data
            pygloo.recv(context, recvptr, size=5, datatype=pygloo.glooDataType_t.glooFloat32, peer=0, tag=tag)
            assert np.allclose(recv_buf, value_to_send), "Small fail"
            print("Small send/recv passed")

    def test_network_bandwidth_without_glue(self):
        if subprocess.run("which iperf3", shell=True).returncode != 0:
            subprocess.run("sudo apt-get install iperf3 -y", shell=True)

        peer_rank = 0 if self.rank == 1 else 1
        peer_ip = self.hostnames[peer_rank]
        self_ip = self.hostnames[self.rank]
        port = 1026

        if self.rank == 0:
            subprocess.check_call(f"sudo iperf3 -s -B {self_ip} -p {port} --one-off", shell=True)
        else:
            time.sleep(1)
            subprocess.check_call(f"sudo iperf3 -c {peer_ip} -p {port} -O 1 --bytes 5G --bidir", shell=True)

        print('iperf3 done')

    def test_large_p2p_comm(self, context):
        import pygloo
        print('Starting large P2P communication')
        tag = 0

        shape =  10 * 2**30
        value_to_send = np.ones(shape, dtype=np.uint8)
        if self.rank == 0:
            send_buf = value_to_send
            sendptr = send_buf.ctypes.data
            for tag in range(5):
                start = time.time()
                pygloo.send(context, sendptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=1, tag=tag)
                dur_s = time.time() - start
                print(f'send took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s')
        else:
            recv_buf = np.empty(shape, dtype=np.uint8)
            recvptr = recv_buf.ctypes.data
            for tag in range(5):
                start = time.time()
                pygloo.recv(context, recvptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=0, tag=tag)
                dur_s = time.time() - start
                print(f'recv took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s')


def schedule_on_node(node, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node['NodeID'], soft=soft
    )

def alive(nodes):
    for node in nodes:
        if node['Alive']:
            yield node

redis_actor = ray.get_actor(name='redis_actor', namespace='redis_actor_namespace')
redis_hostname, redis_port = ray.get(redis_actor.get_connection_info.remote())

alive_nodes = list(alive(ray.nodes()))

state_manager_actors = [
    NodeStateManager.options(
        scheduling_strategy=schedule_on_node(node)
    ).remote() for node in alive_nodes
]

ray.get([a.setup.remote() for a in state_manager_actors])

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

ray.get(actors[0].clear_redis.remote())
hostnames = {rank: hostname for rank, hostname in ray.get([a.get_hostname_with_rank.remote() for a in actors])}
ray.get([a.set_hostnames.remote(hostnames) for a in actors])
ray.get([actor.test_gloo.remote(prefix) for actor in actors])

#while True:
#    import time
#    time.sleep(1)

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
