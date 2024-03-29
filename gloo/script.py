#!/usr/bin/env python3

import ray
import subprocess
import time
import numpy as np
import socket

ray.init()

@ray.remote(num_cpus=0)
class NodeStateManager:
    def setup(self):
        cache_dir = "/home/ray/.cache/gloo-deps-src"
        # htop iproute2
        if subprocess.run(f"ls {cache_dir} > /dev/null", shell=True).returncode != 0:
            subprocess.check_call(f"mkdir {cache_dir}", shell=True)
        
        if subprocess.run(f"ls {cache_dir}/ray > /dev/null", shell=True).returncode != 0:
            print('Shallow clone ray')
            subprocess.check_call(
                f"git clone --depth 1 https://github.com/ray-project/ray.git "
                f"{cache_dir}/ray",
                shell=True
            )

        if subprocess.run("ls ~/bin/bazel > /dev/null", shell=True).returncode != 0:
            # install bazel
            subprocess.check_call("sudo apt-get install curl", shell=True)
            subprocess.check_call(f"cd {cache_dir}/ray; bash ./ci/env/install-bazel.sh", shell=True)
            subprocess.check_call("ls ~/bin/bazel", shell=True)

        # Compiling fork
        require_recompile = False
        if require_recompile:
            subprocess.run(f"rm -rf {cache_dir}/pygloo", shell=True)

        if subprocess.run(f"ls {cache_dir}/pygloo > /dev/null", shell=True).returncode != 0:
            print('Shallow clone pygloo')
            subprocess.check_call(
                # Has my modifications
                f"git clone --depth 1 https://github.com/cadedaniel/pygloo.git "
                f"{cache_dir}/pygloo",
                shell=True
            )
        
        if subprocess.run("pip show pygloo > /dev/null", shell=True).returncode != 0 or require_recompile:
            subprocess.check_call(f"cd {cache_dir}/pygloo; python setup.py install", shell=True)

        ## This negatively impacts throughput. TCP window scaling allows OS to go beyond 65k window size.
        #use_max_tcp_window_size = True
        #if use_max_tcp_window_size:
        #    # https://serverfault.com/a/778178/381697
        #    subprocess.run("sudo bash -c \"echo 1 > /proc/sys/net/ipv4/tcp_window_scaling\"", shell=True)

        ## This has no impact on throughput. I'm not sure if TCP window scaling is able to go higher with this.
        #use_max_tcp_window_size = True
        #if use_max_tcp_window_size:
        #    # https://serverfault.com/questions/778503/possible-to-force-tcp-window-scaling-to-a-higher-value
        #    subprocess.run("sudo bash -c 'echo 33554432 > /proc/sys/net/core/rmem_max'", shell=True)
        #    subprocess.run("sudo bash -c 'echo \"4096 33554432 33554432\" > /proc/sys/net/ipv4/tcp_rmem'", shell=True)

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
        import os
        print('niceness', os.nice(0))
        subprocess.run(f"sudo renice 0 -p {os.getpid()}", shell=True)
        print('niceness', os.nice(0))

        import pygloo

        context = pygloo.rendezvous.Context(self.rank, self.world_size)

        attr = pygloo.transport.tcp.attr(self.get_hostname())
        dev = pygloo.transport.tcp.CreateDevice(attr)
        print('device:', dev, 'speed:', dev.getInterfaceSpeed())
        store = pygloo.rendezvous.RedisStore(self.redis_hostname, self.redis_port)

        print('connectFullMesh start')
        context.connectFullMesh(store, dev)

        self.test_small_p2p_comm(context)
        self.test_network_bandwidth_without_glue()
        self.test_large_p2p_comm(context)
        #self.test_multi_thread_comm(context)

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
        if subprocess.run("which iperf3 > /dev/null", shell=True).returncode != 0:
            subprocess.run("sudo apt-get install iperf3 -y", shell=True)

        peer_rank = 0 if self.rank == 1 else 1
        peer_ip = self.hostnames[peer_rank]
        self_ip = self.hostnames[self.rank]
        port = 1026

        report_interval_s = 5
        total_time_s = 5
        if self.rank == 0:
            subprocess.check_call(
                f"sudo iperf3 -s -B {self_ip} -p {port} --one-off -i {report_interval_s}",
                shell=True,
            )
        else:
            time.sleep(1)
            subprocess.check_call(
                f"sudo iperf3 -c {peer_ip} -p {port} -O 1 --time {total_time_s} -P 6 -i {report_interval_s}",
                shell=True,
            )

        print('iperf3 done')


    def test_multi_thread_comm(self, context):
        import concurrent.futures
        import pygloo

        repeats = 20
        def send_recv(base_tag, repeats):
            #shape =  10 * 2**30
            shape = 40 * 2**20

            #print(f'Starting larger P2P communication (size={shape/2**20:.02f}MB)')

            value_to_send = np.ones(shape, dtype=np.uint8)
            if self.rank == 0:
                send_buf = value_to_send
                sendptr = send_buf.ctypes.data
                for tag in range(repeats):
                    tag = base_tag + tag
                    print(f'send tag {tag}\n', end='')
                    start = time.time()
                    pygloo.send(context, sendptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=1, tag=tag)
                    dur_s = time.time() - start
                    print(f'send took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s\n', end='')
            else:
                recv_buf = np.empty(shape, dtype=np.uint8)
                recvptr = recv_buf.ctypes.data
                for tag in range(repeats):
                    tag = base_tag + tag
                    print(f'recv tag {tag}\n', end='')
                    start = time.time()
                    pygloo.recv(context, recvptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=0, tag=tag)
                    dur_s = time.time() - start
                    print(f'recv took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s\n', end='')

        print('Starting multi-thread-comm')
        pool_size = 2
        comm_repeats = 200 * 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
            futures = [pool.submit(send_recv, i * comm_repeats, comm_repeats) for i in range(pool_size)]
            [f.result() for f in futures]
        print('multi-thread-comm done')
            

    def test_large_p2p_comm(self, context):
        # This might have issues on my fork of pygloo (we release gil)
        import pygloo
        tag = 0

        shape =  10 * 2**30
        #shape = 40 * 2**20
        repeats = 20

        print(f'Starting larger P2P communication (size={shape/2**20:.02f}MB)')

        value_to_send = np.ones(shape, dtype=np.uint8)
        if self.rank == 0:
            send_buf = value_to_send
            sendptr = send_buf.ctypes.data
            for tag in range(repeats):
                start = time.time()
                pygloo.send(context, sendptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=1, tag=tag)
                dur_s = time.time() - start
                print(f'send took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s')
        else:
            recv_buf = np.empty(shape, dtype=np.uint8)
            recvptr = recv_buf.ctypes.data
            for tag in range(repeats):
                start = time.time()
                pygloo.recv(context, recvptr, size=shape, datatype=pygloo.glooDataType_t.glooUint8, peer=0, tag=tag)
                dur_s = time.time() - start
                print(f'recv took {dur_s:.02f} s, {(8 * (shape/2**30)/ dur_s):.02f} Gbit/s')

@ray.remote(num_cpus=0)
class RedisActor:
    def __init__(self):
        self.redis_server_proc = None
        self.port = None

    def start(self, port):
        subprocess.run("sudo apt-get install redis -y", shell=True)
        subprocess.run("pkill redis-server", shell=True)
        self.redis_server_proc = subprocess.Popen(f"redis-server --port {port} --protected-mode no", shell=True)
        self.port = port
        
        import redis
        # TODO add loop here
        redis_client = redis.Redis(host='localhost', port=self.port)

        return socket.gethostname()

    def get_connection_info(self):
        return socket.gethostname(), self.port

    def stop(self):
        print('Killing redis')
        self.redis_server_proc.kill()
        self.redis_server_proc = None

def get_or_start_redis_actor(name, namespace):
    try:
        redis_actor = ray.get_actor(name=name, namespace=namespace)
    except ValueError as e:
        if not 'Failed to look up actor with name' in str(e):
            raise
        print('redis actor not found, recreating')
        redis_actor = RedisActor.options(
            name=name,
            namespace=namespace,
            lifetime='detached',
        ).remote()
        ray.get(redis_actor.start.remote(port=7777))
    return redis_actor

def schedule_on_node(node, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node['NodeID'], soft=soft
    )

def alive(nodes):
    for node in nodes:
        if node['Alive']:
            yield node

redis_actor_name = 'redis_actor'
redis_actor_namespace = 'redis_actor_namespace'

redis_actor = get_or_start_redis_actor(redis_actor_name, redis_actor_namespace)
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
        #scheduling_strategy=schedule_on_node(node),
        num_gpus=1 if rank == 0 else 0,
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

'''
Getting about 8.8 Gbit/s communication.
There are two issues I see with pygloo:
    * currently sends are sync, so only single thread can be used
    * currently creates send buffer each time (instead of pinned memory)

not that hard to fix these things. but i wonder if using different lib would be easier.

ucx and libfabric are good but I think another level of optimization
I should start with pygloo on multiple threads (I can only saturate 25Gbit/s link
with 4 iperf streams). if I can get to ~25Gbit/s that way, then done (until RDMA).
Otherwise will have to use something else (ucx looks easier).

---

Before moving to ucx or libfabric, I should add a method to pygloo which calls send but not waitcompletion.
We can keep track of the outstanding bytes to limit them. Hm, I doubt 10GB is gunna be bottlenecked by our wait completion.
'''
