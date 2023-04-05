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
        pass
        subprocess.run("conda env remove -n ucx -y", shell=True)
        subprocess.run("sudo apt-get install libnuma-dev -y", shell=True)
        subprocess.run("sudo apt-get install iproute2 -y", shell=True)
        subprocess.run("ip addr show", shell=True)
        subprocess.run("conda create -n ucx -c conda-forge -c rapidsai ucx-proc=*=cpu ucx ray-default=2.3.0 ucx-py python=3.8 -y", shell=True)


@ray.remote(runtime_env={"conda": "ucx"})
class NodeActor:
    def __init__(self, rank):
        self.rank = rank
        self.server = None
        import os
        os.environ['UCXPY_IFNAME'] = 'ens5'
        os.environ['UCX_TLS'] = 'tcp'
        os.environ['UCX_NET_DEVICES'] = 'ens5'
        os.environ['UCX_LOG_LEVEL'] = 'INFO'

        import logging
        logging.basicConfig(level=logging.DEBUG)

        self.num_iters = 100

    def get_addr(self):
        import ucp
        return (self.rank, ucp.get_address())

    async def start_server(self, port):
        import ucp
        #callback = self.make_echo_server(lambda n: bytearray(n))
        callback = self.make_benchmark_server(lambda n: bytearray(n))
        self.server = ucp.create_listener(callback, port=port)
        print(f'Listening on {ucp.get_address()}:{self.server.port}')

        while True:
            import asyncio
            await asyncio.sleep(0.1)
            #ucp.comm.flush_ep(self.server)
            
        #return (ucp.get_address(), self.server.port)

    async def run_client(self, server_addr, server_port):
        print(f'run client. server is at {server_addr}:{server_port}')
        import ucp
        print('client address', ucp.get_address())
        
        size = 2**30

        #msg = bytearray(b"m" * size)
        msg = np.arange(size, dtype=np.uint8)
        msg_size = np.array([len(msg)], dtype=np.uint64)

        client = await ucp.create_endpoint(server_addr, server_port)
        
        for _ in range(self.num_iters):
            await client.send(msg_size)
            await client.send(msg)
            resp = bytearray(10)
            await client.recv(resp)
            #assert resp == msg
            print(f'client response size {len(resp)}')

    def make_benchmark_server(self, create_empty_data):
        create_empty_data = lambda n: np.empty(n, dtype=np.uint8)
        async def benchmark_server(ep):
            for _ in range(self.num_iters):
                msg_size = np.empty(1, dtype=np.uint64)
                await ep.recv(msg_size)
                msg = create_empty_data(msg_size[0])
                msg_res = bytearray(b"m" * 10)
                await ep.recv(msg)
                await ep.send(msg_res)
                print(f'server size {msg_size}')
            await ep.close()
        return benchmark_server
    
    def make_echo_server(self, create_empty_data):
        async def echo_server(ep):
            msg_size = np.empty(1, dtype=np.uint64)
            await ep.recv(msg_size)
            msg = create_empty_data(msg_size[0])
            await ep.recv(msg)
            await ep.send(msg)
            await ep.close()
        return echo_server

            

def schedule_on_node(node, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node['NodeID'], soft=soft
    )

def alive(nodes):
    for node in nodes:
        if node['Alive']:
            yield node

def run_single_pair():
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
            #scheduling_strategy=schedule_on_node(alive_nodes[0])
        ).remote(rank) for rank, node in enumerate(alive_nodes)
    ]
    
    server_actor, client_actor = actors
    #server_actor, client_actor = list(reversed(actors))
    
    #addrs = {rank: addr for rank, addr in ray.get([a.get_addr.remote() for a in actors])}
    
    from random import randrange
    #print(randrange(10))
    
    server_port = randrange(60000) + 1024
    server_actor.start_server.remote(server_port)
    _, server_addr = ray.get(server_actor.get_addr.remote())
    
    time.sleep(1)
    
    ray.get(client_actor.run_client.remote(server_addr, server_port))
    
    #ray.get([a.test.remote(addrs) for a in actors])

def run_pairs(num_pairs):
    alive_nodes = list(alive(ray.nodes()))
    
    state_manager_actors = [
        NodeStateManager.options(
            scheduling_strategy=schedule_on_node(node)
        ).remote() for node in alive_nodes
    ]
    
    ray.get([a.setup.remote() for a in state_manager_actors])

    pairs = []
    client_run_info = []
    
    for _ in range(num_pairs):
        actors = [
            NodeActor.options(
                scheduling_strategy=schedule_on_node(node)
            ).remote(rank) for rank, node in enumerate(alive_nodes)
        ]
        
        server_actor, client_actor = actors
        
        from random import randrange
        
        server_port = randrange(60000) + 1024
        server_actor.start_server.remote(server_port)
        _, server_addr = ray.get(server_actor.get_addr.remote())

        pairs.extend(actors)
        client_run_info.append((client_actor, server_addr, server_port))
        
    time.sleep(1)
    
    ray.get([c.run_client.remote(server_addr, server_port) for c, server_addr, server_port in client_run_info])
    #ray.get(client_actor.run_client.remote(server_addr, server_port))
    
    #ray.get([a.test.remote(addrs) for a in actors])

#run_single_pair()
run_pairs(3)
