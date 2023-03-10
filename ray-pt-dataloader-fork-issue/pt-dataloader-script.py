#!/usr/bin/env python3

import ray
import time
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


"""
When `kill -SIGTERM` is used, the process stays alive but in cleanup mode.
Each of the torch forked processes dies one-by-one, in a slow fashion.
But they do die.
This was on Ray 2.3, torch 1.13.0+cu116.

things to check:
* multiprocessing.set_start_method('spawn') vs fork (what am I using?)
* is there a inf timeout set in the user code?
* the dataloader implementation sends a final None to the workers
    to tell them to die. They wait for this (indefinitely?).
    They still die if their ppid!=original ppid.
    Is there a case where parent doesn't send None but stays alive?
* Does the dataloader die when ppid is 1, or ppid != old ppid?
    No -- it dies when the ppid changes.

* Does the manager process have multiple threads?


Ricky says:
* When a task creates an actor, and the driver dies, the actor is leaked.
    I tested this, but looks like the forks get reparented.

* When the driver dies, the raylet notices and sends a KillWorker RPC
    to core worker. That will kill the actor ungracefully.

    I tested this, but looks like the forks get reparented.

"""

import subprocess
print('killing previous')
subprocess.run('ps aux | grep Fork | grep Actor | awk \'{print $2}\' | xargs kill -9', shell=True)

@ray.remote(num_gpus=1)
class ForkActor:
    def __init__(self):
        self.dl = None
        self.rows = []
        self.labels = []
        print('ForkActor pid', os.getpid())
        print('ForkActor ppid', os.getppid())
        print('ForkActor cwd', os.getcwd())
        
    def fork(self):
        training_data = datasets.FashionMNIST(
            root="/data/cache/torch-datasets",
            train=True,
            download=True,
            transform=ToTensor()
        )

        local_rank = os.environ['CUDA_VISIBLE_DEVICES']

        self.dl = DataLoader(
            training_data,
            batch_size=64,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            pin_memory_device=f'cuda:{local_rank}',
            prefetch_factor=10,
            persistent_workers=True,
        )
        while True:
            self.rows = []
            self.labels = []
            print(f'Looping over data loader')
            for i, (row, label) in enumerate(self.dl):
                #print(f'iteration {i} of data loader {row.device} pid {os.getpid()}')
                self.rows.append(row.cuda())
                self.labels.append(label.cuda())

@ray.remote
class ProxyActor:
    def __init__(self):
        self.handle = None
    def create(self):
        self.handle = ForkActor.options().remote()
        self.handle.fork.remote()
        return self.handle

@ray.remote
def task():
    pass
    a = ForkActor.options().remote()
    ray.get(a.fork.remote())

#a = ForkActor.options(name='cade', lifetime='detached').remote()
#ray.get(
#a.fork.remote()
proxy = ProxyActor.remote()
handle = ray.get(proxy.create.remote())

for i in range(5):
    print(f'waiting {i}')
    time.sleep(1)

#print('killing')
#ray.kill(proxy, no_restart=True)

while True:
    print('killed')
    time.sleep(1)
