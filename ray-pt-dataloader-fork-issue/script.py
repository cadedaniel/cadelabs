#!/usr/bin/env python3

import ray
import time
import os

@ray.remote
class ForkActor:
    def __init__(self):
        pass

    def fork(self):
        print('forking')
        pid = os.fork()
        print('fork done')
        if pid == 0:
            return self.do_child()
        else:
            return self.do_parent()

    def do_child(self):
        while True:
            with open('~/workspace-project-cade-dev/ray-pt-dataloader-fork-issue/log', 'a') as f:
                print(f'child pid {os.getpid()} parent {os.getppid()}', file=f)
            time.sleep(1)

    def do_parent(self):
        while True:
            with open('~/workspace-project-cade-dev/ray-pt-dataloader-fork-issue/log', 'a') as f:
                print(f'parent pid {os.getpid()} parent {os.getppid()}', file=f)
            time.sleep(1)

a = ForkActor.remote()
ray.get(a.fork.remote())
