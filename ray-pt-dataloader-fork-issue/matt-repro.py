#!/usr/bin/env python3

import ray
import time
from multiprocessing import Process

@ray.remote
class MyActor:

    def run(self):
       p = Process(target = lambda: time.sleep(10), daemon=True)
       p.start()
       p.join()

actor = MyActor.remote()
ray.get(actor.run.remote())
