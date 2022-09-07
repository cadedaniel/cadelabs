#!/usr/bin/env python3

import ray
import gc
import os

ray.init(address='localhost:6379')

@ray.remote
def work(input_data):
    return os.urandom(1 << 29)

while True:
    input_data = os.urandom(1 << 29)
    ref = work.remote(input_data)
    value = ray.get(ref)
    
    input('computed value')
    del ref
    del value
    input('deleted objrefs')
    gc.collect()
    input('gc ran')
