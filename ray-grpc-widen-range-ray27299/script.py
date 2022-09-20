#!/usr/bin/env python3
import ray
ray.init()


@ray.remote
def task(argument):
    import grpc
    print(argument, grpc.__version__)

ray.get(task.remote('Hello world'))
