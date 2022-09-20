#!/usr/bin/env python3
import ray
ray.init()


@ray.remote
def task(argument):
    import grpc
    import platform
    print(argument, grpc.__version__, platform.python_version())

ray.get(task.remote('Hello world'))
