#!/usr/bin/env python3
import ray
ray.init()


@ray.remote
def task(argument):
    print(argument)

ray.get(task.remote('Hello world'))
