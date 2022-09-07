#!/usr/bin/env python3

import ray
import os
import gc
import time

ray.init(address="localhost:6379")
arg_data_size = 3 * 1 << 20  # 3MB
rval_data_size = 2 * 1 << 20  # 2MB


@ray.remote
def rval_only_worker():
    return os.urandom(rval_data_size)


@ray.remote
def arg_only_worker(x):
    assert len(x) == arg_data_size


@ray.remote
def arg_and_rval_worker(x):
    assert len(x) == arg_data_size
    return os.urandom(rval_data_size)


def test_method(method):
    needs_arg = (
        "arg_only_worker" in method._function_name
        or "arg_and_rval_worker" in method._function_name
    )
    start_mem = ray.available_resources()["object_store_memory"]

    print(
        "index name (available_at_start-available_before_alloc) (available_at_start-available_after_gc)"
    )

    for i in range(5):
        before = ray.available_resources()["object_store_memory"]

        arg = os.urandom(arg_data_size)
        if needs_arg:
            ref = method.remote(arg)
        else:
            ref = method.remote()
        rval = ray.get(ref)

        if rval:
            assert len(rval) == rval_data_size

        del arg
        del rval
        del ref
        gc.collect()

        time.sleep(
            0.1
        )  # Allow Ray async gc to catch up, flaky but works in simple case.

        after = ray.available_resources()["object_store_memory"]

        nice_name = method._function_name.split(".")[1]
        print(
            f"{i} {nice_name}: {int(start_mem-before)//(1 << 20)}MB -> {int(start_mem-after)//(1 << 20)}MB"
        )


test_method(arg_only_worker)
test_method(rval_only_worker)
test_method(arg_and_rval_worker)
