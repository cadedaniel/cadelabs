#!/usr/bin/env python3


"""
Remaining work:
    * Benchmark.
    * Shim.
    * Improve allocator to support sizes that are larger than block size.
"""

import click
from collections.abc import Iterable
import time
import ray
import sys
import cupy as cp
import numpy as np
import os
import math

skip_stream_synchronization = True  # will lose correctness
skip_sync_free_offset = True
alloc_ahead_window = 2

ray.init(address="auto")


class SameSizeAllocator:
    def __init__(self, size, total_size):
        self._block_size = size
        self._next_offset = 0
        self._total_size = total_size
        self._available = []

    def get_offset(self):
        if not self._available:
            self._alloc_new()
        return self._available.pop()

    def _alloc_new(self):
        if self._next_offset + self._block_size > self._total_size:
            raise ValueError(f"not enough memory in {type(self)}")
        self._available.append(self._next_offset)
        self._next_offset += self._block_size

    @property
    def available(self):
        return len(self._available)

    def free(self, offset):
        self._available.append(offset)


def test_allocator():
    print("test_allocator")
    num_allocs = 1024
    block_size = 128
    s = SameSizeAllocator(block_size, block_size * num_allocs)

    offsets = [s.get_offset() for _ in range(num_allocs)]
    assert offsets[1] - offsets[0] == block_size

    try:
        offsets.append(s.get_offset())
    except ValueError:
        pass
    assert len(offsets) == num_allocs
    assert s.available == 0

    [s.free(offsets.pop()) for _ in range(num_allocs)]
    assert s.available == num_allocs


@ray.remote(num_gpus=0.1)
class GpuCommBuffer:
    def __init__(self, block_size, buf_size):
        # print('running on:', os.environ['CUDA_VISIBLE_DEVICES'])
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        self._buf_size = buf_size
        self._buffer = cp.cuda.alloc(self._buf_size)
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._buffer.ptr)
        self._allocator = SameSizeAllocator(block_size, self._buf_size)

    def get_ipc_handle(self):
        return self._ipc_handle

    def get_bounce_buffer(self, batch_size=1):
        """
        TODO add variable size
        """
        batch = [self._allocator.get_offset() for _ in range(batch_size)]
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def size(self):
        return self._buf_size

    def free_bounce_buffer(self, offset):
        self._allocator.free(offset)

    def free(self):
        cp.cuda.runtime.free(self._buffer.ptr)
        self._buffer = None


def test_gpu_comm_buffer():
    print("test_gpu_comm_buffer")
    comm_buffer = GpuCommBuffer.remote(1 << 20, 1 << 30)
    assert type(ray.get(comm_buffer.get_ipc_handle.remote())) == bytes

    offsets = [
        ray.get(offset)
        for offset in [comm_buffer.get_bounce_buffer.remote() for _ in range(128)]
    ]
    [comm_buffer.free_bounce_buffer.remote(offset) for offset in offsets]
    ray.get(comm_buffer.free.remote())


class IpcEventSource:
    """
    TODO need to add a callback on the receiver side to free these
    also to free the offset
    """

    def __init__(self):
        self._events = []

    def get_event(self):
        if not self._events:
            self._create_event()
        return self._events.pop()

    def _create_event(self):
        new_event = cp.cuda.Event(
            block=False, disable_timing=True, interprocess=True)
        self.recycle(new_event)

    def recycle(self, event):
        self._events.append(event)


class BounceBuffer:
    def __init__(self, device, gpu_comm_buffer: GpuCommBuffer):
        self._comm_buffer = gpu_comm_buffer
        self._ipc_handle = None
        self._buffer = None
        self._device = device
        self._pid = os.getpid()  # This assumes one bounce buffer per process.
        self._event_source = IpcEventSource()

        self._preallocated_offsets = []
        self._preallocated_offsets_max = alloc_ahead_window

    def copy_into_gpu_bounce_buffer(self, tensor):
        with cp.cuda.Device(self._device):
            base = self._get_base_offset()
            offset = self._get_new_offset()

            source = tensor.data
            destination = base + offset
            size = tensor.nbytes

            destination.copy_from_device(source, size)

            # event = self._event_source.get_event()
            # event.record(cp.cuda.get_current_stream())
            # event_ipc_handle = cp.cuda.runtime.ipcGetEventHandle(event.ptr)
            event_ipc_handle = None

            if not skip_stream_synchronization:
                stream = cp.cuda.get_current_stream()
                cp.cuda.runtime.streamSynchronize(stream.ptr)

        return (offset, tensor.shape, size, tensor.dtype, self._pid, event_ipc_handle)

    def copy_from_gpu_bounce_buffer(self, tensor_meta):
        offset, shape, nbytes, dtype, other_pid, event_ipc_handle = tensor_meta

        # TODO open ipc handle, cache?
        # Once we wait on the event, can we tell the sender that the event is complete?

        with cp.cuda.Device(self._device):
            base = self._get_base_offset()

            src = base + offset
            dst = cp.empty(shape=shape, dtype=dtype)
            dst.data.copy_from_device(src, nbytes)

            if not skip_stream_synchronization:
                # TODO replace with event
                stream = cp.cuda.get_current_stream()
                cp.cuda.runtime.streamSynchronize(stream.ptr)

        # TODO need to trigger this only once the cuda event is done.
        # How to do this without polling?
        # Can use cudaLaunchHostFunc but it will block the stream.. maybe this is OK for a
        # signalling function.
        self._free_offset(offset, get=not skip_sync_free_offset)

        return dst

    def _get_new_offset(self):
        if not len(self._preallocated_offsets):
            self._preallocated_offsets = ray.get(
                self._comm_buffer.get_bounce_buffer.remote(
                    self._preallocated_offsets_max
                )
            )
            if not isinstance(self._preallocated_offsets, Iterable):
                self._preallocated_offsets = [self._preallocated_offsets]

        return self._preallocated_offsets.pop(0)

    def _get_base_offset(self):
        if not self._ipc_handle:
            self._map_memory()
        return self._buffer

    def _free_offset(self, offset, get=True):
        future = self._comm_buffer.free_bounce_buffer.remote(offset)
        if get:
            return ray.get(future)

    def _map_memory(self):
        assert not self._ipc_handle
        self._ipc_handle = ray.get(self._comm_buffer.get_ipc_handle.remote())

        mapped = cp.cuda.BaseMemory()
        mapped.ptr = cp.cuda.runtime.ipcOpenMemHandle(self._ipc_handle)
        mapped.size = ray.get(self._comm_buffer.size.remote())
        mapped.device_id = self._device
        self._buffer = cp.cuda.MemoryPointer(mapped, 0)


def test_copy_into():
    print("test_copy_into")
    device = 0
    with cp.cuda.Device(device):
        tensor = cp.ones(1 << 20, dtype=np.uint8)

    comm_buffer = GpuCommBuffer.remote(1 << 20, 1 << 30)
    bb = BounceBuffer(device, comm_buffer)

    pid = os.getpid()

    # I think the order of offsets is reversed.
    assert bb.copy_into_gpu_bounce_buffer(tensor)[:-1] == (
        0,
        (1 << 20,),
        1 << 20,
        np.uint8,
        pid,
    )
    assert bb.copy_into_gpu_bounce_buffer(tensor)[:-1] == (
        1 << 20,
        (1 << 20,),
        1 << 20,
        np.uint8,
        pid,
    )


def test_same_proc_bounce_buffer():
    print("test_same_proc_bounce_buffer")
    device = 0
    with cp.cuda.Device(device):
        tensor = cp.ones(1 << 20, dtype=np.uint8)

    comm_buffer = GpuCommBuffer.remote(1 << 20, 1 << 30)
    bb = BounceBuffer(device, comm_buffer)

    meta = bb.copy_into_gpu_bounce_buffer(tensor)
    output_tensor = bb.copy_from_gpu_bounce_buffer(meta)

    input_as_cpu = cp.asnumpy(tensor)
    output_as_cpu = cp.asnumpy(output_tensor)
    np.testing.assert_equal(input_as_cpu, output_as_cpu)


def test_poc_copy_into_buffer():
    print("test_poc_copy_into_buffer")
    copy_size = 1 << 20
    buffer_size = 1 << 30
    dtype = np.uint8

    with cp.cuda.Device(0):
        src = cp.ones(copy_size, dtype=dtype)
        dst = cp.cuda.alloc(buffer_size)
        dst.memset(0, copy_size)
        dst.copy_from_device(src.data, copy_size)
        cp.cuda.runtime.deviceSynchronize()

    # assert correct copy
    gpu_copied = cp.ndarray(shape=copy_size, dtype=dtype, memptr=dst)
    cpu_copied = cp.asnumpy(gpu_copied)
    np.testing.assert_equal(cpu_copied, np.ones(copy_size, dtype=dtype))


def test_poc_copy_from_buffer():
    print("test_poc_copy_from_buffer")
    copy_size = 1 << 20
    buffer_size = 1 << 30
    dtype = np.uint8

    with cp.cuda.Device(0):
        src = cp.cuda.alloc(buffer_size)
        src.memset(1, copy_size)
        dst = cp.empty(shape=copy_size, dtype=dtype)
        dst.data.memset(0, copy_size)
        dst.data.copy_from_device(src, copy_size)
        cp.cuda.runtime.deviceSynchronize()

    cpu_copied = cp.asnumpy(dst)
    np.testing.assert_equal(cpu_copied, np.ones(copy_size, dtype=dtype))


@ray.remote(num_gpus=0.1)
class TestMapMemory:
    def __init__(self, gpu_comm_buffer):
        self._comm_buffer = gpu_comm_buffer

    def map_memory(self):
        ipc_handle = ray.get(self._comm_buffer.get_ipc_handle.remote())
        mapped_buf = cp.cuda.runtime.ipcOpenMemHandle(ipc_handle)
        assert type(mapped_buf) == int


def test_map_memory():
    print("test_map_memory")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)
    mapper = TestMapMemory.remote(gpu_comm_buffer)

    ray.get(mapper.map_memory.remote())


def test_cross_proc_bounce_buffer():
    print("test_cross_proc_bounce_buffer")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)

    p = Producer.remote(0, gpu_comm_buffer)
    c = Consumer.remote(0, gpu_comm_buffer)

    meta = ray.get(p.produce.remote())
    ray.get(c.consume.remote(meta))


def test_local_worker_bounce_buffer():
    print("test_local_worker_bounce_buffer")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)

    p = Producer.remote(0, gpu_comm_buffer)

    bb = BounceBuffer(0, gpu_comm_buffer)

    def consume(tensor_meta):
        output_tensor = bb.copy_from_gpu_bounce_buffer(tensor_meta)
        return inner_consume(output_tensor)

    def inner_consume(tensor):
        cpu_tensor = cp.asnumpy(tensor)
        np.testing.assert_equal(cpu_tensor, np.ones(1 << 10, dtype=np.uint8))

    meta = ray.get(p.produce.remote())
    consume(meta)


@ray.remote(num_gpus=0.1)
class Producer:
    def __init__(self, device, comm_buffer):
        self._device = device
        self._bb = BounceBuffer(self._device, comm_buffer)

    def produce(self):
        rval = self.inner_produce()
        meta = self._bb.copy_into_gpu_bounce_buffer(rval)
        return meta

    def inner_produce(self):
        return cp.ones(1 << 10, dtype=np.uint8)


@ray.remote(num_gpus=0.1)
class Consumer:
    def __init__(self, device, comm_buffer):
        self._device = device
        self._bb = BounceBuffer(self._device, comm_buffer)

    def consume(self, tensor_meta):
        output_tensor = self._bb.copy_from_gpu_bounce_buffer(tensor_meta)
        return self.inner_consume(output_tensor)

    def inner_consume(self, tensor):
        cpu_tensor = cp.asnumpy(tensor)
        np.testing.assert_equal(cpu_tensor, np.ones(1 << 10, dtype=np.uint8))


def test_bidirectional_cross_proc_bounce_buffer():
    print("test_bidirectional_cross_proc_bounce_buffer")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    meta = ray.get(p1.produce.remote(shape=1 << 10, dtype=np.uint8))
    ray.get(p2.consume.remote(meta))

    meta = ray.get(p2.produce.remote(shape=1 << 10, dtype=np.uint8))
    ray.get(p1.consume.remote(meta))


def test_different_shape_and_dtype():
    print("test_different_shape_and_dtype")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    meta = ray.get(p1.produce.remote(shape=[10, 20], dtype=np.float16))
    output_tensor = ray.get(p2.consume.remote(meta))
    np.testing.assert_equal(output_tensor, np.ones([10, 20], dtype=np.float16))

    meta = ray.get(p2.produce.remote(shape=100, dtype=np.float64))
    output_tensor = ray.get(p1.consume.remote(meta))
    np.testing.assert_equal(output_tensor, np.ones(100, dtype=np.float64))


@ray.remote(num_gpus=0.1)
class ProducerConsumer:
    def __init__(self, device, comm_buffer):
        self._device = device
        self._bb = BounceBuffer(self._device, comm_buffer)

    def consume(self, tensor_meta, return_cpu_tensor=True):
        output_tensor = self._bb.copy_from_gpu_bounce_buffer(tensor_meta)
        return self.inner_consume(output_tensor, return_cpu_tensor)

    def inner_consume(self, tensor, return_cpu_tensor):
        if return_cpu_tensor:
            return cp.asnumpy(tensor)
        else:
            return

    def produce(self, shape, dtype, populate=True):
        rval = self.inner_produce(shape, dtype, populate)
        meta = self._bb.copy_into_gpu_bounce_buffer(rval)
        return meta

    def inner_produce(self, shape, dtype, populate):
        if populate:
            return cp.ones(shape, dtype=dtype)
        else:
            return cp.empty(shape, dtype=dtype)

    def produce_without_bb(self, shape, dtype, populate):
        tensor = self.inner_produce(shape, dtype, populate)
        cpu_tensor = cp.asnumpy(tensor)
        return cpu_tensor

    def consume_without_bb(self, tensor, return_cpu_tensor):
        gpu_tensor = cp.asarray(tensor)
        return self.inner_consume(gpu_tensor, return_cpu_tensor)

    def produce_batch(self, use_bb, batch_size, shape, dtype, populate):
        if use_bb:
            return [self.produce(shape, dtype, populate) for _ in range(batch_size)]
        else:
            return [self.produce_without_bb(shape, dtype, populate) for _ in range(batch_size)]

    def consume_batch(self, use_bb, tensors, return_cpu_tensor):
        if use_bb:
            return [self.consume(tensor, return_cpu_tensor) for tensor in tensors]
        else:
            return [self.consume_without_bb(tensor, return_cpu_tensor) for tensor in tensors]


def test_bidirectional_loop():
    print("test_bidirectional_loop")
    gpu_comm_buffer = GpuCommBuffer.remote(1 << 10, 1 << 20)
    num_bidirectional = 1 << 4

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    for _ in range(num_bidirectional):
        meta = ray.get(p1.produce.remote(shape=[10, 20], dtype=np.float16))
        output_tensor = ray.get(p2.consume.remote(meta))
        np.testing.assert_equal(
            output_tensor, np.ones([10, 20], dtype=np.float16))

        meta = ray.get(p2.produce.remote(shape=100, dtype=np.float64))
        output_tensor = ray.get(p1.consume.remote(meta))
        np.testing.assert_equal(output_tensor, np.ones(100, dtype=np.float64))


def benchmark_bidirectional_loop_helper(use_bb):
    """
    Kourosh says by default tensor size is 3 * 84 * 84 * 1 = 21168
    """
    message_size = 1 << 24
    gpu_comm_buffer = GpuCommBuffer.remote(message_size, 1 << 30)
    num_bidirectional = 16
    num_warmup = 10

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    def test_code():
        if use_bb:
            meta = ray.get(
                p1.produce.remote(shape=message_size,
                                  dtype=np.uint8, populate=False)
            )
            ray.get(p2.consume.remote(meta, False))

            meta = ray.get(
                p2.produce.remote(shape=message_size,
                                  dtype=np.uint8, populate=False)
            )
            ray.get(p1.consume.remote(meta, False))
        else:
            meta = ray.get(
                p1.produce_without_bb.remote(
                    shape=message_size, dtype=np.uint8, populate=False
                )
            )
            ray.get(p2.consume_without_bb.remote(meta, False))

            meta = ray.get(
                p2.produce_without_bb.remote(
                    shape=message_size, dtype=np.uint8, populate=False
                )
            )
            ray.get(p1.consume_without_bb.remote(meta, False))

    for _ in range(num_warmup):
        test_code()

    start_time = time.time()
    for _ in range(num_bidirectional):
        test_code()

    elapsed = time.time() - start_time
    print("Warmup iterations (not measured):", num_warmup)
    print("Total bidirectional communications, in series:", num_bidirectional)
    print("Size of each message (MB):", message_size / (1 << 20))
    print("Total elapsed time (s):", elapsed)
    print(
        "Time per bidirectional communication (ms):",
        1000 * elapsed / (2 * num_bidirectional),
    )
    print()


def benchmark_bidirectional_loop_with_bb():
    print("benchmark_bidirectional_loop_with_bb")
    benchmark_bidirectional_loop_helper(True)


def benchmark_bidirectional_loop_without_bb():
    print("benchmark_bidirectional_loop_without_bb")
    benchmark_bidirectional_loop_helper(False)


def benchmark_batch_loop(use_bb):
    message_size = 1 << 24
    gpu_comm_buffer = GpuCommBuffer.remote(message_size, 1 << 30)
    num_bidirectional = 16
    num_warmup = 10
    batch_size = 20

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    def test_code():
        metas = p1.produce_batch.remote(
            use_bb, batch_size, shape=message_size, dtype=np.uint8, populate=False)
        ray.get(p2.consume_batch.remote(
            use_bb, metas, return_cpu_tensor=False))

        metas = p2.produce_batch.remote(
            use_bb, batch_size, shape=message_size, dtype=np.uint8, populate=False)
        ray.get(p1.consume_batch.remote(
            use_bb, metas, return_cpu_tensor=False))

    for _ in range(num_warmup):
        test_code()

    start_time = time.time()
    for _ in range(num_bidirectional):
        test_code()

    elapsed = time.time() - start_time
    print("Warmup iterations (not measured):", num_warmup)
    print("Total batch communications, in series:", num_bidirectional)
    print("Batch size:", batch_size)
    print("Size of each message (MB):", message_size / (1 << 20))
    print("Total elapsed time (s):", elapsed)
    print(
        "Time per bidirectional communication (ms):",
        1000 * elapsed / (2 * num_bidirectional),
    )
    print()


def benchmark_batch_loop_with_bb():
    print("benchmark_batch_loop_with_bb")
    benchmark_batch_loop(True)


def benchmark_batch_loop_without_bb():
    print("benchmark_batch_loop_without_bb")
    benchmark_batch_loop(False)


def benchmark_demo():
    ray_throughputs = []
    bb_throughputs = []

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    #batch_sizes = [32768]
    #batch_sizes = [65536]
    #batch_sizes = [262144]
    #batch_sizes = [131072]

    for bs in batch_sizes:
        print("Without bounce buffer")
        ray_throughput = benchmark_demo_helper(False, bs)
        ray_throughputs.append(ray_throughput)

        print("With bounce buffer")
        bb_throughput = benchmark_demo_helper(True, bs)
        bb_throughputs.append(bb_throughput)

    print()
    print('\n'.join([str(x) for x in ray_throughputs]))
    print()
    print('\n'.join([str(x) for x in bb_throughputs]))


'''
1
2
4
8
16
32
64
128
256
512
1024
2048
4096
'''


def benchmark_demo_helper(use_bb, batch_size):
    single_batch_nbytes = 3 * 84 * 84
    message_size = single_batch_nbytes * batch_size
    #message_size = 1 << 24
    #message_size = 1 << 32
    gpu_comm_buffer = GpuCommBuffer.remote(message_size, int(math.pow(2, 32)))
    num_iters = 4
    num_warmup = 1

    p1 = ProducerConsumer.remote(0, gpu_comm_buffer)
    p2 = ProducerConsumer.remote(0, gpu_comm_buffer)

    def test_code():
        if use_bb:
            meta = ray.get(
                p1.produce.remote(shape=message_size,
                                  dtype=np.uint8, populate=False)
            )
            ray.get(p2.consume.remote(meta, False))
        else:
            meta = ray.get(
                p1.produce_without_bb.remote(
                    shape=message_size, dtype=np.uint8, populate=False
                )
            )
            ray.get(p2.consume_without_bb.remote(meta, False))

    for _ in range(num_warmup):
        test_code()

    start_time = time.time()
    for _ in range(num_iters):
        test_code()

    elapsed = time.time() - start_time
    print("Warmup iterations (not measured):", num_warmup)
    print("Total communications, in series:", num_iters)
    print("Size of each message (MB):", message_size / (1 << 20))
    print("Batch size of each message:", message_size / single_batch_nbytes)
    print("Total elapsed time (s):", elapsed)
    print(
        "Time per communication (ms):",
        1000 * elapsed / num_iters,
    )
    print()
    return 1000 * elapsed / num_iters


@click.group()
@click.option('--optimized-events/--no-optimized-events', default=False)
@click.option('--use-async-free/--no-use-async-free', default=True)
def cli(optimized_events, use_async_free):
    global skip_stream_synchronization
    global skip_sync_free_offset
    skip_stream_synchronization = optimized_events
    skip_sync_free_offset = use_async_free


@cli.command()
def test():
    test_allocator()
    test_gpu_comm_buffer()
    test_poc_copy_into_buffer()
    test_poc_copy_from_buffer()
    test_copy_into()
    test_same_proc_bounce_buffer()
    test_map_memory()
    test_cross_proc_bounce_buffer()
    test_local_worker_bounce_buffer()
    test_bidirectional_cross_proc_bounce_buffer()
    test_different_shape_and_dtype()
    test_bidirectional_loop()


@cli.command()
def benchmark():
    benchmark_bidirectional_loop_with_bb()
    benchmark_bidirectional_loop_without_bb()
    benchmark_batch_loop_with_bb()
    benchmark_batch_loop_without_bb()


@cli.command()
@click.option('--alloc-ahead-window-size', default=10)
def demo_benchmark(alloc_ahead_window_size):
    global alloc_ahead_window
    alloc_ahead_window = alloc_ahead_window_size
    benchmark_demo()


if __name__ == '__main__':
    sys.exit(cli())
