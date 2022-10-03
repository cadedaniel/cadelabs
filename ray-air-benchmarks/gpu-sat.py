#!/usr/bin/env python

import torch

def cuda_sleep():
    # Warm-up CUDA.
    torch.empty(1, device="cuda")

    # From test/test_cuda.py in PyTorch.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1000000)
    end.record()
    end.synchronize()
    cycles_per_ms = 1000000 / start.elapsed_time(end)

    def cuda_sleep(seconds):
        torch.cuda._sleep(int(seconds * cycles_per_ms * 1000))

    return cuda_sleep

def test(overlap_io_and_compute: bool, source_tensor: torch.Tensor):

    compute_stream = torch.cuda.Stream()
    if overlap_io_and_compute:
        io_stream = torch.cuda.Stream()
    else:
        io_stream = compute_stream

    cuda_sleep_estimated_seconds = cuda_sleep()

    io_start = torch.cuda.Event(enable_timing=True)
    io_end = torch.cuda.Event(enable_timing=True)

    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)

    io_data_ready = torch.cuda.Event()
    
    with torch.cuda.stream(io_stream):
        # Schedule IO on the io stream.
        io_start.record()
        dev_tensor = cpu_tensor.to('cuda', non_blocking=True)
        io_end.record()
        io_data_ready.record()

    with torch.cuda.stream(compute_stream):
        # Schedule compute on the compute stream.
        compute_start.record()
        cuda_sleep_estimated_seconds(1)
        compute_end.record()

        # Wait for the IO to be done.
        io_data_ready.wait()
    
    compute_end.synchronize()
    print(f'---- overlap_io_and_compute is {overlap_io_and_compute}')
    print(f'IO time: {io_start.elapsed_time(io_end) / 1000:.02f} s')
    print(f'Compute time: {compute_start.elapsed_time(compute_end) / 1000:.02f} s')
    print(f'End-to-end time: {io_start.elapsed_time(compute_end) / 1000:.02f} s')

cpu_tensor = torch.ones(8 << 30, dtype=torch.uint8, device='cpu').pin_memory()

test(True, cpu_tensor)
test(False, cpu_tensor)
