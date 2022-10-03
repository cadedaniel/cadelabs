#!/usr/bin/env python

import torch


def main():
    """
    $ ./gpu-sat.py
    ---- overlap_io_and_compute is True
    IO time: 0.78 s
    Compute time: 0.85 s
    End-to-end time: 0.86 s
    ---- overlap_io_and_compute is False
    IO time: 1.01 s
    Compute time: 0.96 s
    End-to-end time: 1.98 s
    """

    cpu_tensor = torch.ones(8 << 30, dtype=torch.uint8, device="cpu").pin_memory()

    test(True, cpu_tensor)
    test(False, cpu_tensor)


def test(overlap_io_and_compute: bool, source_tensor: torch.Tensor) -> None:

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

    end_to_end_done = torch.cuda.Event(enable_timing=True)

    io_data_ready = torch.cuda.Event()

    with torch.cuda.stream(io_stream):
        # Schedule IO on the io stream.
        io_start.record()
        dev_tensor = source_tensor.to("cuda", non_blocking=True)
        io_end.record()
        io_data_ready.record()

    with torch.cuda.stream(compute_stream):
        # Schedule compute on the compute stream.
        compute_start.record()
        cuda_sleep_estimated_seconds(1)
        compute_end.record()

        # Wait for the IO to be done.
        io_data_ready.wait()
        end_to_end_done.record()

    end_to_end_done.synchronize()
    print(f"---- overlap_io_and_compute is {overlap_io_and_compute}")
    print(f"IO time: {io_start.elapsed_time(io_end) / 1000:.02f} s")
    print(f"Compute time: {compute_start.elapsed_time(compute_end) / 1000:.02f} s")
    print(f"End-to-end time: {io_start.elapsed_time(end_to_end_done) / 1000:.02f} s")


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


if __name__ == "__main__":
    import sys

    sys.exit(main())
