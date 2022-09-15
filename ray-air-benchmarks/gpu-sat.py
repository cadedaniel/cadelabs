#!/usr/bin/env python

import torch
import time



def do_test(with_nonblocking, with_pin):
    print('Nonblocking copy' if with_nonblocking else 'Blocking copy')

    tensor = torch.ones((1024, 1024, 512 // 4), dtype=torch.float32, device='cpu')
    torch.cuda.synchronize()

    print(f'Tensor size {tensor.nelement() * tensor.element_size() / (1 << 20)} MB')
    if with_pin:
        tensor = tensor.pin_memory()
    print(f'tensor is pinned? {tensor.is_pinned()}')
    
    h2d_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    num_iters = 32
    start = time.time()

    for i in range(num_iters):
        dev_tensor = tensor.to('cuda', non_blocking=with_nonblocking)
        #dev_tensor.sum()
        dev_tensor.matmul(dev_tensor[:, :128])

        #with torch.cuda.stream(h2d_stream):
        #    dev_tensor = tensor.to('cuda', non_blocking=with_nonblocking)

        #with torch.cuda.stream(compute_stream):
        #    dev_tensors.append(dev_tensor.sum())

        #dev_tensors.append(tensor.to('cuda', non_blocking=with_nonblocking).sum())

    torch.cuda.synchronize()
    end = time.time()
    duration_ms = (end - start) * 1e3

    print(f'{duration_ms/num_iters:0.2f}ms per iteration')

do_test(with_nonblocking=True, with_pin=True)
do_test(with_nonblocking=False, with_pin=True)
#do_test(with_nonblocking=True, with_pin=False)
#do_test(with_nonblocking=False, with_pin=False)
