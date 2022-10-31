#!/usr/bin/env python3


def basic():
    import cupy as cp
    tensor = cp.random.rand(100)
    print(tensor)

def ray():
    import cupy as cp
    import ray
    import uuid
    
    @ray.remote(num_gpus=0.1)
    def creator_task(size, output_gpu_tensor):
        tag = ('gpu' if output_gpu_tensor else 'cpu') + f':{size*4}B'
        cp.cuda.nvtx.RangePush(f'creator_task_{tag}')
        tensors = [cp.random.rand(size, dtype=cp.float32) for _ in range(1)]
        #tensors = [cp.asnumpy(tensor) for tensor in tensors]

        if output_gpu_tensor:
            output = tensors[0]
        else:
            # Approximate size of handle
            output = str(uuid.uuid4())

        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePush(f'between_tasks_{tag}')
        return output

    @ray.remote(num_gpus=0.1)
    def printer_task(tensor):
        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePush('printer_task')
        #print(tensor)
        #print(tensor)

        #print(tensor)
        #cpu_tensor = tensor
        #gpu_tensor = cp.asarray(cpu_tensor)
        cp.cuda.nvtx.RangePop()

    for _ in range(128):
        ray.get(printer_task.remote(creator_task.remote(1, False)))
        ray.get(printer_task.remote(creator_task.remote(1, True)))

    for _ in range(128):
        ray.get(printer_task.remote(creator_task.remote((1 << 10) // 4, False)))
        ray.get(printer_task.remote(creator_task.remote((1 << 10) // 4, True)))

    for _ in range(128):
        ray.get(printer_task.remote(creator_task.remote((1 << 20) // 4, False)))
        ray.get(printer_task.remote(creator_task.remote((1 << 20) // 4, True)))
    
ray()
