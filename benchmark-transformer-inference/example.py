import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(f'Start, local rank {local_rank} world size {world_size}\n', end='')

#model_name = "EleutherAI/gpt-neo-2.7B"
#model_name = "EleutherAI/gpt-neox-20b"
model_name = "EleutherAI/gpt-neo-125m"

if local_rank == 0 or True:
    generator = pipeline(
        'text-generation',
        model=model_name,
        #device='cpu',
        device=local_rank,
    )
else:
    import time
    time.sleep(123123123)

print('pipeline created')

print('World size', world_size)

generator.model = deepspeed.init_inference(
    generator.model,
    mp_size=world_size,
    dtype=torch.float16,
    replace_with_kernel_inject=False,
    checkpoint=None,
)

string = generator("DeepSpeed is", do_sample=True, min_length=20)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

import time
print(f'Sleeping {local_rank}')
while True:
    time.sleep(1)
