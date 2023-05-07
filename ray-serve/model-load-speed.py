#!/usr/bin/env python3

import io
import os
from typing import Optional

CACHE_PATH = "/home/ubuntu/.cache"

def run_stable_diffusion():
"""
Remaining steps:
* Install nvcc. If I am on host, then maybe I kill the container
and reinstall CUDA completely. That way I am not in this weird state
where my apt packages are broken.
* install flash-attn and triton
"""

print('starting')

import time
import os
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH

#model_to_use = "cerebras/Cerebras-GPT-111M"
#model_to_use = "cerebras/Cerebras-GPT-6.7B"
#model_to_use = "EleutherAI/gpt-neo-125m"
#model_to_use = "facebook/opt-13b"


from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained(
    model_to_use,
    trust_remote_code=True,
    cache_dir=CACHE_PATH,
)
#if model_to_use == 'mosaicml/mpt-7b-instruct':
#    config.attn_config['attn_impl'] = 'triton'

with deepspeed.OnDevice(dtype=torch.float16, device="meta"):

model = AutoModelForCausalLM.from_pretrained(
    model_to_use,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=CACHE_PATH,
)
print('sending model to gpu')

model = model.eval()


print(f'dur_s {dur_s:.02f} count {count} bs {bs}')
