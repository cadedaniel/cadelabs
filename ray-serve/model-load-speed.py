#!/usr/bin/env python3

import time
start_time = time.time()
print(f'{time.time() - start_time:.02f} starting')

import io
import os
from typing import Optional

CACHE_PATH = "/home/ray/.cache"

"""
Remaining steps:
* Install nvcc. If I am on host, then maybe I kill the container
and reinstall CUDA completely. That way I am not in this weird state
where my apt packages are broken.
* install flash-attn and triton
"""


import os
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH

#model_to_use = "cerebras/Cerebras-GPT-111M"
#model_to_use = "cerebras/Cerebras-GPT-6.7B"
#model_to_use = "EleutherAI/gpt-neo-125m"
#model_to_use = "facebook/opt-13b"
model_to_use = "mosaicml/mpt-7b-instruct"

revision = None
if model_to_use == "mosaicml/mpt-7b-instruct":
    revision = 'bd1748ec173f1c43e11f1973fc6e61cb3de0f327'


from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import deepspeed

print(f'{time.time() - start_time:.02f} autoconfig')

config = AutoConfig.from_pretrained(
    model_to_use,
    trust_remote_code=True,
    cache_dir=CACHE_PATH,
    revision=revision,
)
config.init_device = 'cuda:0'
if model_to_use == 'mosaicml/mpt-7b-instruct':
    config.attn_config['attn_impl'] = 'triton'

print(f'{time.time() - start_time:.02f} model creation')

model = AutoModelForCausalLM.from_pretrained(
    model_to_use,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=CACHE_PATH,
    revision=revision,
)
model = model.eval()

print(f'{time.time() - start_time:.02f} model to GPU')

#print(f'dur_s {dur_s:.02f} count {count} bs {bs}')
