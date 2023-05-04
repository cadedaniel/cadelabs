#!/usr/bin/env python3

import subprocess
#subprocess.check_call("pip install -U accelerate 'numpy<1.24' transformers", shell=True)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
import torch
print('imports done')

small_model = "cerebras/Cerebras-GPT-111M"
full_model = "cerebras/Cerebras-GPT-6.7B"

config = AutoConfig.from_pretrained(full_model)
print('config loaded')
model = AutoModelForCausalLM.from_config(config)
print('cpu model instantiated')
model = model.eval()
print('model eval set')
model = model.to(torch.float16)
print('float16 set')
model = model.to('cuda:0')
print('sent to gpu')

input('waiting')
