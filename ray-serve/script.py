#!/usr/bin/env python3

import subprocess
#subprocess.check_call("pip install -U accelerate 'numpy<1.24' transformers", shell=True)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
import torch
print('imports done')

config = AutoConfig.from_pretrained("cerebras/Cerebras-GPT-111M")
model = AutoModelForCausalLM.from_config(config)
model = model.to(torch.float16).eval().to('cuda:0')

print('model loaded to gpu')

input('waiting')
