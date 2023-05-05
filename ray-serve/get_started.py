#import subprocess
#subprocess.run("pip install modal-client", shell=True)
#uodal token new
#import modal

#stub = modal.Stub("example-get-started")
#
#
#@stub.function()
#def square(x):
#    print("This code is running on a remote worker!")
#    return x**2
#
#
#@stub.local_entrypoint()
#def main():
#    print("the square is", square.call(42))

import modal
import io
import os
from typing import Optional

from modal import Image, Secret, SharedVolume, Stub, web_endpoint
stub = Stub("example-stable-diff-bot")

CACHE_PATH = "/root/model_cache"

volume = SharedVolume().persist("stable-diff-model-vol-2")

@stub.function(
    gpu="A100",
    cpu=10.5,
    image=(
        Image.debian_slim()
        .run_commands(
            "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"
        )
        .pip_install("diffusers", "transformers", "scipy", "ftfy", "accelerate")
    ),
    shared_volumes={CACHE_PATH: volume},
    #secret=Secret.from_name("huggingface-secret"),
)
async def run_stable_diffusion():
    import time
    start_time = time.time()
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print(f'imports done {time.time()-start_time:.02f}')

    small_model = "cerebras/Cerebras-GPT-111M"
    full_model = "cerebras/Cerebras-GPT-6.7B"
    model_to_use = small_model

    config = AutoConfig.from_pretrained(model_to_use, cache_dir=CACHE_PATH)
    print(f'config loaded {time.time()-start_time:.02f}')
    model = AutoModelForCausalLM.from_config(config)
    print(f'cpu model instantiated {time.time()-start_time:.02f}')
    model = model.eval()
    print(f'eval mode set {time.time()-start_time:.02f}')
    model = model.to(torch.float16)
    print(f'fp16 set {time.time()-start_time:.02f}')
    model = model.to('cuda:0')
    print(f'sent to gpu {time.time()-start_time:.02f}')

    print(f'memory available {torch.cuda.get_device_properties(0).total_memory / 2**30:.02f}')



@stub.local_entrypoint()
def run():
    run_stable_diffusion.call()
    print('done')

