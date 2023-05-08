#!/usr/bin/env python3

import io
import os
import ray
import time

@ray.remote(num_gpus=1)
def run():
    import time
    start_time = time.time()
    print(f'{time.time() - start_time:.02f} starting')

    import os

    #model_to_use = "cerebras/Cerebras-GPT-111M"
    #model_to_use = "cerebras/Cerebras-GPT-6.7B"
    #model_to_use = "EleutherAI/gpt-neo-125m"
    #model_to_use = "facebook/opt-13b"

    model_to_use = "mosaicml/mpt-7b-instruct"
    tokenizer_to_use = "EleutherAI/gpt-neox-20b"

    #model_to_use = "EleutherAI/gpt-neo-125m"
    #tokenizer_to_use = "EleutherAI/gpt-neo-125m"

    import torch
    dtype = torch.bfloat16

    revision = None
    if model_to_use == "mosaicml/mpt-7b-instruct":
        revision = 'bd1748ec173f1c43e11f1973fc6e61cb3de0f327'

    from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM

    print(f'{time.time() - start_time:.02f} creating config')

    config = AutoConfig.from_pretrained(
        model_to_use,
        trust_remote_code=True,
        revision=revision,
    )
    config.init_device = 'cuda:0'
    if model_to_use == 'mosaicml/mpt-7b-instruct':
        config.attn_config['attn_impl'] = 'triton'

    print(f'{time.time() - start_time:.02f} creating tokenizer')
    print(f'config.eos_token_id is {config.eos_token_id}')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_to_use,
        pad_token_id=config.eos_token_id,
        padding=True,
    )

    print(f'tokenizer.eos_token_id is {tokenizer.eos_token_id}, eos_token {tokenizer.eos_token}')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f'tokenizer.pad_token is {tokenizer.pad_token}')

    print(f'{time.time() - start_time:.02f} creating model')

    model = AutoModelForCausalLM.from_pretrained(
        model_to_use,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        revision=revision,
    )
    model = model.eval()

    print(f'{time.time() - start_time:.02f} creating pipeline')
    pipe = pipeline(
        'text-generation',
        model=model,
        config=config,
        tokenizer=tokenizer,
        device='cuda:0',
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print(f'{time.time() - start_time:.02f} starting inference')

    bs = 1
    test_duration_s = 60

    context_tokens = 512
    new_tokens = 64
    total_tokens = context_tokens + new_tokens

    def data_gen():
        prompt = ("The " * context_tokens)[:-1]
        while True:
            yield prompt

    count = 0
    measure_start_time = time.time()
    for out in pipe(
        data_gen(),
        do_sample=True,
        min_length=total_tokens,
        max_length=total_tokens,
        batch_size=bs,
    ):
        print(f'Inference done, count {count}')
        count += 1

        dur_s = time.time() - measure_start_time
        if dur_s >= test_duration_s:
            break

    tok_per_s = count * bs * total_tokens / dur_s

    print(f'{time.time() - start_time:.02f} printing results')
    print(f'dur_s {dur_s:.02f} count {count} bs {bs} tok/s {tok_per_s:.02f}')

ray.get(run.remote())
