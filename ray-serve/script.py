#!/usr/bin/env python3

import subprocess
if False:
    # Not sure if I need scipy, diffusers, ftfy, einops
    subprocess.check_call(
        "pip install -U fastertransformer transformers accelerate 'numpy<1.24' deepspeed diffusers scipy ftfy einops flash-attn==1.0.3.post0 triton==2.0.0.dev20221202",
        shell=True
    )

import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import io
import os
import ray
import time
import torch

def gen_random_prompts(tokenizer, vocab_range, num_context_tokens, num_prompts):
    import random
    
    random.seed(0xCADE)
    prompts = []
    for _ in range(num_prompts):
        input_ids = [
            random.randint(*vocab_range)
            for _ in range(num_context_tokens)
        ]
        prompt = tokenizer.decode(input_ids)
        prompts.append(prompt)

    return prompts

@ray.remote
def run_experiments():
    # Attention config
    attn_impls = ['torch']
    assert all(attn_impl in ['torch', 'triton', 'flash'] for attn_impl in attn_impls)
    
    # Alibi enables longer sequence lengths,
    # not supported by flash
    use_alibi = [False]

    batch_sizes = [1, 2, 4, 8, 16]
    test_duration_s = 60

    context_tokens = 512
    new_tokens = 64

    use_random_prompt = False
    use_full_vocab_in_random_prompt = True

    dtypes = [torch.bfloat16, torch.float16]

    #model_to_use = "cerebras/Cerebras-GPT-111M"
    #model_to_use = "cerebras/Cerebras-GPT-6.7B"
    #model_to_use = "EleutherAI/gpt-neo-125m"
    #model_to_use = "facebook/opt-13b"

    model_to_use = "mosaicml/mpt-7b-instruct"
    tokenizer_to_use = "EleutherAI/gpt-neox-20b"

    #model_to_use = "EleutherAI/gpt-neo-125m"
    #tokenizer_to_use = "EleutherAI/gpt-neo-125m"

    from collections import OrderedDict

    experiment_configs = []
    for bs in batch_sizes:
        for dtype in dtypes:
            for attn_impl in attn_impls:
                for use_alibi_ in use_alibi:
                    experiment_descriptor = OrderedDict(
                        model_to_use=model_to_use,
                        tokenizer_to_use=tokenizer_to_use,
                        attn_impl=attn_impl,
                        use_alibi=use_alibi_,
                        bs=bs,
                        test_duration_s=test_duration_s,
                        context_tokens=context_tokens,
                        new_tokens=new_tokens,
                        use_random_prompt=use_random_prompt,
                        use_full_vocab_in_random_prompt=use_full_vocab_in_random_prompt,
                        dtype=dtype,
                    )
                    experiment_configs.append(experiment_descriptor)


    print(f'Running {len(experiment_configs)} experiments')

    stagger_runs = False
    if stagger_runs:
        num_gpus = 8
        saturation = 0
        experiment_results = []
        to_stagger = experiment_configs[:num_gpus]
        remaining = experiment_configs[num_gpus:]

        for config in to_stagger:
            experiment_results.append(run.remote(config))
            import random
            import time
            time.sleep(random.uniform(10, 30))
        experiment_results += [run.remote(config) for config in remaining]
    else:
        experiment_results = [run.remote(config) for config in experiment_configs]

    experiment_results = ray.get(experiment_results)

    return experiment_results


@ray.remote(num_gpus=1)
def run(experiment_config):
    import time
    start_time = time.time()
    print(f'{time.time() - start_time:.02f} starting')

    
    attn_impl = experiment_config['attn_impl']
    use_alibi = experiment_config['use_alibi']
    bs = experiment_config['bs']
    test_duration_s = experiment_config['test_duration_s']
    context_tokens = experiment_config['context_tokens']
    new_tokens = experiment_config['new_tokens']
    dtype = experiment_config['dtype']
    use_random_prompt = experiment_config['use_random_prompt']
    use_full_vocab_in_random_prompt = experiment_config['use_full_vocab_in_random_prompt']
    model_to_use = experiment_config['model_to_use']
    tokenizer_to_use = experiment_config['tokenizer_to_use']

    import os

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
        config.attn_config['attn_impl'] = attn_impl
        config.attn_config['alibi'] = use_alibi
    else:
        assert attn_impl == 'torch'
        assert use_alibi == False

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
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
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

    total_tokens = context_tokens + new_tokens

    def data_gen():
        if not use_random_prompt:
            prompt = ("The " * context_tokens)[:-1]
            while True:
                yield prompt
        else:

            if use_full_vocab_in_random_prompt:
                vocab_range = (1, config.vocab_size-1)
            else:
                vocab_range = (1, 512)
            
            prompts = gen_random_prompts(
                tokenizer,
                vocab_range=vocab_range,
                num_context_tokens=context_tokens,
                num_prompts=25
            )
            i = 0
            while True:
                prompt = prompts[i % len(prompts)]
                try:
                    yield prompt
                except:
                    # TODO(cade) there is some bug that causes MPT to fail
                    # I suspect the padding token is being generated.
                    # Should either fix the padding token or improve
                    # random prompt generation..
                    print(f'Exception, prompt was "{prompt}"')
                    print(f'tokenized: {tokenizer(prompt)}')
                    raise
                i += 1


    memory_stats = {}
    torch.cuda.reset_peak_memory_stats()
    memory_stats['before'] = {
        'memory_allocated': torch.cuda.memory_allocated(),
        'memory_reserved': torch.cuda.memory_reserved(),
    }

    count = 0
    measure_start_time = time.time()
    for out in pipe(
        data_gen(),
        do_sample=True,
        min_length=total_tokens,
        max_length=total_tokens,
        batch_size=bs,
    ):
        print(f'{time.time() - measure_start_time:.02f} Inference done, count {count}')
        count += 1

        dur_s = time.time() - measure_start_time
        if dur_s >= test_duration_s:
            break

    memory_stats['after'] = {
        'memory_allocated': torch.cuda.memory_allocated(),
        'max_memory_allocated': torch.cuda.max_memory_allocated(),
        'memory_reserved': torch.cuda.memory_reserved(),
        'max_memory_reserved': torch.cuda.max_memory_reserved(),
    }
    memory_stats['after'] = {k: v / 2**30 for k, v in memory_stats['after'].items()}
    memory_stats['before'] = {k: v / 2**30 for k, v in memory_stats['before'].items()}
    print(f'max_memory_reserved {memory_stats["after"]["max_memory_reserved"]:.02f}')

    tok_per_s = count * bs * total_tokens / dur_s

    print(f'{time.time() - start_time:.02f} printing results')
    print(f'dur_s {dur_s:.02f} count {count} bs {bs} tok/s {tok_per_s:.02f}')

    from collections import OrderedDict
    results_descriptor = OrderedDict(
        tok_per_s=tok_per_s,
        max_memory_reserved_gib=memory_stats["after"]["max_memory_reserved"],
    )
    
    print(experiment_config)
    return (experiment_config, results_descriptor)

experiment_results = ray.get(run_experiments.remote())

print('Got results')
for config, results in experiment_results:
    print(config)
    print(results)
