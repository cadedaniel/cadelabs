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
        #Image.debian_slim()
        Image.from_dockerhub(
            "rayproject/ray:2.4.0-gpu",
            #"nvidia/cuda:11.7.0-devel-ubuntu18.04"
        )
        .run_commands(
            "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"
        )
        .pip_install("diffusers", "transformers", "scipy", "ftfy", "accelerate", "einops", "flash-attn==1.0.3.post0", "triton==2.0.0.dev20221202")
    ),
    shared_volumes={CACHE_PATH: volume},
    #secret=Secret.from_name("huggingface-secret"),
    timeout=2*3600,
)
async def run_stable_diffusion():
    print('starting')

    import time
    import os
    os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH

    #model_to_use = "cerebras/Cerebras-GPT-111M"
    #model_to_use = "cerebras/Cerebras-GPT-6.7B"
    #model_to_use = "EleutherAI/gpt-neo-125m"
    #model_to_use = "facebook/opt-13b"

    bs = 1
    if True:

        model_to_use = "mosaicml/mpt-7b-instruct"
        tokenizer_to_use = "EleutherAI/gpt-neox-20b"

        #model_to_use = "EleutherAI/gpt-neo-125m"
        #tokenizer_to_use = "EleutherAI/gpt-neo-125m"

        from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM
        import torch
        #generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

        print('creating config')

        #assert model_to_use == 'mosaicml/mpt-7b-instruct'
        config = AutoConfig.from_pretrained(
            model_to_use,
            trust_remote_code=True,
            cache_dir=CACHE_PATH,
        )
        if model_to_use == 'mosaicml/mpt-7b-instruct':
            config.attn_config['attn_impl'] = 'triton'

        print('creating tokenizer')
        print(f'config.eos_token_id is {config.eos_token_id}')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_to_use,
            cache_dir=CACHE_PATH,
            pad_token_id=config.eos_token_id,
            padding=True,
        )

        print(f'tokenizer.eos_token_id is {tokenizer.eos_token_id}, eos_token {tokenizer.eos_token}')
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f'tokenizer.pad_token is {tokenizer.pad_token}')
        print('creating model')

        model = AutoModelForCausalLM.from_pretrained(
            model_to_use,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=CACHE_PATH,
        )
        print('sending model to gpu')

        model = model.eval()
        #model = model.to(device='cuda:0')

        print('creating pipeline')
        generator = pipeline(
            'text-generation',
            model=model,
            config=config,
            tokenizer=tokenizer,
            device='cuda:0',
            #torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print('starting inference')

        if False:
            prompt = ("The " * 512)[:-1]
            count = 0
            start_time = time.time()
            while True:
                dur_s = time.time() - start_time
                if dur_s >= 60:
                    break

                out = generator(prompt, do_sample=True, min_length=512 + 64, max_length=512 + 64)
                print(out)
                count += 1
        else:
            pipe = generator

            def data_gen():
                prompt = ("The " * 512)[:-1]
                while True:
                    yield prompt
            
            bs = 4
            count = 0
            start_time = time.time()
            for out in pipe(
                data_gen(),
                do_sample=True,
                min_length=512+64,
                max_length=512+64,
                batch_size=bs,
            ):
                print(out)
                count += 1

                dur_s = time.time() - start_time
                if dur_s >= 60:
                    break
    else:
        model_to_use = "EleutherAI/gpt-neo-125m"

        from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM
        import torch
        #generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
        
        print('creating pipeline')
        pipe = pipeline(
            'text-generation',
            model=model_to_use,
            device='cuda:0',
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        def data_gen():
            prompt = ("The " * 512)[:-1]
            while True:
                yield prompt

        count = 0
        start_time = time.time()
        for out in pipe(data_gen(), do_sample=True, min_length=512+64, max_length=512+64):
            #out = generator(prompt, do_sample=True, min_length=512 + 64, max_length=512 + 64)
            print(out)
            count += 1

            dur_s = time.time() - start_time
            if dur_s >= 60:
                break


    print(f'dur_s {dur_s:.02f} count {count} bs {bs}')

    #import time
    #start_time = time.time()
    #import torch
    #from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
    #from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    #from transformers.generation.logits_process import LogitsProcessorList, LogitNormalization
    #print(f'imports done {time.time()-start_time:.02f}')

    ##model_to_use = "cerebras/Cerebras-GPT-111M"
    ##model_to_use = "cerebras/Cerebras-GPT-6.7B"
    ##model_to_use = "EleutherAI/gpt-neo-125m"
    #model_to_use = "facebook/opt-13b"

    #config = AutoConfig.from_pretrained(model_to_use, cache_dir=CACHE_PATH)
    #print(f'config loaded {time.time()-start_time:.02f}')
    #model = AutoModelForCausalLM.from_config(config)
    #print(f'cpu model instantiated {time.time()-start_time:.02f}')
    #model = model.eval()
    #print(f'eval mode set {time.time()-start_time:.02f}')
    #model = model.to(torch.float16)
    #print(f'fp16 set {time.time()-start_time:.02f}')
    #model = model.to('cuda:0')
    #print(f'sent to gpu {time.time()-start_time:.02f}')

    #print(f'memory available {torch.cuda.get_device_properties(0).total_memory / 2**30:.02f}')
    #
    #batch_size = 1
    #context_length = 512
    #device = 'cuda:0'
    #vocab_size = 50257

    ##input_ids = (torch.rand(batch_size, context_length, device=device) * vocab_size).to(torch.int64)


    #from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
    #tokenizer = AutoTokenizer.from_pretrained(model_to_use, cache_dir=CACHE_PATH)
    #inputs = tokenizer("Hello, I'm am conscious and", return_tensors="pt")
    ##inputs = {k: v.to(device) for k, v in inputs.items()}
    #inputs = inputs['input_ids'].to(device)
    #print(inputs)
    #outputs = model.generate(inputs, min_length=50, max_length=50, do_sample=True)
    #print(outputs)
    #print(tokenizer.batch_decode(outputs))

    #lm_logits = model.forward(
    #    input_ids,
    #    return_dict=False,
    #    output_attentions=False,
    #    output_hidden_states=False,
    #)
    #print([type(x) for x in lm_logits])

    ##torch.Size([1, 512, 50257])
    #print(lm_logits[0].shape)
    #print([type(x) for x in lm_logits[1][0]])
    #print([x.shape for x in lm_logits[1][0]])


    #next_token_logits = lm_logits[:, -1, :]
    #processors = LogitsProcessorList()
    #processors.append(LogitNormalization())
    #next_token_scores = processors(input_ids, next_token_logits)
    #probs = nn.functional.softmax(next_token_scores, dim=-1)
    #next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

    #print(next_tokens.shape)

    #for _ in range(64):
    #    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    #next_token_logits = outputs.logits[:, -1, :]
    #probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    #next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    #print(next_token_logits)



@stub.local_entrypoint()
def run():
    run_stable_diffusion.call()
    print('done')

