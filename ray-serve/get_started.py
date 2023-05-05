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
    timeout=2*3600,
)
async def run_stable_diffusion():
    print('starting')

    import os
    os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH

    #model_to_use = "cerebras/Cerebras-GPT-111M"
    #model_to_use = "cerebras/Cerebras-GPT-6.7B"
    #model_to_use = "EleutherAI/gpt-neo-125m"
    #model_to_use = "facebook/opt-13b"
    model_to_use = "mosaicml/mpt-7b-instruct"

    from transformers import pipeline
    import torch
    #generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

    print('creating pipeline')
    generator = pipeline('text-generation', model=model_to_use, device='cuda:0', torch_dtype=torch.float16, trust_remote_code=True)
    print('starting inference')
    for _ in range(5):
        out = generator("The quick brown fox jumps over", do_sample=True, min_length=50, max_length=50)
        print(out)

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

