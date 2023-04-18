#!/usr/bin/env python3

import subprocess
subprocess.check_call("pip install -U accelerate 'numpy<1.24' transformers deepspeed", shell=True)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
import torch
import csv

import ray

@ray.remote(num_cpus=0)
class MetricsActor:
    def __init__(self):
        self.rows = []
        self.FIELDS = ["model", "dtype", "bs", "ctx_len", "num_blks", "tput"]

    def report(self, checkpoint, dtype, batch_size, context_length, num_blks, dur_s, throughput):
        row = {
            "model": checkpoint,
            "dtype": dtype,
            "bs": batch_size,
            "ctx_len": context_length,
            "num_blks": num_blks,
            "tput": throughput,
        }
        print(row)
        self.rows.append(row)

    def write(self):
        import sys
        write_header = True
        writer = csv.DictWriter(sys.stdout, fieldnames=self.FIELDS)
        if write_header:
            writer.writeheader()

        for row in self.rows:
            writer.writerow(row)

class Trial:
    def __init__(self, checkpoint, dtype):
        self.device = 'cuda:0'
        self.checkpoint = checkpoint
        self.model = None
        self.config = None
        self.dtype = dtype

        self.gb_used_by_model = None

    def load_model(self):
        config = AutoConfig.from_pretrained(self.checkpoint)
        print(self.checkpoint, config.hidden_size)
        
        torch.ones(1).to(self.device)
        before_size = torch.cuda.memory_reserved()
        
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(self.dtype)

        model.eval()

        device_map = {
            "transformer": 0,
            "lm_head": 0,
        }
        dispatched = dispatch_model(model, device_map=device_map)
        
        after_size = torch.cuda.memory_reserved()

        self.gb_used_by_model = (after_size - before_size) / 2**30
        self.model = model
        self.config = config

    def load_only_transformer_block(self):
        config = AutoConfig.from_pretrained(self.checkpoint)
        print(self.checkpoint, config.hidden_size)
        
        torch.ones(1).to(self.device)
        before_size = torch.cuda.memory_reserved()
        
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(self.dtype)
        model.eval()
        #before_len = len(model.transformer.h)
        #model = model.transformer.h[:1]
        model = model.transformer.h
        print(f'num transformer blocks: {len(model)}')

        #device_map = {
        #    "transformer": 0,
        #    "lm_head": 0,
        #}
        #dispatched = dispatch_model(model, device_map=device_map)
        model.to(self.device)
        
        after_size = torch.cuda.memory_reserved()

        self.gb_used_by_model = (after_size - before_size) / 2**30
        print('GB used by model', self.gb_used_by_model)
        self.model = model
        self.config = config

    def load_only_transformer_block_with_ds(self):
        self.load_only_transformer_block()

        import deepspeed

        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(
            self.model,
            mp_size=4,
            dtype=torch.float16,
            checkpoint=None,
            replace_with_kernel_inject=True,
        )
        model = ds_engine.module
        output = model('Input String')

    def normal_model_fwd_pass(self, batch_size, context_length):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        input_ids = (torch.rand(batch_size, context_length, device=self.device) * self.config.vocab_size).to(torch.int64)

        start_event.record()
        outputs = self.model.forward(input_ids)
        end_event.record()
        
        end_event.synchronize()
        dur_s = start_event.elapsed_time(end_event) / 1000
        return dur_s

    def only_transformer_fwd_pass(self, batch_size, context_length, num_blks):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        hidden_input = torch.rand(batch_size, context_length, self.config.hidden_size, device=self.device).to(torch.float16)
        start_event.record()
        for blk in self.model[:num_blks]:
            hidden_input = blk.forward(hidden_input)
            hidden_input = hidden_input[0]
        end_event.record()

        end_event.synchronize()
        dur_s = start_event.elapsed_time(end_event) / 1000
        return dur_s

    def run_trial(self, metrics_actor, batch_size, context_length, num_blks):
        try:
            durations = []
            for i in range(6):
                #dur_s = self.normal_model_fwd_pass(batch_size, context_length)
                dur_s = self.only_transformer_fwd_pass(batch_size, context_length, num_blks)
                durations.append(dur_s)

            had_exception = False
        except Exception as e:
            print(e)
            had_exception = True

        #durations.sort()
        dur_s = min(durations)
        throughput = batch_size / dur_s
        ray.get(metrics_actor.report.remote(self.checkpoint, self.dtype, batch_size, context_length, num_blks, dur_s, throughput))


checkpoint_125m = "EleutherAI/gpt-neo-125m"
checkpoint_1p3b = "EleutherAI/gpt-neo-1.3B"
checkpoint_2p7b = "EleutherAI/gpt-neo-2.7B"

@ray.remote(num_gpus=1)
def trial(checkpoint, dtype, metrics_actor):
    t = Trial(checkpoint, dtype)
    #t.load_model()
    #t.load_only_transformer_block()
    t.load_only_transformer_block_with_ds()

    for num_blks in [1, 2, 4, 8]:
        for bs in [1]:
            for ctx_len in [2048]:
                t.run_trial(metrics_actor, bs, ctx_len, num_blks)

metrics_actor = MetricsActor.remote()

#futures = []
#for dtype in [torch.float16]:
#    for checkpoint in [checkpoint_125m, checkpoint_1p3b, checkpoint_2p7b]:
#        futures.append(trial.remote(checkpoint, dtype, metrics_actor))
#
#ray.get(futures)

ray.get(trial.remote(checkpoint_125m, torch.float16, metrics_actor))
ray.get(trial.remote(checkpoint_1p3b, torch.float16, metrics_actor))
ray.get(trial.remote(checkpoint_2p7b, torch.float16, metrics_actor))

ray.get(metrics_actor.write.remote())

#checkpoint = checkpoint_2p7b
#device = 'cuda:0'
#dtype = torch.float16
#
#config = AutoConfig.from_pretrained(checkpoint)
#print(checkpoint, config.hidden_size)
#
#torch.ones(1).to(device)
#before_size = torch.cuda.memory_reserved()
#
#model = AutoModelForCausalLM.from_config(config)
#model = model.to(dtype)
##model.transformer.h = model.transformer.h[:1]
#model.eval()
#
#device_map = {
#    "transformer": 0,
#    "lm_head": 0,
#}
#dispatched = dispatch_model(model, device_map=device_map)
#
#after_size = torch.cuda.memory_reserved()
#print('space taken', (after_size - before_size) / 2**30)
