#!/usr/bin/env python3

from ray import air, tune
import os


class Trainable(tune.Trainable):
    def setup(self, config: dict):
        self.x = 0
        import torch
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        from torch.utils.data import DataLoader

        training_data = datasets.FashionMNIST(
            root="/data/cache/torch-datasets",
            train=True,
            download=True,
            transform=ToTensor()
        )

        local_rank = os.environ['CUDA_VISIBLE_DEVICES']

        self.dl = DataLoader(
            training_data,
            batch_size=64,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            pin_memory_device=f'cuda:{local_rank}',
            persistent_workers=True,
        )
        self.dl_iter = iter(self.dl)

    def step(self):
        self.x += 1
        try:
            n = next(self.dl_iter)
        except StopIteration:
            self.dl_iter = iter(self.dl)
        return {"score": 1}

def run():
    #trainable = Trainable()
    trainable = tune.with_resources(Trainable, resources={"cpu": 1, "gpu": 1})
    
    #failure_config=FailureConfig(max_failures=-1)
    
    #run_config=RunConfig(name="test",
    #                     local_dir=results_dir,
    #                     failure_config=failure_config,
    #                     log_to_file=True
    #)
    
    #tune_config = TuneConfig(num_samples=1,
    #                       reuse_actors=False
    #                       )
    
    #tuner=Tuner(new_trainable, run_config=run_config, tune_config=tune_config, param_space=cfg)
    tuner = tune.Tuner(trainable, param_space={'a': 5})
    
    tuner.fit()

run()
