#!/usr/bin/env python3

import ray

from ray.data import DataContext

ctx = DataContext.get_current()
ctx.strict_mode = True

import ray
from ray.data.datasource.partitioning import Partitioning

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/val/"

# The S3 directory structure is {s3_uri}/{class_id}/{*.JPEG}
partitioning = Partitioning("dir", field_names=["class"], base_dir=s3_uri)

ds = ray.data.read_images(
    s3_uri,
    partitioning=partitioning,
    mode="RGB",
)

ds = ds.repeat(times=10)

#print(ds.schema())

#single_batch = ds.take_batch(10)

from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms

weights = ResNet18_Weights.IMAGENET1K_V1

# Load the pretrained resnet model and move to GPU
model = models.resnet18(weights=weights).cuda()

imagenet_transforms = weights.transforms
transform = transforms.Compose([transforms.ToTensor(), imagenet_transforms()])

import torch

#transformed_batch = [transform(image) for image in single_batch["image"]]
#with torch.inference_mode():
#    prediction_results = model(torch.stack(transformed_batch).cuda())
#    classes = prediction_results.argmax(dim=1).cpu()

#print(classes)
# tensor([490, 395, 103, 932, 324, 845, 476, 331, 684, 770])

def preprocess_images(image):
    return {"original_image": image["image"], "transformed_image": transform(image["image"])}

transformed_ds = ds.map(preprocess_images)

class TorchModel:
    def __init__(self):
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights).cuda()
        self.model.eval()

    def __call__(self, batch):
        torch_batch = torch.stack(list(batch["transformed_image"])).cuda()
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            return {"class": prediction.argmax(dim=1).detach().cpu().numpy()}


predictions = transformed_ds.map_batches(
    TorchModel,
    compute=ray.data.ActorPoolStrategy(size=2),
    num_gpus=1,  # Specify 1 GPU per worker.
    batch_size=1024
    #batch_size=392
)

print(predictions.show(5))
