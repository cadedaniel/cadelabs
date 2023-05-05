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

volume = SharedVolume().persist("stable-diff-model-vol")

@stub.function(
    gpu="A10G",
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
async def run_stable_diffusion(prompt, channel_name = None):
    from diffusers import StableDiffusionPipeline
    from torch import float16

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        #use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        revision="fp16",
        torch_dtype=float16,
        cache_dir=CACHE_PATH,
        device_map="auto",
    )

    image = pipe(prompt, num_inference_steps=1).images[0]

    # Convert PIL Image to PNG byte array.
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

    if channel_name:
        # `post_image_to_slack` is implemented further below.
        post_image_to_slack.call(prompt, channel_name, img_bytes)

    return img_bytes

@stub.local_entrypoint()
def run(
    prompt: str = "oil painting of a shiba",
    output_dir: str = "/tmp/stable-diffusion",
):
    os.makedirs(output_dir, exist_ok=True)
    img_bytes = run_stable_diffusion.call(prompt)
    output_path = os.path.join(output_dir, "output.png")
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Wrote data to {output_path}")

