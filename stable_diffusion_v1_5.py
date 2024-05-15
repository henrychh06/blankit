import time

from diffusers import StableDiffusionPipeline
import torch


def test_stable_diffusion_v1_5():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   # safety_checker=None,
                                                   use_safetensors=True,
                                                   variant="fp32",
                                                   torch_dtype=torch.float32,
                                                   )
    pipe = pipe.to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.show()

test_stable_diffusion_v1_5()