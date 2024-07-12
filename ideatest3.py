import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from transparent_background import Remover
from PIL import Image, ImageOps
import requests
from io import BytesIO

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)

pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to("cuda")


def remove_background(img):
    remover = Remover()  # default setting
    remover = Remover(mode='base')  # nightly release checkpoint
    fg_mask = remover.process(img, type='map')  # default setting - transparent background
    mask = ImageOps.invert(fg_mask)
    return mask

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def fetch_and_prepare_image(control_image, size=(512, 512)):
    response = requests.get(control_image)
    img = Image.open(BytesIO(response.content))
    img = resize_with_padding(img, size)
    return img

def generate_controlnet_image(pipeline, prompt, img, mask, seed, cond_scale=1.0, num_inference_steps=50):
    img = resize_with_padding(img, (512, 512))
    generator = torch.Generator(device='cuda').manual_seed(seed)
    with torch.autocast("cuda"):
        result = pipeline(
            prompt=prompt, image=img, mask_image=mask, control_image=mask,
            num_images_per_prompt=1, generator=generator,
            num_inference_steps=num_inference_steps, guess_mode=False,
            controlnet_conditioning_scale=cond_scale
        )
    return result.images[0]

def generate_image(control_image, prompt, seed=13):
    pipeline = pipe
    img = fetch_and_prepare_image(control_image)
    mask = remove_background(img)
    controlnet_image = generate_controlnet_image(pipeline, prompt, img, mask, seed)
    return controlnet_image

control_image = 'https://lh4.googleusercontent.com/proxy/p-1dRttrHDLLjtC8E-V6g9uW5uP3jk6yLPHzozss9csThf62LccKm4wOmof_-N8v5WkxnjK-8gkKefMkE_eFC2q5i-mrl9SVVVSc_aqpWI_gN7IRxwrE2nL0e4T9JRqZq3In'

#control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
prompt = "A swan in a futuristic city, 3d, 4k"
controlnet_image = generate_image(control_image, prompt)

image = pipe(prompt, control_image=controlnet_image ,controlnet_conditioning_scale=0.7).images[0]
#controlnet_image = generate_image(control_image, prompt)
image.save(f"{prompt}.png")
