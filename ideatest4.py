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

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def fetch_and_prepare_image(image_url, size=(512, 512)):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = resize_with_padding(img, size)
    return img

def remove_background(image_url):
    img = fetch_and_prepare_image(image_url)
    img = resize_with_padding(img, (512, 512))
    remover = Remover()  # default setting
    remover = Remover(mode='base')  # nightly release checkpoint
    fg_mask = remover.process(img, type='map')  # default setting - transparent background
    mask = ImageOps.invert(fg_mask)
    return mask

image_url = 'https://lh4.googleusercontent.com/proxy/p-1dRttrHDLLjtC8E-V6g9uW5uP3jk6yLPHzozss9csThf62LccKm4wOmof_-N8v5WkxnjK-8gkKefMkE_eFC2q5i-mrl9SVVVSc_aqpWI_gN7IRxwrE2nL0e4T9JRqZq3In'
prompt = "A swan in a futuristic city, 3d, 4k"

control_image = remove_background(image_url)

image = pipe(prompt, control_image=control_image, controlnet_conditioning_scale=0.7).images[0]
image.save(f"{prompt}.png")