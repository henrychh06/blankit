from sanic import Sanic
from sanic.response import html, file
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps
import requests
from io import BytesIO
import torch
from transparent_background import Remover
import os

app = Sanic("ImageGeneratorApp")

def initialize_pipeline(model_id="yahoo-inc/photo-background-generation"):
    pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
    pipeline = pipeline.to('cuda')
    return pipeline

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

def remove_background(img):
    remover = Remover()  # default setting
    remover = Remover(mode='base')  # nightly release checkpoint
    fg_mask = remover.process(img, type='map')  # default setting - transparent background
    mask = ImageOps.invert(fg_mask)
    return mask

def generate_controlnet_image(pipeline, prompt, img, mask, seed, cond_scale=1.0, num_inference_steps=20):
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

def generate_image(image_url, prompt, seed=13):
    pipeline = initialize_pipeline()
    img = fetch_and_prepare_image(image_url)
    mask = remove_background(img)
    controlnet_image = generate_controlnet_image(pipeline, prompt, img, mask, seed)
    return controlnet_image

@app.route("/")
async def index(request):
    html_content = """
    <html>
    <body>
        <h1>Generar Imagen</h1>
        <form action="/generate" method="get">
            <label for="prompt">Prompt:</label><br>
            <input type="text" id="prompt" name="prompt" value="A dark swan in a bedroom"><br>
            <input type="submit" value="Generar Imagen">
        </form>
    </body>
    </html>
    """
    return html(html_content)

@app.route("/generate")
async def generate(request):
    prompt = request.args.get('prompt', 'A swan in a bedroom')
    image_url = 'https://lh4.googleusercontent.com/proxy/p-1dRttrHDLLjtC8E-V6g9uW5uP3jk6yLPHzozss9csThf62LccKm4wOmof_-N8v5WkxnjK-8gkKefMkE_eFC2q5i-mrl9SVVVSc_aqpWI_gN7IRxwrE2nL0e4T9JRqZq3In'
    controlnet_image = generate_image(image_url, prompt)

    # Guardar la imagen generada en un archivo temporal
    image_path = "/tmp/generated_image.png"
    controlnet_image.save(image_path)

    html_content = f"""
    <html>
    <body>
        <h1>Imagen Generada</h1>
        <img src="/image" alt="Generated Image">
        <br>
        <a href="/">Generar otra imagen</a>
    </body>
    </html>
    """
    return html(html_content)

@app.route("/image")
async def serve_image(request):
    return await file("/tmp/generated_image.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
