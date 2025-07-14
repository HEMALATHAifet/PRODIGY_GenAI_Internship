# ✅ Import necessary libraries
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# ✅ Enter your Hugging Face token here
HUGGINGFACE_TOKEN = " enter your access token here "  # <-- Replace this!

# ✅ Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

# ✅ Define image generation function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# ✅ Launch Gradio app
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt"),
    outputs=gr.Image(type="pil"),
    title="🎨 AI Image Generator using Stable Diffusion",
    description="Type any creative text prompt to generate an AI image."
).launch()
