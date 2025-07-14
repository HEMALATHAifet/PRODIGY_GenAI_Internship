# 🎨 AI Image Generator using Stable Diffusion

This project is developed as part of **Internship Task 2** under the **Artificial Intelligence Internship Program** at **Prodigy Infotech**.

It is an AI-powered image generator that uses a **pre-trained Stable Diffusion model** to generate images based on simple text prompts. The interface is built using **Gradio**, making it easy for anyone to interact with the model and generate creative visuals.

---

## 📌 Project Overview

- 🏢 **Internship**: Prodigy Infotech – Artificial Intelligence
- 📄 **Task**: Image Generation using Pre-trained Models
- 🧠 **Model Used**: Stable Diffusion (`CompVis/stable-diffusion-v1-4`)
- 🛠️ **Tech Stack**: Python, Hugging Face Diffusers, Gradio, Google Colab

---

## 🚀 Features

- Generate high-quality AI images from text prompts
- Gradio-powered web interface for easy access
- Fast execution using Google Colab and GPU
- Works best with short and simple prompts

---

## 🧠 Understanding Pre-trained Models

Pre-trained models are machine learning models that have been trained on large datasets to perform specific tasks. In this case, **Stable Diffusion** is trained on billions of image-caption pairs. Instead of training from scratch, we reuse this model to generate images from text prompts efficiently.

---

## 🖼️ How Image Generation Works

Stable Diffusion follows a process called **text-to-image synthesis**:
- It starts with a **text prompt**.
- The text is converted into **embeddings** (machine-readable features).
- A random noise image is gradually **refined through a diffusion process** using these embeddings.
- The result is a high-quality, realistic image that reflects the input prompt.

---

## ✅ Model Selection and Alternatives

- **Model Used**: `CompVis/stable-diffusion-v1-4` from Hugging Face
- **Why Chosen**: Open-source, optimized for Colab GPU, produces detailed images
- **Alternatives**:
  - `SDXL` (newer, more powerful)
  - DALL·E by OpenAI
  - Craiyon (lightweight DALL·E mini)
  - MidJourney (commercial)

---

## 🌍 Social Responsibility Through AI

This project promotes:
- **Creative empowerment**: Allows non-designers to create visuals easily.
- **Educational value**: Helps students and AI enthusiasts understand generative models.
- **Accessibility**: Makes advanced AI tools usable in browser-based environments.

---

## 🔐 Hugging Face Token Generation

To access pre-trained models:
1. Sign up: https://huggingface.co/join
2. Go to: https://huggingface.co/settings/tokens
3. Click “New Token” → Give a name → Choose “Read” access → Copy the token

Use this token securely in your Python environment to load models.

---

## 🔗 Why Use Hugging Face?

Hugging Face provides:
- Thousands of ready-to-use ML models
- Pre-trained pipelines for NLP, CV, and diffusion tasks
- Python libraries (`diffusers`, `transformers`) that simplify ML integration

---

## ⚙️ Google Colab and GPU Runtime

Since image generation is **compute-intensive**, we switch to **GPU runtime**:
- Go to: `Runtime → Change runtime type → GPU`
- GPUs like **T4** are available in Colab Free and are fast enough for 512x512 image outputs.
- This speeds up model loading and image generation by 10x or more.

---

## 🌫️ Role of Diffusers Library

The `diffusers` library by Hugging Face:
- Provides tools to run diffusion-based models like Stable Diffusion
- Handles model loading, denoising steps, and pipeline abstraction
- Can be used for tasks like **image inpainting**, **super-resolution**, and **audio synthesis**

---

## 🔥 Role of PyTorch in This Project

`PyTorch` is the deep learning engine under the hood:
- Performs tensor operations and neural network computations
- Enables GPU acceleration in model inference
- Also used widely in training custom models in NLP, vision, and reinforcement learning

---

## 🔧 Setup Instructions

1. Open the Colab notebook.
2. Enable GPU:  
   Go to **Runtime → Change runtime type → GPU**
3. Install required libraries:

```bash
!pip install diffusers transformers accelerate safetensors gradio
````

4. Add your Hugging Face access token:

```python
HUGGINGFACE_TOKEN = "your_huggingface_token_here"
```

5. Run the Gradio interface and enter any text prompt.

---

## 💬 Example Prompts

Try using simple prompts like:

* `A cat sitting on a laptop`
* `A robot holding a light bulb`
* `A student reading a book`
* `A sunrise over mountains`
* `A graduation cap on a desk`

These work best in Colab with a 512x512 image output.

---

## 📷 Sample Output

| Prompt           | Output Image           |
| ---------------- | ---------------------- |
| `Cat on laptop`  | ![cat](sample-cat.jpg) |
| `Graduation cap` | ![cap](sample-cap.jpg) |

*(Replace with actual output images if submitting screenshots)*

---

## 🧾 Code Walkthrough – Line-by-Line Explanation

```python
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
```

* Importing libraries: PyTorch for computation, `diffusers` for the model, and `gradio` for UI.

```python
HUGGINGFACE_TOKEN = "your_token_here"
```

* Set your Hugging Face access token to authenticate and download the model.

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")
```

* Load the pre-trained Stable Diffusion model in half-precision (`fp16`) to save memory.
* Move it to the GPU using `.to("cuda")` for faster performance.

```python
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image
```

* Define a function that accepts a text prompt and returns the generated image.

```python
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt"),
    outputs=gr.Image(type="pil"),
    title="🎨 AI Image Generator using Stable Diffusion",
    description="Type any creative text prompt to generate an AI image."
).launch()
```

* Build a simple web UI using Gradio.
* The user enters a prompt, and the AI-generated image appears on screen.

---
