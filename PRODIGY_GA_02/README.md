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
- Works with simple and short prompts

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

