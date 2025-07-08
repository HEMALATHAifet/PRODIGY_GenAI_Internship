# 🧠 Text Generation with GPT-2  
**Prodigy Internship – Task 1 at Generative_AI**

This project demonstrates how to build a simple *text generation application using OpenAI's GPT-2 language model*. It includes an interactive web interface powered by Gradio, allowing users to input a prompt and receive AI-generated text responses.

---

## 📂 Files Included

- `Text_Generation_with_GPT2.py` – Main Python script that loads the GPT-2 model, defines a text generation function, and launches a Gradio interface.
- `requirements.txt` – Lists all Python dependencies needed to run the project.

---

## 🔧 Installation

You can set up the environment using the following steps:

```bash

# Install dependencies
pip install -r requirements.txt
```
Or, if you're using Google Colab:
```
!pip install gradio --quiet
!pip install transformers --quiet
```
----
## 📜 How It Works

- **Model & Tokenizer**:  
  Loads the pre-trained GPT-2 model and tokenizer from Hugging Face's Transformers library.

- **Text Generation Logic**:  
  Uses beam search (with 5 beams and early stopping) to generate coherent and creative text output based on user input.

- **User Interface**:  
  A simple Gradio web interface takes user input through a textbox and displays the generated result in the browser.
---
## ▶️ Run the Application
To launch the web interface locally:
```
python Text_Generation_with_GPT2.py
```
This will open a browser window (or provide a link) where you can:

- Enter a prompt

- Get AI-generated text

- Interact repeatedly
---
## 📦 Dependencies

Make sure you’ve installed the following packages (already included in `requirements.txt`):

- `gradio`– for building the frontend interface  
- `transformers` – for loading GPT-2 from Hugging Face  
- `tensorflow` – backend for running the GPT-2 model
---
## ✨ Example Usage
**Input Prompt:**
```
Once upon a time in a distant galaxy,
```
**Generated Output:**
```
Once upon a time in a distant galaxy, there lived a group of explorers who sought to uncover the secrets of ancient civilizations. They traveled through time and space...
```
![Output_3](https://github.com/user-attachments/assets/584f5d26-c745-4cba-ae2f-8fbe6cdb6ef0)
---
## 📚 References

- [🤗 Hugging Face Transformers](https://huggingface.co/transformers/)
- [🧪 Gradio Documentation](https://www.gradio.app/docs)
- [📄 GPT-2 Paper by OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**Happy coding!** 🚀
---
