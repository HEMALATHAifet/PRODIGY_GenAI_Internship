#Install Gradio (frontend interface)
!pip install gradio --quiet

#Install Hugging Face Transformers (GPT-2 model)
!pip install transformers --quiet

#Import necessary libraries
import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#Define text generation function using beam search
def generate_text(inp):
  input_ids = tokenizer.encode(inp, return_tensors='tf')  # Tokenize input text
  beam_output = model.generate(
      input_ids,
      max_length=100,
      num_beams=5,
      no_repeat_ngram_size=2,
      early_stopping=True
  )
  output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return ".".join(output.split(".")[:-1]) + "."  # Clean up and return the generated text

#Create a Gradio interface to run in browser
gr.Interface(
    fn=generate_text,  # Function to call
    inputs=gr.Textbox(label="Enter your prompt"),  # Input UI element
    outputs=gr.Textbox(label="Generated Text"),    # Output UI element
    title="GPT-2",  # Title of the app
    description="OpenAI's GPT-2 is an unsupervised language model that can generate coherent text."  # Description
).launch()  # Launch the Gradio app
