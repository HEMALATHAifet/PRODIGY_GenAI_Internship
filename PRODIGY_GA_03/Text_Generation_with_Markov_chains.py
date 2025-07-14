!pip install gradio --q
# ✅ Import Libraries
import random
import gradio as gr
from collections import defaultdict

# ✅ Build Markov Chain (word-based)
def build_markov_chain(text, n=2):
    words = text.split()
    markov_chain = defaultdict(list)
    
    for i in range(len(words) - n):
        prefix = tuple(words[i:i+n])
        next_word = words[i+n]
        markov_chain[prefix].append(next_word)

    return markov_chain

# ✅ Generate Text using Markov Chain
def generate_markov_text(text, prefix_length=2, output_length=50):
    words = text.split()
    if len(words) < prefix_length + 1:
        return "Please provide more text to learn from."

    chain = build_markov_chain(text, n=prefix_length)

    # Start from a random prefix
    current_prefix = random.choice(list(chain.keys()))
    generated_words = list(current_prefix)

    for _ in range(output_length):
        possible_next_words = chain.get(current_prefix)
        if not possible_next_words:
            break
        next_word = random.choice(possible_next_words)
        generated_words.append(next_word)
        current_prefix = tuple(generated_words[-prefix_length:])

    return ' '.join(generated_words)

# ✅ Gradio UI
gr.Interface(
    fn=generate_markov_text,
    inputs=[
        gr.Textbox(lines=10, label="Enter Training Text"),
        gr.Slider(1, 5, step=1, value=2, label="Prefix Length (n)"),
        gr.Slider(10, 100, step=5, value=50, label="Output Length (Number of Words)")
    ],
    outputs="text",
    title="✨ Markov Chain Text Generator",
    description="This app generates new text using a word-level Markov Chain model. Paste a paragraph and watch it remix your input!"
).launch()
