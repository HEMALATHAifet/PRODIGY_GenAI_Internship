import tensorflow as tf
import os
import numpy as np
from glob import glob
from tensorflow_examples.models.pix2pix import pix2pix
import gradio as gr
from PIL import Image

# Dataset path
DATA_DIR = "/content/edge2face/train"

# Load and preprocess one image pair
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    w = tf.shape(image)[1] // 2
    input_image = image[:, :w, :]
    target_image = image[:, w:, :]
    input_image = tf.image.resize(input_image, [256, 256])
    target_image = tf.image.resize(target_image, [256, 256])
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
    target_image = (tf.cast(target_image, tf.float32) / 127.5) - 1
    return input_image, target_image

# Load dataset safely
def load_train_dataset():
    files = glob(os.path.join(DATA_DIR, "*.png"))
    files = tf.constant(files, dtype=tf.string)  # ✅ Fix: Ensure string type
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(87).batch(1)  # You have 87 images
    return dataset

train_dataset = load_train_dataset()

# Pix2Pix model
OUTPUT_CHANNELS = 3
generator = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='batchnorm')
discriminator = pix2pix.discriminator(norm_type='batchnorm', target=True)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (LAMBDA * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_optimizer.apply_gradients(zip(
        gen_tape.gradient(gen_loss, generator.trainable_variables),
        generator.trainable_variables
    ))

    discriminator_optimizer.apply_gradients(zip(
        disc_tape.gradient(disc_loss, discriminator.trainable_variables),
        discriminator.trainable_variables
    ))

# Train the model
EPOCHS = 3  # Increase if needed
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for input_image, target in train_dataset.take(87):  # Use exact image count
        train_step(input_image, target)

# Gradio Inference
def predict(image):
    image = tf.image.resize(image, [256, 256])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.expand_dims(image, 0)
    prediction = generator(image, training=False)
    prediction = (prediction[0] + 1) / 2  # Rescale to [0, 1]
    return prediction.numpy()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Sketch (256x256)", type="numpy"),
    outputs=gr.Image(label="Generated Face", type="numpy"),
    title="Edge2Face Generator using Pix2Pix"
)

demo.launch()

