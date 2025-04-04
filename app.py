# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13kBa0c6INdPPQYzP_nRlZyid0a5wuBss
"""

import gradio as gr
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load BLIP-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch_dtype
).to(device)

# Function to process image and generate description
def describe_image(image):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device, torch_dtype)
    output = model.generate(**inputs, max_length=100)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Create Gradio Interface
interface = gr.Interface(
    fn=describe_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ContextVision: AI-Powered Scene Assistant",
    description="Upload an image and get a description using BLIP-2."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)