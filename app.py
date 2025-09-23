from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

app = Flask(__name__)

# Initialize as None
processor = None
model = None

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

@app.route("/", methods=["GET", "POST"])
def index():
    global processor, model
    caption = None

    # Lazy load
    if processor is None or model is None:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-small", use_auth_token=HF_TOKEN
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-small", use_auth_token=HF_TOKEN
        )

    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        image = Image.open(file).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    return render_template("index.html", caption=caption)
