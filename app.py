from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

app = Flask(__name__)

# Lazy load (None until needed)
model = None
feature_extractor = None
tokenizer = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model, feature_extractor, tokenizer
    caption = None

    if request.method == "POST":
        # Load model only on first request
        if model is None:
            model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        if "image" not in request.files:
            return "No file uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        image = Image.open(file).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

        # Disable gradient tracking → saves memory
        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=16, num_beams=2)

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return render_template("index.html", caption=caption)
