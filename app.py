from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os  
app = Flask(__name__)

# Load processor and model once (not on every request)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-small")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small")

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        # Open image
        image = Image.open(file).convert("RGB")

        # Process and generate caption
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
    return render_template("index.html", caption=caption)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
