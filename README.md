# Image Captioning App

A Python Flask web app that generates **AI-powered captions for uploaded images** using deep learning.  
Easily upload any image and get a human-like descriptive caption instantly.

---

## Features

- Upload images from your computer
- Generate captions using advanced AI models (BLIP with Transformers)
- Attractive, responsive web interface
- Easy setup — run locally on any desktop
- Clear instructions for installation and usage

---
## Getting Started

Follow these steps to set up and run the app on your machine:

### 1. **Clone or Download the Repository**
git clone https://github.com/MadhumithraA1426/image-captioning-app.git
cd image-captioning-app

Or click **Code → Download ZIP** on GitHub, then extract.

---

### 2. **Create a Python Virtual Environment**
On **Windows**:
python -m venv venv
venv\Scripts\activate

On **Mac/Linux**:
python3 -m venv venv
source venv/bin/activate

---

### 3. **Install Required Packages**
pip install -r requirements.txt

text

---

### 4. **Run the Flask App**
python app.py

---

### 5. **Open the App in Your Browser**
Go to:
http://127.0.0.1:5000/

---

## Project Structure

├── app.py # Main backend code
├── requirements.txt # List of required Python packages
├── static/ # Contains CSS (style.css) and static assets
├── templates/ # Contains index.html template
├── .gitignore # Git ignore rules (venv, etc.)
├── LICENSE # License
└── README.md # This help file

text

---

## Usage

- Click "Choose File" to upload an image
- Hit "Generate Caption"
- See the caption displayed instantly below the image

---

## Requirements

- Python 3.8 or higher
- Packages listed in `requirements.txt`
  - Flask
  - torch
  - torchvision
  - transformers
  - pillow

---

## Deployment & Sharing

- To share your app online, use ngrok/localtunnel or deploy to a cloud service
- See more deployment advice in the project discussions

---

## License

This project is licensed under the **MIT License** — feel free to use and adapt for learning or demo purposes!

---

## Author

**Madhu Mithra**  
[GitHub Profile](https://github.com/MadhumithraA1426)


