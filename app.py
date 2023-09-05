from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import os
import pytesseract
import re

# Function to download Tesseract data
def download_tessdata_from_github():
    tessdata_url = "https://github.com/jatin2088/Punjabi/raw/main/tesseract-ocr/4.00/tessdata/"
    tessdata_files = ['pan.traineddata']

    tessdata_dir = './tessdata/'
    os.makedirs(tessdata_dir, exist_ok=True)

    for file in tessdata_files:
        response = requests.get(f"{tessdata_url}{file}", allow_redirects=True)
        with open(os.path.join(tessdata_dir, file), 'wb') as f:
            f.write(response.content)

# Download Tesseract data
download_tessdata_from_github()

# Set TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = './tessdata/'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# GitHub repository URL for annotations
GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

# Preprocessing function
def preprocess_image(pil_img):
    nparr = np.array(pil_img)
    gray = cv2.cvtColor(nparr, cv2.COLOR_BGR2GRAY)
    return gray

# Perform OCR using Tesseract
def perform_ocr(img):
    text = pytesseract.image_to_string(img, lang='pan', config='--psm 6')
    return text

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    file_extension = image_file.filename.split('.')[-1].lower()

    if file_extension != 'png':
        return render_template("index.html", text="Check file format.")

    pil_img = Image.open(BytesIO(image_file.read()))
    img = preprocess_image(pil_img)

    unique_id = pil_img.info.get("uniqueID", "")

    # Attempt to fetch annotations first
    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    if response.status_code == 200:
        # Use annotations for text extraction (implement your existing logic)
        pass
    else:
        # Fallback to Tesseract OCR
        extracted_text = perform_ocr(img)
        return render_template("index.html", text=extracted_text)

    return render_template("index.html", text="An unexpected error occurred.")

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
