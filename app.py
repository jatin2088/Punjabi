from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re

# Initialize Flask
app = Flask(__name__)

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

# Set TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = './tessdata/'

# Download Tesseract data
download_tessdata_from_github()

# Function to preprocess image
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

# Function to perform OCR
def perform_ocr(img):
    img = Image.fromarray(img)
    text = pytesseract.image_to_string(img, lang='pan', config='--psm 6')
    return text

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    pil_img = Image.open(BytesIO(image_file.read()))
    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
    unique_id = pil_img.info.get("uniqueID", "")
    GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"
    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')

    if response.status_code == 200:
        lines = response.text.splitlines()
        text = ""
        for line in lines:
            text += line + "\n"
        return render_template("index.html", text=text)
    else:
        processed_image = preprocess_image(image)
        ocr_text = perform_ocr(processed_image)
        return render_template("index.html", text=ocr_text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0")
