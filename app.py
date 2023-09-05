from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
import os
import re
import zipfile
import gdown
import pytesseract
from flask_cors import CORS
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# GitHub repo URL
GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

# Download and unzip tesseract data
gdown.download("https://drive.google.com/uc?id=1gMCrUtn4cbScmFkcHUcKmRbDLJ_z8vOO", "tesseract-ocr.zip", quiet=False)
with zipfile.ZipFile("tesseract-ocr.zip", "r") as zip_ref:
    zip_ref.extractall(".")

# Tesseract setup
os.environ['TESSDATA_PREFIX'] = './tesseract-ocr/4.00/tessdata/'

# Function definitions
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh

def filter_punjabi(text):
    pattern = re.compile('[^\u0A00-\u0A7F  \n]')
    return pattern.sub('', text)

def ocr_from_image(image, lang='pan'):
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 6')
    return filter_punjabi(text)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    file_extension = image_file.filename.split('.')[-1].lower()

    if file_extension != 'png':
        return render_template("index.html", text="Check file format.")

    image_data = BytesIO(image_file.read())
    pil_img = Image.open(image_data)
    meta = pil_img.info
    unique_id = meta.get("uniqueID", "")

    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')

    if response.status_code != 200:
        extracted_text = ocr_from_image(image)
        if extracted_text:
            return render_template("index.html", text=extracted_text)
        else:
            return render_template("index.html", text="No text could be extracted.")

    lines = response.text.splitlines()
    text = ''
    prev_y_center = 0

    for line in lines:
        line = line.strip()
        if not line:
            text += ' '
            continue

        line_data = line.split()
        char, x_center, y_center, w, h = line_data[0], float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4])
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height

        if y_center - prev_y_center > h:
            text += '\n'

        prev_y_center = y_center
        text += char

    return render_template("index.html", text=text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
