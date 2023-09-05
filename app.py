from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import os
import pytesseract
import zipfile

app = Flask(__name__)
CORS(app)

GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"
TESSDATA_URL = "https://github.com/jatin2088/Punjabi/raw/main/tesseract-ocr/4.00/tessdata.zip"
TESSDATA_DIR = "/tmp/tessdata"

# Download Tesseract data files
os.makedirs(TESSDATA_DIR, exist_ok=True)
r = requests.get(TESSDATA_URL)
with open(f"{TESSDATA_DIR}/tessdata.zip", "wb") as f:
    f.write(r.content)

# Unzip Tesseract data files
with zipfile.ZipFile(f"{TESSDATA_DIR}/tessdata.zip", 'r') as zip_ref:
    zip_ref.extractall(TESSDATA_DIR)

# Update TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh

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
    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)

    response = requests.get(GITHUB_REPO_URL + image_file.filename.replace('.png', '') + '.txt')

    if response.status_code == 200:
        text = response.text
        return render_template("index.html", text=text)
    else:
        processed_img = preprocess_image(image)
        text = pytesseract.image_to_string(processed_img, lang='pan', config='--psm 6')
        return render_template("index.html", text=text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0")
