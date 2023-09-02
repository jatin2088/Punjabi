from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
import json
from flask_cors import CORS
from PIL import Image
from PIL import PngImagePlugin
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

OCR_SPACE_API_KEY = "K81439199088957"
GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    file_extension = image_file.filename.split('.')[-1].lower()

    if file_extension != 'png':
        return render_template("index.html", text="Check file format.")

    # Convert uploaded image to a byte array and read it as a PIL Image to fetch metadata
    image_data = BytesIO(image_file.read())
    pil_img = Image.open(image_data)
    meta = pil_img.info

    # Convert PIL image to OpenCV format
    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = img.shape

    # Try to fetch annotation using unique_id from metadata
    unique_id = meta.get("uniqueID", "")
    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    if response.status_code != 200:
        # Use OCR.space API to extract text if annotation not available
        headers = {"apikey": OCR_SPACE_API_KEY}
        files = {"file": ("image.png", image_data.getvalue(), 'image/png')}
        ocr_response = requests.post("https://api.ocr.space/parse/image", headers=headers, files=files)
        ocr_result = json.loads(ocr_response.text)
        if ocr_result.get('IsErroredOnProcessing') == False and 'ParsedResults' in ocr_result:
            text = ocr_result['ParsedResults'][0]['ParsedText']
        else:
            text = ""
        return render_template("index.html", text=text)

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
