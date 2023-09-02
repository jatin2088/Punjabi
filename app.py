from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
from PIL import PngImagePlugin
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']

    # Read the uploaded image to PIL Image to fetch metadata
    pil_img = Image.open(BytesIO(image_file.read()))
    meta = pil_img.info
    unique_id = meta.get("uniqueID", "random_unique_id")
    
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

    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    if response.status_code != 200:
        return render_template("index.html", text="Random Output")
    
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
