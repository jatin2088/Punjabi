from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
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

    # Additional image processing (if needed, you can uncomment this block)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(gray, (5, 5), 0)

    height, width = image.shape[:2]

    # Fetch annotation using unique_id from metadata
    unique_id = meta.get("uniqueID", "")
    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    if response.status_code == 200:
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
    else:
        return render_template("index.html", text="Annotation not found.")

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
