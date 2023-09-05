from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import pytesseract
import os

app = Flask(__name__)
CORS(app)

# Update the TESSDATA_PREFIX as per your setup
os.environ['TESSDATA_PREFIX'] = '/tmp/tessdata'

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

    pil_img = Image.open(BytesIO(image_file.read()))
    meta = pil_img.info
    unique_id = meta.get("uniqueID", "")

    # Check for annotations first
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
            text += line_data[0]
        
        return render_template("index.html", text=text)

    # Fallback to OCR if annotations not found
    else:
        nparr = np.array(pil_img)
        image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(img, lang='pan', config='--psm 6')
        
        return render_template("index.html", text=text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
