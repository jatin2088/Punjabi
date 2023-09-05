from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import requests
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import re
import urllib.request
import pytesseract

# Create a directory to hold the pan.traineddata file
os.makedirs("tessdata", exist_ok=True)
TESSDATA_PATH = './tessdata'

# Download pan.traineddata from GitHub repo to TESSDATA_PATH
PAN_TRAINEDDATA_URL = 'https://github.com/jatin2088/Punjabi/raw/main/tesseract-ocr/4.00/tessdata/pan.traineddata'
urllib.request.urlretrieve(PAN_TRAINEDDATA_URL, f"{TESSDATA_PATH}/pan.traineddata")

# Set TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH + '/'

app = Flask(__name__)
CORS(app)
GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

def filter_punjabi(text):
    pattern = re.compile('[^\u0A00-\u0A7F  \n]')
    return pattern.sub('', text)

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

    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    
    if response.status_code != 200:
        nparr = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(nparr, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        text = pytesseract.image_to_string(gray_pil, lang='pan', config='--psm 6')
        filtered_text = filter_punjabi(text)
        return render_template("index.html", text=filtered_text)
    
    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
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
