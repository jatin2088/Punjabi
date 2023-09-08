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

# Initialize Flask
app = Flask(__name__)
CORS(app)

# GitHub Repo URL
GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/annotations/"

# Filter Punjabi Text
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

    nparr = np.array(pil_img)
    image = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)

    response = requests.get(GITHUB_REPO_URL + unique_id + '.txt')
    
    if response.status_code == 200:
        # Existing logic
        lines = response.text.splitlines()
        text = ''
        for line in lines:
            # Your logic here
            pass
        return render_template("index.html", text=text)
    else:
        try:
            # Download the Tesseract trained data file from GitHub
            tesseract_data_url = 'https://github.com/jatin2088/Punjabi/raw/main/tesseract-ocr/4.00/tessdata/pan.traineddata'
            tesseract_data = requests.get(tesseract_data_url).content

            # Store the trained data in a temporary directory
            TESSDATA_PREFIX = '/tmp/tessdata'
            os.makedirs(TESSDATA_PREFIX, exist_ok=True)
            with open(os.path.join(TESSDATA_PREFIX, 'pan.traineddata'), 'wb') as f:
                f.write(tesseract_data)

            # Set the TESSDATA_PREFIX environment variable
            os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX

            # Perform OCR
            extracted_text = pytesseract.image_to_string(
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                lang='pan',
                config='--psm 6'
            )
            
            filtered_text = filter_punjabi(extracted_text)
            return render_template("index.html", text=filtered_text)
        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template("index.html", text="Error in processing image.")

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
