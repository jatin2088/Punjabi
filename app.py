from flask import Flask, render_template, request, jsonify
import os
import re
import cv2
import random
import requests
import numpy as np
from google.cloud import vision
from google.oauth2.service_account import Credentials

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify(error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file")
    
    # Convert the uploaded file to OpenCV image
    nparr = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert processed image back to the format suitable for Google Vision API
    is_success, im_buf_arr = cv2.imencode(".jpg", processed)
    byte_im = im_buf_arr.tobytes()

    # Fetch the service account key from the URL
    service_account_url = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/service_account.json"
    response = requests.get(service_account_url)
    service_account_info = response.json()

    # Create credentials from the service account info
    credentials = Credentials.from_service_account_info(service_account_info)

    # Initialize the Vision API client with the credentials
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=byte_im)

    # Perform text detection
    response = client.text_detection(image=image)
    entire_text = response.text_annotations[0].description

    # Regular expression to match Punjabi characters and spaces
    punjabi_pattern = re.compile("[\u0A00-\u0A7F\s]+")

    # Extract lines while retaining line breaks
    lines = entire_text.split('\n')
    filtered_lines = []

    for line in lines:
        # Filter out any non-Punjabi characters from each line
        filtered_line = ''.join(punjabi_pattern.findall(line))
        
        # Mimic common Tesseract errors
        if random.random() < 0.2:
            filtered_line = filtered_line.replace('ਸ', 'ਸ਼')
        if random.random() < 0.2:
            filtered_line = filtered_line.replace('ਫ', 'ਫ਼')
        if random.random() < 0.2:
            filtered_line = filtered_line.replace('ਜ', 'ਜ਼')
        if random.random() < 0.2:
            filtered_line = filtered_line.replace('ਲ', 'ਲ਼')
        if random.random() < 0.2:
            filtered_line = filtered_line.replace('ਿ', 'ੀ')
        if random.random() < 0.2:
            filtered_line = ' '.join(list(filtered_line))
        if random.random() < 0.2:
            filtered_line = filtered_line.replace(' ', '')
               
        filtered_lines.append(filtered_line)

    # Combine the filtered lines with line breaks
    final_text = '\n'.join(filtered_lines)

    return jsonify(text=final_text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0")
