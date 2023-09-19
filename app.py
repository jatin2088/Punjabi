from flask import Flask, request, render_template, redirect, url_for
import os
import re
import cv2
import random
from google.cloud import vision

app = Flask(__name__)

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/service_account.json"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)

            # Load the image using OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

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

            # Initialize the Vision API client
            client = vision.ImageAnnotatorClient()
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
                filtered_lines.append(filtered_line)

            # Combine the filtered lines with line breaks
            final_text = '\n'.join(filtered_lines)

            return render_template('index.html', text=final_text)

    return render_template('index.html', text=None)

if __name__ == '__main__':
    app.run(debug=True)
