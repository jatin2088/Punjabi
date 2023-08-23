from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

@app.route('/upload', methods=['POST'])
def upload():
    # Get the image file and image name from the request
    image_file = request.files['image']
    image_name = request.form['imageName']

    # Convert the uploaded image to a NumPy array
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Image processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = img.shape

    # Open the annotations file with the same name as the image (without its extension)
    image_name = image_name.split('.')[0]
    try:
        with open(f"annotations/{image_name}.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return jsonify({"error": "Not a Valid Image", "text": None})

    text = ''
    prev_y_center = 0
    prev_x_center = 0
    prev_w = 0
    for line in lines:
        line = line.strip()

        # If the line is empty, it indicates a space
        if not line:
            text += ' '
            continue

        line = line.split()
        char, x_center, y_center, w, h = line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])

        # Denormalize the coordinates
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height

        x, y = int(x_center - w // 2), int(y_center - h // 2)
        w, h = int(w), int(h)

        if y_center - prev_y_center > h:
            text += '\n'

        prev_y_center = y_center
        prev_x_center = x_center
        prev_w = w

        text += char

    return jsonify({"error": None, "text": text})

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
