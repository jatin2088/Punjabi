from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.json

    # Decode base64 image
    image_data = base64.b64decode(data["image"].split(",")[-1])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

    # Your image processing code starts here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, width = img.shape

    annotation = data["annotation"].splitlines()

    text = ''
    prev_y_center = 0
    prev_x_center = 0
    prev_w = 0
    for line in annotation:
        line = line.split()
        char, x_center, y_center, w, h = line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])

        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x, y = int(x_center - w//2), int(y_center - h//2)
        w, h = int(w), int(h)

        if y_center - prev_y_center > h:
            text += '\n'
        else:
            prev_right_edge = prev_x_center + prev_w / 2
            curr_left_edge = x_center - w / 2

            if prev_right_edge < curr_left_edge:
                text += ' '

        prev_y_center = y_center
        prev_x_center = x_center
        prev_w = w
        text += char

    # Your image processing code ends here

    return jsonify({'result': text})

if __name__ == '__main__':
    app.run(debug=True)
