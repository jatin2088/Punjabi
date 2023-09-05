from flask import Flask, request, render_template
import cv2
import numpy as np
import requests
from flask_cors import CORS
import os
import tensorflow as tf
import pickle
import gdown

app = Flask(__name__)
CORS(app)

GITHUB_REPO_URL = "https://raw.githubusercontent.com/jatin2088/Punjabi/main/"
MODEL_URL = 'https://drive.google.com/uc?id=1lhkyQuBPfJVqU5N2KyxjfuTcMiRB0uh0'
MODEL_PATH = 'trained_model.h5'
LABEL_BINARIZER_PATH = "label_binarizer.pkl"

# Download and load the trained model and LabelBinarizer
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
model = tf.keras.models.load_model(MODEL_PATH)

response = requests.get(GITHUB_REPO_URL + LABEL_BINARIZER_PATH)
with open(LABEL_BINARIZER_PATH, "wb") as f:
    f.write(response.content)

with open(LABEL_BINARIZER_PATH, "rb") as f:
    lb = pickle.load(f)

# Function to predict labels for a new image
def predict_labels(image):
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    predicted_chars_probs = model.predict(image)
    predicted_chars = lb.inverse_transform(predicted_chars_probs.squeeze())
    return ''.join(predicted_chars).strip()

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    if not image_file.filename.lower().endswith('.png'):
        return render_template("index.html", text="Check file format")

    image_name, _ = os.path.splitext(image_file.filename)
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    response = requests.get(GITHUB_REPO_URL + "annotations/" + image_name + '.txt')
    
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
            x_center, y_center, w, h = x_center * 128, y_center * 128, w * 128, h * 128  # Adjust according to your preprocessing
            
            if y_center - prev_y_center > h:
                text += '\n'
            prev_y_center = y_center
            text += char
    else:
        text = predict_labels(image)

    return render_template("index.html", text=text)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
