from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Load the model
model_path = "C:/Users/suday/OneDrive/Desktop/Main_project/model.h5"
loaded_model_hdf5 = load_model(model_path)
image_size = 256

class_names = {
    0: "Safe Driving", 1: "Texting - Right Hand", 2: "Talking on the Phone - Right",
    3: "Texting - Left Hand", 4: "Talking on the Phone - Left", 5: "Operating the Radio",
    6: "Drinking", 7: "Reaching Behind", 8: "Hair and Makeup", 9: "Talking to Passenger"
}

def process_image(img):
    img = tf.image.resize(img, (image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('combined_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist('file')
        predictions = []

        for file in files:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = tf.convert_to_tensor(img)
            img = process_image(img)

            # Make prediction
            predictions_hdf5 = loaded_model_hdf5.predict(img)
            predicted_class_hdf5 = class_names[np.argmax(predictions_hdf5[0])]

            predictions.append(predicted_class_hdf5)

        return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)