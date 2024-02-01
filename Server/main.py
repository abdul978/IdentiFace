import flask
from flask import json, make_response, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import face_recognition
import cv2
import os
import pickle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = flask.Flask(__name__)

# Load your trained or pre-trained object detection model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load face recognition data
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("No file part in the request")
        response = make_response(json.dumps({"message": "No file part in the request"}))
        response.status_code = 400
        return response

    file = request.files['file']
    if file:
        # Read the image data
        image_data = file.read()

        # Perform object detection using the loaded model
        object_detection_results = perform_object_detection(image_data)
        face_recognition_results = perform_face_recognition(image_data)

        results = {
            # "object_detection": object_detection_results,
            "face_recognition": face_recognition_results
        }

        # Return the results as JSON
        response = make_response(json.dumps(results))
        response.status_code = 200
        return response

    response = make_response(json.dumps({"message": "Error processing the image"}))
    response.status_code = 400
    return response

def perform_object_detection(image_data):
    # Preprocess the image data (e.g., resizing, normalization)
    image = preprocess_image(image_data)

    # Use the loaded model for object detection
    predictions = model.predict(image)

    # Post-process the model's output to get object labels
    labels = decode_predictions(predictions, top=5)[0]

    # Return the top object labels and their probabilities
    results = [{"label": label, "probability": float(prob)} for (_, label, prob) in labels]
    return results

def perform_face_recognition(image_data):
    # Convert image to OpenCV format
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert to RGB for face recognition
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # Perform face recognition
    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)

    # Label recognized faces
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if (name in counts and counts[name] >= 6):
            cv2.putText(image, 'Criminal', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Save the annotated image
    annotated_image_path = 'annotated_image.jpg'
    cv2.imwrite(annotated_image_path, image)

    # Return recognized names and the path to the annotated image
    return names, annotated_image_path

# Preprocess the image data (adjust to your model's requirements)
def preprocess_image(image_data):
    image = tf.image.decode_image(image_data)
    image = tf.image.resize(image, (224, 224))  # Adjust dimensions
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Batch dimension
    return image

if __name__ == '__main__':
    app.run(host="192.168.100.2", port=3000, debug=True,threaded=True)
