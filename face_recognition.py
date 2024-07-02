import cv2
from keras.models import model_from_json
import numpy as np
import os

class FaceRecognition:
    def __init__(self, json_file_path, weights_file_path):
        # Load the model architecture
        with open(json_file_path, "r") as json_file:
            model_json = json_file.read()
        self.model = model_from_json(model_json)
        
        # Check if the weights file exists and load the weights
        if os.path.exists(weights_file_path):
            self.model.load_weights(weights_file_path)
        else:
            raise FileNotFoundError("The weights file does not exist.")
        
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define labels for emotion classes
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    def extract_features(self, image):
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        return image

    def recognize_emotion(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        predictions = []
        for (p, q, r, s) in faces:
            # Extract the region of interest (ROI) which contains the face
            face_image = gray_frame[q:q + s, p:p + r]
            face_image = cv2.resize(face_image, (48, 48))
            img = self.extract_features(face_image)
            pred = self.model.predict(img)
            prediction_label = self.labels[np.argmax(pred)]
            predictions.append(((p, q, r, s), prediction_label))
        return predictions
