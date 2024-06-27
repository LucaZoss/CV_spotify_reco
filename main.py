import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace
import music_classifier.music_emotions_classifier_app_version as mc
import music_classifier.utils2

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

# Function to reset the page
def reset_page():
    st.session_state.page = 'intro'
    st.experimental_rerun()

# Function for emotion recognition
def image_emotion_recognition(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    try:
        result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']  # Assuming we take the dominant emotion of the first face detected
        else:
            emotion = result['dominant_emotion']
        return emotion
    except ValueError as e:
        return str(e)

# Intro page
if st.session_state.page == 'intro':
    st.title("Welcome to the Image App")
    st.write("Choose how you would like to provide the image:")
    
    option = st.selectbox(
        "Select an option",
        ("Choose an option", "Upload an image", "Take a picture with the webcam")
    )
    
    if option == "Upload an image":
        st.session_state.page = 'upload'
        st.experimental_rerun()
    elif option == "Take a picture with the webcam":
        st.session_state.page = 'camera'
        st.experimental_rerun()

# Upload page
if st.session_state.page == 'upload':
    st.title("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Image uploaded successfully!")
        
        # Call the image emotion recognition function
        emotion = image_emotion_recognition(image)
        st.write("Emotions detected:", emotion)
        
        # Use the detected emotion with the converter function
        converted_output = mc.converter(emotion)
        st.write("Converted Output:", converted_output)
    else:
        st.write("Please upload an image file.")
    
    if st.button("Back"):
        reset_page()

# Camera page
if st.session_state.page == 'camera':
    st.title("Take a Picture")
    picture = st.camera_input("Take a picture")
    
    if picture:
        image = Image.open(picture)
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.write("Picture taken successfully!")
        
        # Call the image emotion recognition function
        emotion = image_emotion_recognition(image)
        st.write("Emotions detected:", emotion)
        
        # Use the detected emotion with the converter function
        converted_output = mc.converter(emotion)
        st.write("Converted Output:", converted_output)
    
    if st.button("Back"):
        reset_page()
