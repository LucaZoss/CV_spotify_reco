import music_emotions_classifier_app_version as mc
# from deepface import DeepFace
import numpy as np
from PIL import Image
import streamlit as st
import sys
import os
import base64
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv('.env')


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'emotion' not in st.session_state:
    st.session_state.emotion = ''
if 'converted_output' not in st.session_state:
    st.session_state.converted_output = ''
if 'image' not in st.session_state:
    st.session_state.image = None

# Function to reset the page


def reset_page():
    st.session_state.page = 'home'
    st.experimental_rerun()

# Function for emotion recognition


def image_emotion_recognition(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    try:
        result = DeepFace.analyze(
            image_np, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            # Assuming we take the dominant emotion of the first face detected
            emotion = result[0]['dominant_emotion']
        else:
            emotion = result['dominant_emotion']
        return emotion
    except ValueError as e:
        return str(e)


# Home page
if st.session_state.page == 'home':
    st.title("Welcome to the Image Emotion Analyzer")
    st.write("Whether you're feeling happy, sad, or anything in between, our app can find the perfect music to accompany your mood. This app uses advanced machine learning techniques to analyze the dominant emotion in any image you upload or capture. Based on the detected emotion, it suggests a curated list of songs that match the mood.")

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
    st.write(
        "Upload an image from your device. The image should be in JPG, JPEG, or PNG format.")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Image uploaded successfully!")

        if st.button("Analyze Emotions"):
            st.session_state.image = image
            st.session_state.page = 'loading'
            st.experimental_rerun()
    else:
        st.write("Please upload an image file.")

    if st.button("Back"):
        reset_page()

# Camera page
if st.session_state.page == 'camera':
    st.title("Take a Picture")
    st.write("Use your webcam to take a picture.")
    picture = st.camera_input("Take a picture")

    if picture:
        image = Image.open(picture)
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.write("Picture taken successfully!")

        if st.button("Analyze Emotions"):
            st.session_state.image = image
            st.session_state.page = 'loading'
            st.experimental_rerun()

    if st.button("Back"):
        reset_page()

# Loading page
if st.session_state.page == 'loading':
    st.title("Analyzing...")
    st.write("Please wait while we analyze the emotions in the image.")

    with st.spinner("Analyzing emotions..."):
        emotion = image_emotion_recognition(st.session_state.image)
        st.session_state.emotion = emotion
        converted_output = mc.converter(emotion)

        # Print the emotion and the output to the terminal
        print(f"Detected Emotion: {emotion}")
        print("Converted Output:")
        for track in converted_output:
            print(
                f"Name: {track['name']}, Artist: {track['artist']}, Preview URL: {track['preview_url']}")

        # Extract only "name", "artist", and "preview_url" from the converted output
        st.session_state.converted_output = [
            {"name": track["name"], "artist": track["artist"],
                "preview_url": track["preview_url"]}
            for track in converted_output
        ]

        st.session_state.page = 'results'
        st.experimental_rerun()

# Results page
if st.session_state.page == 'results':
    st.title("Emotion Analysis Results")
    st.image(st.session_state.image,
             caption='Analyzed Image', use_column_width=True)
    st.write(f"Detected Emotion: {st.session_state.emotion}")
    st.write("Recommended Songs:")

    for song in st.session_state.converted_output:
        song_name = song['name']
        artist_name = song['artist']
        # Use .get to safely access the dictionary key
        preview_url = song.get('preview_url')

        if preview_url:  # Check if preview URL exists

            st.write(f"Name: {song_name}")
            st.write(f"Artist: {artist_name}")
            st.audio(preview_url)
        else:
            print("Preview URL not found.")

    if st.button("Back"):
        reset_page()
