import sys
import os
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace
import music_classifier.music_emotions_classifier_app_version as mc

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'emotion' not in st.session_state:
    st.session_state.emotion = ''
if 'converted_output' not in st.session_state:
    st.session_state.converted_output = ''
if 'image' not in st.session_state:
    st.session_state.image = None

# Read and encode the logo image
with open(r"C:\Users\Usuario\Downloads\DALL·E 2024-06-27 18.34.49 - A modern and minimalistic logo for an app that recognizes emotions from facial expressions and recommends music. The logo features a stylized face sho.webp", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ecf0f1;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        max-width: 800px;
    }
    h1, h2, h3 {
        color: #34495e;
        text-align: center;
    }
    p {
        color: #7f8c8d;
        font-size: 18px;
        text-align: center;
    }
    .stButton button {
        background-color: #2980b9;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
        display: block;
        margin: 20px auto;
    }
    .stButton button:hover {
        background-color: #3498db;
        color: white;
    }
    .uploaded-image, .analyzed-image {
        text-align: center;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .result-item {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
    }
    .result-item h4 {
        margin: 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: #bdc3c7;
    }
    .navbar {
        background-color: #2980b9;
        overflow: hidden;
    }
    .navbar a {
        float: left;
        display: block;
        color: white;
        text-align: center;
        padding: 14px 20px;
        text-decoration: none;
        font-size: 17px;
    }
    .navbar a:hover {
        background-color: #3498db;
        color: white;
    }
    .navbar .logo {
        float: left;
        padding: 0 20px;
        height: 100px;
    }
    .navbar img {
        height: 100px;
        padding: 5px 10px;
    }
    .accordion {
        background-color: #ecf0f1;
        color: #444;
        cursor: pointer;
        padding: 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 18px;
        transition: 0.4s;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .active, .accordion:hover {
        background-color: #bdc3c7;
    }
    .panel {
        padding: 0 18px;
        background-color: white;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
    }
    .icon {
        margin-right: 10px;
    }
    .white-text-button .stButton button {
        color: white !important;
    }
    </style>
    <script>
    var acc = document.getElementsByClassName("accordion");
    for (var i = 0; i < acc.length; i++) {
        acc[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var panel = this.nextElementSibling;
            if (panel.style.maxHeight) {
                panel.style.maxHeight = null;
            } else {
                panel.style.maxHeight = panel.scrollHeight + "px";
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)

# Function to reset the page
def reset_page():
    st.session_state.page = 'home'
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

# Navbar
st.markdown("""
    <div class="navbar">
        <img src="data:image/png;base64,{}" class="logo" alt="App Logo">
        <a href="javascript:location.reload()">Home</a>
        <a href="javascript:st.session_state.page='about';window.location.reload();">About</a>
        <a href="javascript:st.session_state.page='contact';window.location.reload();">Contact</a>
    </div>
    """.format(encoded_string), unsafe_allow_html=True)

# Home page
if st.session_state.page == 'home':
    st.markdown("<div class='main'><h1>Welcome to the Image Emotion Analyzer</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Whether you're feeling happy, sad, or anything in between, our app can find the perfect music to accompany your mood. This app uses advanced machine learning techniques to analyze the dominant emotion in any image you upload or capture. Based on the detected emotion, it suggests a curated list of songs that match the mood.</p></div>", unsafe_allow_html=True)
    
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

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>", unsafe_allow_html=True)

# Upload page
if st.session_state.page == 'upload':
    st.markdown("<div class='main'><h2>Upload an Image</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Upload an image from your device. The image should be in JPG, JPEG, or PNG format.</p></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.markdown("<div class='main'><p>Image uploaded successfully!</p></div>", unsafe_allow_html=True)
        
        if st.button("Analyze Emotions", key="analyze_button"):
            st.session_state.image = image
            st.session_state.page = 'loading'
            st.experimental_rerun()
    else:
        st.markdown("<div class='main'><p>Please upload an image file.</p></div>", unsafe_allow_html=True)
    
    if st.button("Back", key="back_button"):
        reset_page()

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>", unsafe_allow_html=True)

# Camera page
if st.session_state.page == 'camera':
    st.markdown("<div class='main'><h2>Take a Picture</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Use your webcam to take a picture.</p></div>", unsafe_allow_html=True)
    picture = st.camera_input("Take a picture")
    
    if picture:
        image = Image.open(picture)
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.markdown("<div class='main'><p>Picture taken successfully!</p></div>", unsafe_allow_html=True)
        
        if st.button("Analyze Emotions", key="analyze_button_camera"):
            st.session_state.image = image
            st.session_state.page = 'loading'
            st.experimental_rerun()
    
    if st.button("Back", key="back_button_camera"):
        reset_page()

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>", unsafe_allow_html=True)

# Loading page
if st.session_state.page == 'loading':
    st.markdown("<div class='main'><h2>Analyzing...</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Please wait while we analyze the emotions in the image.</p></div>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing emotions..."):
        emotion = image_emotion_recognition(st.session_state.image)
        st.session_state.emotion = emotion
        converted_output = mc.converter(emotion)
        
        # Extract only "name" and "artist" from the converted output
        st.session_state.converted_output = [
            {"name": track["name"], "artist": track["artist"]}
            for track in converted_output
        ]
        
        st.session_state.page = 'results'
        st.experimental_rerun()

# Results page
if st.session_state.page == 'results':
    st.markdown("<div class='main'><h2>Emotion Analysis Results</h2></div>",
                unsafe_allow_html=True)
    st.image(st.session_state.image, caption='Analyzed Image', use_column_width=True)
    st.markdown(
        f"<div class='main'><h2 style='text-align: center; color: black; font-size: 48px;'>{st.session_state.emotion}</h2></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='main'><h3 style='text-align: center; color: grey; font-size: 36px;'>Recommended Songs:</h3></div>",
        unsafe_allow_html=True
    )
    for song in st.session_state.converted_output:
        song_name = song['name']
        artist_name = song['artist']
        preview_url = song.get('preview_url')  # Use .get to safely access the dictionary key

        st.markdown(
            f"<div class='result-item'><h4><strong>Name:</strong> {song_name}<br><strong>Artist:</strong> {artist_name}</h4></div>",
            unsafe_allow_html=True
        )
        
        if preview_url:  # Check if preview URL exists
            st.audio(preview_url)
        else:
            st.markdown(
                "<div class='result-item'><p>Preview not available.</p></div>",
                unsafe_allow_html=True
            )

    if st.button("Back", key="back_button_results"):
        reset_page()

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>",
                unsafe_allow_html=True)

# About page
if st.session_state.page == 'about':
    st.markdown("<div class='main'><h2>About Image Emotion Analyzer</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>This app uses advanced machine learning techniques to analyze the dominant emotion in any image you upload or capture. Based on the detected emotion, it suggests a curated list of songs that match the mood.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Whether you're feeling happy, sad, or anything in between, our app can find the perfect music to accompany your mood.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><h3>How it Works</h3></div>", unsafe_allow_html=True)
    st.markdown("""
        <button class="accordion">Emotion Detection</button>
        <div class="panel">
          <p>We use DeepFace, a cutting-edge facial recognition library, to detect emotions from images.</p>
        </div>
        <button class="accordion">Song Recommendations</button>
        <div class="panel">
          <p>Our system uses a trained model to recommend songs that match the detected emotions.</p>
        </div>
        <button class="accordion">Easy Integration</button>
        <div class="panel">
          <p>The app is designed to be user-friendly and easily integrates with your existing music library.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Back to Home", key="back_button_about"):
        reset_page()

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>", unsafe_allow_html=True)

# Contact page
if st.session_state.page == 'contact':
    st.markdown("<div class='main'><h2>Contact Us</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>If you have any questions, feedback, or need support, feel free to reach out to us.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><h3>Contact Information</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Email: support@imageemotionanalyzer.com</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><p>Phone: +1 234 567 890</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='main'><h3>Send Us a Message</h3></div>", unsafe_allow_html=True)
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    if st.button("Send Message", key="send_message_button"):
        if name and email and message:
            st.markdown("<div class='main'><p>Thank you for your message! We will get back to you soon.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='main'><p>Please fill out all fields before sending your message.</p></div>", unsafe_allow_html=True)

    if st.button("Back to Home", key="back_button_contact"):
        reset_page()

    st.markdown("<div class='footer'>© 2023 Image Emotion Analyzer</div>", unsafe_allow_html=True)
