import pandas as pd
import numpy as np
import streamlit as st

import requests
import spotipy
from dotenv import load_dotenv
import os
import tensorflow as tf
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model
import joblib

import warnings
warnings.filterwarnings('ignore')

# Function to get client credentials manager

# Set client credentials
base_dir = os.path.dirname(os.path.abspath(__file__))
# Load environment variables
env_path = os.path.join(base_dir, '.env')
# load_dotenv(env_path)
# print('Env.Variables loaded at ' + env_path)
# Load environement for streamlit
load_dotenv(env_path)
st.write(st.secrets.load_if_toml_exists())
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET")

# SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
print(SPOTIFY_CLIENT_ID)
# SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
print(SPOTIFY_CLIENT_SECRET)

client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
print('Client credentials manager created')

# Model Loading Part
# Load the trained TFLite model
model_tflite_path = os.path.join(base_dir, 'music_emotion_classifier.tflite')
interpreter = tf.lite.Interpreter(model_path=model_tflite_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler_path = os.path.join(base_dir, 'scaler.joblib')
scaler = joblib.load(scaler_path)
print('Scaler loaded successfully.')

label_encoder_path = os.path.join(base_dir, 'label_encoder.joblib')
label_encoder = joblib.load(label_encoder_path)

# Def Fetch only one song


def fetch_song(song_name):
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    result = sp.search(song_name)
    song = result['tracks']['items'][0]
    return song

# Get Song from Playlist


def fetch_playlist_songs(playlist_id, limit=100):
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    playlist = sp.playlist_tracks(playlist_id, limit=limit)
    songs = playlist['items']
    song_ids = []
    for song in songs:
        song_ids.append(song['track']['id'])
    return song_ids


# Get track preview URL
def fetch_track_preview_url(track_id):
    track_info = sp.track(track_id)
    # This is the direct link to the track's 30-second preview
    preview_url = track_info['preview_url']
    return preview_url


# Get Song Features
def get_songs_features(ids):
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids = meta['id']
    preview_url = meta['preview_url']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track_values = [name, album, artist, ids, preview_url, release_date, popularity,
                    length, acousticness, danceability, liveness, loudness, speechiness]
    columns = ['name', 'album', 'artist', 'ids', 'preview_url', 'release_date', 'popularity',
               'length', 'acousticness', 'danceability', 'liveness', 'loudness', 'speechiness']

    return track_values, columns

# Model Prediction


def predict_track_mood(track_id):
    # Get the features of the song
    track_values, _ = get_songs_features(track_id)
    # Pre-process the features to input the model
    preds_features = np.array(track_values[8:]).reshape(1, -1)
    # Standardize the input features using the loaded scaler
    preds_features_scaled = scaler.transform(preds_features)
    # Set the model input
    interpreter.set_tensor(
        input_details[0]['index'], preds_features_scaled.astype(np.float32))
    # Run inference
    interpreter.invoke()
    # Get the model output
    preds = interpreter.get_tensor(output_details[0]['index'])
    # Get the predicted class
    predicted_class = np.argmax(preds, axis=1)
    # Decode the predicted class back to the label
    predicted_emotion = label_encoder.inverse_transform(predicted_class)

    # Print predicted emotion
    # print(predicted_emotion[0])

    return predicted_emotion[0]


# testing
if __name__ == '__main__':
    song = fetch_song('Happy')
    print(song['id'])
    # song_features = get_songs_features(song['id'])
    print(predict_track_mood(song['id']))

    # playlist_songs_ids = fetch_playlist_songs('37i9dQZF1DXcBWIGoYBM5M')
    # print(playlist_songs_ids)
