import pandas as pd
import numpy as np

import requests
import spotipy
from dotenv import load_dotenv
import os

from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model
import joblib

import warnings
warnings.filterwarnings('ignore')

# Function to get client credentials manager

# Set client credentials
# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# Load environment variables
env_path = os.path.join(base_dir, '.env')
load_dotenv(env_path)
print('Env.Variables loaded at ' + env_path)

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
print(SPOTIFY_CLIENT_ID)
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
print(SPOTIFY_CLIENT_SECRET)

# Setting up the client credentials manager
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
print('Client credentials manager created')

# Model Loading Part
# Load the trained model
model_path = os.path.join(base_dir, 'music_emotion_classifier.h5')
model = load_model(model_path)

print('Model loaded successfully.')

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

# Get Track Preview URL


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
    preds = get_songs_features(track_id)
    # Pre-process the features to input the model
    preds_features = np.array(preds[0][8:]).reshape(1, -1)
    # Standardize the input features using the loaded scaler
    preds_features_scaled = scaler.transform(preds_features)
    # Predict the emotion
    preds = model.predict(preds_features_scaled)
    # Get the predicted class
    predicted_class = np.argmax(preds, axis=1)
    # Decode the predicted class back to the label
    predicted_emotion = label_encoder.inverse_transform(predicted_class)

    # Print predictied emotion
    print(predicted_emotion[0])

    return predicted_emotion[0]


# testing
if __name__ == '__main__':
    # song = fetch_song('Happy')
    # print(song['id'])
    # #song_features = get_songs_features(song['id'])
    # #print(predict_track_mood(song['id']))

    playlist_songs_ids = fetch_playlist_songs('37i9dQZF1DWTJNOeepZTGy')
    # predict_track_mood(playlist_songs_ids)
    # Only display the mood
    preds_mood = []
    for track_id in playlist_songs_ids:
        # Get the features of the track
        track_values, columns = get_songs_features(track_id)

        # Create a dictionary for the track metadata
        track_meta = dict(zip(columns, track_values))

        # Predict the mood of the track
        predicted_mood = predict_track_mood(track_id)

        # Add the predicted mood to the track metadata
        track_meta['predicted_mood'] = predicted_mood

        preds_mood.append(predicted_mood)

    print(np.unique(preds_mood))

    # output = [0 1 2 3]
