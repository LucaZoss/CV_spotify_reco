import spotipy
from spotipy.oauth2 import SpotifyOAuth


def get_song_features(song_id):
    features = sp.audio_features(song_id)[0]
    return {
        'danceability': features['danceability'],
        'energy': features['energy'],
        'key': features['key'],
        'loudness': features['loudness'],
        'mode': features['mode'],
        'speechiness': features['speechiness'],
        'acousticness': features['acousticness'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'valence': features['valence'],
        'tempo': features['tempo']
    }

# Collect features for the user's saved tracks
user_song_features = []
for item in results['items']:
    track = item['track']
    song_id = track['id']
    features = get_song_features(song_id)
    user_song_features.append(features)

# Create a DataFrame
user_songs_df = pd.DataFrame(user_song_features)
print(user_songs_df)



# Standardize the user's song features
user_songs_scaled = scaler.transform(user_songs_df)

# Predict emotions
predicted_emotions = model.predict(user_songs_scaled)
predicted_emotion_labels = label_encoder.inverse_transform(predicted_emotions.argmax(axis=1))

# Add the predicted emotions to the DataFrame
user_songs_df['emotion'] = predicted_emotion_labels
print(user_songs_df)


#recommend
def recommend_songs(emotion, n=5):
    recommended_songs = user_songs_df[user_songs_df['emotion'] == emotion].sample(n)
    return recommended_songs

# Example usage
emotion_detected = 'happy'  # This would be the output from the YOLO model
recommended_songs = recommend_songs(emotion_detected)
print(recommended_songs)
