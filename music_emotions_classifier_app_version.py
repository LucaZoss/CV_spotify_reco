import music_classifier.utils2 as utils2
import pandas as pd


def playlist_track_classifier(playlist_id):
    # Fetch the track IDs from the playlist
    track_ids = utils2.fetch_playlist_songs(playlist_id, limit=100)

    # Dictionary to hold the results
    results = {}

    # Iterate over each track ID
    for track_id in track_ids:
        # Get the features of the track
        track_values, columns = utils2.get_songs_features(track_id)

        # Create a dictionary for the track metadata
        track_meta = dict(zip(columns, track_values))

        # Predict the mood of the track
        predicted_mood = utils2.predict_track_mood(track_id)

        # Add the predicted mood to the track metadata
        track_meta['predicted_mood'] = predicted_mood

        # Add the track metadata with the predicted mood to the results
        results[track_id] = track_meta

    return results


def converter(output_of_cv_model: str, playlist_id: str = '37i9dQZF1DXcBWIGoYBM5M'):
    # Simulating the function call to a hypothetical playlist_track_classifier
    results = playlist_track_classifier(playlist_id)

    label_dict = {
        2: ['happy', 'surprised'],  # "Happy",
        3: ['sad', 'disgusted'],    # "Sad",
        0: ['neutral'],             # "Calm",
        1: ['angry', 'fearful']     # "Energetic"
    }

    mood_index = None
    for key, values in label_dict.items():
        if output_of_cv_model in values:
            mood_index = key
            break

    if mood_index is not None:
        output = [{
            "id": track,
            "name": results[track]['name'],
            "artist": results[track]['artist'],
            "album": results[track]['album'],
            # Including preview_url in the output
            "preview_url": results[track]['preview_url'],
            "predicted_mood": results[track]['predicted_mood']
        } for track in results if results[track]['predicted_mood'] == mood_index]
        return output
    else:
        print("Are you a Zombie Today?")
        return None


if __name__ == '__main__':
    # Test the function
    # 37i9dQZF1DWZUozJiHy44Y #37i9dQZF1DZ06evO3M0Fbi
    output = converter('neutral', playlist_id='26xMbGyBhFCtrwAYsXefRq')
    print(output)
