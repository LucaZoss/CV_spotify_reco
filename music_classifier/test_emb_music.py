# app.py
import streamlit as st
import utils  # This imports the utility functions from utils.py


def display_tracks(playlist_id):
    track_ids = utils.fetch_playlist_songs(playlist_id)
    for track_id in track_ids:
        # Assuming 'sp' is also accessible from utils or recreate the Spotify client here
        track_info = utils.sp.track(track_id)
        track_name = track_info['name']
        track_artist = track_info['artists'][0]['name']
        preview_url = utils.fetch_track_preview_url(track_id)
        if preview_url:
            st.write(f"**{track_name}** by {track_artist}")
            st.audio(preview_url)
        else:
            st.write(
                f"**{track_name}** by {track_artist} (Preview not available)")


if __name__ == "__main__":
    st.title("Spotify Track Sampler")
    playlist_input = st.text_input("Enter a Spotify Playlist ID:")
    if st.button("Load Tracks"):
        display_tracks(playlist_input)
