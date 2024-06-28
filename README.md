# Facial Music Emotion Recommendation System
_(Name to be changed)_
_(Add image)_
---
## Business Part?

## Technicall Stuff

### Computer Vision Task - Facial Emotion Recognition
_(Constantin)_
__Data Preparation__


__Neural Network Structure & Results__



__Production Script__


### MLP Classification Task - DNN Classifier to label Music Mood
_(Luca)_

__Neural Network Structure & Results__

The second part of this project was to build a classifer that can be used to classify music songs depending on their mood. To train our model we used a Kaggle Dataset that had already label over 3000 spotify songs using the audio components of each tracks.

_Here if we had more time we would have maybe train a neural network to this labelling work in a "unsupervised way" using auto-encoders?_

Our neural network as the following structure 5 - 64 -32 -4, using relu activation function within the 2 hidden layer and a softmax activation function in the output layer. We use an Adam optimizer with sparse-cross-entropy loss function looking for accuracy. We did a 80/20 split and trained the network over 50 epochs, resulting on an accury on test of 98,947%, with the followinf confusion matrix:

!['confusion_matrix'](conf_matrix.png)


__Production Script__

_Saving the model components_

In order to put our train model into production for our final application we had to perform several tasks.
First we saved the initial trained model using keras `model.save` method and then in order to lighter the model for production purposes we parsed it using `TFLiteConverter`[green] method. We then also used joblib to load the label encoder + our scaler of our input audio features.

_Production Scripts_

On this part the directory contains the previous mentioned model components, a utils.py (used for initial trained model) and utils2.py (= used with the lighter model) and the key scripts:

- music_emotion_classifier.py

- music_emotion_classifier_app_version.py _(that was then moved to main folder to solve package dependency issues)_

In practice, the following scripts does this following workflow:

`playlist_track_classifier`: 

By passing a spotify playlist idea this function uses the `fetch_playlist_songs`from utils.py/ utils2.py to fetch the different spotify track ids with a limit of 20 per call. - _Here note, that we can fine tune this to avoid an API call 425 errors of the spotify api._

Then we iterate over each track id on the utils.py/utils2.py `get_songs_features` method to fetch the audio features that will be serve as an input in our model to predict the mood of each tracks with the utils.py/utils2.py `predict_track_mood`method.

`converter`:

This function takes as parameters the output of the computer vision model and a spotify playlist id (by default one is already set). It then by using the previous mentioned function `playlist_track_classifier` do the inference and map the the output of the cv model to the 4 different track mood as follows:

```python
label_dict = {
    2: ['happy', 'surprised'],  # "Happy",
    3: ['sad', 'disgusted'],    # "Sad",
    0: ['neutral'],             # "Calm",
    1: ['angry', 'fearful']     # "Energetic"
}
```
_Here note that we could revise the way we map the different outputs_

The converer also retrieve the meta_data of the track such as the id, name, artist, album and preview_url (which will be used with streamlit st.audio component for user experience).

For the `main.py`script only the `converter` function is used.

### How-to

easy-way: `link-to-streamlit-app on public cloud`

1) First git clone the repo

2) Set your Spotify API credentials: client id + client secret

3) Run ```streamlit run main.py``` on your terminal.






