import time
import random
import subprocess
import os
import cv2
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from music21 import stream, note, midi, chord
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import base64

app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load models safely
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Hugging Face text emotion model
try:
    text_emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
except Exception as e:
    logging.error(f"Error loading text emotion model: {e}")
    text_emotion_model = None

# Load the facial emotion model
try:
    class FaceEmotionModel(nn.Module):
        def __init__(self):
            super(FaceEmotionModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.fc1 = nn.Linear(32 * 24 * 24, 7)  # Assuming 7 emotions

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    face_emotion_model = FaceEmotionModel().to(device)
    face_emotion_model.load_state_dict(torch.load('emotion_model/emotion_model.pth', map_location=device))
    face_emotion_model.eval()
except Exception as e:
    logging.error(f"Error loading facial emotion model: {e}")
    face_emotion_model = None

class AudioEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioEmotionModel, self).__init__()

        # CNN layers
        self.cnn1 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 2, 128)  # Bidirectional LSTM output
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.cnn1(x)))  # Apply CNN1
        x = torch.relu(self.batchnorm2(self.cnn2(x)))  # Apply CNN2

        x = x.permute(0, 2, 1)  # Change shape for LSTM (batch, seq_len, features)
        x, _ = self.lstm(x)  # LSTM processing

        x = self.fc1(torch.relu(x[:, -1, :]))  # Take the last LSTM output (final time step)
        x = self.dropout1(x)
        x = self.fc2(x)  # Final output
        return x

# Initialize the model
input_size = 40  # Number of input features (MFCC features)
num_classes = 7  # Number of emotions (output classes)
audio_emotion_model = AudioEmotionModel(num_classes)

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Spotify credentials
SPOTIFY_CLIENT_ID = 'da97eacd47724370b42c50a0e2666a56'
SPOTIFY_CLIENT_SECRET = 'fdaa127b31d64359918fdd12960b68f9'

access_token = None
token_expires_at = 0  # Timestamp when token expires

def get_spotify_access_token():
    global access_token, token_expires_at

    # Check if the token is still valid
    if access_token and time.time() < token_expires_at:
        return access_token  # Return cached token if valid

    # Encode client_id and client_secret as base64
    auth_string = f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'
    b64_auth_string = base64.b64encode(auth_string.encode()).decode()

    # Request a new token
    headers = {
        'Authorization': f'Basic {b64_auth_string}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}

    response = requests.post('https://accounts.spotify.com/api/token', headers=headers, data=data)

    if response.status_code == 200:
        response_data = response.json()
        access_token = response_data['access_token']
        expires_in = response_data['expires_in']  # Typically 3600 seconds (1 hour)
        token_expires_at = time.time() + expires_in - 60  # Refresh 1 min before expiration
        logging.info(f"Access Token: {access_token}")  # Log the access token

        logging.info(f"Spotify Access Token refreshed. Expires in {expires_in} seconds.")
        return access_token
    else:
        logging.error(f"Error retrieving Spotify access token: {response.json()}")
        return None

# Function to get Spotify recommendations based on emotion
def get_spotify_recommendation(emotion):
    access_token = get_spotify_access_token()
    if not access_token:
        return {"error": "Unable to fetch Spotify access token"}

    # Log the received emotion to verify
    logging.debug(f"Received emotion: {emotion}")

    # Normalize emotion to capitalize the first letter
    emotion = emotion.capitalize()

    spotify_playlists = {
        "Joy": "6NtPUvOEl8fvZYbnfM5BkU",
        "Sadness": "7MAWFKUIo84IugV9StM0lU",
        "Neutral": "51R2TAHW47vO1KsWqE8vu1",
        "Fear": "3EYrdOiSjz3Ki8i57f9au5",
        "Surprise": "5bpM6C4xbahBVxLLf3uycJ",
        "Anger": "3ZRaqe2sjBeN2aMEfqugst",
        "Disgust": "3ZRaqe2sjBeN2aMEfqugst"
    }

    # Log if emotion isn't in the dictionary
    if emotion not in spotify_playlists:
        logging.error(f"Emotion '{emotion}' not found in the playlist dictionary.")
        return {"error": f"Unknown emotion '{emotion}'"}

    # Get the playlist ID for the emotion
    playlist_id = spotify_playlists.get(emotion, "6NtPUvOEl8fvZYbnfM5BkU")  # Default to "Happy" playlist

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    # Fetch playlist tracks
    try:
        response = requests.get(f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks', headers=headers)
        if response.status_code == 200:
            results = response.json()
            track_list = [{"name": item['track']['name'], "url": item['track']['external_urls']['spotify']} for item in results['items']]
            return {"music": track_list}
        else:
            logging.error(f"Error fetching playlist tracks: {response.json()}")
            return {"error": "Error fetching playlist tracks"}
    except Exception as e:
        logging.error(f"Error in get_spotify_recommendation: {str(e)}")
        return {"error": str(e)}

def handle_music_request(emotion, play_generated):
    """Decides whether to generate music or return a Spotify recommendation."""
    if play_generated:
        return generate_music(emotion)  # Return MP3 file path
    else:
        # Fetch dynamic Spotify recommendations
        recommendations = get_spotify_recommendation(emotion)
        return recommendations  # Return the list of recommended tracks

@app.route('/detect-emotion-text', methods=['POST'])
def detect_emotion_text():
    if text_emotion_model is None:
        return jsonify({'error': 'Text emotion model not initialized'}), 500
    try:
        data = request.get_json()
        text = data['text']
        play_generated = data.get('play_generated', True)  # Get play_generated from the request (default to True)

        # Predict emotion from text
        prediction = text_emotion_model(text)
        detected_emotion = prediction[0]['label']

        # Get music recommendation or generated music based on user preference
        music_or_recommendations = handle_music_request(detected_emotion, play_generated)

        # Return response based on play_generated flag
        if play_generated:
            music_url = f"http://192.168.0.105:5000/music/{os.path.basename(music_or_recommendations)}" if music_or_recommendations else None
            return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
        else:
            return jsonify({'emotion': detected_emotion, 'spotify_recommendations': music_or_recommendations}), 200

    except Exception as e:
        logging.error(f"Error in detect_emotion_text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-emotion-audio', methods=['POST'])
def detect_emotion_audio(analyze_audio_emotion=None):
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        # Get play_generated flag from request
        data = request.get_json()
        play_generated = data.get('play_generated', True)  # Default to True if not provided

        audio_file = request.files['audio']
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) == 0:
            return jsonify({'error': 'Audio file is empty or corrupted'}), 400

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        detected_emotion = analyze_audio_emotion(mfccs)

        # Get music recommendation or generate music
        music_or_recommendations = handle_music_request(detected_emotion, play_generated)

        if play_generated:
            music_url = f"http://192.168.0.105:5000/music/{os.path.basename(music_or_recommendations)}" if music_or_recommendations else None
            return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
        else:
            return jsonify({'emotion': detected_emotion, 'spotify_recommendations': music_or_recommendations}), 200

    except Exception as e:
        logging.error(f"Error in detect_emotion_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-emotion-face', methods=['POST'])
def detect_emotion_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Get play_generated flag from form-data
        play_generated = request.form.get('play_generated', True)  # Default to True if not provided

        image_file = request.files['image']
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        # Here you would preprocess the image and pass it through the facial emotion model
        detected_emotion = "Neutral"  # For now, returning Neutral as a placeholder

        # Get music recommendation or generate music
        music_or_recommendations = handle_music_request(detected_emotion, play_generated)

        if play_generated:
            music_url = f"http://192.168.0.105:5000/music/{os.path.basename(music_or_recommendations)}" if music_or_recommendations else None
            return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
        else:
            return jsonify({'emotion': detected_emotion, 'spotify_recommendations': music_or_recommendations}), 200

    except Exception as e:
        logging.error(f"Error in detect_emotion_face: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/music/<filename>')
def serve_music(filename):
    return send_from_directory('static', filename)

def generate_music(emotion):
    try:
        # Initialize the music stream
        s = stream.Stream()

        # Define note ranges and durations to randomize
        note_range = {
            "Happy": ["C4", "E4", "G4", "D4", "F4", "A4"],
            "Sad": ["A3", "C4", "E4", "D3", "F3"],
            "Neutral": ["D4", "F4", "A4", "C4", "E3"],
            "Fear": ["F3", "A3", "C4", "D3"],
            "Surprise": ["G3", "B3", "D4", "E3"],
            "Angry": ["E3", "G3", "B3", "C4", "A3"],
            "Disgust": ["F3", "D4", "B3", "C3"]
        }

        # Randomly select notes for the given emotion
        selected_notes = note_range.get(emotion, note_range["Happy"])

        # Randomly generate a melody
        for _ in range(10):  # Generate 10 notes for the melody
            pitch = random.choice(selected_notes)
            duration = random.choice([0.5, 1.0, 1.5])  # Random durations: 0.5, 1.0, or 1.5 beats
            s.append(note.Note(pitch, quarterLength=duration))

        # Save the stream to a MIDI file
        midi_path = f"generated_music/{emotion}_{random.randint(1000, 9999)}.mid"
        mf = midi.translate.music21ObjectToMidiFile(s)
        mf.open(midi_path, 'wb')
        mf.write()
        mf.close()

        # Convert MIDI to MP3 (using FluidSynth or other methods)
        mp3_path = f"static/{emotion}_{random.randint(1000, 9999)}.mp3"
        subprocess.run(["fluidsynth", "-ni", "soundfont.sf2", midi_path, "-F", mp3_path, "-r", "44100"])

        return mp3_path
    except Exception as e:
        logging.error(f"Error generating music: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000 ,threaded=True)
