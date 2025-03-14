import time
import random
import subprocess
import os
import cv2
import numpy as np
import librosa
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
import torch
from torch import nn
import torch.nn.functional as F
from music21 import stream, note, midi, chord ,tempo , instrument
import logging


app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral','Sad', 'Surprise']

# Load models safely
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Hugging Face text emotion model
try:
    text_emotion_model = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt",  # Force PyTorch instead of TensorFlow
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading text emotion model: {e}")
    text_emotion_model = None

# Load the facial emotion model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

try:
    face_emotion_model = EmotionCNN().to(device)
    state_dict = torch.load('D:/mscproject/enigmasoundbackend/emotion_model/best_emotion_model.pth', map_location=device)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral','Sad', 'Surprise']
    face_emotion_model.load_state_dict(state_dict)
    face_emotion_model.eval()
except Exception as e:
    logging.error(f"Error loading facial emotion model: {e}")
    face_emotion_model = None

# Face detection
face_cascade = cv2.CascadeClassifier('D:\M.Sc.IT\partII\CV practical\Images\haarcascades\haarcascade_frontalface_default.xml')

if face_cascade.empty():
    logging.error("Error loading face cascade classifier.")

class AudioEmotionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioEmotionModel, self).__init__()
        self.cnn1 = torch.nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm1d(128)
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(64 * 2, 128)  # *2 due to bidirectional LSTM
        self.dropout1 = torch.nn.Dropout(0.3)  # Tuning might be needed
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through CNN layers with batch normalization and ReLU
        x = torch.relu(self.batchnorm1(self.cnn1(x)))
        x = torch.relu(self.batchnorm2(self.cnn2(x)))

        # LSTM expects input shape: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, seq_len, channels)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Use the last time step's output (x[:, -1, :]) after LSTM
        x = self.fc1(torch.relu(x[:, -1, :]))  # Last time step (bidirectional)
        x = self.dropout1(x)  # Apply dropout for regularization
        x = self.fc2(x)  # Output layer

        return x

# Load the trained model from .pth file
MODEL_PATH = "D:/mscproject/enigmasoundbackend/emotion_model/audio_emotion_model.pth"  # Replace with actual path to model file
num_classes = 8  # Number of emotions (as you have from RAVDESS)
emotion_labels = ['Happy', 'Sad', 'Neutral', 'Fear', 'Surprise', 'Angry', 'Disgust', 'Calm']

# Initialize the model and load the weights
audio_emotion_model = AudioEmotionModel(num_classes)
audio_emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
audio_emotion_model.eval()  # Set model to evaluation mode

def analyze_audio_emotion(mfccs):
    """Predict emotion from MFCC features using the loaded PyTorch model."""
    try:
        # Convert MFCCs to PyTorch tensor and reshape to the expected input shape
        mfccs = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)  # Shape: (1, seq_len, 40)

        # Get model prediction
        with torch.no_grad():  # Ensure no gradient tracking during inference
            output = audio_emotion_model(mfccs)
            predicted_label = torch.argmax(output, dim=1).item()  # Get the predicted class index

        # Return the corresponding emotion label
        return emotion_labels[predicted_label]

    except Exception as e:
        # Enhanced error handling for debugging
        print(f"Error in analyze_audio_emotion: {str(e)}")
        print(f"MFCC input shape: {mfccs.shape}")  # Log shape of input for troubleshooting
        return "Unknown"

@app.route('/detect-emotion-text', methods=['POST'])
def detect_emotion_text():
    if text_emotion_model is None:
        logging.error("Text emotion model is not initialized!")
        return jsonify({'error': 'Text emotion model not initialized'}), 500

    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        prediction = text_emotion_model(text)
        detected_emotion = prediction[0]['label']

        play_generated = data.get('play_generated', True)
        if play_generated:
            music_path = generate_music(detected_emotion)
            if music_path and os.path.exists(music_path):
                music_url = f"http://192.168.0.104:5000/static/{os.path.basename(music_path)}"
                return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
            else:
                return jsonify({'error': 'Music file not found'}), 404

        return jsonify({'emotion': detected_emotion}), 200

    except Exception as e:
        logging.error(f"Error in emotion detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-emotion-audio', methods=['POST'])
def detect_emotion_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        data = request.form
        play_generated = data.get('play_generated', 'true').lower() == 'true'

        audio_file = request.files['audio']
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) == 0:
            return jsonify({'error': 'Audio file is empty or corrupted'}), 400

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        detected_emotion = analyze_audio_emotion(mfccs)

        if play_generated:
            music_path = generate_music(detected_emotion)
            if music_path and os.path.exists(music_path):
                music_url = f"http://192.168.0.104:5000/static/{os.path.basename(music_path)}"
                return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
            else:
                return jsonify({'error': 'Music file not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-emotion-face', methods=['POST'])
def detect_emotion_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Get play_generated flag from form-data (Convert to Boolean)
        play_generated = request.form.get('play_generated', 'true').lower() == 'true'

        image_file = request.files['image']
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        # Process the first detected face (assuming one face)
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert face to grayscale
        face = cv2.resize(face, (48, 48))  # Resize to match model input size
        face = face.astype('float32') / 255  # Normalize pixel values
        face = (face - 0.5) / 0.5  # Match normalization during training

        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=0)  # Add channel dimension (grayscale)

        face_tensor = torch.from_numpy(face).float().to(device)

        with torch.no_grad():
            output = face_emotion_model(face_tensor)
            _, predicted = torch.max(output, 1)
            detected_emotion = emotion_labels[predicted.item()]

        # Generate music based on detected emotion
        if play_generated:
            music_path = generate_music(detected_emotion)
            if music_path and os.path.exists(music_path):  # Ensure the music file was generated successfully
                music_url = f"http://192.168.0.104:5000/static/{os.path.basename(music_path)}"
                return jsonify({'emotion': detected_emotion, 'music_url': music_url}), 200
            else:
                logging.error("Generated music file not found!")
                return jsonify({'error': 'Music file not found'}), 404
        else:
            return jsonify({'emotion': detected_emotion}), 200

    except Exception as e:
        logging.error(f"Error in detect_emotion_face: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Define instruments for each emotion
instrument_mapping = {
    "Happy": [instrument.Piano(), instrument.AcousticGuitar(), instrument.Flute()],
    "Sad": [instrument.Violin(), instrument.Piano(), instrument.Violoncello()],
    "Neutral": [instrument.Piano(), instrument.Clarinet(), instrument.Vibraphone()],
    "Angry": [instrument.ElectricGuitar(), instrument.Trumpet(), instrument.BassDrum()],
    "Fear": [instrument.Harp(), instrument.Bassoon(), instrument.TubularBells()],
    "Surprise": [instrument.Marimba(), instrument.Xylophone(), instrument.Trumpet()],
    "Disgust": [instrument.Bassoon(), instrument.Clarinet(), instrument.Harpsichord()],
    "Love": [instrument.Piano(), instrument.Violin(), instrument.Flute()]
}

# Emotion-based note selection
note_range = {
    "Happy": ["C4", "E4", "G4", "D4", "F4", "A4", "B4", "C5"],
    "Sad": ["A3", "C4", "E4", "D3", "F3", "G3"],
    "Neutral": ["D4", "F4", "A4", "C4", "E3", "G3"],
    "Angry": ["E3", "G3", "B3", "C4", "A3", "D4"],
    "Fear": ["F3", "A3", "C4", "D3", "E4"],
    "Surprise": ["G3", "B3", "D4", "E3", "C5"],
    "Disgust": ["F3", "D4", "B3", "C3", "G3"],
    "Love": ["C4", "E4", "G4", "A4", "F4", "D4", "B4"]

}

# Tempo range (BPM)
tempo_ranges = {
    "Happy": (120, 150),
    "Sad": (50, 80),
    "Neutral": (90, 110),
    "Angry": (140, 170),
    "Fear": (100, 130),
    "Surprise": (110, 140),
    "Disgust": (90, 120),
    "Love": (100, 130)

}

# Chord progressions
chord_progressions = {
    "Happy": [["C4", "E4", "G4"], ["F4", "A4", "C5"], ["G4", "B4", "D5"]],
    "Sad": [["A3", "C4", "E4"], ["D3", "F3", "A3"], ["E3", "G3", "B3"]],
    "Neutral": [["D4", "F4", "A4"], ["C4", "E4", "G4"], ["G3", "B3", "D4"]],
    "Angry": [["E3", "G3", "B3"], ["D3", "F3", "A3"], ["C4", "E4", "G4"]],
    "Fear": [["F3", "A3", "C4"], ["D3", "F3", "A3"], ["E3", "G3", "B3"]],
    "Surprise": [["G3", "B3", "D4"], ["E3", "G3", "C4"], ["C4", "E4", "A4"]],
    "Disgust": [["F3", "D4", "B3"], ["C3", "E3", "G3"], ["A3", "C4", "E4"]],
    "Love": [["C4", "E4", "G4"], ["A3", "C4", "E4"], ["F4", "A4", "D5"]]

}
label_mapping = {
    "joy": "Happy",
    "happy": "Happy",
    "sad": "Sad",
    "sadness":"Sad",
    "fear": "Fear",
    "surprise": "Surprise",
    "anger": "Angry",
    "love": "Love",
    "disgust": "Disgust",
    "neutral": "Neutral"
}

def generate_music(detected_emotion):
    try:
        timestamp = str(time.time())  # Use current timestamp to create a unique seed
        random.seed(timestamp)
        # Normalize emotion case to match dictionary keys
        detected_emotion = label_mapping.get(detected_emotion.lower(), detected_emotion.capitalize())

        if detected_emotion not in instrument_mapping:
            logging.warning(f"Emotion '{detected_emotion}' not found. Using 'Neutral' as fallback.")
            detected_emotion = "Neutral"

        # Create a new music stream
        s = stream.Stream()

        # Get emotion-based settings
        instruments = instrument_mapping.get(detected_emotion, instrument_mapping["Neutral"])
        selected_notes = note_range.get(detected_emotion, note_range["Neutral"])
        tempo_range = tempo_ranges.get(detected_emotion, tempo_ranges["Neutral"])
        chords = chord_progressions.get(detected_emotion, chord_progressions["Neutral"])

        # Ensure the main instrument is not duplicated
        if not any(isinstance(el, instrument.Instrument) for el in s):
            s.append(instruments[0])

        # Set tempo
        tempo_value = random.randint(*tempo_range)
        s.append(tempo.MetronomeMark(number=tempo_value))

        beats_per_second = tempo_value / 60.0
        total_beats_needed = 30 * beats_per_second
        current_beats = 0
        chord_counter = 0
        instrument_switched = False

        while current_beats < total_beats_needed:
            section = "Verse" if current_beats < total_beats_needed / 2 else "Chorus"

            # Add second instrument only once in the chorus
            if section == "Chorus" and not instrument_switched and len(instruments) > 1:
                s.append(instruments[1])
                instrument_switched = True

            if random.random() < 0.3:  # 30% chance of a chord
                harmony = chord.Chord(chords[chord_counter % len(chords)])
                harmony.quarterLength = random.choice([0.5, 1.0, 1.5, 2.0])
                s.append(harmony)
                current_beats += harmony.quarterLength
                chord_counter += 1
            else:
                pitch = random.choice(selected_notes)
                duration = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
                melody_note = note.Note(pitch, quarterLength=duration)

                # Occasionally add the third instrument, ensuring no duplication
                if random.random() < 0.1 and len(instruments) > 2 and not any(isinstance(el, type(instruments[2])) for el in s):
                    s.append(instruments[2])

                s.append(melody_note)
                current_beats += duration

        # Save as MIDI
        random_number = random.randint(1000, 9999)
        midi_path = f"static/{detected_emotion}_{random_number}.mid"
        mf = midi.translate.music21ObjectToMidiFile(s)
        mf.open(midi_path, 'wb')
        mf.write()
        mf.close()

        # Convert MIDI to MP3
        mp3_path = f"static/{detected_emotion}_{random_number}.mp3"
        subprocess.run(["D:/Aaa/fluidsynth-2.4.3/bin/fluidsynth", "-ni", "soundfonts/FluidR3_GM.sf2", midi_path, "-F", mp3_path, "-r", "44100"])

        if os.path.exists(mp3_path):
            logging.info(f"Generated MP3 Path: {mp3_path}")
            return mp3_path
        else:
            logging.error(f"MP3 file not found: {mp3_path}")
            return None

    except Exception as e:
        logging.error(f"Error generating music: {str(e)}")
        return None

@app.route('/generate_music', methods=['POST'])
def handle_generate_music():
    try:
        # Parse the emotion from the request
        data = request.get_json()
        print(data)
        detected_emotion = data.get('detected_emotion')
        play_generated = data.get('play_generated', True)  # Default to True if not provided

        if not detected_emotion:
            return jsonify({'error': 'Emotion not provided'}), 400


        music_path = generate_music(detected_emotion)

        if music_path and os.path.exists(music_path):  # Check if the music was generated successfully
            music_url = f"http://192.168.0.104:5000/static/{os.path.basename(music_path)}"
            return jsonify({'music_url': music_url}), 200
        else:
            return jsonify({'error': 'Music generation failed'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename, mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000 ,threaded=True)
