# Enigma_Sound 

# Published IEEE research paper :
🔗 https://ieeexplore.ieee.org/document/10932090/

# Objective :
The Enigma Sound App aimed to provide a seamless and intuitive music experience by leveraging AI to enhance personalization. It sought to solve common issues such as poor sound quality, lack of personalized recommendations, and inefficient user experience in existing music apps.

It utilizes AI models to analyze text, voice, and facial expressions to detect emotions and generate melodies using  Music21 and FluidSynth or recommend Spotify songs.

# Tech Stack
-> Backend: Python (Flask,Music21,FluidSynth)

-> Frontend: Flutter  

-> AI & Machine Learning: PyTorch, TensorFlow, Librosa, CNN-LSTM model for audio, FER model for face detection

# AI Model Architecture & Training Details

## Text Emotion Model
**Pretrained Model: bhadresh-savani/distilbert-base-uncased-emotion via Hugging Face pipeline**

 -> Input: User text input (chat, voice-to-text transcription)
 
 -> Output: 6 emotion classes (sadness, joy, love, anger, fear, surprise)
 
 -> Integration: Labels are mapped to application-specific music emotions using label_mapping

## Facial Emotion Model (CNN)
**Architecture:**

 -> Input: Grayscale 48×48 facial images (detected using OpenCV)
 
 -> Convolutional Layers: 4-layer CNN with BatchNorm
 
 -> Conv2d(1→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→512)
 
 -> Each conv layer followed by: ReLU + BatchNorm2d + MaxPool2d(2×2)
 
 -> Fully Connected Layers: (512×3×3) → 512 → 256 → 7 (emotion classes)
 
 -> Regularization: Dropout(0.4) after first FC layer (512 units)
 
 -> Output: 7 emotion classes (Happy, Sad, Angry, Neutral, Surprised, Disgust, Fear)

***************************
**Training Configuration:**

 -> Loss: CrossEntropyLoss
 
 -> Optimizer: Adam (learning_rate=0.0003, weight_decay=0.0001)
 
 -> Scheduler: StepLR (step_size=15, gamma=0.5) — reduces LR by half every 15 epochs
 
 -> Augmentation: RandomHorizontalFlip, RandomRotation(15°), RandomPerspective
 
 -> Epochs: 150 with early stopping based on training accuracy

## Audio Emotion Model (CNN-LSTM)
**Architecture:**

 -> Input: 40-dimensional MFCC features extracted from audio using Librosa
 
 -> CNN Feature Extractor: 1D CNN with BatchNorm
     -  Conv1d(40→64) → BatchNorm1d
      - Conv1d(64→128) → BatchNorm1d
 
 -> LSTM Sequence Modeler: 2-layer Bidirectional LSTM (hidden_size=64)
 
 -> Fully Connected Layers: 128 → 128 → 8 (emotion classes)
 
 -> Regularization: Dropout(0.3)
 
 -> Output: 8 emotion classes (Happy, Sad, Neutral, Fear, Surprise, Angry, Disgust, Calm)

***********************************
**Training Configuration:**

 -> Loss: CrossEntropyLoss
 
 -> Optimizer: Adam (learning_rate=0.0003, weight_decay=0.0001)
 
 -> Scheduler: StepLR (step_size=15, gamma=0.5)
 
 -> Epochs: 100+ with early stopping

## Datasets
1. Facial: FER-2013 (7 classes, 48×48 grayscale images)
2. Audio: RAVDESS (8 classes, 40 MFCC features)
3. Text: Pretrained DistilBERT (no training needed)


# 🌐 Community & Research Recognition

This project is actively discussed within the global AI developer ecosystem and is featured on the **Google AI Developers Forum**. 
* **Official Discussion Thread:** [https://discuss.ai.google.dev/t/enigma-sound-multi-modal-emotion-to-music-pipeline-fusing-face-voice-and-text-ieee-research/137178]

**Live Visual Architecture Demo Now Hosted on Hugging Face Spaces!**

https://huggingface.co/spaces/ApurvaDev111/enigma-sound-ai
---

# 🛠 Setup & Installation

1. Clone the repository:

- git clone https://github.com/ApurvaPatil2401/Enigma_Sound.git
- cd enigma-sound


2. Install backend dependencies:

- cd enigmasoundbackend
- pip install -r requirements.txt

  **Note: Due to file size limits on GitHub, you must download the SoundFont manually: 
  Place the .sf2 file in: enigmasoundbackend/soundfonts/

  🔗 https://drive.google.com/drive/folders/1Afpft75F2IBZz-L-B_y5zrsaZIPeEBV0?usp=drive_link

**FluidSynth Configuration:**
Download FluidSynth from: https://drive.google.com/file/d/1c3vnRnS1JOv3cXpBEy_QWWFO5y29JJw1/view?usp=drive_link
After downloading, update line 24 in `enigmasoundbackend/app.py`:
```python
FLUIDSYNTH_PATH = "your/path/to/fluidsynth/bin/fluidsynth"
```

3. Run the backend server:

- python app.py  


4. Navigate to the Flutter frontend and install dependencies:

- cd emotion 

- flutter pub get  


5. Run the Flutter app:

- flutter run

**Frontend Configuration:**
Update your backend IP in `emotion/lib/config.dart`:
```dart
static const String baseUrl = 'http://YOUR_IP:5000';
```

# Low-Resource Edge Device Demo : 


https://github.com/user-attachments/assets/ceb6a8cb-315c-4668-a3c7-f6a8139d0253




**Technical Note on Demo: > "This demo was captured during a live test on a mid-range mobile device to demonstrate the model's efficiency on Edge Devices without cloud-side GPU acceleration. The focus is on real-time emotional mapping logic  rather than high-fidelity recording."


I am now exploring how to scale this architecture by migrating the custom LSTM layers to Vertex AI or leveraging Gemini’s multimodal embeddings for more complex orchestration.

I’d love to connect with anyone working on ‘emotional latency’ in real-time generative apps or those who have moved custom LSTM pipelines to Google’s managed services.









