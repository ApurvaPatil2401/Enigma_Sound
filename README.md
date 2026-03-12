# Enigma_Sound

# Published IEEE research paper :
🔗 https://ieeexplore.ieee.org/document/10932090/

# Objective :
The Enigma Sound App aimed to provide a seamless and intuitive music experience by leveraging AI to enhance personalization. It sought to solve common issues such as poor sound quality, lack of personalized recommendations, and inefficient user experience in existing music apps.

It utilizes AI models to analyze text, voice, and facial expressions to detect emotions and generate melodies using  Music21 and FluidSynth or recommend Spotify songs.

# Tech Stack
-> Frontend: Flutter  

-> Backend: Python (Flask,Music21,FluidSynth)

-> AI & Machine Learning: TensorFlow, Librosa, CNN-LSTM model for audio, FER model for face detection

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


3. Run the backend server:

- python app.py  


4. Navigate to the Flutter frontend and install dependencies:

- cd emotion 

- flutter pub get  


5. Run the Flutter app:

- flutter run



# Low-Resource Edge Device Demo : 

https://github.com/user-attachments/assets/03a5bae2-eeaf-41f7-864d-bca3daa6ca39



**Technical Note on Demo: > "This demo was captured during a live test on a mid-range mobile device to demonstrate the model's efficiency on Edge Devices without cloud-side GPU acceleration. The focus is on real-time emotional mapping logic  rather than high-fidelity recording."









