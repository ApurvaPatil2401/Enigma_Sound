import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

def detect_emotion_with_gemini(text: str) -> str:
    """
    Detect emotion from text using Google Gemini.
    Falls back to None if API key is missing.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
        Analyze the emotion in this text and respond with ONLY one word 
        from: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral, Love
        
        Text: "{text}"
        
        Emotion:
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        emotion = response.text.strip()
        return emotion

    except Exception as e:
        print(f"Gemini API error: {e}")
        return None