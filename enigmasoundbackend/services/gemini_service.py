import os
import logging
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
ALLOWED = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral", "Love"]


def detect_emotion_with_gemini(text: str):
    """Return one emotion label, or None so the caller falls back to DistilBERT."""
    if not GEMINI_API_KEY:
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "Classify the emotion in this text. Reply with exactly one word "
            f"from: {', '.join(ALLOWED)}.\n\nText: \"{text}\"\n\nEmotion:"
        )
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        label = response.text.strip().strip(".").capitalize()
        return label if label in ALLOWED else None
    except Exception as e:
        logging.warning("Gemini failed, using fallback: %s", e)
        return None