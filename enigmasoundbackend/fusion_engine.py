import numpy as np

def fuse_emotions(text_dist, audio_dist, face_dist, quality_scores):
    """
    Fuses multi-modal emotional probability distributions using normalized
    reliability-weighted linear combinations.
    """
    # 1. Define base static reliability weights for each modality
    STATIC_WEIGHTS = {
        'text': 1.0,   # Text/NLP is highly stable for explicit emotion
        'audio': 0.8,  # Audio models capture tone well but can be noisy
        'face': 0.75   # Face models depend heavily on lighting/framing
    }

    # 2. Extract dynamic input-quality heuristics
    q_text = quality_scores.get('text', 1.0)
    q_audio = quality_scores.get('audio', 1.0)
    q_face = quality_scores.get('face', 1.0)

    # 3. Calculate absolute reliability coefficients (W_m = Static * Quality)
    w_text = STATIC_WEIGHTS['text'] * q_text
    w_audio = STATIC_WEIGHTS['audio'] * q_audio
    w_face = STATIC_WEIGHTS['face'] * q_face

    total_weight = w_text + w_audio + w_face

    # Graceful degradation fallback: If all streams are totally dark/ruined
    if total_weight == 0:
        return {'Happy': 0.0, 'Sad': 0.0, 'Neutral': 1.0, 'Angry': 0.0}

    # 4. Global project vocabulary layout (Strict array indexing alignment)
    labels = ['Happy', 'Sad', 'Neutral', 'Angry']

    # Convert input dicts to raw numpy vectors aligned to standard labels
    v_text = np.array([text_dist.get(l, 0.0) for l in labels])
    v_audio = np.array([audio_dist.get(l, 0.0) for l in labels])
    v_face = np.array([face_dist.get(l, 0.0) for l in labels])

    # 5. Compute Normalized Weighted Linear Combination
    fused_vector = (w_text * v_text + w_audio * v_audio + w_face * v_face) / total_weight

    # Return back as a clean dictionary mapping
    return dict(zip(labels, fused_vector.tolist()))


def map_fused_to_music_controls(fused_dict):
    """
    Maps a fused emotional probability profile directly into explicit
    Valence (-1 to +1) and Arousal (-1 to +1) coordinates to drive Music21.
    """
    # Define directional anchor vectors for each base emotion
    # Coordinate system: [Valence, Arousal]
    EMOTION_COORDINATES = {
        'Happy':   np.array([ 0.8,  0.6]), # Positive, High Energy
        'Sad':     np.array([-0.8, -0.7]), # Negative, Low Energy
        'Angry':   np.array([-0.6,  0.8]), # Negative, Ultra High Energy
        'Neutral': np.array([ 0.0,  0.0])  # Center Anchor
    }

    # Calculate expected value (weighted average coordinate vector)
    target_coords = np.zeros(2)
    for emotion, prob in fused_dict.items():
        if emotion in EMOTION_COORDINATES:
            target_coords += prob * EMOTION_COORDINATES[emotion]

    v, a = target_coords[0], target_coords[1]

    # Map coordinates directly to practical, dynamic Music21 parameters
    # Tempo (BPM range: 60 to 150)
    calculated_tempo = int(105 + (a * 45))

    # Chord Modality Profile selection criteria
    if v >= 0.1:
        modality = "major"
    elif v <= -0.1:
        modality = "minor"
    else:
        modality = "dorian" # Balanced modal fallback for neutral blends

    # Chord harmony density generation probability
    harmony_probability = float(np.clip(0.3 + (a * 0.3), 0.1, 0.7))

    return {
        'valence': float(v),
        'arousal': float(a),
        'tempo_bpm': calculated_tempo,
        'key_modality': modality,
        'harmony_prob': harmony_probability
    }