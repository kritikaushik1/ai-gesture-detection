
import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "static_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def normalize_landmarks(landmarks):
    if len(landmarks) != 21:
        raise ValueError("Exactly 21 hand landmarks are required")

    wrist_x, wrist_y, wrist_z = landmarks[0]

    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            x - wrist_x,
            y - wrist_y,
            z - wrist_z
        ])

    scale = max(abs(value) for value in normalized)

    if scale == 0:
        raise ValueError("Invalid landmarks: scale is zero")

    return [
        value / scale
        for value in normalized
    ]


def get_feature_names():
    columns = []

    for landmark_id in range(21):
        columns.extend([
            f"landmark_{landmark_id}_x",
            f"landmark_{landmark_id}_y",
            f"landmark_{landmark_id}_z"
        ])

    return columns


def predict_gesture(landmarks):
    features = normalize_landmarks(landmarks)

    input_data = pd.DataFrame(
        [features],
        columns=get_feature_names()
    )

    probabilities = model.predict_proba(input_data)[0]

    predicted_index = int(np.argmax(probabilities))

    gesture = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(probabilities[predicted_index])

    return {
        "gesture": gesture,
        "confidence": round(confidence, 4)
    }
