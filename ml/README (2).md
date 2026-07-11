
# AI Gesture Gaming Platform - Static Gesture ML

## Overview

This module performs static hand gesture recognition using MediaPipe hand landmarks and XGBoost.

## Pipeline

Webcam
→ MediaPipe
→ 21 Hand Landmarks
→ 63 Features (x, y, z)
→ Landmark Normalization
→ XGBoost
→ Gesture Prediction

## Supported Gestures

- fist
- ok
- open
- peace
- point
- rock
- thumb

## Model Performance

Accuracy: 98.00%
Precision: 98.04%
Recall: 98.00%
F1 Score: 98.01%

## Files

- static_landmarks.csv: Training landmark dataset
- static_model.pkl: Trained XGBoost model
- label_encoder.pkl: Gesture label encoder
- predict.py: Gesture prediction interface
- retrain.py: Future custom gesture retraining pipeline
- train_static.ipynb: Model training notebook

## Prediction Output

Example:

{
    "gesture": "fist",
    "confidence": 0.9926
}

## Custom Gesture Retraining

New gesture landmark samples must follow the same 63-feature format.

Use:

retrain("new_gesture.csv")

The pipeline appends new samples, retrains XGBoost, and saves an updated model and label encoder.

## Important

The same landmark normalization used during dataset collection must be used during real-time prediction.
