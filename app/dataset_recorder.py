import cv2
import csv
import os

from camera import Camera
from hand_detector import HandDetector


GESTURES = {
    ord("1"): "open",
    ord("2"): "fist",
    ord("3"): "peace",
    ord("4"): "thumb",
    ord("5"): "point",
    ord("6"): "ok",
    ord("7"): "rock",
}

MAX_SAMPLES = 250

DATA_PATH = "data/static_landmarks.csv"


def normalize_landmarks(landmarks):
    wrist_x, wrist_y, wrist_z = landmarks[0]

    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            x - wrist_x,
            y - wrist_y,
            z - wrist_z,
        ])

    scale = max(abs(value) for value in normalized)

    if scale == 0:
        return None

    return [value / scale for value in normalized]


def create_csv():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        return

    columns = []

    for landmark_id in range(21):
        columns.extend([
            f"landmark_{landmark_id}_x",
            f"landmark_{landmark_id}_y",
            f"landmark_{landmark_id}_z",
        ])

    columns.append("gesture_label")

    with open(DATA_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def save_sample(features, gesture):
    with open(DATA_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(features + [gesture])


def main():
    create_csv()

    camera = Camera()
    detector = HandDetector()

    current_gesture = None
    sample_count = 0

    print("Dataset Recorder Started")

    while True:
        frame = camera.get_frame()

        if frame is None:
            print("Frame is None")
            break

        frame, hands = detector.detect(frame)

        key = cv2.waitKey(1) & 0xFF

        if key in GESTURES:
            current_gesture = GESTURES[key]
            sample_count = 0

            print(f"Collecting: {current_gesture}")

        if (
            current_gesture is not None
            and sample_count < MAX_SAMPLES
            and len(hands) > 0
        ):
            landmarks = hands[0]["landmarks"]

            features = normalize_landmarks(landmarks)

            if features is not None:
                save_sample(features, current_gesture)
                sample_count += 1

        cv2.putText(
            frame,
            f"Gesture: {current_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{MAX_SAMPLES}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Dataset Recorder", frame)

        if (
         current_gesture is not None
        and sample_count >= MAX_SAMPLES
        ):
         print(f"{current_gesture} completed")
         current_gesture = None
         sample_count = 0
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()