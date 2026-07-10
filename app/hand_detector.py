import cv2
import mediapipe as mp


class HandDetector:

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        hands_data = []

        if results.multi_hand_landmarks:

            for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):

                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                label = hand_info.classification[0].label

                # Draw LEFT / RIGHT
                h, w, _ = frame.shape

                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)

                cv2.putText(
                    frame,
                    label,
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                landmark_list = []

                for lm in hand_landmarks.landmark:

                    landmark_list.append(
                        (
                            lm.x,
                            lm.y,
                            lm.z,
                        )
                    )

                hands_data.append(
                    {
                        "label": label,
                        "landmarks": landmark_list,
                    }
                )

        return frame, hands_data