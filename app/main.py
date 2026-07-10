import cv2

from camera import Camera
from hand_detector import HandDetector


def main():

    camera = Camera()
    detector = HandDetector()

    while True:

        frame = camera.get_frame()

        if frame is None:
            break

        frame, hands = detector.detect(frame)

        # Print landmark data
        for hand in hands:

            print(hand["label"])

            print(hand["landmarks"][0])

        cv2.imshow(
            "AI Gesture Gaming Platform",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()