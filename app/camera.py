import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

    def get_frame(self):
        success, frame = self.cap.read()

        print("Frame Success:", success)   # Debug

        if not success:
            return None

        return frame

    def release(self):
        self.cap.release()