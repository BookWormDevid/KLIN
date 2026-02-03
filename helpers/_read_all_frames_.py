import cv2
import pathlib
import numpy as np
class ReadAllFrames:

    @staticmethod
    def read_all_frames(path: pathlib.Path) -> list[np.ndarray] | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return None
        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            return None
        return frames