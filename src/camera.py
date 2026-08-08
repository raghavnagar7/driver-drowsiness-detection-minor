"""Camera handler with FPS tracking."""
import sys
import time
from collections import deque

import cv2
import numpy as np


class Camera:
    """Webcam capture wrapper used by the drowsiness detector."""

    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

        self.frame_times = deque(maxlen=30)
        self.last_time = None
        self.reported_fps = float(fps)
        self.actual_width = width
        self.actual_height = height

    def start(self):
        """Initialize camera and request the configured resolution/FPS."""
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # MJPEG codec removes USB bandwidth bottleneck on Windows — major FPS boost.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_width = actual_w
        self.actual_height = actual_h
        if actual_fps and actual_fps > 1:
            self.reported_fps = float(actual_fps)
        self.last_time = None
        self.frame_times.clear()
        print(f"Camera started: requested {self.width}x{self.height}@{self.fps}, actual {actual_w}x{actual_h}@{actual_fps:.1f}")

    def read(self):
        """Read one frame and update rolling FPS measurement."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if ret:
            current_time = time.perf_counter()
            if self.last_time is not None:
                elapsed = max(current_time - self.last_time, 1e-6)
                self.frame_times.append(1.0 / elapsed)
            self.last_time = current_time

        return ret, frame

    def get_fps(self):
        """Return measured average FPS over the last 30 frames."""
        if not self.frame_times:
            return 0.0
        return float(np.mean(self.frame_times))

    def get_reported_fps(self):
        """Return the camera-reported FPS, falling back to configured FPS."""
        return float(self.reported_fps or self.fps or 0.0)

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
