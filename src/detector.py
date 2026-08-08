"""MediaPipe face detection with EAR/MAR landmarks and head pose."""
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class FaceDetector:
    """MediaPipe Face Mesh detector optimised for real-time drowsiness detection."""

    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    MOUTH_TOP    = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT   = 78
    MOUTH_RIGHT  = 308

    POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

    # Generic 3-D face model (mm) for solvePnP — nose tip, chin, eye corners, mouth corners.
    FACE_3D_MODEL = np.array([
        [  0.0,    0.0,   0.0],   # nose tip
        [  0.0,  -63.6, -12.5],   # chin
        [-43.3,   32.7, -26.0],   # left eye outer corner
        [ 43.3,   32.7, -26.0],   # right eye outer corner
        [-28.9,  -28.9, -24.1],   # left mouth corner
        [ 28.9,  -28.9, -24.1],   # right mouth corner
    ], dtype=np.float64)

    DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe is required for FaceDetector")

        try:
            from mediapipe.python.solutions import face_mesh as fm
            face_mesh_mod = fm
        except ImportError:
            face_mesh_mod = mp.solutions.face_mesh

        self.face_mesh = face_mesh_mod.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,          # keep fast; set True for iris tracking later
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._camera_matrix = None
        self._frame_size: tuple[int, int] | None = None
        # Cache previous rvec for SQPNP warm-start (improves stability on fast motion).
        self._prev_rvec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_matrix(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Build a pinhole camera matrix for solvePnP (cached per resolution)."""
        if self._frame_size == (frame_w, frame_h):
            return self._camera_matrix  # type: ignore[return-value]

        focal_length = frame_w          # reasonable approximation for a webcam
        self._camera_matrix = np.array([
            [focal_length, 0,            frame_w / 2],
            [0,            focal_length, frame_h / 2],
            [0,            0,            1          ],
        ], dtype=np.float64)
        self._frame_size = (frame_w, frame_h)
        return self._camera_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Return 468 face landmarks as (N, 2) pixel coords, or None."""
        h, w = frame.shape[:2]
        # Resize to half size for MediaPipe: 4x fewer pixels = ~4x faster detection.
        # Landmarks are normalised [0..1] so they still map correctly to the full frame.
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._prev_rvec = None          # reset pose cache on face loss
            return None

        face = results.multi_face_landmarks[0]
        return np.array(
            [[lm.x * w, lm.y * h] for lm in face.landmark],
            dtype=np.float64,
        )

    def get_eyes(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_eye, right_eye) landmark arrays."""
        return landmarks[self.LEFT_EYE], landmarks[self.RIGHT_EYE]

    def get_mouth(self, landmarks: np.ndarray) -> np.ndarray:
        """Return [top, bottom, left, right] mouth landmark array."""
        return np.array([
            landmarks[self.MOUTH_TOP],
            landmarks[self.MOUTH_BOTTOM],
            landmarks[self.MOUTH_LEFT],
            landmarks[self.MOUTH_RIGHT],
        ])

    def get_head_pose(self, landmarks: np.ndarray,
                      frame_w: int, frame_h: int) -> dict | None:
        """Estimate pitch / yaw / roll in degrees via cv2.solvePnP.

        Uses SQPNP (more numerically stable than ITERATIVE) and seeds from
        the previous frame's rotation vector when available.
        """
        if landmarks is None or len(landmarks) <= max(self.POSE_LANDMARK_IDS):
            return None

        image_pts = np.array(
            [landmarks[i] for i in self.POSE_LANDMARK_IDS],
            dtype=np.float64,
        )
        cam_mat = self._get_camera_matrix(frame_w, frame_h)

        # SQPNP is more robust to near-degenerate configurations than ITERATIVE.
        try:
            success, rvec, _ = cv2.solvePnP(
                self.FACE_3D_MODEL,
                image_pts,
                cam_mat,
                self.DIST_COEFFS,
                rvec=self._prev_rvec,
                useExtrinsicGuess=(self._prev_rvec is not None),
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            # Fallback: SQPNP not available in older OpenCV builds.
            success, rvec, _ = cv2.solvePnP(
                self.FACE_3D_MODEL,
                image_pts,
                cam_mat,
                self.DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            self._prev_rvec = None
            return None

        self._prev_rvec = rvec.copy()

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = float(angles[0])
        yaw   = float(angles[1])
        roll  = float(angles[2])

        # Normalise roll into [-90, 90].
        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    def draw_landmarks(self, frame: np.ndarray,
                       landmarks: np.ndarray) -> np.ndarray:
        """Draw anti-aliased eye outlines and mouth crosshair on the frame."""
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]
        thick = 2 if min(h, w) >= 540 else 1

        # Eye outlines — bright green polylines.
        for eye_idx in (self.LEFT_EYE, self.RIGHT_EYE):
            pts = landmarks[eye_idx].round().astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [pts], isClosed=True,
                color=(0, 255, 0), thickness=thick, lineType=cv2.LINE_AA,
            )

        # Mouth — red crosshair (vertical + horizontal).
        mouth = self.get_mouth(landmarks).round().astype(np.int32)
        cv2.line(frame, tuple(mouth[0]), tuple(mouth[1]),
                 (0, 80, 255), thick, cv2.LINE_AA)
        cv2.line(frame, tuple(mouth[2]), tuple(mouth[3]),
                 (0, 80, 255), thick, cv2.LINE_AA)

        return frame

    def cleanup(self) -> None:
        self.face_mesh.close()
