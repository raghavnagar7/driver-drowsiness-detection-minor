"""Drowsiness metrics: EAR, MAR, PERCLOS, blink rate, microsleep."""
from collections import deque

import numpy as np


def _ear(eye: np.ndarray) -> float:
    """Eye Aspect Ratio via numpy (avoids scipy overhead for 6-point arrays)."""
    # Vertical distances: (p2-p6) and (p3-p5)
    A = float(np.linalg.norm(eye[1] - eye[5]))
    B = float(np.linalg.norm(eye[2] - eye[4]))
    # Horizontal distance: p1-p4
    C = float(np.linalg.norm(eye[0] - eye[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def _mar(mouth: np.ndarray) -> float:
    """Mouth Aspect Ratio — vertical / horizontal opening."""
    vert  = float(np.linalg.norm(mouth[0] - mouth[1]))
    horiz = float(np.linalg.norm(mouth[2] - mouth[3]))
    return vert / horiz if horiz > 0 else 0.0


class DrowsinessMetrics:
    """Calculate EAR, MAR, PERCLOS, blink rate, and microsleep duration."""

    CALIBRATION_SECONDS = 30
    CALIBRATION_RATIO   = 0.80
    FALLBACK_THRESHOLD  = 0.21
    CLOSED_EYE_RATIO    = 0.90   # closed_threshold = ear_threshold * this
    MICROSLEEP_SECONDS  = 1.25

    def __init__(
        self,
        fps: float = 30,
        perclos_window_seconds: float = 5,
        microsleep_seconds: float | None = None,
        calibration_ratio: float | None = None,
        fallback_threshold: float | None = None,
        closed_eye_ratio: float | None = None,
    ):
        self.fps = max(float(fps), 1.0)
        self.perclos_window_seconds = float(perclos_window_seconds)
        self.microsleep_seconds  = float(microsleep_seconds  if microsleep_seconds  is not None else self.MICROSLEEP_SECONDS)
        self.calibration_ratio   = float(calibration_ratio   if calibration_ratio   is not None else self.CALIBRATION_RATIO)
        self.fallback_threshold  = float(fallback_threshold  if fallback_threshold  is not None else self.FALLBACK_THRESHOLD)
        self.closed_eye_ratio    = float(closed_eye_ratio    if closed_eye_ratio    is not None else self.CLOSED_EYE_RATIO)

        self.ear_threshold        = self.fallback_threshold
        self.closed_eye_threshold = self.ear_threshold * self.closed_eye_ratio

        self.ear_history   = deque(maxlen=max(1, int(self.fps * self.perclos_window_seconds)))
        self.blink_history = deque(maxlen=max(1, int(self.fps * 60)))

        self.eye_was_open    = True
        self._closed_frames  = 0
        self._frame_index    = 0
        self._blink_events: deque[int] = deque()

        self._min_blink_frames    = max(2,  int(self.fps * 0.07))
        # Minimum sustained frames to count a closure toward PERCLOS.
        # 0.10s at any FPS is enough to catch real blinks without noise spikes.
        self._closure_min_frames  = max(2,  int(self.fps * 0.10))
        self._microsleep_frames   = max(1,  int(self.fps * self.microsleep_seconds))
        # Grace period: allow up to this many consecutive open frames without
        # resetting the microsleep counter (handles EAR noise spikes mid-closure).
        self._grace_frames        = max(2,  int(self.fps * 0.08))
        self._open_grace_count    = 0

        self._cal_samples: list[float] = []
        self._cal_target  = int(self.CALIBRATION_SECONDS * self.fps)
        self.calibrating  = True
        self.baseline_ear: float | None = None

        # Per-eye asymmetry tracking (large asymmetry can signal ptosis/drowsiness)
        self._left_ear_history:  deque[float] = deque(maxlen=max(1, int(self.fps * 3)))
        self._right_ear_history: deque[float] = deque(maxlen=max(1, int(self.fps * 3)))

    # ------------------------------------------------------------------
    # FPS management
    # ------------------------------------------------------------------

    def update_fps(self, fps: float) -> None:
        """Resize all FPS-dependent windows once the real camera FPS is known."""
        if fps <= 5:
            return
        self.fps = float(fps)

        old_ear   = list(self.ear_history)
        old_blink = list(self.blink_history)

        ear_maxlen   = max(1, int(self.fps * self.perclos_window_seconds))
        blink_maxlen = max(1, int(self.fps * 60))
        self.ear_history   = deque(old_ear[-ear_maxlen:],   maxlen=ear_maxlen)
        self.blink_history = deque(old_blink[-blink_maxlen:], maxlen=blink_maxlen)

        self._cal_target           = int(self.CALIBRATION_SECONDS * self.fps)
        self._min_blink_frames     = max(2,  int(self.fps * 0.07))
        self._closure_min_frames   = max(2,  int(self.fps * 0.10))
        self._microsleep_frames    = max(1,  int(self.fps * self.microsleep_seconds))
        self._grace_frames         = max(2,  int(self.fps * 0.08))
        self._open_grace_count     = 0
        self._left_ear_history     = deque(maxlen=max(1, int(self.fps * 3)))
        self._right_ear_history    = deque(maxlen=max(1, int(self.fps * 3)))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @property
    def calibration_progress(self) -> float:
        return min(len(self._cal_samples) / max(self._cal_target, 1), 1.0)

    def _finish_calibration(self) -> None:
        samples = np.array(self._cal_samples, dtype=np.float64)
        samples = samples[samples > 0.05]

        if len(samples) < 10:
            self.baseline_ear     = self.fallback_threshold
            self.ear_threshold    = self.fallback_threshold
            print("\nCalibration had too few samples — using fallback EAR threshold.")
        else:
            low_cut  = np.percentile(samples, 40)
            high_cut = np.percentile(samples, 98)
            open_samples = samples[(samples >= low_cut) & (samples <= high_cut)]
            if len(open_samples) < 10:
                open_samples = samples

            self.baseline_ear        = float(np.median(open_samples))
            self.ear_threshold       = round(self.baseline_ear * self.calibration_ratio, 4)
            self.closed_eye_threshold = round(self.ear_threshold * self.closed_eye_ratio, 4)

            print(
                f"\nCalibration complete — baseline EAR: {self.baseline_ear:.3f} | "
                f"threshold: {self.ear_threshold:.3f} | "
                f"closed threshold: {self.closed_eye_threshold:.3f}"
            )

        self.closed_eye_threshold = round(self.ear_threshold * self.closed_eye_ratio, 4)
        self.calibrating          = False
        self.ear_history.clear()
        self.blink_history.clear()
        self._blink_events.clear()
        self._closed_frames    = 0
        self._open_grace_count = 0
        self._frame_index      = 0
        self.eye_was_open      = True
        self._left_ear_history.clear()
        self._right_ear_history.clear()

    def recalibrate(self) -> None:
        self._cal_samples         = []
        self.calibrating          = True
        self.baseline_ear         = None
        self.ear_threshold        = self.fallback_threshold
        self.closed_eye_threshold = self.fallback_threshold * self.closed_eye_ratio
        self.ear_history.clear()
        self.blink_history.clear()
        self._blink_events.clear()
        self._closed_frames    = 0
        self._open_grace_count = 0
        self._frame_index      = 0
        self.eye_was_open      = True
        self._left_ear_history.clear()
        self._right_ear_history.clear()
        print("\nRecalibrating — look straight ahead for 30 seconds.")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def reset_eye_state(self) -> None:
        """Clear live closure state when face / eyes are unreliable."""
        self._closed_frames   = 0
        self._open_grace_count = 0
        self.eye_was_open     = True
        if self.blink_history:
            self.blink_history.append(True)

    # ------------------------------------------------------------------
    # Per-frame metric calculation
    # ------------------------------------------------------------------

    def calculate_ear(self, eye: np.ndarray) -> float:
        return _ear(eye)

    def calculate_mar(self, mouth: np.ndarray) -> float:
        return _mar(mouth)

    def calculate_perclos(self) -> float:
        """PERCLOS — fraction of window where eyes were sustainedly closed."""
        if not self.ear_history:
            return 0.0

        closed = 0
        run    = 0
        for ear in self.ear_history:
            if ear < self.closed_eye_threshold:
                run += 1
            else:
                if run >= self._closure_min_frames:
                    closed += run
                run = 0
        if run >= self._closure_min_frames:
            closed += run

        return closed / len(self.ear_history)

    def get_ear_asymmetry(self) -> float:
        """Return mean |left_EAR - right_EAR| over the last 3 s.

        Values > 0.06 can indicate unilateral ptosis or tracking noise.
        """
        if len(self._left_ear_history) < 5:
            return 0.0
        diffs = np.abs(
            np.array(self._left_ear_history) - np.array(self._right_ear_history)
        )
        return float(np.mean(diffs))

    def _update_blink_and_closure(self, ear: float) -> None:
        closed = ear < self.closed_eye_threshold

        if closed:
            self._open_grace_count = 0
            self._closed_frames += 1
        else:
            self._open_grace_count += 1
            if self._open_grace_count > self._grace_frames:
                # Genuinely open long enough — finalise the previous closure.
                if self._min_blink_frames <= self._closed_frames < self._microsleep_frames:
                    self._blink_events.append(self._frame_index)
                self._closed_frames = 0
            # else: brief noise spike — treat as still closed, don't reset counter

        self.eye_was_open = not closed
        self.blink_history.append(self.eye_was_open)

    def get_blink_rate(self) -> float | None:
        """Return blinks/min once 20 s of data is available, else None."""
        if self._frame_index < int(self.fps * 20):
            return None

        window_frames = int(self.fps * 60)
        cutoff        = self._frame_index - window_frames
        while self._blink_events and self._blink_events[0] < cutoff:
            self._blink_events.popleft()

        duration_s = min(self._frame_index, window_frames) / self.fps
        if duration_s <= 0:
            return None
        return (len(self._blink_events) / duration_s) * 60

    def get_microsleep_duration(self) -> float:
        return self._closed_frames / self.fps

    # ------------------------------------------------------------------
    # Main update entry-point
    # ------------------------------------------------------------------

    def update(self, left_eye: np.ndarray, right_eye: np.ndarray,
               eyes_reliable: bool = True) -> dict:
        left_ear  = _ear(left_eye)
        right_ear = _ear(right_eye)
        avg_ear   = (left_ear + right_ear) / 2.0

        # Always track per-eye history for asymmetry metric.
        self._left_ear_history.append(left_ear)
        self._right_ear_history.append(right_ear)

        # ---- Calibration phase ----------------------------------------
        if self.calibrating:
            if eyes_reliable and avg_ear > 0.05:
                self._cal_samples.append(avg_ear)
            if len(self._cal_samples) >= self._cal_target:
                self._finish_calibration()
            return {
                "ear": avg_ear,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "ear_asymmetry": 0.0,
                "perclos": None,
                "blink_rate": None,
                "microsleep_duration": 0.0,
                "microsleeping": False,
                "calibrating": True,
                "eyes_reliable": eyes_reliable,
            }

        # ---- Post-calibration ----------------------------------------
        self._frame_index += 1

        if not eyes_reliable:
            # Clamp to threshold so extreme head angles don't spike PERCLOS.
            safe_ear = max(avg_ear, self.ear_threshold)
            self.ear_history.append(safe_ear)
            self.reset_eye_state()
            return {
                "ear": avg_ear,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "ear_asymmetry": self.get_ear_asymmetry(),
                "perclos": self.calculate_perclos(),
                "blink_rate": self.get_blink_rate(),
                "microsleep_duration": 0.0,
                "microsleeping": False,
                "calibrating": False,
                "eyes_reliable": False,
            }

        self.ear_history.append(avg_ear)
        self._update_blink_and_closure(avg_ear)
        microsleep_dur = self.get_microsleep_duration()

        return {
            "ear": avg_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear_asymmetry": self.get_ear_asymmetry(),
            "perclos": self.calculate_perclos(),
            "blink_rate": self.get_blink_rate(),
            "microsleep_duration": microsleep_dur,
            "microsleeping": microsleep_dur >= self.microsleep_seconds,
            "calibrating": False,
            "eyes_reliable": True,
        }
