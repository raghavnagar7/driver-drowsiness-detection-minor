"""Driver Drowsiness Detection System — Laptop Demo."""
import time
import cv2
import yaml
from pathlib import Path

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DrowsinessMetrics
from src.alerts import AlertSystem
from src.logger import SessionLogger


# ── Colour palette (BGR) ────────────────────────────────────────────────────
_C = {
    "ok":       (80,  200, 80),
    "low":      (60,  230, 230),
    "medium":   (40,  165, 255),
    "high":     (40,   40, 255),
    "critical": (20,   20, 180),
    "white":    (240, 240, 240),
    "muted":    (160, 160, 160),
    "dark":     (30,   30,  30),
    "panel":    (20,   20,  20),
}

_ALERT_COLORS = {
    "OK":       _C["ok"],
    "LOW":      _C["low"],
    "MEDIUM":   _C["medium"],
    "HIGH":     _C["high"],
    "CRITICAL": _C["critical"],
}

_ALERT_BORDER = {
    "OK":       None,
    "LOW":      (60,  230, 230),
    "MEDIUM":   (40,  165, 255),
    "HIGH":     (40,   40, 255),
    "CRITICAL": (20,   20, 180),
}


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draw a filled rounded rectangle — no alpha, no frame copy, fast."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)


def _pill(frame, text, x, y, fg, bg, scale=0.40, pad_x=10, pad_y=5):
    """Draw a pill-shaped text badge (solid fill, zero heap alloc)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    w, h = tw + pad_x * 2, th + pad_y * 2
    _draw_rounded_rect(frame, (x, y), (x + w, y + h), bg, radius=h // 2)
    cv2.putText(frame, text, (x + pad_x, y + pad_y + th - 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
    return w, h


class DrowsinessDetectionSystem:
    ALERT_ORDER = ["OK", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self):
        config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        cam_cfg    = self.config.get("camera",    {})
        det_cfg    = self.config.get("detection", {})
        alert_cfg  = self.config.get("alerts",    {})
        display_cfg = self.config.get("display",  {})
        drowsy_cfg = self.config.get("drowsiness",{})

        self.camera = Camera(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
        )
        self.detector = FaceDetector(
            min_detection_confidence=det_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=det_cfg.get("min_tracking_confidence", 0.5),
        )
        self.metrics = DrowsinessMetrics(
            fps=cam_cfg.get("fps", 30),
            perclos_window_seconds=drowsy_cfg.get("perclos_window_seconds", 5),
            microsleep_seconds=drowsy_cfg.get("microsleep_seconds", 1.25),
            calibration_ratio=drowsy_cfg.get("calibration_ratio", 0.80),
            fallback_threshold=drowsy_cfg.get("fallback_ear_threshold", 0.21),
            closed_eye_ratio=drowsy_cfg.get("closed_eye_ratio", 0.90),
        )
        self.alerts = AlertSystem(cooldown_seconds=alert_cfg.get("cooldown_seconds", 3))
        self.logger = SessionLogger()

        # ── Drowsiness thresholds from config ───────────────────────────
        self.mar_threshold               = drowsy_cfg.get("mar_threshold", 0.6)
        self.ear_consecutive_frames      = drowsy_cfg.get("ear_consecutive_frames", 15)
        self.mar_consecutive_frames      = drowsy_cfg.get("mar_consecutive_frames", 15)
        self.perclos_threshold           = drowsy_cfg.get("perclos_threshold", 0.20)
        self.score_low                   = drowsy_cfg.get("score_low", 25)
        self.score_medium                = drowsy_cfg.get("score_medium", 45)
        self.score_high                  = drowsy_cfg.get("score_high", 70)
        self.score_critical              = drowsy_cfg.get("score_critical", 90)
        self.pitch_threshold             = drowsy_cfg.get("pitch_threshold", 18.0)
        self.yaw_threshold               = drowsy_cfg.get("yaw_threshold", 30.0)
        self.roll_threshold              = drowsy_cfg.get("roll_threshold", 25.0)
        self.pose_score_max              = drowsy_cfg.get("pose_score_max", 30)
        self.eye_unreliable_up_pitch     = drowsy_cfg.get("eye_unreliable_up_pitch", 25.0)
        self.eye_unreliable_yaw          = drowsy_cfg.get("eye_unreliable_yaw", 45.0)
        self.microsleep_high_seconds     = drowsy_cfg.get("microsleep_high_seconds", 1.5)
        self.microsleep_critical_seconds = drowsy_cfg.get("microsleep_critical_seconds", 2.5)
        self.pose_medium_score           = drowsy_cfg.get("pose_medium_score", 12)
        self.pose_high_score             = drowsy_cfg.get("pose_high_score", 22)

        # ── Display flags ────────────────────────────────────────────────
        self.show_landmarks = display_cfg.get("show_landmarks", True)
        self.show_metrics   = display_cfg.get("show_metrics", True)
        self.show_fps       = display_cfg.get("show_fps", True)
        self.show_keyhelp   = True

        # ── Runtime state ────────────────────────────────────────────────
        self.frame_count              = 0
        self.consecutive_drowsy       = 0
        self.consecutive_yawn         = 0
        self.consecutive_pitch_down   = 0
        self.consecutive_yaw_away     = 0
        self.consecutive_roll_tilt    = 0
        self.drowsiness_score         = 0.0
        self.alert_level              = "OK"
        self._pending_alert_level     = "OK"
        self._pending_alert_frames    = 0
        self.pose_score               = 0.0
        self.no_face_frames           = 0
        self._last_fps_sync_loop      = 0
        self._last_pose               = None
        self.pose_update_interval     = det_cfg.get("pose_update_interval", 2)

    # ────────────────────────────────────────────────────────────────────
    # Reliability gate
    # ────────────────────────────────────────────────────────────────────

    def eyes_are_reliable(self, pose: dict | None) -> bool:
        if pose is None:
            return True
        vals = (pose.get("pitch"), pose.get("yaw"), pose.get("roll"))
        if any(v is None for v in vals):
            return False
        if any(not (-180.0 <= float(v) <= 180.0) for v in vals):
            return False
        if pose["pitch"] < -self.eye_unreliable_up_pitch:
            return False
        if abs(pose["yaw"]) > self.eye_unreliable_yaw:
            return False
        return True

    # ────────────────────────────────────────────────────────────────────
    # Pose score
    # ────────────────────────────────────────────────────────────────────

    def calculate_pose_score(self, pose: dict | None) -> float:
        if pose is None:
            self.consecutive_pitch_down  = 0
            self.consecutive_yaw_away    = 0
            self.consecutive_roll_tilt   = 0
            self.pose_score = max(0.0, self.pose_score - 0.5)
            return self.pose_score

        fps = self.camera.get_fps() or self.metrics.fps or 30

        if pose["pitch"] < -self.eye_unreliable_up_pitch:
            self.consecutive_pitch_down = 0

        if pose["pitch"] > self.pitch_threshold:
            self.consecutive_pitch_down += 1
        else:
            self.consecutive_pitch_down = max(0, self.consecutive_pitch_down - 5)

        if abs(pose["yaw"]) > self.yaw_threshold:
            self.consecutive_yaw_away += 1
        else:
            self.consecutive_yaw_away = max(0, self.consecutive_yaw_away - 5)

        if abs(pose["yaw"]) < 20.0:
            if abs(pose["roll"]) > self.roll_threshold:
                self.consecutive_roll_tilt += 1
            else:
                self.consecutive_roll_tilt = max(0, self.consecutive_roll_tilt - 5)

        pitch_score = _clamp((self.consecutive_pitch_down / fps - 0.8) / 1.7) * 24
        yaw_score   = _clamp((self.consecutive_yaw_away   / fps - 2.0) / 2.5) * 5
        roll_score  = _clamp((self.consecutive_roll_tilt  / fps - 1.5) / 2.0) * 7

        raw = min(self.pose_score_max, pitch_score + yaw_score + roll_score)
        alpha = 0.25 if raw > self.pose_score else (0.45 if raw == 0 else 0.18)
        self.pose_score = (1 - alpha) * self.pose_score + alpha * raw
        return self.pose_score

    # ────────────────────────────────────────────────────────────────────
    # Fusion score
    # ────────────────────────────────────────────────────────────────────


    def calculate_fusion_score(self, ear: float, mar: float, perclos: float,
                                blink_rate: float | None, microsleep_duration: float,
                                pose_score: float, eyes_reliable: bool = True) -> float:
        perclos = perclos or 0.0

        if eyes_reliable:
            ear_drop        = max(0.0, self.metrics.closed_eye_threshold - ear)
            ear_score       = _clamp(ear_drop / max(self.metrics.closed_eye_threshold * 0.45, 1e-6)) * 18
            perclos_score   = _clamp((perclos - 0.10) / 0.30) * 30
            microsleep_score = _clamp(microsleep_duration / max(self.microsleep_critical_seconds, 1e-6)) * 42
        else:
            ear_score = perclos_score = microsleep_score = 0.0

        yawn_score = _clamp(self.consecutive_yawn / max(self.mar_consecutive_frames, 1)) * 8

        # Gradual blink-rate penalty (ramp, not step).
        blink_score = 0.0
        if eyes_reliable and blink_rate is not None:
            if blink_rate < 8:
                # Hypoblink: ramp 0 → 6 as rate drops from 8 → 0.
                blink_score = _clamp(1.0 - blink_rate / 8.0) * 6
            elif blink_rate > 45:
                # Hyperblink: ramp 0 → 4 as rate rises from 45 → 80.
                blink_score = _clamp((blink_rate - 45) / 35.0) * 4

        raw = min(100.0, ear_score + perclos_score + microsleep_score +
                  yawn_score + blink_score + pose_score)

        alpha = 0.22 if (raw > self.drowsiness_score or not eyes_reliable) else 0.06
        self.drowsiness_score = (1 - alpha) * self.drowsiness_score + alpha * raw
        return self.drowsiness_score

    # ────────────────────────────────────────────────────────────────────
    # Alert state machine
    # ────────────────────────────────────────────────────────────────────

    def _target_alert_from_score(self, score: float, microsleep_duration: float,
                                  eyes_reliable: bool = True) -> str:
        if not eyes_reliable:
            return "OK"
        if self.pose_score >= self.pose_high_score:
            return "HIGH"
        if self.pose_score >= self.pose_medium_score:
            return "MEDIUM"
        if eyes_reliable and microsleep_duration >= self.microsleep_critical_seconds:
            return "CRITICAL"
        if eyes_reliable and microsleep_duration >= self.microsleep_high_seconds:
            return "HIGH"
        if score >= self.score_critical:
            return "CRITICAL"
        if score >= self.score_high:
            return "HIGH"
        if score >= self.score_medium:
            return "MEDIUM"
        if score >= self.score_low:
            return "LOW"
        return "OK"

    def update_alert_state(self, score: float, microsleep_duration: float,
                           eyes_reliable: bool = True) -> str:
        if not eyes_reliable and self.pose_score < 2:
            self.alert_level          = "OK"
            self._pending_alert_level = "OK"
            self._pending_alert_frames = 0
            return self.alert_level

        target = self._target_alert_from_score(score, microsleep_duration, eyes_reliable)

        if target == self.alert_level:
            self._pending_alert_level  = target
            self._pending_alert_frames = 0
            return self.alert_level

        if target != self._pending_alert_level:
            self._pending_alert_level  = target
            self._pending_alert_frames = 1
        else:
            self._pending_alert_frames += 1

        current_i = self.ALERT_ORDER.index(self.alert_level)
        target_i  = self.ALERT_ORDER.index(target)
        fps       = self.camera.get_fps() or self.metrics.fps or 30

        seconds_needed = (
            0.6 if target == "CRITICAL" else
            0.9 if target == "HIGH" else
            0.8 if target_i > current_i else
            1.5
        )

        if self._pending_alert_frames >= int(fps * seconds_needed):
            self.alert_level           = target
            self._pending_alert_frames = 0

        return self.alert_level

    # ────────────────────────────────────────────────────────────────────
    # HUD rendering
    # ────────────────────────────────────────────────────────────────────

    def _draw_calibration_overlay(self, frame: "cv2.Mat", current_ear: float) -> None:
        h, w = frame.shape[:2]
        progress     = self.metrics.calibration_progress
        seconds_done = int(DrowsinessMetrics.CALIBRATION_SECONDS * progress)
        seconds_left = DrowsinessMetrics.CALIBRATION_SECONDS - seconds_done

        # Dark header band.
        frame[0:120, :] = (frame[0:120, :] * 0.45).astype(frame.dtype)

        cv2.putText(frame,
                    "CALIBRATING  -  look straight ahead, eyes open",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 240, 80), 2, cv2.LINE_AA)

        # Progress bar.
        bx, by, bh = 20, 50, 18
        bw = w - 40
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        fill = int(bw * progress)
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (70, 210, 120), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)
        cv2.putText(frame, f"{int(progress * 100)}%",
                    (bx + bw // 2 - 18, by + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,
                    f"{seconds_left}s remaining  |  samples: {len(self.metrics._cal_samples)}"
                    f"  |  EAR: {current_ear:.3f}",
                    (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (180, 180, 180), 1, cv2.LINE_AA)

    def draw_ui(self, frame: "cv2.Mat", ear: float, mar: float, perclos: float,
                blink_rate: float | None, alert_level: str, fps: float, score: float,
                microsleep_duration: float, pose: dict | None = None,
                pose_score: float = 0.0, eyes_reliable: bool = True,
                ear_asymmetry: float = 0.0) -> "cv2.Mat":
        h, w = frame.shape[:2]

        # ── Alert border flash ───────────────────────────────────────────
        border_color = _ALERT_BORDER.get(alert_level)
        if border_color:
            bw = 4 if alert_level in ("LOW", "MEDIUM") else 6
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, bw)

        # ── Header panel ────────────────────────────────────────────────
        header_h = 108
        cv2.rectangle(frame, (0, 0), (w, header_h), _C["panel"], -1)
        cv2.rectangle(frame, (0, header_h - 1), (w, header_h + 1), (60, 60, 60), -1)

        if self.show_fps:
            fps_text  = f"FPS {fps:.0f}"
            fps_color = (80, 200, 80) if fps >= 25 else (40, 165, 255) if fps >= 15 else (40, 40, 255)
            _pill(frame, fps_text, w - 90, 10, _C["dark"], fps_color, scale=0.42)

        if self.show_metrics:
            ear_color = (40, 40, 255) if ear < self.metrics.closed_eye_threshold else (80, 200, 80)
            _pill(frame, f"EAR {ear:.3f}", 12, 10, _C["dark"], ear_color, scale=0.40)

            mar_color = (40, 40, 255) if mar > self.mar_threshold else (60, 200, 60)
            _pill(frame, f"MAR {mar:.3f}", 12, 38, _C["dark"], mar_color, scale=0.40)

            _pill(frame, f"PERCLOS {perclos:.1%}", 12, 66, _C["dark"], (70, 140, 200), scale=0.40)

            blink_str = "blink --" if blink_rate is None else f"blink {blink_rate:.0f}/min"
            _pill(frame, blink_str, 200, 10, _C["dark"], (100, 100, 160), scale=0.38)

            score_color = (80, 200, 80) if score < 25 else (40, 165, 255) if score < 65 else (40, 40, 255)
            _pill(frame, f"Score {score:.0f}", 200, 38, _C["dark"], score_color, scale=0.40)

            ms_color = (40, 40, 255) if microsleep_duration >= self.microsleep_high_seconds else (100, 100, 100)
            _pill(frame, f"msleep {microsleep_duration:.1f}s", 200, 66, _C["dark"], ms_color, scale=0.38)

            if pose is not None:
                pose_str = f"P{pose['pitch']:+.0f} Y{pose['yaw']:+.0f} R{pose['roll']:+.0f}"
            else:
                pose_str = "Pose —"
            _pill(frame, pose_str, 420, 10, _C["dark"], (90, 110, 90), scale=0.38)

            # Asymmetry indicator (subtle warning if large).
            if ear_asymmetry > 0.06:
                _pill(frame, f"Asym {ear_asymmetry:.2f}", 560, 38, _C["dark"], (40, 165, 255), scale=0.38)

            if not eyes_reliable:
                _pill(frame, "EYE UNRELIABLE", 420, 66, _C["dark"], (40, 40, 255), scale=0.38)

        # ── Baseline hint ────────────────────────────────────────────────
        if self.metrics.baseline_ear is not None:
            cv2.putText(frame, f"baseline {self.metrics.baseline_ear:.3f}",
                        (w - 185, h - 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1, cv2.LINE_AA)

        # ── Score bar ────────────────────────────────────────────────────
        bar_w    = min(600, w - 60)
        bar_x    = (w - bar_w) // 2
        bar_y    = h - 82
        bar_h_px = 10

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (50, 50, 50), -1)
        fill_w = min(int((score / 100.0) * bar_w), bar_w)
        if fill_w > 0:
            fill_col = (80, 200, 80) if score < 25 else (40, 165, 255) if score < 65 else (40, 40, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), fill_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (90, 90, 90), 1)

        # Threshold tick marks.
        for pct, label in [(0.25, "25"), (0.45, "45"), (0.70, "70"), (0.90, "90")]:
            tx = bar_x + int(pct * bar_w)
            cv2.line(frame, (tx, bar_y - 3), (tx, bar_y + bar_h_px + 3), (100, 100, 100), 1)
            cv2.putText(frame, label, (tx - 8, bar_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Alert banner ─────────────────────────────────────────────────
        if alert_level and alert_level != "OK":
            banner_h = 52
            cv2.rectangle(frame, (0, h - banner_h), (w, h), (20, 20, 20), -1)
            color = _ALERT_COLORS[alert_level]
            label = {
                "LOW":      "LOW DROWSINESS",
                "MEDIUM":   "MODERATE DROWSINESS",
                "HIGH":     "HIGH DROWSINESS  -  TAKE A BREAK",
                "CRITICAL": "CRITICAL  -  PULL OVER NOW",
            }.get(alert_level, alert_level)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
            cv2.putText(frame, label, ((w - tw) // 2, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2, cv2.LINE_AA)

        # ── Key hints (bottom-right corner) ─────────────────────────────
        if self.show_keyhelp:
            hints = "[Q] quit  [L] landmarks  [R] recalibrate  [M] metrics  [H] help"
            cv2.putText(frame, hints, (8, h - (58 if alert_level != "OK" else 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (70, 70, 70), 1, cv2.LINE_AA)

        return frame

    # ────────────────────────────────────────────────────────────────────
    # Frame processing
    # ────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: "cv2.Mat") -> "cv2.Mat":
        landmarks = self.detector.detect(frame)

        if landmarks is None:
            self.no_face_frames    += 1
            self.consecutive_yawn   = 0       # BUG FIX: was never reset on face loss
            self.metrics.reset_eye_state()
            self.drowsiness_score  *= 0.85
            self.pose_score        *= 0.85
            fps = self.camera.get_fps() or self.metrics.fps or 30
            if self.no_face_frames > int(fps):
                self.alert_level           = "OK"
                self._pending_alert_level  = "OK"
                self._pending_alert_frames = 0
            cv2.putText(frame, "No face detected", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (40, 165, 255), 2, cv2.LINE_AA)
            if self.metrics.calibrating:
                self._draw_calibration_overlay(frame, 0.0)
            return frame

        self.no_face_frames = 0

        left_eye, right_eye = self.detector.get_eyes(landmarks)
        mouth                = self.detector.get_mouth(landmarks)
        h_f, w_f             = frame.shape[:2]

        if self.frame_count % max(1, int(self.pose_update_interval)) == 0:
            self._last_pose = self.detector.get_head_pose(landmarks, w_f, h_f)
        pose          = self._last_pose
        eyes_reliable = self.eyes_are_reliable(pose)

        metrics_data     = self.metrics.update(left_eye, right_eye, eyes_reliable=eyes_reliable)
        ear              = metrics_data["ear"]
        perclos          = metrics_data["perclos"]
        blink_rate       = metrics_data["blink_rate"]
        microsleep_dur   = metrics_data["microsleep_duration"]
        ear_asymmetry    = metrics_data.get("ear_asymmetry", 0.0)

        mar = self.metrics.calculate_mar(mouth) if mouth is not None else 0.0

        if metrics_data["calibrating"]:
            if self.show_landmarks:
                frame = self.detector.draw_landmarks(frame, landmarks)
            self._draw_calibration_overlay(frame, ear)
            return frame

        if eyes_reliable and ear < self.metrics.closed_eye_threshold:
            self.consecutive_drowsy += 1
        else:
            self.consecutive_drowsy = 0

        if mar > self.mar_threshold:
            self.consecutive_yawn += 1
        else:
            self.consecutive_yawn = 0

        pose_score  = self.calculate_pose_score(pose)
        score       = self.calculate_fusion_score(
            ear, mar, perclos, blink_rate, microsleep_dur,
            pose_score, eyes_reliable=eyes_reliable,
        )
        alert_level = self.update_alert_state(score, microsleep_dur, eyes_reliable=eyes_reliable)

        if alert_level != "OK":
            self.alerts.trigger(
                alert_level,
                {"ear": ear, "perclos": perclos or 0.0,
                 "score": score, "microsleep": microsleep_dur},
            )

        self.logger.log(
            frame=self.frame_count, ear=ear, mar=mar,
            perclos=perclos or 0.0, blink_rate=blink_rate,
            alert_level=alert_level, score=score,
            microsleep_duration=microsleep_dur, pose=pose,
            pose_score=pose_score, eyes_reliable=eyes_reliable,
        )
        self.frame_count += 1

        if self.show_landmarks:
            frame = self.detector.draw_landmarks(frame, landmarks)

        return self.draw_ui(
            frame, ear, mar, perclos or 0.0, blink_rate,
            alert_level, self.camera.get_fps(), score,
            microsleep_dur, pose, pose_score, eyes_reliable, ear_asymmetry,
        )

    # ────────────────────────────────────────────────────────────────────
    # Main loop
    # ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.camera.start()
            time.sleep(0.1)

            real_fps = self.camera.get_reported_fps()
            if real_fps > 5:
                self.metrics.update_fps(real_fps)
                print(
                    f"Camera running at {real_fps:.1f} fps  "
                    f"(calibration target: {self.metrics._cal_target} frames)"
                )

            win_cfg  = self.config.get("display", {})
            win_name = win_cfg.get("window_name", "Drowsiness Detection")
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name,
                             win_cfg.get("window_width", 960),
                             win_cfg.get("window_height", 540))

            print("\nStarting calibration — look straight ahead for 30 s.\n")
            print("Keys:  Q quit  |  L landmarks  |  R recalibrate  |  M metrics  |  H key-hints\n")

            loop_frames = 0
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame.")
                    break

                loop_frames += 1
                if loop_frames - self._last_fps_sync_loop >= 120:
                    measured = self.camera.get_fps()
                    if measured > 5:
                        self.metrics.update_fps(measured)
                        self._last_fps_sync_loop = loop_frames

                frame = self.process_frame(frame)
                cv2.imshow(win_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord("m"):
                    self.show_metrics = not self.show_metrics
                    print(f"Metrics overlay {'ON' if self.show_metrics else 'OFF'}")
                elif key == ord("h"):
                    self.show_keyhelp = not self.show_keyhelp
                elif key == ord("r"):
                    self._full_reset()

        finally:
            self.cleanup()

    def _full_reset(self) -> None:
        self.metrics.recalibrate()
        self.alerts.reset_counts()
        self.consecutive_drowsy     = 0
        self.consecutive_yawn       = 0
        self.consecutive_pitch_down = 0
        self.consecutive_yaw_away   = 0
        self.consecutive_roll_tilt  = 0
        self.drowsiness_score       = 0.0
        self.pose_score             = 0.0
        self.alert_level            = "OK"
        self._pending_alert_level   = "OK"
        self._pending_alert_frames  = 0
        self._last_pose             = None

    def cleanup(self) -> None:
        self.logger.close()
        self.camera.release()
        self.detector.cleanup()
        cv2.destroyAllWindows()
        print("\nSession ended.")


if __name__ == "__main__":
    system = DrowsinessDetectionSystem()
    system.run()
