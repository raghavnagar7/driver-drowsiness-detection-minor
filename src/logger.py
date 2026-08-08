"""Session logger for per-frame drowsiness metrics."""
import csv
from pathlib import Path
from datetime import datetime


class SessionLogger:
    COLUMNS = [
        "timestamp",
        "frame",
        "ear",
        "mar",
        "perclos",
        "blink_rate",
        "score",
        "microsleep_duration",
        "pose_pitch",
        "pose_yaw",
        "pose_roll",
        "pose_score",
        "eyes_reliable",
        "alert_level",
    ]

    def __init__(self, log_dir: str = "data/sessions"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        filename = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
        self.path = Path(log_dir) / filename

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._rows_written = 0

    def _round_or_blank(self, value, digits):
        if value is None:
            return ""
        return round(value, digits)

    def log(
        self,
        frame: int,
        ear: float,
        mar: float,
        perclos: float,
        blink_rate: float,
        alert_level: str,
        score: float = 0.0,
        microsleep_duration: float = 0.0,
        pose=None,
        pose_score: float = 0.0,
        eyes_reliable: bool = True,
    ):
        pose = pose or {}
        self._writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame,
            round(ear, 4),
            round(mar, 4),
            round(perclos, 4),
            self._round_or_blank(blink_rate, 2),
            round(score, 2),
            round(microsleep_duration, 3),
            self._round_or_blank(pose.get("pitch"), 2),
            self._round_or_blank(pose.get("yaw"), 2),
            self._round_or_blank(pose.get("roll"), 2),
            round(pose_score, 2),
            int(bool(eyes_reliable)),
            alert_level,
        ])
        self._rows_written += 1
        if self._rows_written % 30 == 0:
            self._file.flush()

    def close(self):
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()
        print(f"Session saved -> {self.path}")
