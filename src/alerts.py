"""Alert system with per-level cooldowns."""
import time


# CRITICAL gets a shorter cooldown so rapid re-alerting is allowed at peak danger.
# Lower severity levels get progressively longer cooldowns to avoid noise.
_LEVEL_COOLDOWNS = {
    "CRITICAL": 1.5,
    "HIGH": 2.0,
    "MEDIUM": 3.0,
    "LOW": 4.0,
}
_DEFAULT_COOLDOWN = 3.0


class AlertSystem:
    """Alert system with per-level cooldown and escalation tracking."""

    def __init__(self, cooldown_seconds: float = 3.0):
        self.default_cooldown = cooldown_seconds
        self._last_alert_time: dict[str, float] = {}
        self._alert_counts: dict[str, int] = {
            "OK": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0
        }

    def _cooldown_for(self, level: str) -> float:
        """Return the cooldown for this level, scaled by default_cooldown ratio."""
        base = _LEVEL_COOLDOWNS.get(level, _DEFAULT_COOLDOWN)
        # Honour any user-configured global cooldown scaling.
        ratio = self.default_cooldown / _DEFAULT_COOLDOWN
        return base * ratio

    def should_alert(self, level: str) -> bool:
        """Return True when enough time has passed since the last alert at this level."""
        last = self._last_alert_time.get(level, 0.0)
        return (time.time() - last) >= self._cooldown_for(level)

    def trigger(self, level: str, metrics: dict) -> bool:
        """Fire an alert if the per-level cooldown has elapsed."""
        if not self.should_alert(level):
            return False

        self._last_alert_time[level] = time.time()
        self._alert_counts[level] = self._alert_counts.get(level, 0) + 1

        print(
            f"\n[!] {level} ALERT (#{self._alert_counts[level]}) — "
            f"EAR: {metrics.get('ear', 0):.3f} | "
            f"PERCLOS: {metrics.get('perclos', 0):.2%} | "
            f"Score: {metrics.get('score', 0):.1f} | "
            f"Microsleep: {metrics.get('microsleep', 0):.2f}s"
        )
        return True

    def reset_counts(self) -> None:
        """Reset per-session alert counters (call on recalibrate)."""
        self._alert_counts = {k: 0 for k in self._alert_counts}
        self._last_alert_time.clear()
