Real-time driver drowsiness detection built with OpenCV and MediaPipe Face Mesh. The system watches eye closure, blink behavior, yawning, and head pose, fuses them into a single drowsiness score, and raises staged alerts before fatigue becomes dangerous. Every session is logged to CSV and can be reviewed afterward in a Streamlit analytics dashboard.

The project was developed and demoed on a laptop webcam, then deployed and demonstrated live on a Raspberry Pi 4 with a standard USB webcam for the final demo.

How it works
Face landmark detection — MediaPipe Face Mesh extracts 468 facial landmarks per frame. The frame is downscaled before detection to keep this fast enough for real-time use on modest hardware.
Eye Aspect Ratio (EAR) — computed per eye from six landmarks each; a 30-second calibration phase at startup establishes a personal baseline EAR (looking straight ahead, eyes open) instead of relying on a single hardcoded threshold.
Mouth Aspect Ratio (MAR) — tracks mouth opening to flag yawning.
PERCLOS — percentage of a rolling time window where eyes are sustainedly closed, a standard fatigue metric.
Microsleep detection — tracks continuous eye-closure duration and escalates separately from PERCLOS once closure crosses 1.5s / 2.5s thresholds.
Blink rate — blinks per minute, with both hypo- and hyper-blinking penalized.
Head pose (pitch/yaw/roll) — estimated via cv2.solvePnP against a generic 3D face model. A sustained head-down (nodding off) or head-turned-away posture contributes to the score independently of eye state, and also gates eye metrics off when the head angle makes EAR unreliable (e.g., looking far down or to the side).
Fusion score (0–100) — EAR, PERCLOS, microsleep duration, yawning, blink rate, and head pose are combined into one smoothed score. The score drives a five-stage alert state machine (OK → LOW → MEDIUM → HIGH → CRITICAL) with hysteresis so alerts don't flicker on momentary noise.
Session logging — every frame's metrics are written to a timestamped CSV in data/sessions/.
Project structure
driver-drowsiness-detector/
├── main.py                # Entry point: capture loop, scoring, alert state machine, on-screen HUD
├── dashboard.py           # Streamlit dashboard for reviewing past sessions
├── config/
│   └── config.yaml        # Camera, detection, and drowsiness threshold settings
├── src/
│   ├── camera.py          # Webcam capture wrapper with FPS measurement
│   ├── detector.py        # MediaPipe Face Mesh wrapper, EAR/MAR landmarks, head pose
│   ├── metrics.py         # EAR/MAR/PERCLOS/blink-rate/microsleep calculations, calibration
│   ├── alerts.py          # Alert triggering with per-level cooldowns
│   └── logger.py          # Per-frame CSV session logger
├── data/sessions/         # CSV logs, one file per run (auto-created)
├── pyproject.toml
└── uv.lock
Requirements
Python 3.12+
A webcam (built-in laptop camera or any USB UVC webcam — including the one used on the Raspberry Pi demo)
uv for dependency management (recommended), or pip with a virtual environment
Core dependencies (see pyproject.toml): mediapipe, opencv-python, numpy, pyyaml, plus pandas, plotly, and streamlit for the dashboard.

Setup
git clone https://github.com/ishaaanvaidya/driver-drowsiness-detector.git
cd driver-drowsiness-detector
uv sync
If not using uv:

python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e .
Running the detector
uv run python main.py
On startup the system calibrates for 30 seconds — look straight at the camera with your eyes open. After calibration, the live HUD shows EAR, MAR, PERCLOS, blink rate, drowsiness score, microsleep duration, and head pose, with a colored alert banner when drowsiness is detected.

Keyboard controls (while the video window is focused):

Key	Action
Q	Quit and save the session log
L	Toggle landmark overlay
M	Toggle metrics overlay
R	Recalibrate (resets baseline and all counters)
H	Toggle key-hint footer
Reviewing sessions: the dashboard
uv run streamlit run dashboard.py
The dashboard reads every CSV in data/sessions/ and shows risk-score trends over time, eye behavior (EAR/PERCLOS/blink rate), head pose, alert episodes, and run-to-run comparisons — useful for tuning thresholds or reviewing how a demo run went after the fact.

Configuration
All thresholds live in config/config.yaml, so tuning doesn't require touching code.

camera:
  source: 0          # 0 = default camera; change if multiple cameras are attached
  width: 960
  height: 540
  fps: 30

detection:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  pose_update_interval: 2   # recompute head pose every N frames (perf tuning)

drowsiness:
  fallback_ear_threshold: 0.21   # used only if calibration fails
  calibration_ratio: 0.80        # closed-eye threshold = baseline_EAR * this
  mar_threshold: 0.6             # yawn threshold
  perclos_threshold: 0.20
  microsleep_high_seconds: 1.5
  microsleep_critical_seconds: 2.5
  score_low: 25
  score_medium: 45
  score_high: 70
  score_critical: 90
  pitch_threshold: 18.0          # degrees, head-down before pose penalty kicks in
  yaw_threshold: 30.0            # degrees, head-turned-away
  roll_threshold: 25.0           # degrees, head-tilt

alerts:
  cooldown_seconds: 3            # base cooldown; scaled per alert level internally

display:
  show_landmarks: true
  show_metrics: true
  show_fps: true
  window_width: 960
  window_height: 540
Lower score_* and microsleep_* values make the system more sensitive; raise them to reduce false positives for a given setup.

Raspberry Pi deployment notes
The final demo ran this exact codebase on a Raspberry Pi 4 with a generic USB webcam (no Pi-specific code branch — the same main.py runs unmodified on both laptop and Pi). A few practical notes from that deployment:

Resolution and FPS: the default 960x540 @ 30fps in config.yaml is a reasonable starting point for a Pi 4; if frame rate is too low, try dropping resolution further (e.g. 640x480) since MediaPipe Face Mesh is the main CPU cost.
pose_update_interval: increasing this (e.g. to 3–4) reduces how often solvePnP head-pose estimation runs, trading a bit of pose responsiveness for lower CPU load — useful headroom on the Pi.
Camera backend: src/camera.py requests the MJPEG FOURCC, which helps avoid USB bandwidth bottlenecks with UVC webcams on the Pi just as it does on a laptop.
Headless/HDMI display: main.py opens an OpenCV window (cv2.imshow), so the Pi needs a connected display (or X11 forwarding) to show the live HUD during the demo. CSV logging works independently of the display.
Calibration still takes the full 30 seconds on the Pi — plan for that in any live demo timing.
Known limitations
Single-face tracking only (max_num_faces=1 in MediaPipe config) — designed for one driver in frame.
Sunglasses or heavy eye occlusion will degrade EAR-based detection; the head-pose signal still contributes in that case but the system is not glasses/occlusion-robust by design.
No physical alert hardware (buzzer, vibration motor) is wired up yet — alerts are currently on-screen only (HUD banner) and printed to the console with cooldowns; this is a natural next step for the Pi deployment.
Lighting strongly affects MediaPipe's landmark accuracy, and therefore EAR/MAR readings; recalibrate (R) if lighting changes mid-session.
Possible next steps
GPIO-driven buzzer/LED alert output on the Raspberry Pi for a non-visual warning.
Multi-camera or IR-camera support for low-light/night driving conditions.
On-device model quantization or a lighter face-mesh variant if further Pi performance headroom is needed.
