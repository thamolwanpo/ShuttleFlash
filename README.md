# ShuttleFlash 🏸

Badminton footwork trainer using YOLOv8 pose detection.

## Setup (one time)

```bash
pip install ultralytics websockets opencv-python
```

## Run

```bash
# Terminal 1 — start the YOLO server
python server.py

# Then open badminton-trainer.html in your browser
```

## How it works

1. `server.py` opens your webcam, runs YOLOv8-nano-pose at ~30fps, streams ankle keypoints via WebSocket to `localhost:8765`
2. The browser app connects, shows a live ankle tracker, and lets you calibrate your center stance
3. Once calibrated, training starts automatically — the screen flashes white then shows a color/corner
4. Walk to that corner, return to center straddling the cone → next color triggers

## Calibration tips

- Place your MacBook camera at **knee height or lower**, front-facing
- Make sure both feet are fully visible in frame
- Click **Calibrate (5s)**, walk to your center spot, straddle the cone — it auto-captures after 5 seconds
- Adjust **Sensitivity** slider if triggers feel too easy or too hard

## Troubleshooting

**"Cannot reach ws://localhost:8765"** — server.py isn't running, or crashed. Check the terminal.

**Ankles not detected** — improve lighting, lower the camera, or wear contrasting shoes.

**Triggers too easily** — raise the Sensitivity slider (e.g. 0.85).

**Triggers not at all** — lower the Sensitivity slider (e.g. 0.65), or re-calibrate.

**Wrong camera** — edit `CAMERA_IDX = 0` in server.py (try 1, 2…).
