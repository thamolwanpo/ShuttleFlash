#!/usr/bin/env python3
"""
ShuttleFlash — YOLOv8 Pose + Red Cone Detection Server
-------------------------------------------------------
Detects:
  1. Your ankles via YOLOv8-nano-pose
  2. The red center cone via HSV color masking

Straddle = cone X is between left ankle X and right ankle X.
This prevents side-step false triggers.

Install deps:
    pip install ultralytics websockets opencv-python

Run:
    python server.py
"""

import asyncio
import json
import cv2
import numpy as np
import websockets
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────
HOST        = "localhost"
PORT        = 8765
CAMERA_IDX  = 0
MODEL_NAME  = "yolov8n-pose.pt"
CONF_THRESH = 0.4       # min keypoint confidence
TARGET_W    = 640
TARGET_H    = 480

# Red cone HSV range
# Red wraps around 0° in HSV so we need two ranges
CONE_HSV_LOWER1 = np.array([0,   120, 80])   # lower red range
CONE_HSV_UPPER1 = np.array([10,  255, 255])
CONE_HSV_LOWER2 = np.array([165, 120, 80])   # upper red range (wraps)
CONE_HSV_UPPER2 = np.array([180, 255, 255])

CONE_MIN_AREA   = 300   # ignore tiny blobs (px²) — tune if needed
CONE_MAX_AREA   = 40000 # ignore huge blobs (someone wearing red shirt)
# ────────────────────────────────────────────────────────

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16
LEFT_KNEE   = 13
RIGHT_KNEE  = 14

print(f"[ShuttleFlash] Loading {MODEL_NAME}…")
model = YOLO(MODEL_NAME)
print(f"[ShuttleFlash] Model ready. Opening camera…")

def open_camera(idx):
    for backend, name in [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "default")]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            print(f"[ShuttleFlash] Camera {idx} opened via {name} ✓")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    return None

cap = None
for try_idx in [CAMERA_IDX] + [i for i in range(5) if i != CAMERA_IDX]:
    print(f"[ShuttleFlash] Trying camera index {try_idx}…")
    cap = open_camera(try_idx)
    if cap:
        print(f"[ShuttleFlash] Using camera index {try_idx}")
        break

if cap is None:
    print("""
[ShuttleFlash] ✗ No camera found. Try:
  1. System Settings → Privacy & Security → Camera → enable Terminal
  2. Quit Terminal completely and reopen, then run again
""")
    raise RuntimeError("No camera found.")

print(f"[ShuttleFlash] Ready. WebSocket on ws://{HOST}:{PORT}")
connected_clients = set()


# ── Cone detection ───────────────────────────────────────
def find_red_cone(frame):
    """
    Find the largest red blob in the frame.
    Returns dict with normalised cx, cy, width, height, area — or None.
    """
    h, w = frame.shape[:2]

    # Blur to reduce noise, then convert to HSV
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Two masks for red (wraps around hue 0°/180°)
    mask1 = cv2.inRange(hsv, CONE_HSV_LOWER1, CONE_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, CONE_HSV_LOWER2, CONE_HSV_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup — remove small noise, fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area, pick the largest qualifying blob
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if CONE_MIN_AREA <= area <= CONE_MAX_AREA and area > best_area:
            best_area = area
            best = cnt

    if best is None:
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    cx = x + bw / 2
    cy = y + bh / 2

    return {
        "x":      round(cx / w, 4),   # normalised 0-1
        "y":      round(cy / h, 4),
        "width":  round(bw / w, 4),
        "height": round(bh / h, 4),
        "area":   int(best_area),
    }


# ── Pose detection ───────────────────────────────────────
def find_person(results, frame_shape):
    """Extract best-scoring person's ankle/knee keypoints."""
    h, w = frame_shape[:2]
    persons = []

    if results.keypoints is None or len(results.keypoints.data) == 0:
        return None

    for person_kps in results.keypoints.data:
        def kp(idx):
            kpt = person_kps[idx]
            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
            return {
                "x":       round(x / w, 4),
                "y":       round(y / h, 4),
                "conf":    round(conf, 3),
                "visible": conf >= CONF_THRESH,
            }
        persons.append({
            "left_ankle":  kp(LEFT_ANKLE),
            "right_ankle": kp(RIGHT_ANKLE),
            "left_knee":   kp(LEFT_KNEE),
            "right_knee":  kp(RIGHT_KNEE),
        })

    # Pick person whose ankles are most confidently detected
    best = max(persons, key=lambda p: p["left_ankle"]["conf"] + p["right_ankle"]["conf"])
    return best


# ── Straddle check ───────────────────────────────────────
def check_straddle(person, cone):
    """
    True if the cone's X position falls between the two ankles.
    This is the correct straddle check — immune to side-steps.
    """
    if person is None or cone is None:
        return False, "no_data"

    la = person["left_ankle"]
    ra = person["right_ankle"]

    if not la["visible"] or not ra["visible"]:
        return False, "ankles_not_visible"

    left_x  = min(la["x"], ra["x"])
    right_x = max(la["x"], ra["x"])
    cone_x  = cone["x"]

    # Add a small margin so detection isn't pixel-perfect
    margin = 0.02
    straddling = (left_x - margin) < cone_x < (right_x + margin)

    return straddling, "ok"


# ── Frame processing ─────────────────────────────────────
def process_frame(frame):
    frame = cv2.flip(frame, 1)  # mirror for natural left/right

    # Run YOLO pose
    results = model(frame, verbose=False, conf=0.3)[0]
    person  = find_person(results, frame.shape)

    # Run red cone detection (pure OpenCV, very fast)
    cone = find_red_cone(frame)

    # Straddle check
    straddling, reason = check_straddle(person, cone)

    return {
        "type":       "keypoints",
        "person":     person,
        "cone":       cone,          # None if not detected
        "straddling": straddling,    # definitive answer for the browser
        "reason":     reason,        # for debugging
        "frame_w":    frame.shape[1],
        "frame_h":    frame.shape[0],
    }


# ── WebSocket server ─────────────────────────────────────
async def camera_loop():
    global connected_clients
    loop = asyncio.get_event_loop()
    while True:
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            await asyncio.sleep(0.05)
            continue

        payload = process_frame(frame)
        msg     = json.dumps(payload)

        if connected_clients:
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send(msg)
                except websockets.exceptions.ConnectionClosed:
                    dead.add(ws)
            connected_clients -= dead

        await asyncio.sleep(1 / 30)


async def handler(websocket):
    connected_clients.add(websocket)
    print(f"[ShuttleFlash] Browser connected: {websocket.remote_address}")
    try:
        async for msg in websocket:
            pass  # no incoming commands needed yet
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[ShuttleFlash] Browser disconnected: {websocket.remote_address}")


async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"[ShuttleFlash] ✓ Ready — open badminton-trainer.html in your browser\n")
        await camera_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ShuttleFlash] Stopped.")
        cap.release()
