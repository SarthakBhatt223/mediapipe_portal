# Dimensions Portal — Build Notes

## Overview

Use MediaPipe Hands + OpenCV to detect when both thumbs and both index fingers are open, then summon a visual portal on screen.

---

## 1. Project Setup

```bash
pip install mediapipe opencv-python numpy
```

Your Python file will need these imports:

```python
import cv2
import mediapipe as mp
import numpy as np
import math
import time
```

Key MediaPipe objects to create:

```python
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
```

---

## 2. Webcam Capture

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

- `0` = default webcam
- Read frames with `cap.read()` which returns `(ret, frame)`
- Flip horizontally with `cv2.flip(frame, 1)` for mirror view
- Convert BGR to RGB with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — MediaPipe needs RGB

---

## 3. MediaPipe Hands — How It Works

```python
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as hands:
    results = hands.process(rgb)
```

**Key parameters:**
- `static_image_mode=False` — enables tracking mode (faster, uses previous frame)
- `max_num_hands=2` — we need both hands
- `min_detection_confidence` — how sure the model must be before it detects
- `min_tracking_confidence` — how sure to keep tracking once detected

**Results you get back:**
- `results.multi_hand_landmarks` — list of landmark sets (one per hand)
- `results.multi_handedness` — tells you which hand is left/right

---

## 4. Hand Landmarks — The 21 Points

MediaPipe gives 21 landmarks per hand, indexed 0–20. The ones you need:

```
THUMB:  CMC=1, MCP=2,  IP=3,   TIP=4
INDEX:  MCP=5, PIP=6,  DIP=7,  TIP=8
MIDDLE: MCP=9, PIP=10, DIP=11, TIP=12
RING:   MCP=13,PIP=14, DIP=15, TIP=16
PINKY:  MCP=17,PIP=18, DIP=19, TIP=20
```

Each landmark has:
- `.x` — normalized 0.0–1.0 (left–right of image)
- `.y` — normalized 0.0–1.0 (top–bottom of image)
- `.z` — depth (negative = closer to camera)

To convert to pixel coords: `x_pixel = int(lm.x * frame_width)`

---

## 5. Detecting Finger Extension

### Thumb (handedness-aware)
The thumb extends sideways, so we compare X coordinates:

```python
tip = landmarks[4]   # THUMB_TIP
ip  = landmarks[3]   # THUMB_IP
mcp = landmarks[2]   # THUMB_MCP
```

- **Right hand**: thumb is open when `tip.x > ip.x` (thumb points left in mirror)
- **Left hand**:  thumb is open when `tip.x < ip.x`

Check `results.multi_handedness[idx].classification[0].label` to get "Left" or "Right".

### Index Finger
The index extends upward, so compare Y coordinates:

```python
tip = landmarks[8]   # INDEX_TIP
pip = landmarks[6]   # INDEX_PIP
mcp = landmarks[5]   # INDEX_MCP
```

Index is open when `tip.y < pip.y` AND `tip.y < mcp.y` (tip is above both joints in image coords — remember Y increases downward).

---

## 6. Portal Logic

Track a counter of how many consecutive frames the gesture has been held:

```python
if thumbs_extended >= 2 and index_extended >= 2:
    gesture_held_frames += 1
else:
    gesture_held_frames = 0

if gesture_held_frames >= HOLD_THRESHOLD:
    # SUMMON PORTAL
```

Use a hold threshold (e.g. 5–10 frames) to prevent flickering.

---

## 7. Portal Position & Size

Position the portal at the midpoint between index fingertips:

```python
cx = (tip1_x + tip2_x) // 2
cy = (tip1_y + tip2_y) // 2
```

Size from distance between hands:

```python
distance = math.dist((x1, y1), (x2, y2))
radius = int(distance * 0.6)
```

---

## 8. Drawing the Portal

Use OpenCV drawing functions on an overlay, then blend with `cv2.addWeighted`:

### Glow effect
Draw concentric circles with decreasing intensity:

```python
for i in range(glow_radius, 0, -1):
    intensity = int(255 * (1 - i / glow_radius))
    cv2.circle(overlay, (cx, cy), i, (intensity, intensity//2, 255), -1)
```

### Rotating rings
Use `math.sin/cos` to compute points on a circle, increment an angle each frame:

```python
angle += 0.03  # each frame
x = cx + radius * math.cos(angle)
y = cy + radius * math.sin(angle)
```

### Particles
Store particle objects with position, velocity, lifetime, color. Each frame:
- Spawn a few at portal center
- Update: `x += vx`, `y += vy`, `life -= dt`
- Draw with alpha based on remaining life
- Remove dead ones

---

## 9. Visual Polish Ideas

- **Additive blending** — use `cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)`
- **Counter-rotating rings** — outer ring rotates +angle, inner ring rotates -angle
- **Spokes** — lines from center outward that pulse in length
- **Bright center** — white circle in the middle for the "core"
- **Bloom** — draw larger, fainter circles around bright parts

---

## 10. Performance Tips

- Use `cv2.waitKey(1)` with small delay (1ms) for real-time
- Show FPS with `1 / (current_time - prev_time)` to gauge performance
- Lower webcam resolution or skip every Nth frame if slow
- Reuse arrays and avoid creating new objects every frame if possible

---

## 11. Complete Flow

```
LOOP:
    1. Capture frame from webcam
    2. Flip horizontally (mirror)
    3. Convert BGR → RGB
    4. Run MediaPipe Hands
    5. For each detected hand:
       a. Draw landmarks (optional)
       b. Check thumb extension (handedness-aware)
       c. Check index extension
       d. Store finger tip positions
    6. If 2 thumbs AND 2 index fingers extended:
       → Hold counter ++
       → If held long enough: position portal at midpoint, activate
    7. Else: deactivate portal, reset counter
    8. Update portal animation (angle, particles)
    9. Draw portal overlay on frame
    10. Show frame with cv2.imshow
    11. If 'q' pressed: break
```

---

## 12. Reference

- MediaPipe Hands: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- OpenCV drawing: `cv2.circle`, `cv2.line`, `cv2.putText`, `cv2.addWeighted`
- Landmark diagram: search "MediaPipe hand landmarks 21 points"
