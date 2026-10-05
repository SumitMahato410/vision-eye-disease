"""Colour-based analysis of a close-up eye photo (no training data needed).
Finds the visible 'white of the eye' (sclera) area and measures how much of it is red/pink or yellow.
Thresholds below are heuristics - adjust them if your photos behave differently."""
import base64, io
import numpy as np
from PIL import Image

CLEAR_MAX = 0.20      # redness ratio below this -> clear eye
MILD_MAX = 0.45       # below this -> mild/moderate redness, above -> severe
YELLOW_MIN = 0.35
MIN_POOL = 0.02       # at least 2% of the central area must look like eye surface

def analyze(img):
    img = img.convert("RGB"); img.thumbnail((500, 500))
    a = np.asarray(img).astype(float)
    h, w, _ = a.shape
    y0, y1, x0, x1 = int(h * .1), int(h * .9), int(w * .1), int(w * .9)
    R, G, B = a[y0:y1, x0:x1, 0], a[y0:y1, x0:x1, 1], a[y0:y1, x0:x1, 2]
    mx, mn = a[y0:y1, x0:x1].max(-1), a[y0:y1, x0:x1].min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    gb = G / (B + 1e-6); rg = R / (G + 1e-6)

    white = (mx > 150) & (sat < 0.22)                                   # healthy sclera
    red = (mx > 70) & (rg > 1.3) & (gb > 0.75) & (gb < 1.15) & (sat > 0.25)  # pink/red (G~B separates it from skin)
    yellow = (R > 150) & (rg < 1.15) & (gb > 1.5) & (sat > 0.3)         # yellow tint
    pool = white | red | yellow
    if pool.mean() < MIN_POOL:
        return {"error": "Could not find the white of the eye. Use a close-up, well-lit photo of one open eye."}

    n = pool.sum()
    red_ratio, yellow_ratio = float(red.sum() / n), float(yellow.sum() / n)
    if red_ratio < CLEAR_MAX: level, symptom = "None", "clear_eye"
    elif red_ratio < MILD_MAX: level, symptom = "Mild-Moderate", "red_eye"
    else: level, symptom = "Severe", "red_eye"

    over = np.asarray(img).copy()
    region = over[y0:y1, x0:x1]
    region[red] = (0.5 * region[red] + 0.5 * np.array([0, 255, 200])).astype(np.uint8)
    over[y0:y1, x0:x1] = region
    buf = io.BytesIO(); Image.fromarray(over).save(buf, "PNG")
    return {"error": None, "symptom": symptom, "redness_percent": round(red_ratio * 100, 1), "redness_level": level,
            "yellow_percent": round(yellow_ratio * 100, 1), "yellowing": yellow_ratio >= YELLOW_MIN,
            "overlay": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
