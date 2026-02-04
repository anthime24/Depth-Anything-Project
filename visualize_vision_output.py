import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
import random

# Paths
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_DIR = Path("visuals")

OUT_DIR.mkdir(exist_ok=True)

# Load data
with open(VISION_JSON, "r", encoding="utf-8") as f:
    vision = json.load(f)

img = cv2.imread(str(IMAGE_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape

segments = vision["sam_output"]["segments"]

# ========== 1) Random colored segments ==========
overlay_random = img.copy()

random.seed(42)
for seg in segments:
    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)

    color = np.array([
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ], dtype=np.uint8)

    overlay_random[mask] = (
        0.5 * overlay_random[mask] + 0.5 * color
    )

cv2.imwrite(
    str(OUT_DIR / "vision_segments_colored.png"),
    cv2.cvtColor(overlay_random, cv2.COLOR_RGB2BGR)
)

# ========== 2) Depth band visualization ==========
overlay_band = img.copy()

band_colors = {
    "front": np.array([255, 0, 0], dtype=np.uint8),   # rouge
    "mid":   np.array([255, 165, 0], dtype=np.uint8), # orange
    "back":  np.array([0, 128, 255], dtype=np.uint8), # bleu
}

for seg in segments:
    band = seg.get("depth_band")
    if band is None:
        continue

    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)
    color = band_colors[band]

    overlay_band[mask] = (
        0.6 * overlay_band[mask] + 0.4 * color
    )

cv2.imwrite(
    str(OUT_DIR / "vision_segments_depth_band.png"),
    cv2.cvtColor(overlay_band, cv2.COLOR_RGB2BGR)
)

# ========== 3) Mean depth heatmap ==========
overlay_heat = img.copy()

for seg in segments:
    mean_depth = seg.get("mean_depth")
    if mean_depth is None:
        continue

    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)

    # colormap simple (bleu -> rouge)
    heat_color = np.array([
        int(255 * mean_depth),        # R
        int(100 * (1 - mean_depth)),  # G
        int(255 * (1 - mean_depth))   # B
    ], dtype=np.uint8)

    overlay_heat[mask] = (
        0.6 * overlay_heat[mask] + 0.4 * heat_color
    )

cv2.imwrite(
    str(OUT_DIR / "vision_segments_depth_heatmap.png"),
    cv2.cvtColor(overlay_heat, cv2.COLOR_RGB2BGR)
)

print("✅ Visuals generated in /visuals")
