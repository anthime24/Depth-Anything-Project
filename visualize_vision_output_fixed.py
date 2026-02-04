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
print(f"Loading {VISION_JSON}...")
with open(VISION_JSON, "r", encoding="utf-8") as f:
    vision = json.load(f)

print(f"Loading {IMAGE_PATH}...")
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape

# Support both old and new structure
# New structure: {"segments": [...]}
# Old structure: {"sam_output": {"segments": [...]}}
if "segments" in vision:
    # New structure (improved)
    segments = vision["segments"]
    print(f"✅ Using new VisionOutput structure")
elif "sam_output" in vision and "segments" in vision["sam_output"]:
    # Old structure (original)
    segments = vision["sam_output"]["segments"]
    print(f"✅ Using old VisionOutput structure")
else:
    raise KeyError("Cannot find segments in VisionOutput.json. Expected 'segments' or 'sam_output.segments'")

print(f"Found {len(segments)} segments")

# ========== 1) Random colored segments ==========
print("\n🎨 Generating random colored segments...")
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

output_path = OUT_DIR / "vision_segments_colored.png"
cv2.imwrite(
    str(output_path),
    cv2.cvtColor(overlay_random, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {output_path}")

# ========== 2) Depth band visualization ==========
print("\n🎨 Generating depth band visualization...")
overlay_band = img.copy()

band_colors = {
    "front": np.array([255, 0, 0], dtype=np.uint8),   # rouge
    "mid":   np.array([255, 165, 0], dtype=np.uint8), # orange
    "back":  np.array([0, 128, 255], dtype=np.uint8), # bleu
}

segments_with_depth = 0
for seg in segments:
    band = seg.get("depth_band")
    if band is None:
        continue

    segments_with_depth += 1
    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)
    color = band_colors.get(band, np.array([128, 128, 128], dtype=np.uint8))

    overlay_band[mask] = (
        0.6 * overlay_band[mask] + 0.4 * color
    )

output_path = OUT_DIR / "vision_segments_depth_band.png"
cv2.imwrite(
    str(output_path),
    cv2.cvtColor(overlay_band, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {output_path}")
print(f"   ({segments_with_depth}/{len(segments)} segments have depth info)")

# ========== 3) Mean depth heatmap ==========
print("\n🎨 Generating mean depth heatmap...")
overlay_heat = img.copy()

segments_with_mean_depth = 0
for seg in segments:
    mean_depth = seg.get("mean_depth")
    if mean_depth is None:
        continue

    segments_with_mean_depth += 1
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

output_path = OUT_DIR / "vision_segments_depth_heatmap.png"
cv2.imwrite(
    str(output_path),
    cv2.cvtColor(overlay_heat, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {output_path}")
print(f"   ({segments_with_mean_depth}/{len(segments)} segments have mean_depth)")

print("\n" + "=" * 60)
print("✅ Visuals generated in /visuals")
print("=" * 60)
print(f"  1. vision_segments_colored.png       - {len(segments)} segments with random colors")
print(f"  2. vision_segments_depth_band.png    - Depth bands (red=front, orange=mid, blue=back)")
print(f"  3. vision_segments_depth_heatmap.png - Continuous depth gradient")
