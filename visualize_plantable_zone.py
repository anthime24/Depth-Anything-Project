"""
visualize_plantable_zone.py

Génère des visualisations des zones plantables identifiées.
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path


# Paths
PLANTABLE_JSON = Path("PlantableZone.json")
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_DIR = Path("visuals")

OUT_DIR.mkdir(exist_ok=True)

print("Loading plantable zone data...")
with open(PLANTABLE_JSON, "r", encoding="utf-8") as f:
    plantable = json.load(f)

print("Loading vision output...")
with open(VISION_JSON, "r", encoding="utf-8") as f:
    vision = json.load(f)

print("Loading image...")
img = cv2.imread(str(IMAGE_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape

segments = vision.get("segments", [])
segment_map = {seg["segment_id"]: seg for seg in segments}

# ========== 1) Plantable vs Non-plantable segments ==========
overlay_plantable = img.copy()

plantable_ids = set(plantable["plantable"]["segment_ids"])

for seg in segments:
    seg_id = seg["segment_id"]
    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)
    
    if seg_id in plantable_ids:
        # Vert semi-transparent pour plantable
        color = np.array([0, 255, 0], dtype=np.uint8)
        alpha = 0.4
    else:
        # Rouge semi-transparent pour non-plantable
        color = np.array([255, 50, 50], dtype=np.uint8)
        alpha = 0.2
    
    overlay_plantable[mask] = (
        (1 - alpha) * overlay_plantable[mask] + alpha * color
    )

cv2.imwrite(
    str(OUT_DIR / "plantable_zones.png"),
    cv2.cvtColor(overlay_plantable, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {OUT_DIR / 'plantable_zones.png'}")


# ========== 2) Combined plantable mask only ==========
plantable_rle = plantable["plantable"]["mask_rle"]
plantable_mask = mask_utils.decode(plantable_rle).astype(bool)

overlay_mask_only = img.copy()
# Vert vif sur zone plantable
overlay_mask_only[plantable_mask] = (
    0.3 * overlay_mask_only[plantable_mask] + 
    0.7 * np.array([0, 255, 0], dtype=np.uint8)
)

cv2.imwrite(
    str(OUT_DIR / "plantable_mask_combined.png"),
    cv2.cvtColor(overlay_mask_only, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {OUT_DIR / 'plantable_mask_combined.png'}")


# ========== 3) Anchor points visualization ==========
overlay_anchors = img.copy()

# Afficher la zone plantable en transparence
overlay_anchors[plantable_mask] = (
    0.6 * overlay_anchors[plantable_mask] + 
    0.4 * np.array([0, 255, 0], dtype=np.uint8)
)

# Dessiner les anchor points
anchors = plantable["plantable"]["anchors"]
for anchor in anchors:
    x_norm, y_norm = anchor
    x_px = int(x_norm * W)
    y_px = int(y_norm * H)
    
    # Croix rouge pour chaque anchor
    cv2.drawMarker(
        overlay_anchors,
        (x_px, y_px),
        color=(255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=2
    )

# Texte avec le nombre d'anchors
cv2.putText(
    overlay_anchors,
    f"Anchors: {len(anchors)}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    overlay_anchors,
    f"Coverage: {plantable['plantable']['coverage']:.1%}",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.imwrite(
    str(OUT_DIR / "plantable_anchors.png"),
    cv2.cvtColor(overlay_anchors, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {OUT_DIR / 'plantable_anchors.png'}")


# ========== 4) Segment analysis heatmap (plantability score) ==========
# Créer un score de plantabilité basé sur les features
overlay_score = img.copy()

for detail in plantable["segment_analysis"]:
    seg_id = detail["segment_id"]
    if seg_id not in segment_map:
        continue
    
    seg = segment_map[seg_id]
    rle = seg["mask_rle"]
    mask = mask_utils.decode(rle).astype(bool)
    
    # Score simple : 1.0 si plantable, sinon basé sur le nombre de critères échoués
    if detail["is_plantable"]:
        score = 1.0
        color = np.array([0, 255, 0], dtype=np.uint8)  # Vert
    else:
        # Score inversé basé sur le nombre de rejets
        num_rejections = len(detail["rejection_reasons"])
        score = max(0.0, 1.0 - num_rejections * 0.25)
        
        # Gradient rouge -> orange -> jaune
        if score < 0.33:
            color = np.array([255, 0, 0], dtype=np.uint8)  # Rouge
        elif score < 0.66:
            color = np.array([255, 165, 0], dtype=np.uint8)  # Orange
        else:
            color = np.array([255, 255, 0], dtype=np.uint8)  # Jaune
    
    overlay_score[mask] = (
        0.6 * overlay_score[mask] + 0.4 * color
    )

cv2.imwrite(
    str(OUT_DIR / "plantable_score_heatmap.png"),
    cv2.cvtColor(overlay_score, cv2.COLOR_RGB2BGR)
)
print(f"✅ Saved {OUT_DIR / 'plantable_score_heatmap.png'}")


print("\n" + "=" * 60)
print("VISUALIZATIONS GENERATED")
print("=" * 60)
print(f"Output directory: {OUT_DIR}/")
print(f"  1. plantable_zones.png          - Vert = plantable, Rouge = non-plantable")
print(f"  2. plantable_mask_combined.png  - Zone plantable uniquement")
print(f"  3. plantable_anchors.png        - Points d'ancrage pour IA générative")
print(f"  4. plantable_score_heatmap.png  - Score de plantabilité par segment")
