"""
compute_plantable_zone.py

Analyse les segments enrichis (géométrie + profondeur) pour identifier
les zones potentiellement plantables dans un jardin.

Critères de plantabilité :
- Position : bas de l'image (typiquement y > 0.5)
- Couleur : dominante verte ou brune (végétation/sol)
- Profondeur : front ou mid (proche/moyen)
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Tuple, Dict


# ============ Configuration ============
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("PlantableZone.json")

# Seuils de plantabilité
CONFIG = {
    "position": {
        "min_y_centroid": 0.4,  # segments avec centroid_y > 0.4 (bas de l'image)
    },
    "depth": {
        "allowed_bands": ["front", "mid"],  # pas "back"
    },
    "color": {
        "min_green_ratio": 0.3,   # ratio vert minimum (détection végétation)
        "min_brown_ratio": 0.2,   # ratio brun minimum (détection sol)
    },
    "size": {
        "min_area_ratio": 0.005,  # segments > 0.5% de l'image
    },
    "anchors": {
        "num_points": 15,  # nombre de points d'ancrage à générer
    }
}


def compute_color_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Analyse les caractéristiques de couleur d'un segment.
    
    Returns:
        {
            "mean_rgb": [r, g, b],
            "mean_hsv": [h, s, v],
            "green_ratio": float,  # ratio de pixels verts
            "brown_ratio": float   # ratio de pixels bruns
        }
    """
    pixels = img[mask]
    
    if len(pixels) == 0:
        return {
            "mean_rgb": [0, 0, 0],
            "mean_hsv": [0, 0, 0],
            "green_ratio": 0.0,
            "brown_ratio": 0.0
        }
    
    # Couleur moyenne RGB
    mean_rgb = pixels.mean(axis=0).tolist()
    
    # Conversion HSV pour analyse couleur
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    mean_hsv = pixels_hsv.mean(axis=0).tolist()
    
    # Détection vert : H dans [35-85], S > 20, V > 20
    green_mask = (
        (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) &
        (pixels_hsv[:, 1] > 20) & (pixels_hsv[:, 2] > 20)
    )
    green_ratio = green_mask.sum() / len(pixels)
    
    # Détection brun : H dans [10-25], S > 20, V dans [20-100]
    brown_mask = (
        (pixels_hsv[:, 0] >= 10) & (pixels_hsv[:, 0] <= 25) &
        (pixels_hsv[:, 1] > 20) & 
        (pixels_hsv[:, 2] > 20) & (pixels_hsv[:, 2] < 100)
    )
    brown_ratio = brown_mask.sum() / len(pixels)
    
    return {
        "mean_rgb": [float(x) for x in mean_rgb],
        "mean_hsv": [float(x) for x in mean_hsv],
        "green_ratio": float(green_ratio),
        "brown_ratio": float(brown_ratio)
    }


def is_plantable(segment: Dict, color_features: Dict, config: Dict) -> Tuple[bool, List[str]]:
    """
    Détermine si un segment est plantable selon les critères.
    
    Returns:
        (is_plantable, reasons)
    """
    reasons = []
    
    # Critère 1 : Position (bas de l'image)
    centroid_y = segment.get("centroid", [0, 0])[1]
    if centroid_y < config["position"]["min_y_centroid"]:
        reasons.append(f"position_too_high (y={centroid_y:.2f})")
    
    # Critère 2 : Profondeur
    depth_band = segment.get("depth_band")
    if depth_band not in config["depth"]["allowed_bands"]:
        reasons.append(f"depth_too_far (band={depth_band})")
    
    # Critère 3 : Couleur (vert OU brun)
    green_ratio = color_features["green_ratio"]
    brown_ratio = color_features["brown_ratio"]
    
    is_green = green_ratio >= config["color"]["min_green_ratio"]
    is_brown = brown_ratio >= config["color"]["min_brown_ratio"]
    
    if not (is_green or is_brown):
        reasons.append(f"color_not_vegetation (green={green_ratio:.2f}, brown={brown_ratio:.2f})")
    
    # Critère 4 : Taille minimale
    area_ratio = segment.get("area_ratio", 0)
    if area_ratio < config["size"]["min_area_ratio"]:
        reasons.append(f"too_small (area={area_ratio:.4f})")
    
    is_plantable = len(reasons) == 0
    
    return is_plantable, reasons


def generate_anchor_points(mask: np.ndarray, num_points: int) -> List[List[float]]:
    """
    Génère des points d'ancrage uniformément répartis dans le masque.
    
    Returns:
        List of [x_norm, y_norm] in [0,1] coordinates
    """
    H, W = mask.shape
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) == 0:
        return []
    
    # Échantillonnage uniforme des points
    total_pixels = len(x_coords)
    if total_pixels <= num_points:
        # Prendre tous les pixels
        indices = np.arange(total_pixels)
    else:
        # Sous-échantillonnage uniforme
        indices = np.linspace(0, total_pixels - 1, num_points, dtype=int)
    
    anchors = []
    for idx in indices:
        x_norm = float(x_coords[idx]) / W
        y_norm = float(y_coords[idx]) / H
        anchors.append([x_norm, y_norm])
    
    return anchors


def main():
    print("=" * 60)
    print("PLANTABLE ZONE COMPUTATION")
    print("=" * 60)
    
    # Load vision output
    print(f"\nLoading {VISION_JSON}...")
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    # Load image
    print(f"Loading {IMAGE_PATH}...")
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"Total segments: {len(segments)}")
    
    # Analyse each segment
    print("\nAnalyzing segments for plantability...")
    plantable_segments = []
    plantable_mask_combined = np.zeros((H, W), dtype=bool)
    
    segment_details = []
    
    for seg in segments:
        seg_id = seg["segment_id"]
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        # Compute color features
        color_features = compute_color_features(img, mask)
        
        # Check plantability
        is_plant, reasons = is_plantable(seg, color_features, CONFIG)
        
        detail = {
            "segment_id": seg_id,
            "area_ratio": seg.get("area_ratio"),
            "centroid": seg.get("centroid"),
            "depth_band": seg.get("depth_band"),
            "mean_depth": seg.get("mean_depth"),
            "color_features": color_features,
            "is_plantable": is_plant,
            "rejection_reasons": reasons if not is_plant else []
        }
        segment_details.append(detail)
        
        if is_plant:
            plantable_segments.append(seg_id)
            plantable_mask_combined |= mask
    
    print(f"\n✅ Plantable segments: {len(plantable_segments)} / {len(segments)}")
    
    # Compute coverage
    total_pixels = H * W
    plantable_pixels = plantable_mask_combined.sum()
    coverage = float(plantable_pixels) / total_pixels
    
    print(f"✅ Plantable coverage: {coverage:.2%}")
    
    # Generate anchor points
    print(f"\nGenerating {CONFIG['anchors']['num_points']} anchor points...")
    anchors = generate_anchor_points(plantable_mask_combined, CONFIG["anchors"]["num_points"])
    print(f"✅ Generated {len(anchors)} anchors")
    
    # Encode combined mask as RLE (optional, for storage)
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_mask_combined.astype(np.uint8)))
    # Convert to serializable format
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    # Build output
    output = {
        "version": "plantable_zone_v1",
        "image_id": vision.get("image_id"),
        "image_size": [W, H],
        
        "config": CONFIG,
        
        "plantable": {
            "segments_count": len(plantable_segments),
            "segment_ids": plantable_segments,
            "coverage": coverage,
            "total_pixels": int(plantable_pixels),
            "mask_rle": plantable_rle,
            "anchors": anchors
        },
        
        "segment_analysis": segment_details
    }
    
    # Save
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total segments:       {len(segments)}")
    print(f"Plantable segments:   {len(plantable_segments)} ({len(plantable_segments)/len(segments):.1%})")
    print(f"Plantable coverage:   {coverage:.2%}")
    print(f"Anchor points:        {len(anchors)}")
    
    # Rejection reasons statistics
    print("\nRejection reasons breakdown:")
    rejection_stats = {}
    for detail in segment_details:
        if not detail["is_plantable"]:
            for reason in detail["rejection_reasons"]:
                reason_key = reason.split("(")[0].strip()
                rejection_stats[reason_key] = rejection_stats.get(reason_key, 0) + 1
    
    for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
