"""
compute_plantable_zone.py - VERSION FINALE OPTIMISÉE

Accepte la pelouse en arrière-plan (back) tout en excluant murs et mobilier.
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Tuple, Dict


# ============ Configuration OPTIMISÉE ============
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("PlantableZone.json")

# Seuils OPTIMISÉS : strict sur murs/mobilier, permissif sur pelouse
CONFIG = {
    "position": {
        "min_y_centroid": 0.50,  # ⬇️ Légèrement assoupli pour pelouse haute
        "min_y_bbox": 0.45,      # ⬇️ Assoupli
    },
    "depth": {
        # ✅ CHANGEMENT CLÉ : accepter "back" pour pelouse arrière-plan
        "allowed_bands": ["front", "mid", "back"],  # Tout accepté
        "max_mean_depth": 0.90,   # Objets très proches (mobilier 3D)
        "min_mean_depth": 0.15,   # ⬇️ Assoupli pour pelouse lointaine
    },
    "color": {
        "min_green_ratio": 0.30,      # ⬇️ Légèrement assoupli
        "min_brown_ratio": 0.20,      # ⬇️ Assoupli
        "max_gray_ratio": 0.65,       # ⬆️ Légèrement plus permissif
        "require_saturation": True,
        "min_saturation": 25,         # ⬇️ Assoupli pour herbe sèche
    },
    "shape": {
        "max_aspect_ratio": 3.5,      # ⬆️ Légèrement assoupli
        "min_area_ratio": 0.005,      # ⬇️ Assoupli (0.5%)
        "max_area_ratio": 0.40,       # ⬆️ Assoupli pour grande pelouse
    },
    "texture": {
        "check_edge_density": True,
        "max_edge_ratio": 0.35,       # ⬆️ Légèrement assoupli
    },
    "special_rules": {
        # Règle spéciale : pelouse peut être "back" si grande et verte
        "allow_back_if_large_green": True,
        "large_green_min_area": 0.05,      # > 5% de l'image
        "large_green_min_green_ratio": 0.40,  # > 40% de pixels verts
    },
    "anchors": {
        "num_points": 15,
    }
}


def compute_color_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """Analyse couleur avec détection saturation et gris."""
    pixels = img[mask]
    
    if len(pixels) == 0:
        return {
            "mean_rgb": [0, 0, 0],
            "mean_hsv": [0, 0, 0],
            "green_ratio": 0.0,
            "brown_ratio": 0.0,
            "gray_ratio": 0.0,
            "mean_saturation": 0.0
        }
    
    mean_rgb = pixels.mean(axis=0).tolist()
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    mean_hsv = pixels_hsv.mean(axis=0).tolist()
    mean_saturation = float(pixels_hsv[:, 1].mean())
    
    # Vert (végétation)
    green_mask = (
        (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) &
        (pixels_hsv[:, 1] > 25) &  # Légèrement assoupli
        (pixels_hsv[:, 2] > 25)
    )
    green_ratio = green_mask.sum() / len(pixels)
    
    # Brun (sol)
    brown_mask = (
        (pixels_hsv[:, 0] >= 10) & (pixels_hsv[:, 0] <= 30) &
        (pixels_hsv[:, 1] > 20) &
        (pixels_hsv[:, 2] > 15) & (pixels_hsv[:, 2] < 130)
    )
    brown_ratio = brown_mask.sum() / len(pixels)
    
    # Gris (murs)
    gray_mask = (
        (pixels_hsv[:, 1] < 30) |
        (
            (pixels_hsv[:, 0] >= 15) & (pixels_hsv[:, 0] <= 35) &
            (pixels_hsv[:, 1] < 40) &
            (pixels_hsv[:, 2] > 80)
        )
    )
    gray_ratio = gray_mask.sum() / len(pixels)
    
    return {
        "mean_rgb": [float(x) for x in mean_rgb],
        "mean_hsv": [float(x) for x in mean_hsv],
        "green_ratio": float(green_ratio),
        "brown_ratio": float(brown_ratio),
        "gray_ratio": float(gray_ratio),
        "mean_saturation": mean_saturation
    }


def compute_shape_features(segment: Dict, H: int, W: int) -> Dict:
    """Features géométriques."""
    bbox = segment.get("bbox", [0, 0, 1, 1])
    bbox_w = bbox[2] * W
    bbox_h = bbox[3] * H
    aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 0
    
    return {
        "aspect_ratio": float(aspect_ratio),
        "bbox_y_start": float(bbox[1]),
        "bbox_y_end": float(bbox[1] + bbox[3]),
        "bbox_width": float(bbox_w),
        "bbox_height": float(bbox_h)
    }


def compute_texture_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """Détection contours pour mobilier."""
    if mask.sum() == 0:
        return {"edge_ratio": 0.0}
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    
    edge_pixels = edges[mask].sum() / 255
    total_pixels = mask.sum()
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
    
    return {"edge_ratio": float(edge_ratio)}


def is_plantable(
    segment: Dict,
    color_features: Dict,
    shape_features: Dict,
    texture_features: Dict,
    config: Dict
) -> Tuple[bool, List[str]]:
    """
    Détermine si plantable avec règle spéciale pour pelouse en arrière-plan.
    """
    reasons = []
    
    # Données du segment
    centroid_y = segment.get("centroid", [0, 0])[1]
    bbox_y_start = shape_features["bbox_y_start"]
    depth_band = segment.get("depth_band")
    mean_depth = segment.get("mean_depth", 0)
    area_ratio = segment.get("area_ratio", 0)
    aspect_ratio = shape_features["aspect_ratio"]
    
    green_ratio = color_features["green_ratio"]
    brown_ratio = color_features["brown_ratio"]
    gray_ratio = color_features["gray_ratio"]
    saturation = color_features["mean_saturation"]
    edge_ratio = texture_features["edge_ratio"]
    
    # ===== RÈGLE SPÉCIALE : Grande pelouse verte peut être "back" =====
    is_large_green = (
        area_ratio >= config["special_rules"]["large_green_min_area"] and
        green_ratio >= config["special_rules"]["large_green_min_green_ratio"]
    )
    
    if is_large_green and depth_band == "back":
        # Pelouse en arrière-plan : assouplir les critères
        print(f"  🌱 Segment {segment['segment_id']}: Large green area in background (lawn)")
        
        # Vérifier quand même les critères de base
        if gray_ratio > config["color"]["max_gray_ratio"]:
            reasons.append(f"too_much_gray (gray={gray_ratio:.2f})")
        
        if edge_ratio > config["texture"]["max_edge_ratio"]:
            reasons.append(f"high_edge_density (edges={edge_ratio:.2f})")
        
        if aspect_ratio > config["shape"]["max_aspect_ratio"] * 1.2:  # Plus permissif
            reasons.append(f"too_vertical (ratio={aspect_ratio:.1f})")
        
        # Si pas de raisons rédhibitoires, accepter
        if len(reasons) == 0:
            return True, []
    
    # ===== CRITÈRES STANDARD =====
    
    # Position
    if centroid_y < config["position"]["min_y_centroid"]:
        reasons.append(f"position_too_high (y={centroid_y:.2f})")
    
    if bbox_y_start < config["position"]["min_y_bbox"]:
        reasons.append(f"bbox_starts_too_high (y={bbox_y_start:.2f})")
    
    # Profondeur
    if depth_band not in config["depth"]["allowed_bands"]:
        reasons.append(f"depth_band_invalid (band={depth_band})")
    
    if mean_depth > config["depth"]["max_mean_depth"]:
        reasons.append(f"depth_too_close (depth={mean_depth:.2f})")
    
    if mean_depth < config["depth"]["min_mean_depth"]:
        reasons.append(f"depth_too_far (depth={mean_depth:.2f})")
    
    # Couleur
    is_green = green_ratio >= config["color"]["min_green_ratio"]
    is_brown = brown_ratio >= config["color"]["min_brown_ratio"]
    
    if not (is_green or is_brown):
        reasons.append(f"color_not_vegetation (green={green_ratio:.2f}, brown={brown_ratio:.2f})")
    
    if gray_ratio > config["color"]["max_gray_ratio"]:
        reasons.append(f"too_much_gray (gray={gray_ratio:.2f})")
    
    if config["color"]["require_saturation"] and saturation < config["color"]["min_saturation"]:
        reasons.append(f"low_saturation (sat={saturation:.1f})")
    
    # Forme
    if area_ratio < config["shape"]["min_area_ratio"]:
        reasons.append(f"too_small (area={area_ratio:.4f})")
    
    if area_ratio > config["shape"]["max_area_ratio"]:
        reasons.append(f"too_large (area={area_ratio:.2f})")
    
    if aspect_ratio > config["shape"]["max_aspect_ratio"]:
        reasons.append(f"too_vertical (ratio={aspect_ratio:.1f})")
    
    # Texture
    if config["texture"]["check_edge_density"] and edge_ratio > config["texture"]["max_edge_ratio"]:
        reasons.append(f"high_edge_density (edges={edge_ratio:.2f})")
    
    is_plantable = len(reasons) == 0
    return is_plantable, reasons


def generate_anchor_points(mask: np.ndarray, num_points: int) -> List[List[float]]:
    """Génère anchor points."""
    H, W = mask.shape
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) == 0:
        return []
    
    total_pixels = len(x_coords)
    if total_pixels <= num_points:
        indices = np.arange(total_pixels)
    else:
        indices = np.linspace(0, total_pixels - 1, num_points, dtype=int)
    
    anchors = []
    for idx in indices:
        x_norm = float(x_coords[idx]) / W
        y_norm = float(y_coords[idx]) / H
        anchors.append([x_norm, y_norm])
    
    return anchors


def main():
    print("=" * 70)
    print("PLANTABLE ZONE - VERSION FINALE OPTIMISÉE")
    print("=" * 70)
    
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"Total segments: {len(segments)}")
    
    print("\n🔍 Analyzing segments...")
    plantable_segments = []
    plantable_mask_combined = np.zeros((H, W), dtype=bool)
    segment_details = []
    
    for seg in segments:
        seg_id = seg["segment_id"]
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        color_features = compute_color_features(img, mask)
        shape_features = compute_shape_features(seg, H, W)
        texture_features = compute_texture_features(img, mask)
        
        is_plant, reasons = is_plantable(seg, color_features, shape_features, texture_features, CONFIG)
        
        detail = {
            "segment_id": seg_id,
            "area_ratio": seg.get("area_ratio"),
            "centroid": seg.get("centroid"),
            "depth_band": seg.get("depth_band"),
            "mean_depth": seg.get("mean_depth"),
            "color_features": color_features,
            "shape_features": shape_features,
            "texture_features": texture_features,
            "is_plantable": is_plant,
            "rejection_reasons": reasons if not is_plant else []
        }
        segment_details.append(detail)
        
        if is_plant:
            plantable_segments.append(seg_id)
            plantable_mask_combined |= mask
            print(f"  ✅ Segment {seg_id}: PLANTABLE")
    
    coverage = plantable_mask_combined.sum() / (H * W)
    print(f"\n✅ Plantable segments: {len(plantable_segments)} / {len(segments)}")
    print(f"✅ Coverage: {coverage:.1%}")
    
    anchors = generate_anchor_points(plantable_mask_combined, CONFIG["anchors"]["num_points"])
    
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_mask_combined.astype(np.uint8)))
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    output = {
        "version": "plantable_zone_v3_optimized",
        "image_id": vision.get("image_id"),
        "image_size": [W, H],
        "config": CONFIG,
        "plantable": {
            "segments_count": len(plantable_segments),
            "segment_ids": plantable_segments,
            "coverage": coverage,
            "total_pixels": int(plantable_mask_combined.sum()),
            "mask_rle": plantable_rle,
            "anchors": anchors
        },
        "segment_analysis": segment_details
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Plantable segments:   {len(plantable_segments)}/{len(segments)} ({len(plantable_segments)/len(segments):.1%})")
    print(f"Coverage:             {coverage:.1%}")
    print(f"Anchors:              {len(anchors)}")
    
    print("\n📊 Rejection reasons:")
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
