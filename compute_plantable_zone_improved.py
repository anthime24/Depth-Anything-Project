"""
compute_plantable_zone.py - VERSION AMÉLIORÉE

Analyse stricte des zones plantables avec filtres pour exclure :
- Murs et façades (position haute + vertical)
- Mobilier (chaises, tables)
- Structures non-sol (bbox ratio)
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Tuple, Dict


# ============ Configuration AMÉLIORÉE ============
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("PlantableZone.json")

# Seuils STRICTS pour éviter les faux positifs
CONFIG = {
    "position": {
        "min_y_centroid": 0.55,  # ⬆️ Plus strict : seulement bas de l'image
        "min_y_bbox": 0.50,      # ⬆️ NOUVEAU : bbox doit commencer assez bas
    },
    "depth": {
        "allowed_bands": ["front", "mid"],
        "max_mean_depth": 0.85,  # ⬆️ NOUVEAU : exclure premier plan trop proche (objets)
        "min_mean_depth": 0.25,  # ⬆️ NOUVEAU : exclure arrière-plan trop loin
    },
    "color": {
        "min_green_ratio": 0.35,      # ⬆️ Plus strict
        "min_brown_ratio": 0.25,      # ⬆️ Plus strict
        "max_gray_ratio": 0.6,        # ⬆️ NOUVEAU : exclure murs gris/beiges
        "require_saturation": True,   # ⬆️ NOUVEAU : nécessite couleur saturée
        "min_saturation": 30,         # ⬆️ NOUVEAU : seuil HSV saturation
    },
    "shape": {
        "max_aspect_ratio": 3.0,      # ⬆️ NOUVEAU : exclure formes très allongées (murs)
        "min_area_ratio": 0.008,      # ⬆️ Plus strict : 0.8% minimum
        "max_area_ratio": 0.35,       # ⬆️ NOUVEAU : exclure zones trop grandes (ciel, murs)
    },
    "texture": {
        "check_edge_density": True,   # ⬆️ NOUVEAU : détecter structures artificielles
        "max_edge_ratio": 0.3,        # ⬆️ NOUVEAU : max 30% de contours (mobilier)
    },
    "anchors": {
        "num_points": 15,
    }
}


def compute_color_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Analyse couleur AMÉLIORÉE avec détection de saturation et gris.
    """
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
    
    # RGB moyen
    mean_rgb = pixels.mean(axis=0).tolist()
    
    # HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    mean_hsv = pixels_hsv.mean(axis=0).tolist()
    
    # Saturation moyenne (important pour distinguer végétation de murs)
    mean_saturation = float(pixels_hsv[:, 1].mean())
    
    # Détection VERT (végétation) - critères stricts
    green_mask = (
        (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) &  # Teinte verte
        (pixels_hsv[:, 1] > 30) &   # Saturation minimum (pas gris)
        (pixels_hsv[:, 2] > 30)     # Luminosité minimum
    )
    green_ratio = green_mask.sum() / len(pixels)
    
    # Détection BRUN (sol/terre) - critères stricts
    brown_mask = (
        (pixels_hsv[:, 0] >= 10) & (pixels_hsv[:, 0] <= 30) &  # Teinte brun/orange
        (pixels_hsv[:, 1] > 25) &   # Saturation minimum
        (pixels_hsv[:, 2] > 20) & (pixels_hsv[:, 2] < 120)  # Luminosité modérée
    )
    brown_ratio = brown_mask.sum() / len(pixels)
    
    # Détection GRIS/BEIGE (murs, structures) - À EXCLURE
    gray_mask = (
        (pixels_hsv[:, 1] < 30) |   # Faible saturation = gris
        (
            (pixels_hsv[:, 0] >= 15) & (pixels_hsv[:, 0] <= 35) &  # Teinte beige
            (pixels_hsv[:, 1] < 40) &  # Saturation faible
            (pixels_hsv[:, 2] > 80)    # Luminosité haute (murs clairs)
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
    """
    Calcule features géométriques pour détecter murs et structures.
    """
    bbox = segment.get("bbox", [0, 0, 1, 1])  # [x, y, w, h] normalisé
    
    # Dimensions en pixels
    bbox_w = bbox[2] * W
    bbox_h = bbox[3] * H
    
    # Aspect ratio : hauteur/largeur (>1 = vertical, <1 = horizontal)
    if bbox_w > 0:
        aspect_ratio = bbox_h / bbox_w
    else:
        aspect_ratio = 0
    
    # Position bbox
    bbox_y_start = bbox[1]
    bbox_y_end = bbox[1] + bbox[3]
    
    return {
        "aspect_ratio": float(aspect_ratio),
        "bbox_y_start": float(bbox_y_start),
        "bbox_y_end": float(bbox_y_end),
        "bbox_width": float(bbox_w),
        "bbox_height": float(bbox_h)
    }


def compute_texture_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Analyse de texture pour détecter structures artificielles (mobilier, fenêtres).
    """
    if mask.sum() == 0:
        return {"edge_ratio": 0.0}
    
    # Créer une sous-image du segment
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Détection de contours (Canny)
    edges = cv2.Canny(img_gray, 50, 150)
    
    # Compter les pixels de contours dans le masque
    edge_pixels = edges[mask].sum() / 255  # Normaliser
    total_pixels = mask.sum()
    
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
    
    return {
        "edge_ratio": float(edge_ratio)
    }


def is_plantable(
    segment: Dict,
    color_features: Dict,
    shape_features: Dict,
    texture_features: Dict,
    config: Dict
) -> Tuple[bool, List[str]]:
    """
    Détermine si un segment est plantable avec critères STRICTS.
    """
    reasons = []
    
    # ===== CRITÈRE 1 : Position (BAS de l'image) =====
    centroid_y = segment.get("centroid", [0, 0])[1]
    bbox_y_start = shape_features["bbox_y_start"]
    
    if centroid_y < config["position"]["min_y_centroid"]:
        reasons.append(f"position_too_high (centroid_y={centroid_y:.2f})")
    
    if bbox_y_start < config["position"]["min_y_bbox"]:
        reasons.append(f"bbox_starts_too_high (y={bbox_y_start:.2f})")
    
    # ===== CRITÈRE 2 : Profondeur (ni trop proche, ni trop loin) =====
    depth_band = segment.get("depth_band")
    mean_depth = segment.get("mean_depth", 0)
    
    if depth_band not in config["depth"]["allowed_bands"]:
        reasons.append(f"depth_band_invalid (band={depth_band})")
    
    if mean_depth > config["depth"]["max_mean_depth"]:
        reasons.append(f"depth_too_close (depth={mean_depth:.2f}, objects)")
    
    if mean_depth < config["depth"]["min_mean_depth"]:
        reasons.append(f"depth_too_far (depth={mean_depth:.2f}, background)")
    
    # ===== CRITÈRE 3 : Couleur (VERT ou BRUN, pas GRIS) =====
    green_ratio = color_features["green_ratio"]
    brown_ratio = color_features["brown_ratio"]
    gray_ratio = color_features["gray_ratio"]
    saturation = color_features["mean_saturation"]
    
    is_green = green_ratio >= config["color"]["min_green_ratio"]
    is_brown = brown_ratio >= config["color"]["min_brown_ratio"]
    
    if not (is_green or is_brown):
        reasons.append(f"color_not_vegetation (green={green_ratio:.2f}, brown={brown_ratio:.2f})")
    
    if gray_ratio > config["color"]["max_gray_ratio"]:
        reasons.append(f"too_much_gray (gray={gray_ratio:.2f}, likely wall/structure)")
    
    if config["color"]["require_saturation"] and saturation < config["color"]["min_saturation"]:
        reasons.append(f"low_saturation (sat={saturation:.1f}, likely gray surface)")
    
    # ===== CRITÈRE 4 : Forme (pas trop vertical = mur, pas trop grand) =====
    area_ratio = segment.get("area_ratio", 0)
    aspect_ratio = shape_features["aspect_ratio"]
    
    if area_ratio < config["shape"]["min_area_ratio"]:
        reasons.append(f"too_small (area={area_ratio:.4f})")
    
    if area_ratio > config["shape"]["max_area_ratio"]:
        reasons.append(f"too_large (area={area_ratio:.2f}, likely wall/sky)")
    
    if aspect_ratio > config["shape"]["max_aspect_ratio"]:
        reasons.append(f"too_vertical (ratio={aspect_ratio:.1f}, likely wall/tree)")
    
    # ===== CRITÈRE 5 : Texture (pas trop de contours = mobilier) =====
    if config["texture"]["check_edge_density"]:
        edge_ratio = texture_features["edge_ratio"]
        if edge_ratio > config["texture"]["max_edge_ratio"]:
            reasons.append(f"high_edge_density (edges={edge_ratio:.2f}, likely furniture)")
    
    is_plantable = len(reasons) == 0
    
    return is_plantable, reasons


def generate_anchor_points(mask: np.ndarray, num_points: int) -> List[List[float]]:
    """Génère des points d'ancrage uniformément répartis."""
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
    print("PLANTABLE ZONE COMPUTATION - VERSION AMÉLIORÉE")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {VISION_JSON}...")
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    print(f"Loading {IMAGE_PATH}...")
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"Total segments: {len(segments)}")
    
    # Analyse each segment
    print("\n🔍 Analyzing segments with STRICT criteria...")
    plantable_segments = []
    plantable_mask_combined = np.zeros((H, W), dtype=bool)
    
    segment_details = []
    
    for seg in segments:
        seg_id = seg["segment_id"]
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        # Compute all features
        color_features = compute_color_features(img, mask)
        shape_features = compute_shape_features(seg, H, W)
        texture_features = compute_texture_features(img, mask)
        
        # Check plantability
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
        else:
            print(f"  ❌ Segment {seg_id}: {reasons[0] if reasons else 'unknown'}")
    
    print(f"\n✅ Plantable segments: {len(plantable_segments)} / {len(segments)}")
    
    # Coverage
    total_pixels = H * W
    plantable_pixels = plantable_mask_combined.sum()
    coverage = float(plantable_pixels) / total_pixels
    
    print(f"✅ Plantable coverage: {coverage:.2%}")
    
    # Anchor points
    print(f"\nGenerating {CONFIG['anchors']['num_points']} anchor points...")
    anchors = generate_anchor_points(plantable_mask_combined, CONFIG["anchors"]["num_points"])
    print(f"✅ Generated {len(anchors)} anchors")
    
    # Encode mask
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_mask_combined.astype(np.uint8)))
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    # Build output
    output = {
        "version": "plantable_zone_v2_strict",
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
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - STRICT FILTERING")
    print("=" * 70)
    print(f"Total segments:       {len(segments)}")
    print(f"Plantable segments:   {len(plantable_segments)} ({len(plantable_segments)/len(segments):.1%})")
    print(f"Plantable coverage:   {coverage:.2%}")
    print(f"Anchor points:        {len(anchors)}")
    
    # Rejection statistics
    print("\n📊 Rejection reasons breakdown:")
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
