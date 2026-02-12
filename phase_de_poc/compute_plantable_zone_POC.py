"""
compute_plantable_zone_POC.py

Version POC SIMPLE et ROBUSTE selon les spécifications :
- Filtres simples : position + profondeur + couleur
- Génération d'anchors avec distance minimum
- Output minimal pour le LLM
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Dict
from scipy.ndimage import binary_erosion


# ============ Configuration POC ============
VISION_JSON = Path("../VisionOutput.json")
IMAGE_PATH = Path("../Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("../PlantableZone_POC.json")

# Seuils simples et efficaces
CONFIG = {
    "position": {
        "min_y_centroid": 0.55,  # Sol en bas de l'image
    },
    "depth": {
        "allowed_bands": ["front", "mid"],  # Sol proche ou moyen
    },
    "color": {
        "min_green_ratio": 0.30,   # Végétation
        "min_brown_ratio": 0.20,   # Terre/sol
    },
    "anchors": {
        "num_points": 15,           # 10-20 points
        "min_distance": 0.08,       # Distance min entre anchors (8% de l'image)
        "border_margin": 0.05,      # Marge aux bords (5%)
    },
    "morphology": {
        "cleanup": True,            # Nettoyage morphologique
        "erosion_size": 3,          # Taille du kernel d'érosion
    }
}


def compute_color_ratios(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Calcul SIMPLE des ratios vert/brun en HSV.
    """
    pixels_rgb = img[mask]
    
    if len(pixels_rgb) == 0:
        return {"green_ratio": 0.0, "brown_ratio": 0.0}
    
    # Conversion HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    
    # VERT : H dans [35-85] (teinte verte)
    green_mask = (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85)
    green_ratio = green_mask.sum() / len(pixels_hsv)
    
    # BRUN : H dans [10-30] (teinte orange/brun)
    brown_mask = (pixels_hsv[:, 0] >= 10) & (pixels_hsv[:, 0] <= 30)
    brown_ratio = brown_mask.sum() / len(pixels_hsv)
    
    return {
        "green_ratio": float(green_ratio),
        "brown_ratio": float(brown_ratio)
    }


def is_plantable_simple(segment: Dict, color_ratios: Dict, config: Dict) -> bool:
    """
    Filtre SIMPLE en 3 étapes : A) position, B) profondeur, C) couleur.
    """
    # A) Position : centroid en bas
    centroid = segment.get("centroid", [0, 0])
    if centroid[1] < config["position"]["min_y_centroid"]:
        return False
    
    # B) Profondeur : front ou mid
    depth_band = segment.get("depth_band")
    if depth_band not in config["depth"]["allowed_bands"]:
        return False
    
    # C) Couleur : vert OU brun
    green_ratio = color_ratios["green_ratio"]
    brown_ratio = color_ratios["brown_ratio"]
    
    is_green = green_ratio >= config["color"]["min_green_ratio"]
    is_brown = brown_ratio >= config["color"]["min_brown_ratio"]
    
    return is_green or is_brown


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Nettoyage morphologique : érosion puis dilatation (opening).
    Supprime les petits artefacts.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    
    # Erosion (supprime les petits pixels isolés)
    eroded = binary_erosion(mask, structure=kernel, iterations=1)
    
    # Dilatation (restaure la taille)
    from scipy.ndimage import binary_dilation
    cleaned = binary_dilation(eroded, structure=kernel, iterations=1)
    
    return cleaned


def generate_anchors_with_spacing(
    mask: np.ndarray,
    num_points: int,
    min_distance: float,
    border_margin: float
) -> List[Dict]:
    """
    Génère des anchor points avec :
    - Distance minimum entre points
    - Marge aux bords
    - Score basé sur la profondeur locale
    
    Returns:
        List[{"id": "p1", "x": 0.5, "y": 0.7, "score": 0.9}]
    """
    H, W = mask.shape
    
    # Créer une marge aux bords
    margin_pixels_x = int(W * border_margin)
    margin_pixels_y = int(H * border_margin)
    
    # Masque avec marge
    mask_inner = mask.copy()
    mask_inner[:margin_pixels_y, :] = False    # Top
    mask_inner[-margin_pixels_y:, :] = False   # Bottom
    mask_inner[:, :margin_pixels_x] = False    # Left
    mask_inner[:, -margin_pixels_x:] = False   # Right
    
    # Points candidats
    y_coords, x_coords = np.where(mask_inner)
    
    if len(x_coords) == 0:
        return []
    
    # Sélection avec distance minimum (algorithme glouton simple)
    min_dist_pixels = int(min(W, H) * min_distance)
    
    selected_indices = []
    selected_points = []
    
    # Premier point au hasard
    np.random.seed(42)
    first_idx = np.random.randint(0, len(x_coords))
    selected_indices.append(first_idx)
    selected_points.append((x_coords[first_idx], y_coords[first_idx]))
    
    # Sélectionner les points suivants avec contrainte de distance
    max_attempts = len(x_coords)
    attempts = 0
    
    while len(selected_indices) < num_points and attempts < max_attempts:
        # Prendre un point candidat au hasard
        candidate_idx = np.random.randint(0, len(x_coords))
        candidate_x = x_coords[candidate_idx]
        candidate_y = y_coords[candidate_idx]
        
        # Vérifier la distance avec tous les points déjà sélectionnés
        too_close = False
        for (sx, sy) in selected_points:
            dist = np.sqrt((candidate_x - sx)**2 + (candidate_y - sy)**2)
            if dist < min_dist_pixels:
                too_close = True
                break
        
        if not too_close:
            selected_indices.append(candidate_idx)
            selected_points.append((candidate_x, candidate_y))
        
        attempts += 1
    
    # Construire les anchors
    anchors = []
    for i, (x_px, y_px) in enumerate(selected_points):
        x_norm = float(x_px) / W
        y_norm = float(y_px) / H
        
        # Score simple : position Y (plus bas = meilleur)
        score = float(y_norm)  # 0.5 à 1.0 pour le bas de l'image
        
        anchors.append({
            "id": f"p{i+1}",
            "x": round(x_norm, 3),
            "y": round(y_norm, 3),
            "score": round(score, 2)
        })
    
    return anchors


def main():
    print("=" * 70)
    print("PLANTABLE ZONE - VERSION POC SIMPLE")
    print("=" * 70)
    
    # ===== 1. Charger les données =====
    print(f"\n📂 Loading {VISION_JSON}...")
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    print(f"📂 Loading {IMAGE_PATH}...")
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"✅ Loaded {len(segments)} segments")
    
    # ===== 2. Filtrer les segments plantables =====
    print(f"\n🔍 Filtering plantable segments (simple filters)...")
    
    plantable_segments = []
    plantable_mask_combined = np.zeros((H, W), dtype=bool)
    
    for seg in segments:
        seg_id = seg["segment_id"]
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        # Calculer couleur
        color_ratios = compute_color_ratios(img, mask)
        
        # Test simple
        if is_plantable_simple(seg, color_ratios, CONFIG):
            plantable_segments.append(seg_id)
            plantable_mask_combined |= mask
            
            print(f"  ✅ Segment {seg_id}: plantable "
                  f"(green={color_ratios['green_ratio']:.2f}, "
                  f"brown={color_ratios['brown_ratio']:.2f}, "
                  f"depth={seg.get('depth_band')})")
    
    print(f"\n✅ Found {len(plantable_segments)} plantable segments")
    
    # ===== 3. Nettoyage morphologique =====
    if CONFIG["morphology"]["cleanup"]:
        print(f"\n🧹 Morphological cleanup...")
        pixels_before = plantable_mask_combined.sum()
        plantable_mask_combined = morphological_cleanup(
            plantable_mask_combined,
            CONFIG["morphology"]["erosion_size"]
        )
        pixels_after = plantable_mask_combined.sum()
        print(f"   Removed {pixels_before - pixels_after} noisy pixels")
    
    # ===== 4. Coverage =====
    total_pixels = H * W
    plantable_pixels = int(plantable_mask_combined.sum())
    coverage = float(plantable_pixels) / total_pixels
    
    print(f"\n📊 Coverage: {coverage:.1%} ({plantable_pixels}/{total_pixels} pixels)")
    
    # ===== 5. Générer les anchors =====
    print(f"\n📍 Generating anchors (target: {CONFIG['anchors']['num_points']})...")
    anchors = generate_anchors_with_spacing(
        plantable_mask_combined,
        CONFIG["anchors"]["num_points"],
        CONFIG["anchors"]["min_distance"],
        CONFIG["anchors"]["border_margin"]
    )
    
    print(f"✅ Generated {len(anchors)} anchors with spacing")
    
    # ===== 6. Encoder le masque (optionnel) =====
    print(f"\n💾 Encoding plantable mask...")
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_mask_combined.astype(np.uint8)))
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    # ===== 7. Output SIMPLE pour le LLM =====
    output = {
        "version": "plantable_zone_poc_v1",
        "image_id": vision.get("image_id"),
        "image_size": [W, H],
        
        # Configuration utilisée
        "config": CONFIG,
        
        # RÉSULTATS
        "plantable": {
            "coverage": round(coverage, 4),
            "segment_ids": plantable_segments,
            "total_pixels": plantable_pixels,
            
            # Masque (optionnel mais utile)
            "mask_rle": plantable_rle,
            
            # ANCHORS (crucial pour le LLM)
            "anchors": anchors
        }
    }
    
    # Sauvegarder
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("POC SUMMARY - READY FOR LLM")
    print("=" * 70)
    print(f"Plantable segments:   {len(plantable_segments)}/{len(segments)}")
    print(f"Coverage:             {coverage:.1%}")
    print(f"Anchors generated:    {len(anchors)}")
    print(f"\nOutput structure:")
    print(f"  - plantable.coverage")
    print(f"  - plantable.mask_rle (optional)")
    print(f"  - plantable.anchors ({len(anchors)} points)")
    
    # Afficher quelques anchors
    if anchors:
        print(f"\nSample anchors:")
        for anchor in anchors[:3]:
            print(f"  - {anchor['id']}: x={anchor['x']}, y={anchor['y']}, score={anchor['score']}")
    
    print("\n✅ POC ready for LLM integration!")


if __name__ == "__main__":
    main()
