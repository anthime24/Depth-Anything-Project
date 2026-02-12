"""
compute_plantable_zone_FINAL.py

VERSION FINALE : Ground - Objects - Existing Vegetation

Pipeline:
A) Ground candidate (sol)
B) Objects on ground (mobilier)
C) Existing vegetation (plantes déjà en place)
D) Plantable = Ground - Objects - Existing vegetation
E) Material filter (herbe libre seulement)
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Dict
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes


VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("PlantableZone_FINAL.json")

CONFIG = {
    # A) Ground
    "ground": {
        "min_y_centroid": 0.55,
        "min_area_ratio": 0.03,
        "min_bbox_width": 0.35,
        "max_depth_std": 0.25,
    },
    
    # B) Objects (mobilier)
    "objects": {
        "min_intersection": 0.60,
        "max_area_ratio": 0.05,
        "compact_threshold": 0.65,
    },
    
    # C) Existing vegetation (NOUVEAU)
    "existing_vegetation": {
        "min_green_ratio": 0.50,        # Très vert (plantes denses)
        "min_saturation": 40,            # Couleur vive
        "min_area_ratio": 0.005,         # > 0.5% (pas trop petit)
        "max_area_ratio": 0.15,          # < 15% (pas toute la pelouse)
        "check_texture": True,           # Texture non-uniforme (feuilles)
        "texture_variance_threshold": 30, # Variance de texture
        "exclude_lawn": True,            # Ne pas exclure la pelouse uniforme
        "lawn_min_area": 0.10,           # Pelouse = grande zone (> 10%)
    },
    
    # D) Lawn (pelouse libre)
    "lawn": {
        "min_green_ratio": 0.30,         # Modérément vert
        "max_green_ratio": 0.70,         # Pas trop vert (évite massifs denses)
        "min_saturation": 20,
        "max_saturation": 60,            # Pas trop saturé (massifs très vifs)
        "texture_uniformity": True,      # Texture uniforme
        "max_texture_variance": 25,      # Faible variance
    },
    
    # E) Morphologie
    "morphology": {
        "fill_holes": True,
        "smooth_iterations": 2,
        "kernel_size": 5,
    },
    
    # F) Anchors
    "anchors": {
        "num_points": 15,
        "min_distance": 0.08,
        "border_margin": 0.05,
    }
}


def compute_texture_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcule la variance de texture dans un segment.
    Texture uniforme (pelouse) : variance faible (~10-20)
    Texture complexe (massif) : variance élevée (~40-60)
    """
    if mask.sum() == 0:
        return 0.0
    
    # Convertir en niveaux de gris
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Extraire les pixels du segment
    pixels = img_gray[mask]
    
    # Variance
    variance = float(np.var(pixels))
    
    return variance


def compute_color_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """Analyse couleur HSV complète."""
    pixels_rgb = img[mask]
    
    if len(pixels_rgb) == 0:
        return {
            "green_ratio": 0.0,
            "mean_saturation": 0.0,
            "mean_value": 0.0,
        }
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    
    # Vert
    green_mask = (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85)
    green_ratio = green_mask.sum() / len(pixels_hsv)
    
    # Saturation et valeur moyennes
    mean_sat = float(pixels_hsv[:, 1].mean())
    mean_val = float(pixels_hsv[:, 2].mean())
    
    return {
        "green_ratio": float(green_ratio),
        "mean_saturation": mean_sat,
        "mean_value": mean_val,
    }


def is_existing_vegetation(
    segment: Dict,
    img: np.ndarray,
    mask: np.ndarray,
    config: Dict
) -> bool:
    """
    Détecte si un segment est de la végétation EXISTANTE (massifs, arbustes).
    
    Critères :
    - Très vert (> 50%)
    - Saturation élevée (> 40)
    - Taille moyenne (0.5% - 15%)
    - Texture complexe (variance > 30)
    - PAS la grande pelouse (area < 10%)
    """
    area_ratio = segment.get("area_ratio", 0)
    
    # Exclure la grande pelouse
    if config["exclude_lawn"] and area_ratio >= config["lawn_min_area"]:
        return False
    
    # Taille
    if area_ratio < config["min_area_ratio"] or area_ratio > config["max_area_ratio"]:
        return False
    
    # Couleur
    color_features = compute_color_features(img, mask)
    
    if color_features["green_ratio"] < config["min_green_ratio"]:
        return False
    
    if color_features["mean_saturation"] < config["min_saturation"]:
        return False
    
    # Texture
    if config["check_texture"]:
        texture_var = compute_texture_variance(img, mask)
        
        if texture_var < config["texture_variance_threshold"]:
            return False  # Trop uniforme = pelouse
    
    return True


def is_lawn(
    img: np.ndarray,
    mask: np.ndarray,
    config: Dict
) -> bool:
    """
    Détecte si un segment est de la PELOUSE LIBRE (plantable).
    
    Critères :
    - Vert modéré (30-70%)
    - Saturation modérée (20-60)
    - Texture UNIFORME (variance < 25)
    """
    color_features = compute_color_features(img, mask)
    
    # Vert modéré
    green_ok = (
        config["min_green_ratio"] <= color_features["green_ratio"] <= config["max_green_ratio"]
    )
    
    # Saturation modérée
    sat_ok = (
        config["min_saturation"] <= color_features["mean_saturation"] <= config["max_saturation"]
    )
    
    # Texture uniforme
    if config["texture_uniformity"]:
        texture_var = compute_texture_variance(img, mask)
        texture_ok = texture_var <= config["max_texture_variance"]
    else:
        texture_ok = True
    
    return green_ok and sat_ok and texture_ok


def compute_intersection(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Ratio d'intersection."""
    if mask1.sum() == 0:
        return 0.0
    return float((mask1 & mask2).sum()) / mask1.sum()


def compute_compactness(mask: np.ndarray, bbox: List[float], H: int, W: int) -> float:
    """Compacité = area_mask / area_bbox."""
    area_mask = mask.sum()
    x, y, w, h = bbox
    bbox_area = (w * W) * (h * H)
    return float(area_mask) / bbox_area if bbox_area > 0 else 0.0


def morphological_processing(mask: np.ndarray, config: Dict) -> np.ndarray:
    """Nettoyage morphologique."""
    kernel = np.ones((config["kernel_size"], config["kernel_size"]), dtype=bool)
    
    if config["fill_holes"]:
        mask = binary_fill_holes(mask)
    
    for _ in range(config["smooth_iterations"]):
        mask = binary_erosion(mask, structure=kernel)
        mask = binary_dilation(mask, structure=kernel)
    
    return mask


def generate_anchors_with_spacing(
    mask: np.ndarray,
    num_points: int,
    min_distance: float,
    border_margin: float
) -> List[Dict]:
    """Génère anchors avec espacement."""
    H, W = mask.shape
    
    margin_x = int(W * border_margin)
    margin_y = int(H * border_margin)
    
    mask_inner = mask.copy()
    mask_inner[:margin_y, :] = False
    mask_inner[-margin_y:, :] = False
    mask_inner[:, :margin_x] = False
    mask_inner[:, -margin_x:] = False
    
    y_coords, x_coords = np.where(mask_inner)
    
    if len(x_coords) == 0:
        return []
    
    min_dist_px = int(min(W, H) * min_distance)
    
    selected = []
    selected_points = []
    
    np.random.seed(42)
    first_idx = np.random.randint(0, len(x_coords))
    selected.append(first_idx)
    selected_points.append((x_coords[first_idx], y_coords[first_idx]))
    
    max_attempts = len(x_coords) * 2
    attempts = 0
    
    while len(selected) < num_points and attempts < max_attempts:
        idx = np.random.randint(0, len(x_coords))
        x, y = x_coords[idx], y_coords[idx]
        
        too_close = any(
            np.sqrt((x - sx)**2 + (y - sy)**2) < min_dist_px
            for sx, sy in selected_points
        )
        
        if not too_close:
            selected.append(idx)
            selected_points.append((x, y))
        
        attempts += 1
    
    return [
        {
            "id": f"p{i+1}",
            "x": round(float(x) / W, 3),
            "y": round(float(y) / H, 3),
            "score": round(float(y) / H, 2)
        }
        for i, (x, y) in enumerate(selected_points)
    ]


def main():
    print("=" * 70)
    print("PLANTABLE ZONE - VERSION FINALE (excluant végétation existante)")
    print("=" * 70)
    
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"✅ Loaded {len(segments)} segments")
    
    # ===== A) Ground Candidate =====
    print("\n" + "=" * 70)
    print("A) GROUND CANDIDATE")
    print("=" * 70)
    
    ground_mask = np.zeros((H, W), dtype=bool)
    ground_segments = []
    
    for seg in segments:
        centroid = seg.get("centroid", [0, 0])
        area_ratio = seg.get("area_ratio", 0)
        bbox = seg.get("bbox", [0, 0, 0, 0])
        depth_std = seg.get("depth_std", 0)
        
        is_ground = (
            centroid[1] >= CONFIG["ground"]["min_y_centroid"] and
            area_ratio >= CONFIG["ground"]["min_area_ratio"] and
            bbox[2] >= CONFIG["ground"]["min_bbox_width"] and
            (depth_std is None or depth_std <= CONFIG["ground"]["max_depth_std"])
        )
        
        if is_ground:
            rle = seg["mask_rle"]
            mask = mask_utils.decode(rle).astype(bool)
            ground_mask |= mask
            ground_segments.append(seg["segment_id"])
            print(f"  ✅ Segment {seg['segment_id']}: ground")
    
    print(f"✅ Ground: {len(ground_segments)} segments → {ground_mask.sum()/(H*W):.1%}")
    
    ground_mask = morphological_processing(ground_mask, CONFIG["morphology"])
    
    # ===== B) Objects on Ground =====
    print("\n" + "=" * 70)
    print("B) OBJECTS ON GROUND")
    print("=" * 70)
    
    objects_mask = np.zeros((H, W), dtype=bool)
    object_segments = []
    
    for seg in segments:
        if seg["segment_id"] in ground_segments:
            continue
        
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        intersection = compute_intersection(mask, ground_mask)
        area_ratio = seg.get("area_ratio", 0)
        bbox = seg.get("bbox", [0, 0, 1, 1])
        compactness = compute_compactness(mask, bbox, H, W)
        
        is_object = (
            intersection >= CONFIG["objects"]["min_intersection"] and
            area_ratio <= CONFIG["objects"]["max_area_ratio"] and
            compactness >= CONFIG["objects"]["compact_threshold"]
        )
        
        if is_object:
            objects_mask |= mask
            object_segments.append(seg["segment_id"])
            print(f"  🪑 Segment {seg['segment_id']}: object")
    
    print(f"✅ Objects: {len(object_segments)} segments → {objects_mask.sum()/(H*W):.1%}")
    
    # ===== C) Existing Vegetation =====
    print("\n" + "=" * 70)
    print("C) EXISTING VEGETATION (massifs, arbustes)")
    print("=" * 70)
    
    vegetation_mask = np.zeros((H, W), dtype=bool)
    vegetation_segments = []
    
    for seg in segments:
        if seg["segment_id"] in ground_segments or seg["segment_id"] in object_segments:
            continue
        
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        # Doit être dans/près du ground
        intersection = compute_intersection(mask, ground_mask)
        
        if intersection < 0.3:  # Au moins 30% dans ground
            continue
        
        if is_existing_vegetation(seg, img, mask, CONFIG["existing_vegetation"]):
            vegetation_mask |= mask
            vegetation_segments.append(seg["segment_id"])
            
            color_feat = compute_color_features(img, mask)
            texture_var = compute_texture_variance(img, mask)
            
            print(f"  🌿 Segment {seg['segment_id']}: existing vegetation "
                  f"(green={color_feat['green_ratio']:.2f}, "
                  f"sat={color_feat['mean_saturation']:.0f}, "
                  f"texture_var={texture_var:.0f})")
    
    print(f"✅ Vegetation: {len(vegetation_segments)} segments → {vegetation_mask.sum()/(H*W):.1%}")
    
    # ===== D) Plantable = Ground - Objects - Vegetation =====
    print("\n" + "=" * 70)
    print("D) PLANTABLE = Ground - Objects - Vegetation")
    print("=" * 70)
    
    plantable_raw = ground_mask & ~objects_mask & ~vegetation_mask
    
    print(f"✅ Plantable (raw): {plantable_raw.sum()/(H*W):.1%}")
    
    # ===== E) Filter for LAWN only =====
    print("\n" + "=" * 70)
    print("E) FILTER FOR LAWN (pelouse libre)")
    print("=" * 70)
    
    # Vérifier que c'est bien de la pelouse uniforme
    if is_lawn(img, plantable_raw, CONFIG["lawn"]):
        plantable_final = plantable_raw
        print(f"✅ Zone validated as lawn")
    else:
        # Filtrer pixel par pixel
        print(f"⚠️  Applying pixel-level lawn filter...")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        lawn_mask = (
            (img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85) &  # Vert
            (img_hsv[:, :, 1] >= CONFIG["lawn"]["min_saturation"]) &
            (img_hsv[:, :, 1] <= CONFIG["lawn"]["max_saturation"])
        )
        
        plantable_final = plantable_raw & lawn_mask
    
    coverage_final = plantable_final.sum() / (H * W)
    
    print(f"✅ Plantable (final): {coverage_final:.1%}")
    
    # ===== F) Anchors =====
    print("\n" + "=" * 70)
    print("F) GENERATE ANCHORS")
    print("=" * 70)
    
    anchors = generate_anchors_with_spacing(
        plantable_final,
        CONFIG["anchors"]["num_points"],
        CONFIG["anchors"]["min_distance"],
        CONFIG["anchors"]["border_margin"]
    )
    
    print(f"✅ Generated {len(anchors)} anchors")
    
    # ===== OUTPUT =====
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_final.astype(np.uint8)))
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    output = {
        "version": "plantable_zone_final_v1",
        "image_id": vision.get("image_id"),
        "image_size": [W, H],
        "config": CONFIG,
        "pipeline_steps": {
            "A_ground": {"segments": ground_segments, "coverage": float(ground_mask.sum()/(H*W))},
            "B_objects": {"segments": object_segments, "coverage": float(objects_mask.sum()/(H*W))},
            "C_vegetation": {"segments": vegetation_segments, "coverage": float(vegetation_mask.sum()/(H*W))},
            "D_plantable_raw": {"coverage": float(plantable_raw.sum()/(H*W))},
            "E_plantable_final": {"coverage": float(coverage_final)}
        },
        "plantable": {
            "coverage": float(coverage_final),
            "total_pixels": int(plantable_final.sum()),
            "mask_rle": plantable_rle,
            "anchors": anchors
        }
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"A) Ground:       {ground_mask.sum()/(H*W):.1%}")
    print(f"B) - Objects:    {objects_mask.sum()/(H*W):.1%}")
    print(f"C) - Vegetation: {vegetation_mask.sum()/(H*W):.1%}")
    print(f"D) = Plantable:  {coverage_final:.1%}")
    print(f"E) Anchors:      {len(anchors)} points")


if __name__ == "__main__":
    main()
