"""
compute_plantable_zone_ROBUST.py

Stratégie ROBUSTE : Ground mask → Soustraction des objets → Filtre matière

Pipeline:
A) Construire ground_candidate (grande surface au bas)
B) Détecter objects_on_ground (chaises, pots)
C) Soustraire : plantable = ground - objects
D) Filtrer par matière (vert/brun, pas gris/dalle)
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes


# ============ Configuration ROBUSTE ============
VISION_JSON = Path("VisionOutput.json")
IMAGE_PATH = Path("Inputs/IMG_5177-535x356_preprocessed.jpg")
OUT_JSON = Path("PlantableZone_ROBUST.json")

CONFIG = {
    # ===== ÉTAPE A : Ground Candidate =====
    "ground": {
        "min_y_centroid": 0.55,      # Surface au bas
        "min_area_ratio": 0.03,      # Éviter micro-segments
        "min_bbox_width": 0.35,      # Surface étendue (largeur > 35%)
        "max_depth_std": 0.25,       # Profondeur stable (optionnel)
    },
    
    # ===== ÉTAPE B : Objects on Ground =====
    "objects": {
        "min_intersection": 0.60,    # Objet dans le sol (60% overlap)
        "max_area_ratio": 0.05,      # Petits/moyens objets (< 5%)
        "max_aspect_ratio": 2.5,     # Pas trop vertical
        "compact_threshold": 0.65,   # Compacité (area/bbox_area)
    },
    
    # ===== ÉTAPE C : Morphologie =====
    "morphology": {
        "fill_holes": True,          # Remplir trous dans ground
        "smooth_iterations": 2,      # Lissage
        "kernel_size": 5,            # Taille kernel
    },
    
    # ===== ÉTAPE D : Filtre matière =====
    "material": {
        "min_green_ratio": 0.25,     # Herbe
        "min_brown_ratio": 0.15,     # Terre
        "max_gray_ratio": 0.70,      # Rejeter dalles/terrasse
        "min_saturation": 20,        # Couleur vivante
        "use_excess_green": True,    # ExG = 2G - R - B
        "exg_threshold": 10,         # Seuil ExG
    },
    
    # ===== ÉTAPE E : Anchors =====
    "anchors": {
        "num_points": 15,
        "min_distance": 0.08,
        "border_margin": 0.05,
    }
}


def compute_intersection(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calcule le ratio d'intersection de mask1 dans mask2."""
    if mask1.sum() == 0:
        return 0.0
    intersection = (mask1 & mask2).sum()
    return float(intersection) / mask1.sum()


def compute_compactness(mask: np.ndarray, bbox: List[float], H: int, W: int) -> float:
    """
    Compacité = area_mask / area_bbox.
    Objets compacts (chaises) ont ~0.5-0.8
    Sol étendu a ~0.3-0.5
    """
    area_mask = mask.sum()
    
    # Bbox en pixels
    x, y, w, h = bbox
    bbox_area = (w * W) * (h * H)
    
    if bbox_area == 0:
        return 0.0
    
    return float(area_mask) / bbox_area


def compute_excess_green(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Excess Green Index (ExG) = 2*G - R - B
    Positif → végétation
    Négatif → autre surface
    """
    pixels = img[mask].astype(float)
    
    if len(pixels) == 0:
        return 0.0
    
    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]
    
    exg = 2 * G - R - B
    mean_exg = exg.mean()
    
    return float(mean_exg)


def compute_color_features(img: np.ndarray, mask: np.ndarray) -> Dict:
    """Analyse couleur HSV + ExG."""
    pixels_rgb = img[mask]
    
    if len(pixels_rgb) == 0:
        return {
            "green_ratio": 0.0,
            "brown_ratio": 0.0,
            "gray_ratio": 0.0,
            "mean_saturation": 0.0,
            "excess_green": 0.0
        }
    
    # HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    
    # Vert
    green_mask = (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85)
    green_ratio = green_mask.sum() / len(pixels_hsv)
    
    # Brun
    brown_mask = (pixels_hsv[:, 0] >= 10) & (pixels_hsv[:, 0] <= 30)
    brown_ratio = brown_mask.sum() / len(pixels_hsv)
    
    # Gris (faible saturation)
    gray_mask = pixels_hsv[:, 1] < 30
    gray_ratio = gray_mask.sum() / len(pixels_hsv)
    
    # Saturation moyenne
    mean_sat = float(pixels_hsv[:, 1].mean())
    
    # Excess Green
    exg = compute_excess_green(img, mask)
    
    return {
        "green_ratio": float(green_ratio),
        "brown_ratio": float(brown_ratio),
        "gray_ratio": float(gray_ratio),
        "mean_saturation": mean_sat,
        "excess_green": exg
    }


def morphological_processing(mask: np.ndarray, config: Dict) -> np.ndarray:
    """
    Nettoyage morphologique :
    - Remplir trous
    - Lisser contours
    """
    kernel = np.ones((config["kernel_size"], config["kernel_size"]), dtype=bool)
    
    # Remplir trous
    if config["fill_holes"]:
        mask = binary_fill_holes(mask)
    
    # Lissage : erosion puis dilatation (opening)
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
    """Génère anchors avec distance min et marge."""
    H, W = mask.shape
    
    # Marge
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
    
    # Sélection avec distance min
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
        
        too_close = False
        for (sx, sy) in selected_points:
            if np.sqrt((x - sx)**2 + (y - sy)**2) < min_dist_px:
                too_close = True
                break
        
        if not too_close:
            selected.append(idx)
            selected_points.append((x, y))
        
        attempts += 1
    
    anchors = []
    for i, (x_px, y_px) in enumerate(selected_points):
        anchors.append({
            "id": f"p{i+1}",
            "x": round(float(x_px) / W, 3),
            "y": round(float(y_px) / H, 3),
            "score": round(float(y_px) / H, 2)
        })
    
    return anchors


def main():
    print("=" * 70)
    print("PLANTABLE ZONE - STRATÉGIE ROBUSTE (Ground - Objects)")
    print("=" * 70)
    
    # Charger données
    print(f"\n📂 Loading {VISION_JSON}...")
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision = json.load(f)
    
    print(f"📂 Loading {IMAGE_PATH}...")
    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    
    segments = vision.get("segments", [])
    print(f"✅ Loaded {len(segments)} segments")
    
    # ========== ÉTAPE A : Ground Candidate ==========
    print("\n" + "=" * 70)
    print("ÉTAPE A : CONSTRUIRE GROUND CANDIDATE")
    print("=" * 70)
    
    ground_mask = np.zeros((H, W), dtype=bool)
    ground_segments = []
    
    for seg in segments:
        centroid = seg.get("centroid", [0, 0])
        area_ratio = seg.get("area_ratio", 0)
        bbox = seg.get("bbox", [0, 0, 0, 0])
        depth_std = seg.get("depth_std", 0)
        
        # Critères ground
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
            
            print(f"  ✅ Segment {seg['segment_id']}: ground candidate "
                  f"(y={centroid[1]:.2f}, area={area_ratio:.3f}, w={bbox[2]:.2f})")
    
    print(f"\n✅ Ground candidates: {len(ground_segments)} segments")
    print(f"   Coverage: {ground_mask.sum() / (H*W):.1%}")
    
    # Nettoyage morphologique
    print(f"\n🧹 Morphological processing...")
    ground_mask = morphological_processing(ground_mask, CONFIG["morphology"])
    print(f"   Coverage after cleanup: {ground_mask.sum() / (H*W):.1%}")
    
    # ========== ÉTAPE B : Objects on Ground ==========
    print("\n" + "=" * 70)
    print("ÉTAPE B : DÉTECTER OBJECTS ON GROUND")
    print("=" * 70)
    
    objects_mask = np.zeros((H, W), dtype=bool)
    object_segments = []
    
    for seg in segments:
        seg_id = seg["segment_id"]
        
        # Skip si déjà dans ground
        if seg_id in ground_segments:
            continue
        
        rle = seg["mask_rle"]
        mask = mask_utils.decode(rle).astype(bool)
        
        # Intersection avec ground
        intersection_ratio = compute_intersection(mask, ground_mask)
        
        # Critères objet
        area_ratio = seg.get("area_ratio", 0)
        bbox = seg.get("bbox", [0, 0, 1, 1])
        
        # Aspect ratio
        aspect_ratio = bbox[3] / bbox[2] if bbox[2] > 0 else 0
        
        # Compacité
        compactness = compute_compactness(mask, bbox, H, W)
        
        is_object = (
            intersection_ratio >= CONFIG["objects"]["min_intersection"] and
            area_ratio <= CONFIG["objects"]["max_area_ratio"] and
            aspect_ratio <= CONFIG["objects"]["max_aspect_ratio"] and
            compactness >= CONFIG["objects"]["compact_threshold"]
        )
        
        if is_object:
            objects_mask |= mask
            object_segments.append(seg_id)
            
            print(f"  🪑 Segment {seg_id}: object on ground "
                  f"(intersection={intersection_ratio:.2f}, "
                  f"area={area_ratio:.3f}, compact={compactness:.2f})")
    
    print(f"\n✅ Objects detected: {len(object_segments)} segments")
    print(f"   Coverage: {objects_mask.sum() / (H*W):.1%}")
    
    # ========== ÉTAPE C : Plantable = Ground - Objects ==========
    print("\n" + "=" * 70)
    print("ÉTAPE C : SOUSTRAIRE OBJECTS")
    print("=" * 70)
    
    plantable_raw = ground_mask & ~objects_mask
    
    print(f"✅ Plantable (before material filter): {plantable_raw.sum() / (H*W):.1%}")
    
    # ========== ÉTAPE D : Filtre Matière ==========
    print("\n" + "=" * 70)
    print("ÉTAPE D : FILTRE MATIÈRE (vert/brun, pas gris)")
    print("=" * 70)
    
    # Analyser couleur de la zone plantable
    color_features = compute_color_features(img, plantable_raw)
    
    print(f"\nColor analysis on plantable area:")
    print(f"  - Green ratio:  {color_features['green_ratio']:.2f}")
    print(f"  - Brown ratio:  {color_features['brown_ratio']:.2f}")
    print(f"  - Gray ratio:   {color_features['gray_ratio']:.2f}")
    print(f"  - Saturation:   {color_features['mean_saturation']:.1f}")
    print(f"  - Excess Green: {color_features['excess_green']:.1f}")
    
    # Filtre pixel-wise
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Masque de végétation/sol
    vegetation_mask = (
        # Vert
        ((img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85)) |
        # Brun
        ((img_hsv[:, :, 0] >= 10) & (img_hsv[:, :, 0] <= 30))
    )
    
    # Saturation minimum
    saturation_mask = img_hsv[:, :, 1] >= CONFIG["material"]["min_saturation"]
    
    # Pas trop gris
    not_gray_mask = img_hsv[:, :, 1] >= 30
    
    # Combiner
    material_mask = vegetation_mask & saturation_mask & not_gray_mask
    
    # Appliquer à plantable
    plantable_final = plantable_raw & material_mask
    
    coverage_final = plantable_final.sum() / (H * W)
    
    print(f"\n✅ Plantable (after material filter): {coverage_final:.1%}")
    
    # ========== ÉTAPE E : Anchors ==========
    print("\n" + "=" * 70)
    print("ÉTAPE E : GÉNÉRER ANCHORS")
    print("=" * 70)
    
    anchors = generate_anchors_with_spacing(
        plantable_final,
        CONFIG["anchors"]["num_points"],
        CONFIG["anchors"]["min_distance"],
        CONFIG["anchors"]["border_margin"]
    )
    
    print(f"✅ Generated {len(anchors)} anchors")
    
    # ========== OUTPUT ==========
    plantable_rle = mask_utils.encode(np.asfortranarray(plantable_final.astype(np.uint8)))
    plantable_rle["counts"] = plantable_rle["counts"].decode("utf-8")
    
    output = {
        "version": "plantable_zone_robust_v1",
        "image_id": vision.get("image_id"),
        "image_size": [W, H],
        
        "config": CONFIG,
        
        "pipeline_steps": {
            "A_ground_segments": ground_segments,
            "B_object_segments": object_segments,
            "C_ground_coverage": float(ground_mask.sum() / (H*W)),
            "D_objects_coverage": float(objects_mask.sum() / (H*W)),
            "E_plantable_raw_coverage": float(plantable_raw.sum() / (H*W)),
            "F_plantable_final_coverage": float(coverage_final)
        },
        
        "plantable": {
            "coverage": float(coverage_final),
            "total_pixels": int(plantable_final.sum()),
            "mask_rle": plantable_rle,
            "anchors": anchors,
            "color_features": color_features
        }
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("ROBUST PIPELINE SUMMARY")
    print("=" * 70)
    print(f"A) Ground candidates:  {len(ground_segments)} segments → {ground_mask.sum()/(H*W):.1%}")
    print(f"B) Objects detected:   {len(object_segments)} segments → {objects_mask.sum()/(H*W):.1%}")
    print(f"C) Ground - Objects:   {plantable_raw.sum()/(H*W):.1%}")
    print(f"D) Material filter:    {coverage_final:.1%}")
    print(f"E) Anchors:            {len(anchors)} points")
    print(f"\n✅ Final plantable coverage: {coverage_final:.1%}")


if __name__ == "__main__":
    main()
