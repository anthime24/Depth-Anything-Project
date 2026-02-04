import json
import numpy as np
from pycocotools import mask as mask_utils
from pathlib import Path

# Paths (adapte si besoin)
DEPTH_NPY = Path("Outputs/IMG_5177-535x356_depth.npy")
DEPTH_JSON = Path("Outputs/IMG_5177-535x356_depth.json")
SAM_JSON = Path("Inputs/IMG_5177-535x356_preprocessed_sam_output.json")
PREPROCESS_JSON = Path("Inputs/IMG_5177-535x356_preprocessed.json")

OUT_JSON = Path("VisionOutput.json")

print("Loading depth map...")
depth = np.load(DEPTH_NPY)

print("Loading depth metadata...")
with open(DEPTH_JSON, "r", encoding="utf-8") as f:
    depth_meta = json.load(f)

print("Loading SAM output...")
with open(SAM_JSON, "r", encoding="utf-8") as f:
    sam_data = json.load(f)

print("Loading preprocess metadata...")
preprocess_meta = {}
if PREPROCESS_JSON.exists():
    with open(PREPROCESS_JSON, "r", encoding="utf-8") as f:
        preprocess_meta = json.load(f)
else:
    print(f"⚠️  Preprocess metadata not found: {PREPROCESS_JSON}")

print("Depth shape:", depth.shape)
print("SAM segments:", len(sam_data["sam_output"]["segments"]))

# Safety checks (alignement pixel-à-pixel)
H, W = depth.shape

sam_size = sam_data["sam_output"].get("image_size", None)  # [W,H]
if sam_size and isinstance(sam_size, list) and len(sam_size) == 2:
    if sam_size != [W, H]:
        raise ValueError(f"SAM image_size {sam_size} != depth size {[W,H]}")

depth_size = depth_meta.get("image_size", None)  # [W,H]
if depth_size and isinstance(depth_size, list) and len(depth_size) == 2:
    if depth_size != [W, H]:
        raise ValueError(f"Depth image_size {depth_size} != depth npy size {[W,H]}")

near_is_one = bool(depth_meta.get("near_is_one", True))

def depth_band(x: float):
    """Classifie la profondeur moyenne en 3 bandes"""
    if x >= 0.66:
        return "front"
    if x >= 0.33:
        return "mid"
    return "back"

print("\nComputing depth statistics for all segments...")

segments_out = []
for seg in sam_data["sam_output"]["segments"]:
    seg_id = seg["segment_id"]
    rle = seg["mask_rle"]

    # Decode RLE -> mask binaire (H,W)
    mask = mask_utils.decode(rle).astype(bool)

    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} != depth shape {depth.shape} (segment {seg_id})")

    vals = depth[mask]

    if vals.size == 0:
        mean_depth = None
        depth_std = None
        band = None
    else:
        mean_depth = float(vals.mean())
        depth_std = float(vals.std())
        band = depth_band(mean_depth)

    # On conserve tous les champs SAM existants, et on ajoute profondeur
    seg_enriched = dict(seg)
    seg_enriched["mean_depth"] = mean_depth
    seg_enriched["depth_std"] = depth_std
    seg_enriched["depth_band"] = band

    segments_out.append(seg_enriched)

# ============ Structure complète recommandée ============
vision_output = {
    "version": "vision_segments_v1",
    
    # Identité de l'image
    "image_id": preprocess_meta.get("image_id") or sam_data.get("image_id") or depth_meta.get("image_id"),
    "image_size": [W, H],  # [W, H]
    
    # Métadonnées de prétraitement (source de vérité)
    "preprocess": preprocess_meta,
    
    # Métadonnées de profondeur
    "depth_meta": {
        "model": depth_meta.get("model", "LiheYoung/depth_anything_vitl14"),
        "near_is_one": near_is_one,
        "depth_range": depth_meta.get("depth_range", [0.0, 1.0]),
        "normalized": depth_meta.get("normalized", True),
        "depth_file": str(DEPTH_NPY).replace("\\", "/")
    },
    
    # Métadonnées SAM
    "sam_meta": {
        "sam_file": str(SAM_JSON).replace("\\", "/"),
        "segments_count": len(segments_out)
    },
    
    # Les segments enrichis
    "segments": segments_out
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(vision_output, f, indent=2)

print(f"\n✅ Saved {OUT_JSON} with {len(segments_out)} segments.")
print("\nOutput structure:")
print(f"  - version: {vision_output['version']}")
print(f"  - image_id: {vision_output['image_id']}")
print(f"  - image_size: {vision_output['image_size']}")
print(f"  - segments_count: {vision_output['sam_meta']['segments_count']}")
print(f"  - depth convention: near_is_one={vision_output['depth_meta']['near_is_one']}")
print("\nSample enriched segment:", {
    k: vision_output["segments"][0][k] 
    for k in ["segment_id", "mean_depth", "depth_std", "depth_band"]
})
