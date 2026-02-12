"""
create_vision_output_for_llm.py

Crée un VisionOutput SIMPLIFIÉ pour le LLM.
Le LLM n'a pas besoin de tous les 90 segments, juste :
- Infos image
- Zones plantables
- Depth bands summary
- Anchors
"""

import json
from pathlib import Path


# Inputs
VISION_JSON = Path("../VisionOutput.json")
PLANTABLE_JSON = Path("../PlantableZone_POC.json")

# Output
OUT_JSON = Path("../VisionOutput_LLM.json")


def main():
    print("=" * 70)
    print("CREATE VISION OUTPUT FOR LLM")
    print("=" * 70)
    
    # Charger les données complètes
    print("\n📂 Loading full vision output...")
    with open(VISION_JSON, "r", encoding="utf-8") as f:
        vision_full = json.load(f)
    
    print("📂 Loading plantable zone...")
    with open(PLANTABLE_JSON, "r", encoding="utf-8") as f:
        plantable = json.load(f)
    
    segments_full = vision_full.get("segments", [])
    
    # Stats de profondeur
    print("\n📊 Computing depth statistics...")
    depth_stats = {
        "front": 0,
        "mid": 0,
        "back": 0
    }
    
    for seg in segments_full:
        band = seg.get("depth_band")
        if band in depth_stats:
            depth_stats[band] += 1
    
    # Segments plantables (info minimale)
    plantable_segment_ids = set(plantable["plantable"]["segment_ids"])
    
    plantable_segments_minimal = []
    for seg in segments_full:
        if seg["segment_id"] in plantable_segment_ids:
            # Garder seulement l'essentiel
            plantable_segments_minimal.append({
                "segment_id": seg["segment_id"],
                "centroid": seg["centroid"],
                "area_ratio": seg["area_ratio"],
                "depth_band": seg.get("depth_band"),
                "mean_depth": seg.get("mean_depth")
            })
    
    # ===== OUTPUT POUR LE LLM =====
    output_llm = {
        "version": "vision_output_llm_v1",
        
        # Métadonnées de base
        "image": {
            "image_id": vision_full.get("image_id"),
            "size": vision_full.get("image_size"),
        },
        
        # Stats de profondeur (aperçu global)
        "depth_summary": {
            "bands_count": depth_stats,
            "total_segments": len(segments_full),
            "near_is_one": vision_full.get("depth_meta", {}).get("near_is_one", True)
        },
        
        # Zone plantable (info clé)
        "plantable_zone": {
            "coverage": plantable["plantable"]["coverage"],
            "segment_count": len(plantable_segment_ids),
            
            # Segments plantables (info minimale)
            "segments": plantable_segments_minimal,
            
            # ANCHORS (crucial pour le LLM)
            "anchors": plantable["plantable"]["anchors"],
            
            # Masque (optionnel, pour backend)
            "mask_rle": plantable["plantable"].get("mask_rle")
        },
        
        # Config utilisée (transparence)
        "generation_config": plantable.get("config", {})
    }
    
    # Sauvegarder
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_llm, f, indent=2)
    
    print(f"\n✅ Saved {OUT_JSON}")
    
    # Summary
    print("\n" + "=" * 70)
    print("LLM OUTPUT SUMMARY")
    print("=" * 70)
    print(f"Total segments (original):     {len(segments_full)}")
    print(f"Plantable segments:            {len(plantable_segments_minimal)}")
    print(f"Coverage:                      {plantable['plantable']['coverage']:.1%}")
    print(f"Anchors:                       {len(plantable['plantable']['anchors'])}")
    print(f"\nDepth distribution:")
    print(f"  - Front: {depth_stats['front']} segments")
    print(f"  - Mid:   {depth_stats['mid']} segments")
    print(f"  - Back:  {depth_stats['back']} segments")
    
    print("\n✅ VisionOutput ready for LLM!")
    print(f"\nWhat the LLM gets:")
    print(f"  ✅ Image metadata")
    print(f"  ✅ Depth summary (not 90 full segments)")
    print(f"  ✅ Plantable zone ({len(plantable_segments_minimal)} segments)")
    print(f"  ✅ Anchors ({len(plantable['plantable']['anchors'])} safe points)")
    print(f"  ✅ Mask RLE (for precise operations)")


if __name__ == "__main__":
    main()
