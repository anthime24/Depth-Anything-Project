"""
migrate_vision_output.py

Convertit l'ancien format VisionOutput.json (avec sam_output)
vers le nouveau format (structure plate améliorée).

Usage:
    python migrate_vision_output.py
    
Cela va:
1. Lire VisionOutput.json (ancien format)
2. Le convertir au nouveau format
3. Sauvegarder une backup: VisionOutput_old.json
4. Écraser VisionOutput.json avec la nouvelle structure
"""

import json
from pathlib import Path
import shutil


def migrate_vision_output(input_path: Path, output_path: Path, backup_path: Path = None):
    """
    Migre l'ancien format VisionOutput.json vers le nouveau.
    
    Ancien format:
    {
      "version": "vision_output_v1",
      "image_id": "...",
      "image_size": [W, H],
      "near_is_one": true,
      "inputs": {...},
      "sam_output": {
        "image_size": [W, H],
        "segments_count": N,
        "segments": [...]
      }
    }
    
    Nouveau format:
    {
      "version": "vision_segments_v1",
      "image_id": "...",
      "image_size": [W, H],
      "preprocess": {...},
      "depth_meta": {...},
      "sam_meta": {...},
      "segments": [...]
    }
    """
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    
    # Déterminer si c'est l'ancien ou le nouveau format
    if "segments" in old_data:
        print("✅ Already in new format, no migration needed!")
        return False
    
    if "sam_output" not in old_data:
        raise ValueError("Invalid VisionOutput.json: missing both 'segments' and 'sam_output'")
    
    print("🔄 Converting to new format...")
    
    # Créer backup
    if backup_path:
        print(f"📦 Creating backup: {backup_path}")
        shutil.copy(input_path, backup_path)
    
    # Extraire les données
    sam_output = old_data.get("sam_output", {})
    segments = sam_output.get("segments", [])
    inputs = old_data.get("inputs", {})
    
    # Construire nouveau format
    new_data = {
        "version": "vision_segments_v1",
        "image_id": old_data.get("image_id"),
        "image_size": old_data.get("image_size"),
        
        # Preprocess (vide si pas disponible, sera rempli par fuse_sam_depth_improved.py)
        "preprocess": {},
        
        # Depth metadata
        "depth_meta": {
            "model": "LiheYoung/depth_anything_vitl14",
            "near_is_one": old_data.get("near_is_one", True),
            "depth_range": [0.0, 1.0],
            "normalized": True,
            "depth_file": inputs.get("depth_file", "")
        },
        
        # SAM metadata
        "sam_meta": {
            "sam_file": inputs.get("sam_file", ""),
            "segments_count": sam_output.get("segments_count", len(segments))
        },
        
        # Segments (structure plate)
        "segments": segments
    }
    
    # Sauvegarder
    print(f"💾 Saving new format to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)
    
    print("✅ Migration completed!")
    print(f"   - Version: {old_data.get('version')} → {new_data['version']}")
    print(f"   - Segments: {len(segments)}")
    print(f"   - Structure: nested (sam_output) → flat (segments)")
    
    return True


def main():
    print("=" * 70)
    print("MIGRATION VISIONOUTPUT.JSON")
    print("=" * 70)
    
    input_path = Path("VisionOutput.json")
    output_path = Path("VisionOutput.json")
    backup_path = Path("VisionOutput_old.json")
    
    if not input_path.exists():
        print(f"❌ Error: {input_path} not found!")
        print("\nPlease run fuse_sam_depth.py first to generate VisionOutput.json")
        return 1
    
    try:
        migrated = migrate_vision_output(input_path, output_path, backup_path)
        
        if migrated:
            print("\n" + "=" * 70)
            print("NEXT STEPS")
            print("=" * 70)
            print("1. Your old file is backed up as: VisionOutput_old.json")
            print("2. VisionOutput.json is now in the new format")
            print("3. You can now run:")
            print("   - python visualize_vision_output.py")
            print("   - python compute_plantable_zone.py")
            print("\n⚠️  For best results, regenerate VisionOutput.json with:")
            print("   python fuse_sam_depth_improved.py")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
