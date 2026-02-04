"""
test_pipeline.py

Script de test end-to-end pour valider le pipeline complet.
"""

import json
from pathlib import Path
import sys


def check_file_exists(path: Path, description: str):
    """Vérifie qu'un fichier existe"""
    if not path.exists():
        print(f"❌ ERREUR : {description} introuvable : {path}")
        return False
    print(f"✅ {description} : {path}")
    return True


def validate_json_structure(path: Path, required_fields: list, description: str):
    """Valide la structure d'un fichier JSON"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            print(f"❌ ERREUR : Champs manquants dans {description} : {missing}")
            return False
        
        print(f"✅ Structure {description} valide")
        return True
    except Exception as e:
        print(f"❌ ERREUR : Impossible de valider {description} : {e}")
        return False


def main():
    print("=" * 70)
    print("TEST END-TO-END DU PIPELINE PAYSAGEA")
    print("=" * 70)
    
    success = True
    
    # ========== Étape 1 : Vérifier les inputs ==========
    print("\n📋 ÉTAPE 1 : Vérification des fichiers d'entrée")
    print("-" * 70)
    
    input_files = [
        (Path("Inputs/IMG_5177-535x356_preprocessed.jpg"), "Image prétraitée"),
        (Path("Inputs/IMG_5177-535x356_preprocessed.json"), "Métadonnées preprocessing"),
        (Path("Inputs/IMG_5177-535x356_preprocessed_sam_output.json"), "Sortie SAM"),
    ]
    
    for path, desc in input_files:
        if not check_file_exists(path, desc):
            success = False
    
    # ========== Étape 2 : Vérifier les outputs Depth ==========
    print("\n📋 ÉTAPE 2 : Vérification des sorties Depth Anything")
    print("-" * 70)
    
    depth_files = [
        (Path("Outputs/IMG_5177-535x356_depth.npy"), "Carte de profondeur"),
        (Path("Outputs/IMG_5177-535x356_depth.json"), "Métadonnées depth"),
        (Path("Outputs/IMG_5177-535x356_depth_preview.png"), "Preview depth"),
    ]
    
    for path, desc in depth_files:
        if not check_file_exists(path, desc):
            success = False
    
    # Valider structure depth.json
    if Path("Outputs/IMG_5177-535x356_depth.json").exists():
        validate_json_structure(
            Path("Outputs/IMG_5177-535x356_depth.json"),
            ["version", "image_id", "image_size", "depth_file", "near_is_one", "model"],
            "depth.json"
        )
    
    # ========== Étape 3 : Vérifier VisionOutput.json ==========
    print("\n📋 ÉTAPE 3 : Vérification de VisionOutput.json")
    print("-" * 70)
    
    vision_path = Path("VisionOutput.json")
    if check_file_exists(vision_path, "VisionOutput.json"):
        # Valider structure complète
        required_fields = [
            "version",
            "image_id",
            "image_size",
            "preprocess",
            "depth_meta",
            "sam_meta",
            "segments"
        ]
        
        if validate_json_structure(vision_path, required_fields, "VisionOutput.json"):
            # Vérifier qu'il y a des segments
            with open(vision_path, "r") as f:
                vision = json.load(f)
            
            segments = vision.get("segments", [])
            print(f"  → Nombre de segments : {len(segments)}")
            
            if len(segments) > 0:
                # Vérifier qu'un segment a bien depth_band
                sample = segments[0]
                if "mean_depth" in sample and "depth_band" in sample:
                    print(f"  → Segments enrichis avec profondeur ✅")
                    print(f"    Exemple : segment_id={sample['segment_id']}, "
                          f"mean_depth={sample.get('mean_depth'):.3f}, "
                          f"depth_band={sample.get('depth_band')}")
                else:
                    print("  ❌ Les segments ne sont pas enrichis avec la profondeur")
                    success = False
            else:
                print("  ❌ Aucun segment trouvé")
                success = False
    else:
        success = False
    
    # ========== Étape 4 : Vérifier PlantableZone.json ==========
    print("\n📋 ÉTAPE 4 : Vérification de PlantableZone.json")
    print("-" * 70)
    
    plantable_path = Path("PlantableZone.json")
    if check_file_exists(plantable_path, "PlantableZone.json"):
        required_fields = [
            "version",
            "image_id",
            "config",
            "plantable",
            "segment_analysis"
        ]
        
        if validate_json_structure(plantable_path, required_fields, "PlantableZone.json"):
            with open(plantable_path, "r") as f:
                plantable = json.load(f)
            
            # Statistiques
            total_segments = len(plantable.get("segment_analysis", []))
            plantable_count = plantable["plantable"]["segments_count"]
            coverage = plantable["plantable"]["coverage"]
            anchors_count = len(plantable["plantable"]["anchors"])
            
            print(f"  → Total segments analysés : {total_segments}")
            print(f"  → Segments plantables : {plantable_count} ({plantable_count/total_segments*100:.1f}%)")
            print(f"  → Coverage zone plantable : {coverage:.1%}")
            print(f"  → Nombre d'anchor points : {anchors_count}")
            
            # Vérifier qu'on a bien des anchors
            if anchors_count == 0:
                print("  ⚠️  Aucun anchor point généré")
            else:
                print(f"  ✅ Anchor points générés")
    else:
        success = False
    
    # ========== Étape 5 : Vérifier les visualisations ==========
    print("\n📋 ÉTAPE 5 : Vérification des visualisations")
    print("-" * 70)
    
    visual_files = [
        (Path("visuals/vision_segments_colored.png"), "Segments colorés"),
        (Path("visuals/vision_segments_depth_band.png"), "Depth bands"),
        (Path("visuals/vision_segments_depth_heatmap.png"), "Depth heatmap"),
        (Path("visuals/plantable_zones.png"), "Zones plantables"),
        (Path("visuals/plantable_mask_combined.png"), "Masque combiné"),
        (Path("visuals/plantable_anchors.png"), "Anchor points"),
        (Path("visuals/plantable_score_heatmap.png"), "Score heatmap"),
    ]
    
    for path, desc in visual_files:
        if not check_file_exists(path, desc):
            print(f"  ℹ️  {desc} non généré (optionnel)")
    
    # ========== Résumé ==========
    print("\n" + "=" * 70)
    print("RÉSUMÉ DU TEST")
    print("=" * 70)
    
    if success:
        print("✅ TOUS LES TESTS SONT PASSÉS")
        print("\n🎉 Le pipeline est opérationnel et prêt pour l'IA générative !")
        return 0
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("\n⚠️  Veuillez corriger les erreurs ci-dessus avant de continuer.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
