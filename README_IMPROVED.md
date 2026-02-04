# Paysagea — Pipeline Vision 2D/3D + Zone Plantable

Pipeline complet pour l'analyse de jardins par vision par ordinateur, développé dans le cadre de la **Clinique de l'IA** en partenariat avec **Paysagea**.

## 🎯 Objectif

À partir d'une image de jardin, ce pipeline :

1. **Génère une carte de profondeur** (Depth Anything)
2. **Segmente l'image** (SAM - Segment Anything Model)
3. **Fusionne les données** pour enrichir chaque segment avec sa profondeur 3D
4. **Identifie les zones plantables** pour l'IA générative

---

## 📊 Architecture du pipeline

```
Image brute
    ↓
Prétraitement (resize, orientation)
    ↓
    ├─ preprocessed.jpg
    └─ preprocessed.json (métadonnées)
    ↓
    ├─────────────────────┬────────────────────┐
    ↓                     ↓                    ↓
Depth Anything          SAM              (optionnel)
    ↓                     ↓
depth.npy            sam_output.json
depth.json
    ↓                     ↓
    └─────────────────────┴──→ FUSION
                              ↓
                        VisionOutput.json
                        (segments + profondeur)
                              ↓
                    compute_plantable_zone.py
                              ↓
                        PlantableZone.json
                        (zones plantables + anchors)
                              ↓
                          IA générative
```

---

## 🚀 Installation

```bash
# Créer l'environnement
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.\.venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📝 Utilisation

### Étape 1 : Génération de la carte de profondeur

```bash
python run_depth_paysagea.py \
  --img Inputs/IMG_5177-535x356_preprocessed.jpg \
  --meta Inputs/IMG_5177-535x356_preprocessed.json \
  --outdir Outputs \
  --near-is-one
```

**Sorties :**
- `*_depth.npy` - Carte de profondeur (float32, [0-1])
- `*_depth.json` - Métadonnées (modèle, convention)
- `*_depth_preview.png` - Visualisation (zones claires = proches)

### Étape 2 : Fusion SAM + Depth (améliorée)

```bash
python fuse_sam_depth_improved.py
```

**Sortie :** `VisionOutput.json` avec structure complète :

```json
{
  "version": "vision_segments_v1",
  "image_id": "sha256:...",
  "image_size": [535, 356],
  
  "preprocess": {
    "original_filename": "...",
    "resized_size": [535, 356],
    "orientation": "landscape"
  },
  
  "depth_meta": {
    "model": "LiheYoung/depth_anything_vitl14",
    "near_is_one": true,
    "depth_range": [0.0, 1.0],
    "normalized": true,
    "depth_file": "Outputs/IMG_5177-535x356_depth.npy"
  },
  
  "sam_meta": {
    "sam_file": "Inputs/IMG_5177-535x356_preprocessed_sam_output.json",
    "segments_count": 90
  },
  
  "segments": [
    {
      "segment_id": 0,
      "area_ratio": 0.2872,
      "bbox": [0.0, 0.5983, 1.0, 0.3989],
      "centroid": [0.4967, 0.8464],
      "mask_rle": {...},
      "mean_depth": 0.286,
      "depth_std": 0.169,
      "depth_band": "back"
    },
    ...
  ]
}
```

### Étape 3 : Identification des zones plantables ⭐ NOUVEAU

```bash
python compute_plantable_zone.py
```

**Critères de plantabilité :**

| Critère | Seuil | Justification |
|---------|-------|---------------|
| **Position** | centroid_y > 0.4 | Zone accessible (bas de l'image) |
| **Profondeur** | `front` ou `mid` | Proche/moyen (pas arrière-plan) |
| **Couleur** | green_ratio ≥ 0.3 **OU** brown_ratio ≥ 0.2 | Végétation ou sol |
| **Taille** | area_ratio > 0.5% | Suffisamment grand |

**Sortie :** `PlantableZone.json` :

```json
{
  "version": "plantable_zone_v1",
  "image_id": "sha256:...",
  "image_size": [535, 356],
  
  "config": {
    "position": {"min_y_centroid": 0.4},
    "depth": {"allowed_bands": ["front", "mid"]},
    "color": {
      "min_green_ratio": 0.3,
      "min_brown_ratio": 0.2
    }
  },
  
  "plantable": {
    "segments_count": 15,
    "segment_ids": [1, 2, 3, ...],
    "coverage": 0.42,
    "total_pixels": 80234,
    "mask_rle": {...},
    "anchors": [
      [0.123, 0.456],
      [0.234, 0.567],
      ...
    ]
  },
  
  "segment_analysis": [
    {
      "segment_id": 0,
      "is_plantable": false,
      "rejection_reasons": [
        "depth_too_far (band=back)",
        "position_too_high (y=0.28)"
      ],
      "color_features": {
        "green_ratio": 0.12,
        "brown_ratio": 0.05
      }
    },
    ...
  ]
}
```

### Étape 4 : Visualisations

```bash
# Visualisations de base (segments + profondeur)
python visualize_vision_output.py

# Visualisations des zones plantables
python visualize_plantable_zone.py
```

**Sorties dans `/visuals/` :**

**Vision générale :**
- `vision_segments_colored.png` - Segments en couleurs aléatoires
- `vision_segments_depth_band.png` - Rouge=proche, Orange=moyen, Bleu=loin
- `vision_segments_depth_heatmap.png` - Dégradé de profondeur

**Zones plantables :**
- `plantable_zones.png` - Vert=plantable, Rouge=non-plantable
- `plantable_mask_combined.png` - Masque fusionné des zones plantables
- `plantable_anchors.png` - Points d'ancrage pour l'IA (15 points)
- `plantable_score_heatmap.png` - Score de plantabilité par segment

---

## 🎨 Utilisation des anchors pour l'IA générative

Les **anchor points** servent de guides pour l'IA générative :

```python
# Exemple d'utilisation avec Stable Diffusion inpainting
import json

with open("PlantableZone.json") as f:
    plantable = json.load(f)

# Récupérer la zone et les anchors
mask_rle = plantable["plantable"]["mask_rle"]
anchors = plantable["plantable"]["anchors"]  # 15 points en coordonnées normalisées

# Prompt pour l'IA
prompt = "beautiful garden with flowers and shrubs"

# Utiliser le mask pour l'inpainting
# Les anchors servent à placer des éléments spécifiques
for x_norm, y_norm in anchors[:5]:  # 5 premiers points
    # Placer une plante à chaque anchor
    x_px = int(x_norm * width)
    y_px = int(y_norm * height)
    # ... logique de génération ...
```

---

## 📊 Format de données

### Convention de profondeur

⚠️ **IMPORTANT** : `near_is_one = true`
- **1.0** = proche (premier plan)
- **0.0** = lointain (arrière-plan)

### Bandes de profondeur

| Bande | Intervalle | Signification |
|-------|------------|---------------|
| `front` | [0.66, 1.0] | Premier plan (proche) |
| `mid` | [0.33, 0.66] | Plan moyen |
| `back` | [0.0, 0.33] | Arrière-plan (lointain) |

### Coordonnées normalisées

Tous les points sont en coordonnées `[0, 1]` :
- `[0, 0]` = coin haut-gauche
- `[1, 1]` = coin bas-droite
- `centroid: [0.5, 0.5]` = centre de l'image

---

## 🔧 Configuration avancée

### Ajuster les seuils de plantabilité

Éditez `compute_plantable_zone.py`, section `CONFIG` :

```python
CONFIG = {
    "position": {
        "min_y_centroid": 0.4,  # Plus bas = plus restrictif
    },
    "depth": {
        "allowed_bands": ["front", "mid"],  # Ajouter "back" si besoin
    },
    "color": {
        "min_green_ratio": 0.3,   # Seuil de détection vert
        "min_brown_ratio": 0.2,   # Seuil de détection brun
    },
    "size": {
        "min_area_ratio": 0.005,  # Segments > 0.5% de l'image
    },
    "anchors": {
        "num_points": 15,  # Nombre de points d'ancrage
    }
}
```

---

## 📦 Dépendances principales

- **torch** + **torchvision** - Modèles de deep learning
- **depth-anything-pytorch** - Estimation de profondeur
- **opencv-python** - Traitement d'images
- **numpy** - Calculs numériques
- **pycocotools** - Manipulation de masques RLE
- **transformers** - Chargement de modèles HuggingFace

---

## 🎓 Contexte académique

**Projet** : Clinique de l'IA  
**Partenaire** : Paysagea  
**Objectif** : Moderniser un outil de conception de jardins avec des briques d'IA générative et de vision par ordinateur

**Responsable brique** : Sabri Serradj (Depth Estimation / Vision 3D / Zone Plantable)

---

## 📋 Checklist de livraison

- [x] Génération de depth map (Depth Anything)
- [x] Fusion SAM + Depth avec métadonnées complètes
- [x] Identification des zones plantables (critères multiples)
- [x] Génération d'anchor points pour IA générative
- [x] Visualisations multiples (debug + production)
- [x] Documentation complète
- [ ] Intégration avec module d'IA générative (étape suivante)

---

## 🚨 Notes importantes

1. **Alignement pixel-à-pixel** : Toutes les données sont strictement alignées (depth, SAM, image)
2. **Pas de logique métier** : Le code fournit des primitives (géométrie + profondeur), pas d'interprétation sémantique
3. **Fichiers `.npy`** : Format binaire, ne pas ouvrir dans un éditeur texte
4. **Warnings HuggingFace** : Les warnings `xFormers` ou `HF_TOKEN` ne sont **pas bloquants**

---

## 🔜 Prochaines étapes

1. **Module IA générative** : Utiliser `PlantableZone.json` pour guider l'inpainting
2. **Raffinement des critères** : Ajuster les seuils selon les retours terrain
3. **Classification sémantique** : Distinguer pelouse / haie / allée (optionnel)
4. **Interface utilisateur** : Visualisation interactive des zones plantables

---

## 📞 Support

Pour toute question sur cette brique :
- **Auteur** : Sabri Serradj
- **Rôle** : Vision 3D / Depth Estimation / Zone Plantable
- **Projet** : Clinique de l'IA × Paysagea
